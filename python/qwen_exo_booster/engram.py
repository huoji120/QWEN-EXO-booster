"""Engram: a hashed n-gram table in host memory, read into the residual stream.

Each token position t is keyed by its bigram and trigram (the token plus the
one or two tokens before it). Every n-gram order has ``heads_per_order``
hash heads; each head picks one row of a large table that lives in pinned host
memory (never VRAM). The concatenated rows ``e_t`` feed a small reader that adds
a gated delta to the residual stream at one decoder layer:

    e = RMSNorm(e_t);  k, v = W_k e, W_v e
    g = sigmoid(<RMSNorm(h), RMSNorm(k)> / sqrt(d));  h <- h + g * v

The hash is Qwen3.8-Flash-Next's PLE hash (bit-exact with transformers'
``Qwen4ExpTextNGramEmbedding`` for ngram_size 3): raw token ids; positions
before the sequence start are EOS; an EOS ends the context (if the previous
token is EOS the one before it is EOS too);
``mixed = x0*m0 ^ x1*m1 [^ x2*m2]`` in wrapping int64; the heads of one order
take ``mixed`` modulo distinct sizes and add their row offset.

Token history on the GPU: decode and speculative verify inputs are only on the
device when a batch is built, so each request keeps its recent token ids in a
ring ``ring[req_pool_idx, position % R]``. A forward first writes its own
input tokens there and then reads positions ``p-1`` and ``p-2``: rejected draft
tokens past the commit point get overwritten by the next block before any
position after them is read. Extend batches get exact triples from the host.

This module is pure torch/numpy so it can be tested on CPU; the serving
collaborator lives in ``sglang.srt.model_executor.model_runner_components``.
"""

from __future__ import annotations

from typing import Any, Optional, Sequence

import msgspec
import numpy as np
import torch

ENGRAM_PARAM = "qwen_exo_engram"
ENGRAM_DISABLED_CACHE_MARKER = "qwen-exo-engram=off"
ENGRAM_MIN_RING_WIDTH = 16


def engram_requested(custom_params: Any) -> bool:
    """Engram is on unless the request sends ``qwen_exo_engram: false``."""
    return not (isinstance(custom_params, dict) and custom_params.get(ENGRAM_PARAM) is False)


def engram_radix_extra_key(extra_key: Optional[str], custom_params: Any) -> Optional[str]:
    """Give opted-out requests their own radix namespace.

    Prefix KV and recurrent state computed with Engram must not be reused by a
    request that runs without it, and vice versa.
    """
    if engram_requested(custom_params):
        return extra_key
    if not extra_key:
        return ENGRAM_DISABLED_CACHE_MARKER
    if ENGRAM_DISABLED_CACHE_MARKER in extra_key.split("|"):
        return extra_key
    return f"{extra_key}|{ENGRAM_DISABLED_CACHE_MARKER}"


class EngramHashSpec(msgspec.Struct, frozen=True, kw_only=True):
    vocab_size: int
    eos_id: int
    ngram_size: int
    heads_per_order: int
    multipliers: tuple[int, ...]
    head_sizes: tuple[int, ...]
    head_offsets: tuple[int, ...]

    @property
    def num_heads(self) -> int:
        return (self.ngram_size - 1) * self.heads_per_order

    def validate(self) -> None:
        # The EOS rule in hash_rows ("x2 = EOS if x1 = EOS") equals the general
        # segment rule only for trigrams.
        if self.ngram_size != 3:
            raise ValueError(f"Engram hash supports ngram_size 3, got {self.ngram_size}")
        if len(self.multipliers) != self.ngram_size:
            raise ValueError("Engram hash needs one multiplier per n-gram position")
        if len(self.head_sizes) != self.num_heads or len(self.head_offsets) != self.num_heads:
            raise ValueError("Engram hash needs one size and offset per head")
        if not 0 <= self.eos_id < self.vocab_size:
            raise ValueError(f"Engram EOS id {self.eos_id} outside the vocabulary")


class EngramHashTensors:
    """Device copies of the hash constants (built once per device)."""

    def __init__(self, spec: EngramHashSpec, device: torch.device | str):
        self.eos_id = spec.eos_id
        self.heads_per_order = spec.heads_per_order
        self.multipliers = torch.tensor(spec.multipliers, dtype=torch.int64, device=device)
        self.sizes = torch.tensor(spec.head_sizes, dtype=torch.int64, device=device)
        self.offsets = torch.tensor(spec.head_offsets, dtype=torch.int64, device=device)


def hash_rows(x0: torch.Tensor, x1: torch.Tensor, x2: torch.Tensor, h: EngramHashTensors) -> torch.Tensor:
    """Token ids [N] (current, previous, the one before) -> global table rows [N, heads].

    Pure tensor ops: safe inside CUDA graph capture and identical on CPU and GPU.
    """
    x0, x1, x2 = x0.long(), x1.long(), x2.long()
    x2 = x2.masked_fill(x1 == h.eos_id, h.eos_id)
    mixed2 = (x0 * h.multipliers[0]) ^ (x1 * h.multipliers[1])
    mixed3 = mixed2 ^ (x2 * h.multipliers[2])
    mixed = torch.stack([mixed2, mixed3], dim=-1).repeat_interleave(h.heads_per_order, dim=-1)
    return torch.remainder(mixed, h.sizes) + h.offsets


def segment_reference_rows(tokens: torch.Tensor, spec: EngramHashSpec) -> torch.Tensor:
    """Rows of a whole sequence [L] -> [L, heads], following the official
    Qwen4ExpTextNGramEmbedding segment rule for any ngram_size (independent of
    ``hash_rows``; used to build and test the self-check rows)."""
    tokens = tokens.long()
    eos = spec.eos_id
    pos = torch.arange(tokens.shape[0])
    last_eos = torch.where(tokens == eos, pos, torch.full_like(pos, -1)).cummax(0).values
    seg_start = torch.cat([torch.tensor([-1]), last_eos[:-1]]) + 1

    def shifted(k: int) -> torch.Tensor:
        if k == 0:
            return tokens
        src = torch.cat([torch.full((k,), eos), tokens[:-k]])
        return torch.where((pos - seg_start >= k) & (pos >= k), src, torch.full_like(src, eos))

    sh = [shifted(k) for k in range(spec.ngram_size)]
    sizes = torch.tensor(spec.head_sizes, dtype=torch.int64)
    offsets = torch.tensor(spec.head_offsets, dtype=torch.int64)
    blocks = []
    for order in range(2, spec.ngram_size + 1):
        mixed = sh[0] * spec.multipliers[0]
        for k in range(1, order):
            mixed = mixed ^ (sh[k] * spec.multipliers[k])
        lo = (order - 2) * spec.heads_per_order
        hi = lo + spec.heads_per_order
        blocks.append(torch.remainder(mixed.unsqueeze(-1), sizes[lo:hi]) + offsets[lo:hi])
    return torch.cat(blocks, dim=-1)


def sanitize_token_ids(ids: np.ndarray, *, vocab_size: int, eos_id: int) -> np.ndarray:
    """Out-of-vocabulary ids (multimodal placeholders) end the n-gram context."""
    return np.where((ids < 0) | (ids >= vocab_size), eos_id, ids)


def is_injectable_token(text: str) -> bool:
    """Whitelist: Engram injects only on natural-language tokens, i.e. the token
    is non-empty and every visible character is a letter (CJK included) or
    whitespace. Digits, punctuation and markup -- IP octets, version numbers,
    commit hashes, ``<tool_call>``, code symbols -- are suppressed, where an
    Engram delta only flips predictions the model is already confident about
    (the 10.0.186.74 and <> regressions)."""
    stripped = text.strip()
    if not stripped:
        return False
    return all(ch.isalpha() or ch.isspace() for ch in stripped)


class EngramExtendInputs(msgspec.Struct, kw_only=True):
    """Per-forward Engram inputs of an EXTEND batch (device tensors).

    ``tokens`` rows are (x0, x1, x2) for every extend token. ``tail_*`` seed the
    token ring with each request's last two positions for the following decode.
    """

    tokens: torch.Tensor  # [3, T] int64
    token_mask: torch.Tensor  # [T] bool
    tail_rows: torch.Tensor  # [K] int64 req_pool_idx
    tail_cols: torch.Tensor  # [K] int64 position % ring width
    tail_tokens: torch.Tensor  # [K] int64


class EngramExtendHost(msgspec.Struct, kw_only=True):
    tokens: np.ndarray
    token_mask: np.ndarray
    tail_rows: np.ndarray
    tail_cols: np.ndarray
    tail_tokens: np.ndarray


def extend_inputs_host(
    *,
    fill_ids: Sequence[Sequence[int]],
    req_pool_indices: Sequence[int],
    prefix_lens: Sequence[int],
    extend_lens: Sequence[int],
    request_mask: Sequence[bool],
    ring_width: int,
    vocab_size: int,
    eos_id: int,
) -> EngramExtendHost:
    """Exact (x0, x1, x2) triples of the extend tokens plus the ring seed.

    ``fill_ids[i][prefix_lens[i]:prefix_lens[i] + extend_lens[i]]`` are request
    i's extend tokens (prefix and chunked-prefill continuations included).
    """
    triples, masks, tail_rows, tail_cols, tail_tokens = [], [], [], [], []
    for ids, row, pre, ext, on in zip(fill_ids, req_pool_indices, prefix_lens, extend_lens, request_mask):
        start = max(0, pre - 2)
        seg = np.asarray(ids[start : pre + ext], dtype=np.int64)
        seg = sanitize_token_ids(seg, vocab_size=vocab_size, eos_id=eos_id)
        seq = np.concatenate([np.full(2 - (pre - start), eos_id, dtype=np.int64), seg])
        triples.append(np.stack([seq[2:], seq[1:-1], seq[:-2]]))
        masks.append(np.full(ext, bool(on)))
        # seq[j] holds the token at position pre - 2 + j.
        for pos in (pre + ext - 2, pre + ext - 1):
            if pos >= 0:
                tail_rows.append(row)
                tail_cols.append(pos % ring_width)
                tail_tokens.append(seq[pos - pre + 2])
    return EngramExtendHost(
        tokens=np.concatenate(triples, axis=1) if triples else np.zeros((3, 0), np.int64),
        token_mask=np.concatenate(masks) if masks else np.zeros(0, bool),
        tail_rows=np.asarray(tail_rows, dtype=np.int64),
        tail_cols=np.asarray(tail_cols, dtype=np.int64),
        tail_tokens=np.asarray(tail_tokens, dtype=np.int64),
    )


def ring_width_for(tokens_per_req: int) -> int:
    """Power of two holding a verify block plus the two tokens before it."""
    need = max(ENGRAM_MIN_RING_WIDTH, tokens_per_req + 2)
    return 1 << (need - 1).bit_length()


def ring_update_and_read(
    *,
    ring: torch.Tensor,
    req_pool_indices: torch.Tensor,
    positions: torch.Tensor,
    input_ids: torch.Tensor,
    eos_id: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Decode / verify: record this forward's tokens, then read their history.

    ``positions`` and ``input_ids`` are [bs * width] with a uniform width per
    request (1 for decode, the verify block for DFLASH). Padded CUDA-graph rows
    use req_pool_idx 0, a slot the request pool never hands out.
    """
    bs = req_pool_indices.shape[0]
    wrap = ring.shape[1] - 1
    pos = positions.view(bs, -1).long()
    tok = input_ids.view(bs, -1).long()
    req = req_pool_indices.long().view(bs, 1).expand_as(pos)
    ring[req, pos & wrap] = tok
    x1 = torch.where(pos >= 1, ring[req, (pos - 1) & wrap], eos_id)
    x2 = torch.where(pos >= 2, ring[req, (pos - 2) & wrap], eos_id)
    return tok.reshape(-1), x1.reshape(-1), x2.reshape(-1)


class EngramReaderWeights(msgspec.Struct, kw_only=True):
    """Linear reader (fp32, as trained): key and value projections stacked."""

    in_norm: torch.Tensor  # [embed_dim]
    kv: torch.Tensor  # [2 * hidden, embed_dim]
    h_norm: torch.Tensor  # [hidden]
    k_norm: torch.Tensor  # [hidden]
    eps: float

    @classmethod
    def from_state_dict(cls, state: dict[str, torch.Tensor], *, eps: float, device) -> "EngramReaderWeights":
        def get(name):
            return state[name].to(device=device, dtype=torch.float32).contiguous()

        return cls(
            in_norm=get("in_norm.weight"),
            kv=torch.cat([get("key.weight"), get("value.weight")]).contiguous(),
            h_norm=get("h_norm.weight"),
            k_norm=get("k_norm.weight"),
            eps=eps,
        )

    @property
    def hidden_size(self) -> int:
        return self.h_norm.shape[0]


def _rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * weight


def reader_delta(h: torch.Tensor, e: torch.Tensor, w: EngramReaderWeights) -> torch.Tensor:
    """Gated residual delta [N, hidden] in ``h``'s dtype (fp32 math, as trained)."""
    hidden = w.hidden_size
    e32 = _rms_norm(e.float(), w.in_norm, w.eps)
    k, v = (e32 @ w.kv.T).split(hidden, dim=-1)
    score = (_rms_norm(h.float(), w.h_norm, w.eps) * _rms_norm(k, w.k_norm, w.eps)).sum(-1, keepdim=True)
    gate = torch.sigmoid(score * hidden**-0.5)
    return (gate * v).to(h.dtype)


def validate_engram_server_config(
    *,
    enable_qwen_exo: bool,
    tp_size: int,
    pp_size: int,
    dp_size: int,
    enable_dp_attention: bool,
    enable_two_batch_overlap: bool,
    enable_mixed_chunk: bool,
    enable_torch_compile: bool,
    speculative_algorithm: Optional[str],
    cuda_graph_backend_prefill: Optional[str],
) -> None:
    """Configurations the Engram path does not support yet raise here."""
    problems = []
    if not enable_qwen_exo:
        problems.append("requires --enable-qwen-exo")
    if tp_size != 1 or pp_size != 1 or dp_size != 1 or enable_dp_attention:
        # TP>1 all-reduce fusion drops post_residual_addition and leaves a
        # partial residual; PP/DP are untested.
        problems.append("requires tp/pp/dp size 1 without DP attention")
    if enable_two_batch_overlap:
        problems.append("does not support two-batch overlap")
    if enable_mixed_chunk:
        problems.append("does not support --enable-mixed-chunk (decode rows need the device ring)")
    if enable_torch_compile:
        problems.append("does not support --enable-torch-compile")
    if speculative_algorithm is not None and str(speculative_algorithm).upper() != "DFLASH":
        problems.append("supports no speculative decoding or DFLASH only (tree verify breaks the ring)")
    if cuda_graph_backend_prefill not in (None, "disabled"):
        problems.append("runs prefill eagerly; drop --cuda-graph-backend-prefill or set it to disabled")
    if problems:
        raise ValueError("--qwen-exo-engram-path " + "; ".join(problems))


__all__ = [
    "ENGRAM_DISABLED_CACHE_MARKER",
    "ENGRAM_PARAM",
    "EngramExtendHost",
    "EngramExtendInputs",
    "EngramHashSpec",
    "EngramHashTensors",
    "EngramReaderWeights",
    "engram_radix_extra_key",
    "engram_requested",
    "extend_inputs_host",
    "hash_rows",
    "is_injectable_token",
    "reader_delta",
    "ring_update_and_read",
    "ring_width_for",
    "sanitize_token_ids",
    "segment_reference_rows",
    "validate_engram_server_config",
]
