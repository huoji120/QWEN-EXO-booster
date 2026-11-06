"""Chunked native Qwen4Exp QSA selection for unpadded training and cached decoding.

The dense SDPA/eager mask and native block budget are unchanged. Only the
piecewise-constant index selection runs without autograd; attention is untouched.
Both paths rank equal-score blocks by ascending block index using a stable sort.
This deliberately replaces ambiguous native/previous Top-K ties, not QSA scores.
"""
from __future__ import annotations

import math
import types
import weakref

import torch


class _CausalMaskContract:
    """Share one verified tensor/version across all indexers without retaining it."""

    def __init__(self, chunk_size):
        self.chunk_size = chunk_size
        self.reference = None
        self.version = None

    def check(self, mask, hidden, kv_length):
        batch, query_length, _ = hidden.shape
        if (not isinstance(mask, torch.Tensor)
                or mask.shape != (batch, 1, query_length, kv_length)
                or mask.device != hidden.device
                or not (mask.dtype == torch.bool or mask.is_floating_point())):
            raise ValueError("chunked_qsa requires a [B,1,Q,N] bool/float mask on the hidden-state device")
        # Inference tensors have no version counter: validate but never cache.
        try:
            version = mask._version
        except RuntimeError:
            version = None
        if (version is not None and self.reference is not None
                and self.reference() is mask and self.version == version):
            return
        keys = torch.arange(kv_length, device=mask.device)
        query_offset = kv_length - query_length
        valid = torch.ones((), dtype=torch.bool, device=mask.device)
        for start in range(0, query_length, self.chunk_size):
            stop = min(start + self.chunk_size, query_length)
            queries = keys[query_offset + start:query_offset + stop]
            visible = mask[:, 0, start:stop]
            if mask.dtype != torch.bool:
                visible = visible == 0
            valid.logical_and_((visible == (keys[None, :] <= queries[:, None])).all())
        # A single device synchronization per new mask/version, not per layer or
        # query. Never silently reinterpret padding, packed sequences or holes.
        if not bool(valid):
            raise ValueError("chunked_qsa requires an unpadded contiguous causal mask")
        self.reference = weakref.ref(mask)
        self.version = version


def install_chunked_qsa(model, native, query_chunk_size=128) -> dict:
    """Replace existing indexer forward methods, retaining all parameter owners.

    ``native`` is transformers.models.qwen4_exp.modeling_qwen4_exp. Fresh inputs
    cover the complete unpadded sequence; cached inputs cover its contiguous tail.
    Native update_indexer owns raw key history and rotary inputs cover all keys.
    Masks are checked once per tensor/version. Padding/holes fail closed.
    """
    if (isinstance(query_chunk_size, bool) or not isinstance(query_chunk_size, int)
            or query_chunk_size <= 0):
        raise ValueError("query_chunk_size must be a positive integer")
    indexers = [module for module in model.modules()
                if isinstance(module, native.Qwen4ExpTextQSAIndexer)]
    for module in indexers:
        if (module.index_kv_heads != 1 or module.compress_ratio <= 0
                or module.block_topk < 0 or module.index_n_heads <= 0
                or module.index_head_dim <= 0
                or module.block_topk != module.token_budget // module.compress_ratio):
            raise ValueError("unsupported native QSA indexer geometry")
    contract = _CausalMaskContract(query_chunk_size)
    rotary = native.apply_rotary_pos_emb

    @torch.no_grad()
    def select(self, hidden_states, position_embeddings, attention_mask, past_key_values=None):
        if hidden_states.ndim != 3 or min(hidden_states.shape[:2]) <= 0:
            raise ValueError("chunked_qsa requires nonempty [B,Q,H] hidden states")
        batch, query_length, _ = hidden_states.shape
        cos, sin = position_embeddings
        if (cos.ndim != 3 or sin.shape != cos.shape or cos.shape[0] != batch
                or cos.shape[1] < query_length
                or cos.device != hidden_states.device or sin.device != hidden_states.device):
            raise ValueError("chunked_qsa requires full-key [B,N,D] rotary positions")
        kv_length = cos.shape[1]
        query_offset = kv_length - query_length
        if past_key_values is None and query_offset != 0:
            raise ValueError("chunked_qsa fresh input must cover every rotary position")
        contract.check(attention_mask, hidden_states, kv_length)
        ratio, dim = self.compress_ratio, self.index_head_dim
        qk = self.index_qk_proj(hidden_states)
        q, raw_keys = torch.split(qk, [self.index_n_heads * dim, dim], dim=-1)
        q = self.q_layernorm(q.reshape(batch, query_length, self.index_n_heads, dim))
        q = rotary(q, cos=cos[:, query_offset:], sin=sin[:, query_offset:], unsqueeze_dim=2)
        if past_key_values is not None:
            raw_keys = past_key_values.update_indexer(raw_keys, self.layer_idx)
        if raw_keys.shape != (batch, kv_length, dim):
            raise ValueError("chunked_qsa indexer cache length must match full rotary positions")
        blocks = kv_length // ratio
        if blocks:
            pooled = raw_keys[:, :blocks * ratio].reshape(batch, blocks, ratio, dim).float().mean(2)
            pooled = self.k_layernorm(pooled.to(raw_keys.dtype))
            pooled = rotary(pooled.unsqueeze(2), cos=cos[:, :blocks * ratio:ratio],
                            sin=sin[:, :blocks * ratio:ratio], unsqueeze_dim=2).squeeze(2)
            block_keys = pooled.float().transpose(-1, -2).unsqueeze(1)
        result = torch.empty((batch, 1, query_length, kv_length), dtype=attention_mask.dtype,
                             device=hidden_states.device)
        positions = torch.arange(kv_length, device=hidden_states.device)
        block_ids = torch.arange(blocks, device=hidden_states.device)
        topk = min(self.block_topk, blocks)
        for start in range(0, query_length, query_chunk_size):
            stop = min(start + query_chunk_size, query_length)
            queries = positions[query_offset + start:query_offset + stop]
            complete = (queries + 1) // ratio
            # The incomplete causal tail is always included, even with budget 0.
            chosen = ((positions[None, :] >= (complete * ratio)[:, None])
                      & (positions[None, :] <= queries[:, None]))
            chosen = chosen.unsqueeze(0).expand(batch, -1, -1).clone()
            if topk:
                visible_blocks = block_ids[None, :] < complete[:, None]
                if topk == blocks:
                    selected = visible_blocks.unsqueeze(0).expand(batch, -1, -1)
                else:
                    scores = torch.matmul(q[:, start:stop].float().transpose(1, 2), block_keys)
                    scores = scores.relu_().sum(1) / math.sqrt(dim)
                    scores.masked_fill_(~visible_blocks.unsqueeze(0), -torch.inf)
                    # Columns are ascending block IDs. Stable descending sort
                    # preserves that order for ties regardless of invisible future
                    # columns or whether this query is a fresh/prefill/decode row.
                    indices = scores.argsort(dim=-1, descending=True, stable=True)[..., :topk]
                    selected = torch.zeros((batch, stop - start, blocks), dtype=torch.bool,
                                           device=hidden_states.device)
                    selected.scatter_(-1, indices, True)
                    selected.logical_and_(visible_blocks.unsqueeze(0))
                chosen[:, :, :blocks * ratio].logical_or_(selected.repeat_interleave(ratio, dim=-1))
            if attention_mask.dtype == torch.bool:
                result[:, 0, start:stop] = chosen
            else:
                result[:, 0, start:stop] = torch.where(
                    chosen, attention_mask.new_zeros(()), torch.finfo(attention_mask.dtype).min)
        return result

    def forward(self, hidden_states, position_embeddings, attention_mask, past_key_values=None):
        # Check the caller's gradient mode before the selection-only no-grad
        # region; cached training must fail before mutating native key history.
        if past_key_values is not None and torch.is_grad_enabled():
            raise ValueError("chunked_qsa cache requires no-grad decoding")
        return select(self, hidden_states, position_embeddings, attention_mask, past_key_values)

    for module in indexers:
        module.forward = types.MethodType(forward, module)
    return {"implementation": "chunked_native_qsa", "indexers": len(indexers),
            "query_chunk_size": query_chunk_size,
            "mask_contract": "verified_unpadded_causal_fresh_or_cached_tail",
            "selection_autograd": False, "topk_ties": "stable_descending_score_then_ascending_block_index",
            "indexer_cache": "native_update_indexer_full_rotary_positions_no_grad_only"}
