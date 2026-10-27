"""Opt-in, target-coordinate vocabulary reduction for the Qwen4 MTP head."""

import hashlib
import json
import math
import re
import weakref
from pathlib import Path

import torch

from sglang.srt.layers.logits_processor import (
    LogitsProcessor,
    should_apply_lm_head_quant_method,
)
from sglang.srt.layers.quantization.sm120_dense import is_sm120_int8_linear

_MAX_MAP_BYTES = 8 * 1024 * 1024
_SCHEMA = "qwen4-draft-vocab-v1"


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def _json_object(raw, label):
    try:
        value = json.loads(raw, object_pairs_hook=_unique_object)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid {label} JSON") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _local_file(path, filename):
    path = Path(path)
    return path / filename if path.is_dir() else path


def _positive_int(value, name):
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _validate_tokens(tokens, vocab_size):
    if not isinstance(tokens, list) or not 0 < len(tokens) < vocab_size:
        raise ValueError("token_ids must be a nonempty proper subset of the vocabulary")
    previous = -1
    for token in tokens:
        if type(token) is not int or not previous < token < vocab_size:
            raise ValueError("token_ids must be sorted unique integers in vocabulary range")
        previous = token


def load_draft_vocab_map(path, *, model_path, tokenizer_path) -> torch.Tensor:
    """Load a bounded JSON map bound to local, raw model/tokenizer file hashes.

    This never downloads assets or deserializes executable torch pickle data.
    """
    with Path(path).open("rb") as source:
        raw = source.read(_MAX_MAP_BYTES + 1)
    if len(raw) > _MAX_MAP_BYTES:
        raise ValueError("Draft vocabulary map exceeds 8 MiB")
    mapping = _json_object(raw, "draft vocabulary map")
    if mapping.get("schema") != _SCHEMA:
        raise ValueError(f"Draft vocabulary map schema must be {_SCHEMA}")
    if not isinstance(mapping.get("provenance"), dict):
        raise ValueError("Draft vocabulary map provenance must be an object")

    model_raw = _local_file(model_path, "config.json").read_bytes()
    tokenizer_raw = _local_file(tokenizer_path, "tokenizer.json").read_bytes()
    for name, contents in (
        ("model_config_sha256", model_raw),
        ("tokenizer_sha256", tokenizer_raw),
    ):
        digest = mapping.get(name)
        if not isinstance(digest, str) or re.fullmatch(r"[0-9a-fA-F]{64}", digest) is None:
            raise ValueError(f"{name} must be a SHA256 hex digest")
        if hashlib.sha256(contents).hexdigest() != digest.lower():
            raise ValueError(f"Draft vocabulary map {name} identity mismatch")

    model = _json_object(model_raw, "model config")
    text = model.get("text_config", model)
    if not isinstance(text, dict) or text.get("model_type") != "qwen4_exp_text":
        raise ValueError("Draft vocabulary map requires qwen4_exp_text model config")
    vocab_size = _positive_int(text.get("vocab_size"), "model vocab_size")
    hidden_size = _positive_int(text.get("hidden_size"), "model hidden_size")
    for name, expected in (("vocab_size", vocab_size), ("hidden_size", hidden_size)):
        if _positive_int(mapping.get(name), name) != expected:
            raise ValueError(f"Draft vocabulary map {name} mismatch")
    tokens = mapping.get("token_ids")
    _validate_tokens(tokens, vocab_size)

    required = set()

    def require_token(token):
        if type(token) is not int or not 0 <= token < vocab_size:
            raise ValueError("Required special token must be an integer in vocabulary range")
        required.add(token)

    tokenizer = _json_object(tokenizer_raw, "tokenizer")
    added = tokenizer.get("added_tokens", [])
    if not isinstance(added, list):
        raise ValueError("Tokenizer added_tokens must be a list")
    for entry in added:
        if not isinstance(entry, dict):
            raise ValueError("Tokenizer added_tokens entries must be objects")
        special = entry.get("special")
        if type(special) is not bool:
            raise ValueError("Tokenizer special flag must be boolean")
        if special:
            require_token(entry.get("id"))
    for config in (model, text) if text is not model else (model,):
        for name in ("bos_token_id", "eos_token_id"):
            value = config.get(name)
            if value is not None:
                for token in value if isinstance(value, list) else [value]:
                    require_token(token)
    missing = required.difference(tokens)
    if missing:
        raise ValueError(f"Draft vocabulary map is missing required special tokens: {sorted(missing)}")
    return torch.tensor(tokens, dtype=torch.int64, device="cpu")


def _tensor_version(tensor):
    # Serving weights may be created under inference_mode and have no counter.
    try:
        return tensor._version
    except RuntimeError:
        if tensor.is_inference():
            return None
        raise


class DraftVocabLogitsProcessor(LogitsProcessor):
    """Compute selected BF16 rows while retaining full target token coordinates.

    Only the row snapshot and IDs are buffers. The shared target head is held by
    weak references, never registered again as a child module or parameter.
    Mutating/replacing the source invalidates this snapshot instead of silently
    using stale rows. Inference tensors support identity/storage checks only.
    """

    def __init__(self, config, lm_head, token_ids, *, logit_scale=None):
        super().__init__(config, logit_scale=logit_scale)
        if (
            self.do_tensor_parallel_all_gather
            or self.do_tensor_parallel_all_gather_dp_attn
            or self.use_attn_tp_group
            or self.use_fp32_lm_head
            or self.rl_on_policy_target is not None
            or self.final_logit_softcapping is not None
        ):
            raise ValueError("Draft vocabulary head requires plain single-rank BF16 logits without softcap")
        self._validate_scale()
        self.hidden_size = _positive_int(config.hidden_size, "hidden_size")
        _positive_int(self.vocab_size, "vocab_size")
        self._validate_head(lm_head)
        if (
            not isinstance(token_ids, torch.Tensor)
            or token_ids.dtype != torch.int64
            or token_ids.device.type != "cpu"
            or token_ids.ndim != 1
        ):
            raise ValueError("Draft vocabulary token_ids must be a one-dimensional CPU int64 tensor")
        _validate_tokens(token_ids.tolist(), self.vocab_size)
        weight = lm_head.weight
        self._source_head = weakref.ref(lm_head)
        self._source_weight = weakref.ref(weight)
        self._source_version = _tensor_version(weight)
        self._source_data_ptr = weight.data_ptr()
        self._source_stride = weight.stride()
        self._source_device = weight.device
        # Independent, nonpersistent snapshots cannot be restored from an old
        # state_dict whose rows refer to a different target checkpoint.
        ids = token_ids.to(device=weight.device, copy=True)
        self.register_buffer("token_ids", ids, persistent=False)
        self.register_buffer(
            "selected_weight", weight.detach().index_select(0, ids), persistent=False
        )

    def _validate_scale(self):
        scale = self.logit_scale
        if scale is not None and (
            isinstance(scale, bool)
            or not isinstance(scale, (float, int))
            or not math.isfinite(scale)
            or scale <= 0
        ):
            raise ValueError("Draft vocabulary logit_scale must be finite and positive")

    def _validate_head(self, lm_head):
        weight = getattr(lm_head, "weight", None)
        if (
            not isinstance(weight, torch.Tensor)
            or weight.dtype != torch.bfloat16
            or weight.layout != torch.strided
            or weight.device.type not in ("cpu", "cuda")
            or tuple(weight.shape) != (self.vocab_size, self.hidden_size)
            or not weight.is_contiguous()
        ):
            raise ValueError("Draft vocabulary head requires a full-width contiguous resident BF16 weight")
        if (
            is_sm120_int8_linear(lm_head)
            or should_apply_lm_head_quant_method(lm_head, getattr(lm_head, "quant_method", None))
            or hasattr(lm_head, "set_lora")
            or hasattr(lm_head, "apply_lora")
            or getattr(lm_head, "bias", None) is not None
        ):
            raise ValueError("Draft vocabulary head does not support active quantization, LoRA, or bias")

    def _compute_lm_head(self, hidden_states, lm_head, embedding_bias=None):
        self._validate_head(lm_head)
        self._validate_scale()
        weight = lm_head.weight
        if (
            lm_head is not self._source_head()
            or weight is not self._source_weight()
            or _tensor_version(weight) != self._source_version
            or weight.data_ptr() != self._source_data_ptr
            or weight.stride() != self._source_stride
            or weight.device != self._source_device
        ):
            raise RuntimeError("Draft vocabulary source head changed; rebuild the selected rows")
        if embedding_bias is not None:
            raise ValueError("Draft vocabulary head does not support embedding_bias")
        if (
            hidden_states.dtype != torch.bfloat16
            or hidden_states.device != weight.device
            or hidden_states.ndim != 2
            or hidden_states.shape[-1] != self.hidden_size
            or self.selected_weight.dtype != torch.bfloat16
            or self.selected_weight.device != weight.device
            or self.token_ids.dtype != torch.int64
            or self.token_ids.device != weight.device
        ):
            raise ValueError("Draft vocabulary head requires colocated BF16 activations and row buffers")
        selected_logits = torch.matmul(hidden_states, self.selected_weight.T)
        logits = selected_logits.new_full((hidden_states.shape[0], self.vocab_size), -torch.inf)
        logits.index_copy_(1, self.token_ids, selected_logits)
        return logits
