"""Offline fresh-sequence HC-prefix caching and frozen-NVFP4 mid-layer LoRA.

This is the native HF W4A16 reference, not a serving W4A4/hot-load integration.
The prefix collector runs the ordinary native model under no_grad and stops at
an inter-layer boundary. It is not an optimized online feature-capture hook.
Only the shared expert down projection is adapted; routed experts stay frozen.
No function in this module creates an optimizer or starts a training job.
"""
from __future__ import annotations

import hashlib
import json
import math
import uuid
from contextlib import nullcontext
from dataclasses import asdict, dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from qwen_exo_booster.native_ple_knowledge import NativePLEIdentity, SparseNativePLEDelta
from qwen_exo_booster.native_ple_training_backend import (
    _checkpoint_contexts,
    _tiny_native_ple_fixture,
    _window_context,
    chunked_causal_cross_entropy,
    load_native_ple_training_model,
)

_TARGET = "mlp.shared_expert.down_proj"
_ADAPTER_SCHEMA = "native-mid-lora-shared-down-v2-fp32"


@dataclass(frozen=True)
class MidLoraConfig:
    """Inclusive, zero-based layer range; one shared down projection per layer."""

    first_layer: int = 16
    last_layer: int = 31
    rank: int = 8
    alpha: float = 16.0
    gradient_checkpointing: bool = True
    checkpoint_group_size: int = 1
    offload_checkpoints: bool = False

    def validate(self, num_layers: int) -> None:
        if (type(self.first_layer) is not int or type(self.last_layer) is not int
                or not 0 <= self.first_layer <= self.last_layer < num_layers):
            raise ValueError("mid_lora_layer_range_invalid")
        if type(self.rank) is not int or self.rank < 1:
            raise ValueError("mid_lora_rank_invalid")
        if not isinstance(self.alpha, (int, float)) or not math.isfinite(self.alpha) or self.alpha <= 0:
            raise ValueError("mid_lora_alpha_invalid")
        if type(self.gradient_checkpointing) is not bool:
            raise ValueError("mid_lora_checkpoint_flag_invalid")
        if type(self.checkpoint_group_size) is not int or not 1 <= self.checkpoint_group_size <= num_layers:
            raise ValueError("mid_lora_checkpoint_group_size_invalid")
        if type(self.offload_checkpoints) is not bool:
            raise ValueError("mid_lora_offload_checkpoints_invalid")


def _targets(config: MidLoraConfig) -> list[str]:
    return [f"model.layers.{i}.{_TARGET}" for i in range(config.first_layer, config.last_layer + 1)]


def _zero_delta(identity: NativePLEIdentity, device="cpu") -> SparseNativePLEDelta:
    # The existing strict loader requires a nonempty sparse table. This one-row
    # zero delta is permanently disabled, including checkpoint recomputation.
    return SparseNativePLEDelta(torch.tensor([0], dtype=torch.int64), identity, device=device)


def inspect_mid_lora_profile(profile, config: MidLoraConfig = MidLoraConfig()) -> dict:
    """Validate native inventory on meta/CPU, without materializing the backbone."""
    from qwen_exo_booster.native_ple_checkpoint import IndexedCheckpoint

    root = Path(profile).resolve()
    raw = json.loads((root / "config.json").read_text(encoding="utf-8"))["text_config"]
    config.validate(raw["num_hidden_layers"])
    identity = NativePLEIdentity.from_profile(root)
    report = load_native_ple_training_model(root, _zero_delta(identity), metadata_only=True,
                                            gradient_checkpointing=False)
    source = IndexedCheckpoint(root)
    try:
        targets = []
        for target in _targets(config):
            source_name = "model.language_model." + target[6:] + ".weight"
            info = source.metadata(source_name)
            shape = info["shape"]
            if (len(shape) != 2 or shape[0] != raw["hidden_size"]
                    or shape[1] != raw["shared_expert_intermediate_size"]
                    or info["dtype"] not in {"BF16", "F16", "F32"}):
                raise ValueError(f"mid_lora_target_not_ordinary_shared_projection: {target}")
            targets.append({"name": target, "shape": list(shape), "dtype": info["dtype"],
                            "adapter_parameters": config.rank * sum(shape)})
    finally:
        source.close()
    return {"profile": str(root), "identity": identity.to_dict(),
            "identity_fingerprint": identity.fingerprint(), "config": asdict(config),
            "num_hidden_layers": raw["num_hidden_layers"], "hidden_size": raw["hidden_size"],
            "hc_count": raw["hc_count"], "targets": targets,
            "trainable_parameters": sum(t["adapter_parameters"] for t in targets),
            "native_inventory": report, "boundary": "fresh-sequence pre-layer full HC plus native kwargs",
            "prefix_collection": "reference no_grad; native masks may be quadratic; not an online hook",
            "suffix_storage": "lower ordinary layers released; frozen experts streamed from checkpoint",
            "weight_reference": "frozen NVFP4 decoded W4A16, not serving W4A4",
            "training_started": False, "gpu_validated": False, "training_ready": False}


def load_mid_lora_reference(profile, *, device="cpu", dtype=torch.bfloat16):
    """Load the strict native reference for collecting one or more fresh windows.

    The default destructive suffix requires collecting windows first. The
    persistent builder instead borrows this complete reference for current-adapter
    generation and new prefix collection after updates, retaining lower weights.
    Neither path implies that a full-profile/32K allocation fits the target GPU.
    """
    identity = NativePLEIdentity.from_profile(profile)
    model = load_native_ple_training_model(profile, _zero_delta(identity, device), device=device,
                                           dtype=dtype, gradient_checkpointing=False)
    model.requires_grad_(False)
    model.eval()
    model.set_ple_window(mode="off")
    return model


def _tensor_digest(value: torch.Tensor) -> str:
    return hashlib.sha256(value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def _snapshot(value):
    if isinstance(value, torch.Tensor):
        return value.detach()
    if isinstance(value, (tuple, list)):
        return tuple(_snapshot(v) for v in value)
    if isinstance(value, dict):
        return MappingProxyType({k: _snapshot(v) for k, v in value.items()})
    if value is None or type(value) in (str, int, float, bool):
        return value
    raise ValueError(f"mid_lora_unsupported_native_boundary_value: {type(value).__name__}")


def _tensor_versions(value):
    if isinstance(value, torch.Tensor):
        return ((value, value._version),)
    if isinstance(value, Mapping):
        return tuple(pair for item in value.values() for pair in _tensor_versions(item))
    if isinstance(value, tuple):
        return tuple(pair for item in value for pair in _tensor_versions(item))
    return ()


def _prefix_revision(model, first_layer):
    # No parameter references are retained: lower tensors must be releasable.
    modules = [model.model.embed_tokens, model.model.rotary_emb, *model.model.layers[:first_layer]]
    return tuple((id(tensor), tensor._version) for module in modules
                 for tensor in (*module.parameters(), *module.buffers()) if tensor.device.type != "meta")


@dataclass(frozen=True)
class CompletePrefixState:
    """In-memory, immutable-by-contract cache owned by one frozen reference.

    Full native HF inter-layer HC is [batch, tokens, hc_count * hidden_size].
    HF has already completed residual injection here: no deferred SGLang tuple.
    This is a fresh full sequence, NOT a KV/GDN/PLE incremental continuation.
    Never mutate tensors through .data or reuse with another base/sequence.
    """

    hidden_states: torch.Tensor
    layer_kwargs: Mapping[str, Any]
    input_ids: torch.Tensor
    ple_history: torch.Tensor | None
    first_layer: int
    identity_fingerprint: str
    input_sha256: str
    history_sha256: str | None
    reference_session: str
    prefix_revision: tuple
    _versions: tuple = field(repr=False)


def collect_prefix(reference, input_ids: torch.Tensor, *, first_layer=16,
                   ple_history: torch.Tensor | None = None) -> CompletePrefixState:
    """Capture exact native kwargs, stopping before the first trainable layer.

    Only batch-one, unpadded fresh sequences are accepted. Native mask and RoPE
    creation are intentionally reused rather than approximated. In particular,
    native reference masks can be quadratic: 32K capacity is a separate GPU gate.
    """
    if getattr(reference, "_mid_lora_consumed", False):
        raise ValueError("mid_lora_reference_already_consumed")
    if type(first_layer) is not int or not 0 <= first_layer < len(reference.model.layers):
        raise ValueError("mid_lora_boundary_layer_invalid")
    if (input_ids.dtype != torch.int64 or input_ids.ndim != 2 or input_ids.shape[0] != 1
            or input_ids.shape[1] < 2):
        raise ValueError("mid_lora_requires_batch_one_fresh_sequence")
    if bool(((input_ids < 0) | (input_ids >= reference.config.vocab_size)).any()):
        raise ValueError("mid_lora_input_id_outside_vocabulary")
    if input_ids.device != reference.model.embed_tokens.weight.device:
        raise ValueError("mid_lora_input_device_mismatch")
    if ple_history is not None and (ple_history.dtype != torch.int64 or ple_history.shape != (1, 2)):
        raise ValueError("mid_lora_ple_history_requires_two_causal_tokens")
    if getattr(reference, "_mid_lora_persistent", False):
        if any(p.requires_grad for module in (reference.model.embed_tokens,
                *reference.model.layers[:first_layer]) for p in module.parameters()):
            raise ValueError("mid_lora_prefix_must_remain_frozen")
    else:
        reference.requires_grad_(False)
    reference.gradient_checkpointing_disable()
    reference.eval()
    session = getattr(reference, "_mid_lora_session", None)
    if session is None:
        session = reference._mid_lora_session = uuid.uuid4().hex
    ids = input_ids.detach().clone()
    history = None if ple_history is None else ple_history.detach().clone()
    captured = {}

    class BoundaryReached(Exception):
        pass

    def capture(_module, args, kwargs):
        if len(args) != 1 or kwargs.get("past_key_values") is not None:
            raise ValueError("mid_lora_native_boundary_contract_changed")
        captured["hidden"] = args[0].detach()
        captured["kwargs"] = _snapshot(kwargs)
        raise BoundaryReached()

    handle = reference.model.layers[first_layer].register_forward_pre_hook(capture, with_kwargs=True)
    try:
        with torch.no_grad(), _window_context((history, "off", None)):
            try:
                reference.model(input_ids=ids, use_cache=False, output_router_logits=False)
            except BoundaryReached:
                pass
    finally:
        handle.remove()
    if not captured:
        raise ValueError("mid_lora_native_boundary_not_reached")
    hidden = captured["hidden"]
    if hidden.shape != (*ids.shape, reference.config.hc_count * reference.config.hidden_size):
        raise ValueError("mid_lora_incomplete_hyper_connection_boundary")
    values = (hidden, captured["kwargs"], ids, history)
    return CompletePrefixState(hidden, captured["kwargs"], ids, history, first_layer,
        reference.ple_identity.fingerprint(), _tensor_digest(ids),
        None if history is None else _tensor_digest(history), session,
        _prefix_revision(reference, first_layer), _tensor_versions(values))


class SharedDownLoRA(nn.Module):
    """Ordinary frozen shared projection plus a zero-initialized low-rank update."""

    def __init__(self, base: nn.Linear, config: MidLoraConfig):
        super().__init__()
        if not isinstance(base, nn.Linear) or base.weight.device.type == "meta":
            raise ValueError("mid_lora_target_requires_materialized_native_linear")
        self.base = base.requires_grad_(False)
        self.scale = float(config.alpha) / config.rank
        self.lora_A = nn.Parameter(torch.empty(config.rank, base.in_features, device=base.weight.device, dtype=torch.float32))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, config.rank, device=base.weight.device, dtype=torch.float32))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, hidden):
        result = self.base(hidden)
        # FP32 adapters/Adam moments avoid losing small updates in BF16 MoE
        # projections. Preserve the native activation dtype and input gradient.
        with torch.autocast(device_type=hidden.device.type, enabled=False):
            delta = F.linear(F.linear(hidden.to(self.lora_A.dtype), self.lora_A), self.lora_B)
        return result + delta.to(result.dtype) * self.scale


class NativeMidLoraSuffix(nn.Module):
    """Adapt layers first..end; own resources by default, borrow in persistent mode."""

    def __init__(self, reference, config: MidLoraConfig, *, retain_reference=False):
        super().__init__()
        if (getattr(reference, "_mid_lora_consumed", False)
                or getattr(reference, "_mid_lora_persistent", False)):
            raise ValueError("mid_lora_reference_already_owned")
        config.validate(len(reference.model.layers))
        self.config = config
        self.identity = reference.ple_identity
        self.reference_session = getattr(reference, "_mid_lora_session", None)
        if self.reference_session is None:
            raise ValueError("mid_lora_collect_prefix_before_releasing_reference")
        self.prefix_revision = _prefix_revision(reference, config.first_layer)
        # Validate every target before changing ownership or replacing modules.
        bases = [reference.get_submodule(name) for name in _targets(config)]
        if any(not isinstance(base, nn.Linear) or base.weight.device.type == "meta" for base in bases):
            raise ValueError("mid_lora_target_requires_materialized_native_linear")
        reference.requires_grad_(False)
        reference.gradient_checkpointing_disable()
        reference.eval()
        self.layers = nn.ModuleList(reference.model.layers[config.first_layer:])
        self.hyper_connection_mixer = reference.model.hyper_connection_mixer
        self.lm_head = reference.lm_head
        self.native_checkpoint = reference.native_checkpoint
        self.fp32_linear_accumulation = bool(getattr(reference, "_mid_lora_fp32_accumulation", False))
        self.hc_width = reference.config.hc_count * reference.config.hidden_size
        self.target_names = _targets(config)
        self._closed = False
        for global_index, base in zip(range(config.first_layer, config.last_layer + 1), bases):
            self.layers[global_index - config.first_layer].mlp.shared_expert.down_proj = SharedDownLoRA(base, config)
        self._owns_resources = not retain_reference
        if retain_reference:
            # Borrow, do not register a second module tree or copy base tensors.
            # The caller keeps the reference alive and closes it exactly once.
            object.__setattr__(self, "_reference", reference)
            reference._mid_lora_persistent = True
        else:
            # Destructive legacy path: retained resources transfer to this suffix.
            for layer in reference.model.layers[:config.first_layer]:
                if layer.ple is not None:
                    layer.ple.ple_embedding.ngram_embedding.base.close()
            reference.model.layers = nn.ModuleList()
            reference.model.embed_tokens = None
            reference.model.rotary_emb = None
            reference.model.hyper_connection_mixer = None
            reference.lm_head = None
            del reference.native_checkpoint
            reference._mid_lora_consumed = True
        # Deterministic native reference; LoRA has no dropout. Explicit non-
        # reentrant checkpoints below work while native modules remain in eval.
        self.eval()

    def _validate_state(self, state: CompletePrefixState):
        if self._closed:
            raise ValueError("mid_lora_suffix_closed")
        if (not isinstance(state, CompletePrefixState) or state.first_layer != self.config.first_layer
                or state.identity_fingerprint != self.identity.fingerprint()
                or state.reference_session != self.reference_session or state.prefix_revision != self.prefix_revision):
            raise ValueError("mid_lora_wrong_or_stale_prefix")
        if any(tensor._version != version for tensor, version in state._versions):
            raise ValueError("mid_lora_prefix_tensor_mutated")
        if (state.hidden_states.shape != (*state.input_ids.shape, self.hc_width)
                or state.hidden_states.requires_grad or state.hidden_states.device != self.lm_head.weight.device
                or state.hidden_states.dtype != self.lm_head.weight.dtype):
            raise ValueError("mid_lora_prefix_shape_device_dtype_invalid")
        if (_tensor_digest(state.input_ids) != state.input_sha256
                or (None if state.ple_history is None else _tensor_digest(state.ple_history)) != state.history_sha256):
            raise ValueError("mid_lora_prefix_sequence_identity_invalid")

    def forward_hidden(self, state: CompletePrefixState):
        self._validate_state(state)
        hidden = state.hidden_states
        with _window_context((state.ple_history, "off", None)):
            size = self.config.checkpoint_group_size
            for start in range(0, len(self.layers), size):
                group = tuple(self.layers[start:start + size])
                def run_group(value, group=group):
                    for layer in group:
                        value = layer(value, **state.layer_kwargs)
                    return value
                if self.config.gradient_checkpointing and torch.is_grad_enabled():
                    storage = (torch.autograd.graph.save_on_cpu(pin_memory=True)
                               if self.config.offload_checkpoints else nullcontext())
                    with storage:
                        hidden = checkpoint(run_group, hidden, use_reentrant=False,
                                            context_fn=_checkpoint_contexts)
                else:
                    hidden = run_group(hidden)
            return self.hyper_connection_mixer(hidden)

    def forward(self, state: CompletePrefixState, labels: torch.Tensor, *, chunk_size=128):
        return self.loss(state, labels, chunk_size=chunk_size)

    def loss(self, state: CompletePrefixState, labels: torch.Tensor, *, chunk_size=128):
        if labels.shape != state.input_ids.shape:
            raise ValueError("mid_lora_labels_must_match_cached_sequence")
        if bool(((labels != -100) & (labels != state.input_ids)).any()):
            raise ValueError("mid_lora_labels_must_be_cached_tokens_or_ignored")
        if self.fp32_linear_accumulation:
            from qwen_exo_booster.native_mid_lora_precision import stable_chunked_causal_cross_entropy
            return stable_chunked_causal_cross_entropy(self.forward_hidden(state), self.lm_head.weight, labels, chunk_size)
        return chunked_causal_cross_entropy(self.forward_hidden(state), self.lm_head.weight, labels, chunk_size)

    def adapter_parameters(self):
        for layer in self.layers[:self.config.last_layer - self.config.first_layer + 1]:
            adapter = layer.mlp.shared_expert.down_proj
            yield adapter.lora_A
            yield adapter.lora_B

    def _adapter_tensors(self):
        tensors = {}
        for name, layer in zip(self.target_names, self.layers):
            adapter = layer.mlp.shared_expert.down_proj
            tensors[name + ".lora_A"] = adapter.lora_A
            tensors[name + ".lora_B"] = adapter.lora_B
        return tensors

    def _adapter_metadata(self):
        return {"schema": _ADAPTER_SCHEMA, "identity": self.identity.to_dict(),
                "config": asdict(self.config), "targets": self.target_names,
                "shapes": {name: list(tensor.shape) for name, tensor in self._adapter_tensors().items()},
                "adapter_dtype": "torch.float32",
                "weight_reference": "frozen NVFP4 decoded W4A16, not serving W4A4"}

    def save_adapter(self, path):
        """Save only adapters and exact targets/base binding, never original weights."""
        from safetensors.torch import save_file
        tensors = {name: tensor.detach().cpu().contiguous() for name, tensor in self._adapter_tensors().items()}
        if not all(bool(torch.isfinite(tensor).all()) for tensor in tensors.values()):
            raise ValueError("mid_lora_nonfinite_adapter")
        save_file(tensors, str(path), metadata={"native_mid_lora": json.dumps(self._adapter_metadata(), sort_keys=True)})

    def load_adapter(self, path):
        """Fail closed on any wrong target/range/rank/base/shape before copying."""
        from safetensors import safe_open
        expected = self._adapter_tensors()
        with safe_open(str(path), framework="pt", device="cpu") as source:
            metadata = json.loads((source.metadata() or {}).get("native_mid_lora", "null"))
            if metadata != self._adapter_metadata() or set(source.keys()) != set(expected):
                raise ValueError("mid_lora_adapter_contract_mismatch")
            loaded = {name: source.get_tensor(name) for name in expected}
        for name, tensor in loaded.items():
            if tensor.shape != expected[name].shape or tensor.dtype != expected[name].dtype or not bool(torch.isfinite(tensor).all()):
                raise ValueError(f"mid_lora_adapter_tensor_invalid: {name}")
        # Stage all device copies before changing even the first adapter tensor.
        placed = {name: tensor.to(expected[name].device) for name, tensor in loaded.items()}
        with torch.no_grad():
            for name, tensor in placed.items():
                expected[name].copy_(tensor)

    def close(self):
        if not self._closed:
            if self._owns_resources:
                for layer in self.layers:
                    if layer.ple is not None:
                        layer.ple.ple_embedding.ngram_embedding.base.close()
                self.native_checkpoint.close()
            self._closed = True


def build_mid_lora_suffix(reference, config: MidLoraConfig = MidLoraConfig()) -> NativeMidLoraSuffix:
    """Destructively transfer ownership, freeing all lower ordinary tensors.

    Do not retain external references to lower layers. The consumed reference
    can only be closed afterwards; suffix.close owns remaining disk resources.
    """
    return NativeMidLoraSuffix(reference, config)


def build_persistent_mid_lora(reference, config: MidLoraConfig = MidLoraConfig()) -> NativeMidLoraSuffix:
    """Borrow a complete reference for fresh current-adapter generation/training.

    Unlike the destructive default, lower ordinary tensors stay resident. Caller
    owns reference.close(); suffix.close() only fences further use. Training is
    always fresh full-sequence replay. Optional no-grad generation-cache parity
    gates live in native_mid_lora_experiment, never in the training forward path.
    """
    if getattr(reference, "_mid_lora_session", None) is None:
        reference._mid_lora_session = uuid.uuid4().hex
    return NativeMidLoraSuffix(reference, config, retain_reference=True)


def verify_tiny_mid_lora_backward() -> dict:
    """Explicit CPU proof only; three native layers, no optimizer and no GPU.

    Native GR4/GDN/PLE layer 1 is adapted and native QSA layer 2 remains frozen.
    Tests loss-to-adapter backward through that upper suffix and the real head.
    """
    from dataclasses import replace
    from safetensors import safe_open
    from safetensors.torch import save_file
    from qwen_exo_booster.native_ple_knowledge import sha256_file

    config = MidLoraConfig(first_layer=1, last_layer=1, rank=2, alpha=4.0)
    with _tiny_native_ple_fixture() as (root, model, ids, history, _delta):
        model.requires_grad_(False)
        model.gradient_checkpointing_disable()
        model.eval()
        original_artifacts = {str(p): sha256_file(p) for p in root.rglob("*") if p.is_file()}
        with torch.no_grad():
            full_hidden = model.forward_hidden(ids, ple_history=history, ple_mode="off")
            full_loss = chunked_causal_cross_entropy(full_hidden, model.lm_head.weight, ids, 3)
        state = collect_prefix(model, ids, first_layer=1, ple_history=history)
        suffix = build_mid_lora_suffix(model, config)
        try:
            if list(model.parameters()) or list(model.buffers()):
                raise AssertionError("Consumed full reference still owns lower ordinary tensors")
            with torch.no_grad():
                cached_hidden = suffix.forward_hidden(state)
            torch.testing.assert_close(cached_hidden, full_hidden, rtol=1e-6, atol=1e-6)
            frozen = {name: tensor.detach().clone() for name, tensor in suffix.named_parameters()
                      if not tensor.requires_grad and tensor.device.type != "meta"}
            frozen_buffers = {name: tensor.detach().clone() for name, tensor in suffix.named_buffers()
                              if tensor.device.type != "meta"}
            loss = suffix.loss(state, ids.clone(), chunk_size=3)
            torch.testing.assert_close(loss.detach(), full_loss, rtol=1e-6, atol=1e-6)
            loss.backward()
            parameters = list(suffix.adapter_parameters())
            if not bool(torch.isfinite(loss)) or any(p.grad is None or not bool(torch.isfinite(p.grad).all()) for p in parameters):
                raise AssertionError("Native suffix loss did not produce finite adapter gradients")
            # B starts at zero: A's initially zero gradient is correct.
            if not any(bool((p.grad != 0).any()) for p in parameters[1::2]):
                raise AssertionError("Native suffix loss did not reach LoRA B")
            if any(p.grad is not None for p in suffix.parameters() if not p.requires_grad):
                raise AssertionError("Frozen native weight acquired a gradient")
            if any(not torch.equal(dict(suffix.named_parameters())[name], tensor) for name, tensor in frozen.items()):
                raise AssertionError("Frozen native weight changed")
            if any(not torch.equal(dict(suffix.named_buffers())[name], tensor) for name, tensor in frozen_buffers.items()):
                raise AssertionError("Frozen native buffer changed")
            gradient_max = max(float(p.grad.abs().max()) for p in parameters)
            # Deterministic nonzero fixture values, NOT an optimizer/training step.
            with torch.no_grad():
                for parameter in parameters[1::2]:
                    parameter.copy_(torch.arange(parameter.numel()).reshape_as(parameter).sin() * .01)
                adapted = suffix.forward_hidden(state)
            if torch.equal(adapted, cached_hidden):
                raise AssertionError("Nonzero adapter failed to affect native suffix")
            adapter_path = root / "adapter.safetensors"
            suffix.save_adapter(adapter_path)
            saved = [p.detach().clone() for p in parameters]
            with torch.no_grad():
                for parameter in parameters:
                    parameter.zero_()
            suffix.load_adapter(adapter_path)
            if any(not torch.equal(parameter, value) for parameter, value in zip(parameters, saved)):
                raise AssertionError("Adapter reload changed saved values")
            with torch.no_grad():
                reloaded = suffix.forward_hidden(state)
            torch.testing.assert_close(adapted, reloaded, rtol=0, atol=0)
            rejected = []
            for name, invalid in (("range", replace(state, first_layer=0)),
                                  ("base", replace(state, identity_fingerprint="wrong")),
                                  ("session", replace(state, reference_session="wrong")),
                                  ("tokens", replace(state, input_sha256="wrong"))):
                try:
                    suffix.forward_hidden(invalid)
                except ValueError:
                    rejected.append(name)
                else:
                    raise AssertionError(f"Accepted invalid cached prefix: {name}")
            with safe_open(str(adapter_path), framework="pt", device="cpu") as source:
                payload = {name: source.get_tensor(name) for name in source.keys()}
                metadata = json.loads(source.metadata()["native_mid_lora"])
            for invalid_kind in ("targets", "range"):
                bad = json.loads(json.dumps(metadata))
                if invalid_kind == "targets":
                    bad["targets"] = ["model.layers.1.mlp.experts.0.down_proj"]
                else:
                    bad["config"]["last_layer"] = 2
                bad_path = root / f"wrong-{invalid_kind}.safetensors"
                save_file(payload, str(bad_path), metadata={"native_mid_lora": json.dumps(bad)})
                try:
                    suffix.load_adapter(bad_path)
                except ValueError:
                    pass
                else:
                    raise AssertionError(f"Accepted adapter with wrong {invalid_kind}")
                if any(not torch.equal(parameter, value) for parameter, value in zip(parameters, saved)):
                    raise AssertionError("Rejected adapter partially mutated parameters")
            if any(sha256_file(path) != digest for path, digest in original_artifacts.items()):
                raise AssertionError("Original native fixture artifact bytes changed")
            return {"device": "cpu", "optimizer_steps": 0, "training_started": False,
                    "gpu_validated": False, "training_ready": False,
                    "fixture_loader": "load_native_ple_training_model",
                    "fixture_weight_encoding": "ModelOpt NVFP4 packed E2M1/E4M3+F32 scales",
                    "native_layers": ["GR4", "GDN", "QSA", "PLE"],
                    "full_hc_shape": list(state.hidden_states.shape),
                    "cache_full_forward_max_difference": float((cached_hidden - full_hidden).abs().max()),
                    "loss": float(loss.detach()), "causal_loss_difference": float((loss.detach() - full_loss).abs()),
                    "adapter_gradient_abs_max": gradient_max, "original_gradients": 0,
                    "original_weights_unchanged": True, "original_artifact_bytes_unchanged": True,
                    "lower_ordinary_tensors_released": True, "frozen_upper_suffix_layers": 1,
                    "adapter_roundtrip_exact": True, "invalid_adapter_targets_and_range_rejected": True,
                    "invalid_prefix_rejected": rejected, "targets": suffix.target_names,
                    "nonzero_adapter_hidden_max_difference": float((adapted - cached_hidden).abs().max())}
        finally:
            suffix.close()
