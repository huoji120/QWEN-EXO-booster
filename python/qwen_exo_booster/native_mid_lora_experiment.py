"""Explicitly authorized offline native shared-down LoRA experiments.

Fresh full-prefix decoding is the default. Optional native-cache decoding is
fail-closed behind tiny and actual-prefix parity gates, including QSA/PLE/GDN.
Teacher-forced NLL and lexical/action-format metrics are not reasoning proofs.
This module never starts/stops services, installs packages, or deploys adapters.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import random
import re
import time
from contextlib import contextmanager, nullcontext
from dataclasses import asdict
from pathlib import Path

import torch
from torch.nn import functional as F

from qwen_exo_booster.native_mid_lora import (
    MidLoraConfig, build_persistent_mid_lora, collect_prefix,
    load_mid_lora_reference,
)
from qwen_exo_booster.native_ple_training_backend import (
    _tiny_native_ple_fixture, native_generation_cache_context,
)

def _save_training_state(path, *, optimizer, adapter_path, parameter_names, progress, training):
    """Publish a hash-bound, CPU-loadable optimizer/RNG companion after its adapter."""
    import numpy as np

    def cpu(value):
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().clone()
        if isinstance(value, dict):
            return {key: cpu(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return type(value)(cpu(item) for item in value)
        return value

    path = Path(path)
    pending = path.with_suffix(path.suffix + ".partial")
    if path.exists() or pending.exists():
        raise FileExistsError(f"Refusing to overwrite training state: {path}")
    numpy_state = np.random.get_state()
    state = {
        "schema": "native-mid-lora-training-state-v1",
        "adapter_path": str(Path(adapter_path).resolve()),
        "adapter_sha256": hashlib.sha256(Path(adapter_path).read_bytes()).hexdigest(),
        "parameter_names": list(parameter_names),
        "optimizer": cpu(optimizer.state_dict()),
        "progress": dict(progress), "training": dict(training),
        "rng": {"torch_cpu": torch.get_rng_state(),
                "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else [],
                "python": random.getstate(),
                "numpy": (numpy_state[0], numpy_state[1].tolist(), *numpy_state[2:])},
    }
    with pending.open("xb") as target:
        torch.save(state, target)
        target.flush()
        os.fsync(target.fileno())
    pending.replace(path)
    if os.name != "nt":
        fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "adapter_sha256": state["adapter_sha256"]}


def _load_training_state(path, *, adapter_path, expected_training, optimizer_step, next_sample_index,
                         transfer_dataset=False):
    state = torch.load(path, map_location="cpu", weights_only=True)
    if state.get("schema") != "native-mid-lora-training-state-v1":
        raise ValueError("Unsupported training-state schema")
    if hashlib.sha256(Path(adapter_path).read_bytes()).hexdigest() != state.get("adapter_sha256"):
        raise ValueError("Training state does not match adapter bytes")
    if transfer_dataset:
        old_hash = state["training"].get("prepared_sha256")
        new_hash = expected_training.get("prepared_sha256")
        if not old_hash or not new_hash or old_hash == new_hash or next_sample_index != 0:
            raise ValueError("Dataset transfer requires distinct data hashes and a zero new-data cursor")
    for key, value in expected_training.items():
        if transfer_dataset and key == "prepared_sha256":
            continue
        if state["training"].get(key) != value:
            raise ValueError(f"Training-state configuration mismatch: {key}")
    progress = state["progress"]
    if (progress.get("arm") != "self_think_sft" or progress.get("optimizer_step") != optimizer_step
            or (not transfer_dataset and progress.get("next_sample_index") != next_sample_index)):
        raise ValueError("Training-state step/cursor mismatch")
    session_step = progress.get("optimizer_session_step")
    if type(session_step) is not int or not 0 < session_step <= optimizer_step:
        raise ValueError("Invalid saved optimizer session step")
    return state


def _restore_training_state(state, optimizer, parameter_names):
    """Validate all parameter states before mutating the optimizer or RNG."""
    import numpy as np

    if list(parameter_names) != state["parameter_names"]:
        raise ValueError("Training-state parameter order mismatch")
    saved = state["optimizer"]
    if len(saved["param_groups"]) != len(optimizer.param_groups):
        raise ValueError("Optimizer group count mismatch")
    ids = []
    for previous, current in zip(saved["param_groups"], optimizer.param_groups):
        for key in ("lr", "betas", "eps", "weight_decay", "amsgrad", "maximize", "foreach", "capturable", "differentiable", "fused"):
            if previous.get(key) != current.get(key):
                raise ValueError(f"Optimizer configuration mismatch: {key}")
        if len(previous["params"]) != len(current["params"]):
            raise ValueError("Optimizer parameter count mismatch")
        for identifier, parameter in zip(previous["params"], current["params"]):
            ids.append(identifier)
            item = saved["state"][identifier]
            if item["step"].item() != state["progress"]["optimizer_session_step"]:
                raise ValueError("Optimizer parameter step mismatch")
            for key in ("exp_avg", "exp_avg_sq"):
                tensor = item[key]
                if tensor.shape != parameter.shape or tensor.dtype != parameter.dtype or not torch.isfinite(tensor).all():
                    raise ValueError(f"Invalid optimizer tensor: {key}")
            if (item["exp_avg_sq"] < 0).any():
                raise ValueError("Negative optimizer variance")
    if len(set(ids)) != len(ids) or set(ids) != set(saved["state"]):
        raise ValueError("Unexpected optimizer parameter IDs")
    rng = state["rng"]
    torch.Generator().set_state(rng["torch_cpu"])
    random.Random().setstate(rng["python"])
    numpy_state = (rng["numpy"][0], np.array(rng["numpy"][1], dtype=np.uint32), *rng["numpy"][2:])
    np.random.RandomState().set_state(numpy_state)
    cuda_count = torch.cuda.device_count() if torch.cuda.is_initialized() else 0
    if len(rng["torch_cuda"]) != cuda_count:
        raise ValueError("CUDA RNG device count mismatch")
    optimizer.load_state_dict(saved)
    torch.set_rng_state(rng["torch_cpu"])
    if cuda_count:
        torch.cuda.set_rng_state_all(rng["torch_cuda"])
    random.setstate(rng["python"])
    np.random.set_state(numpy_state)
    return state["progress"]["optimizer_session_step"]



def _sync(device):
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)


def _memory(device):
    result = {}
    if torch.device(device).type == "cuda":
        result.update(allocated_bytes=torch.cuda.memory_allocated(device),
                      reserved_bytes=torch.cuda.memory_reserved(device),
                      peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
                      peak_reserved_bytes=torch.cuda.max_memory_reserved(device))
    for filename in ("/sys/fs/cgroup/memory.current", "/sys/fs/cgroup/memory.max"):
        path = Path(filename)
        if path.exists():
            result[path.name] = path.read_text().strip()
    return result


def _tensor_hash(tensor):
    digest = hashlib.sha256()
    flat = tensor.detach().reshape(-1)
    # Bound host transfers instead of duplicating an entire frozen embedding.
    chunk = max(1, (16 << 20) // tensor.element_size())
    for start in range(0, flat.numel(), chunk):
        digest.update(flat[start:start + chunk].cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def frozen_fingerprint(reference):
    """Hash materialized base parameters/buffers, not read-only mapped expert files."""
    tensors = [("parameter:" + n, p) for n, p in reference.named_parameters() if not p.requires_grad]
    tensors += [("buffer:" + n, p) for n, p in reference.named_buffers()]
    return {name: _tensor_hash(tensor) for name, tensor in tensors if tensor.device.type != "meta"}


def _assert_only_adapter(reference, suffix):
    expected = {id(p) for p in suffix.adapter_parameters()}
    actual = {id(p) for p in reference.parameters() if p.requires_grad}
    if not expected or actual != expected:
        raise RuntimeError("Only shared-down LoRA parameters may be trainable")
    if any(p.grad is not None for p in reference.parameters() if id(p) not in expected):
        raise RuntimeError("Frozen native parameter acquired a gradient")


def _ids(values, reference):
    return torch.tensor([values], dtype=torch.int64, device=reference.model.embed_tokens.weight.device)


@contextmanager
def _cached_native_context(reference):
    """Enable native recurrent state without changing QSA selection semantics."""
    if torch.is_grad_enabled():
        raise ValueError("Cached native decoding cannot retain training gradients")
    with native_generation_cache_context():
        yield


def _finite_native_state(value, *, path="cache", seen=None):
    """Inspect actual cache tensors, without copying them or assuming one cache ABI."""
    if seen is None:
        seen = set()
    if id(value) in seen:
        return 0
    seen.add(id(value))
    if isinstance(value, torch.Tensor):
        if not bool(torch.isfinite(value).all()):
            raise FloatingPointError(f"Nonfinite native state: {path}")
        return 1
    if isinstance(value, dict):
        children = value.items()
    elif isinstance(value, (tuple, list)):
        children = enumerate(value)
    elif hasattr(value, "__dict__"):
        children = vars(value).items()
    else:
        return 0
    return sum(_finite_native_state(child, path=f"{path}.{name}", seen=seen)
               for name, child in children)


def _validate_native_cache(reference, cache, consumed_tokens):
    """Check native DynamicCache ABI against the loaded model, without tensor copies."""
    from transformers.cache_utils import DynamicCache
    if not isinstance(cache, DynamicCache) or len(cache.layers) != len(reference.model.layers):
        raise RuntimeError("Native cache type/layer-count mismatch")
    if cache.get_seq_length() != consumed_tokens:
        raise RuntimeError("Native cache sequence advancement mismatch")
    device = reference.model.embed_tokens.weight.device

    def shape(value, expected, name):
        if not isinstance(value, torch.Tensor) or tuple(value.shape) != expected or value.device != device:
            raise RuntimeError(f"Native cache shape/device mismatch: {name}, expected {expected}")

    shape(getattr(cache, "position_ids", None), (3, 1, consumed_tokens), "position_ids")
    for index, (layer, state) in enumerate(zip(reference.model.layers, cache.layers)):
        if layer.layer_type == "linear_attention":
            attention = layer.linear_attn
            conv = getattr(state, "conv_states", {})
            recurrent = getattr(state, "recurrent_states", {})
            shape(conv.get(0), (1, attention.conv_dim, attention.conv_kernel_size), f"layer{index}.gdn_conv")
            shape(recurrent.get(0), (1, attention.num_v_heads, attention.head_k_dim, attention.head_v_dim),
                  f"layer{index}.gdn_recurrent")
            if layer.ple is not None:
                ple = layer.ple
                shape(conv.get(1), (1, ple.hidden_size * ple.hc_count, ple.short_conv_state_len),
                      f"layer{index}.ple_conv")
                shape(conv.get(2), (1, ple.ple_embedding.context_len), f"layer{index}.ple_history")
                if conv[2].dtype != torch.int64:
                    raise RuntimeError("Native PLE history must retain integer token IDs")
        elif layer.layer_type == "qwen_sparse_attention":
            attention = layer.self_attn
            kv_shape = (1, reference.config.num_key_value_heads, consumed_tokens, attention.head_dim)
            shape(getattr(state, "keys", None), kv_shape, f"layer{index}.keys")
            shape(getattr(state, "values", None), kv_shape, f"layer{index}.values")
            indexer_shape = ((1, consumed_tokens, attention.indexer.index_head_dim)
                if attention.indexer.index_kv_heads == 1 else
                (1, consumed_tokens, attention.indexer.index_kv_heads, attention.indexer.index_head_dim))
            shape(getattr(state, "indexer_keys", None), indexer_shape, f"layer{index}.indexer_keys")
        else:
            raise RuntimeError(f"Unsupported native cache layer type: {layer.layer_type}")


@torch.no_grad()
def verify_native_generation_cache(reference, *, probe_ids=None, allow_numeric_drift_research=False):
    """Fail closed on actual-model native fresh/cached last-logit disagreement.

    Full causal replay uses the actual installed training QSA path. Cached replay
    uses native QSA/GDN/PLE state, including EOS-delimited NGram history. The gate
    is rebound to each adapter revision; caches never survive a generation call.
    """
    if probe_ids is None:
        eos = reference.config.eos_token_id
        eos = eos[0] if isinstance(eos, list) else eos
        vocab = reference.config.vocab_size
        probe_ids = [1 % vocab, 4 % vocab, 7 % vocab, int(eos), 2 % vocab,
                     5 % vocab, 8 % vocab, 9 % vocab, 3 % vocab, 6 % vocab]
    ids = _ids(probe_ids, reference)
    if ids.shape[1] < 6:
        raise ValueError("Cache parity probe requires at least six causal tokens")
    split = ids.shape[1] - 3
    probe_digest = _tensor_hash(ids)
    fresh = []
    reference.eval()
    device = ids.device
    _sync(device)
    started = time.perf_counter()
    for end in range(split, ids.shape[1] + 1):
        hidden = reference.forward_hidden(ids[:, :end], ple_mode="off")
        _finite_native_state(hidden, path="fresh_hidden")
        if end == split:
            fresh_prefill_hidden = hidden
        fresh.append(reference.lm_head(hidden[:, -1, :]).float())
        _finite_native_state(fresh[-1], path="fresh_logits")
        del hidden
    _sync(device)
    fresh_seconds = time.perf_counter() - started
    cached = []
    started = time.perf_counter()
    with _cached_native_context(reference):
        cache = None
        for end in range(split, ids.shape[1] + 1):
            current = ids[:, :end] if cache is None else ids[:, end - 1:end]
            result = reference.model(input_ids=current, past_key_values=cache,
                                     use_cache=True, output_router_logits=False)
            cache = result.past_key_values
            if cache is None:
                raise RuntimeError("Native model did not return a real cache")
            _validate_native_cache(reference, cache, end)
            if result.last_hidden_state.shape != (1, current.shape[1], reference.lm_head.weight.shape[1]):
                raise RuntimeError("Native cached hidden-state shape mismatch")
            if end == split:
                prefill_hidden_equal = torch.equal(result.last_hidden_state, fresh_prefill_hidden)
                del fresh_prefill_hidden
            _finite_native_state(result.last_hidden_state, path="cached_hidden")
            if not _finite_native_state(cache):
                raise RuntimeError("Native cache contains no inspectable tensor state")
            cached.append(reference.lm_head(result.last_hidden_state[:, -1, :]).float())
            _finite_native_state(cached[-1], path="cached_logits")
            del result
    _sync(device)
    cached_seconds = time.perf_counter() - started
    tolerance = (1e-4, 1e-5) if reference.lm_head.weight.dtype == torch.float32 else (.04, .01)
    comparisons = []
    for position, (actual, expected) in enumerate(zip(cached, fresh), start=split):
        if actual.shape != expected.shape:
            raise RuntimeError("Native cached/fresh logit shape mismatch")
        close = torch.isclose(actual, expected, rtol=tolerance[0], atol=tolerance[1])
        comparisons.append({"prefix_tokens": position,
            "bitwise_equal": torch.equal(actual, expected), "within_tolerance": bool(close.all()),
            "outside_tolerance_fraction": float((~close).float().mean()),
            "max_abs_logit_error": float((actual - expected).abs().max()),
            "mean_abs_logit_error": float((actual - expected).abs().mean()),
            "greedy_tokens_equal": torch.equal(actual.argmax(-1), expected.argmax(-1))})
    passed = all(item["within_tolerance"] and item["greedy_tokens_equal"] for item in comparisons)
    prefill_equal = comparisons[0]["bitwise_equal"] and prefill_hidden_equal
    if _tensor_hash(ids) != probe_digest:
        raise RuntimeError("Native cache probe mutated its causal token IDs")
    report = {"passed": passed, "probe_tokens": ids.shape[1], "compared_positions": len(fresh),
              "probe_input_sha256": probe_digest, "positions": comparisons,
              "max_abs_logit_error": max(item["max_abs_logit_error"] for item in comparisons),
              "rtol": tolerance[0], "atol": tolerance[1],
              "greedy_tokens_equal": all(item["greedy_tokens_equal"] for item in comparisons),
              "prefill_bitwise_equal": prefill_equal, "finite_states_and_logits": True,
              "prefill_hidden_bitwise_equal": prefill_hidden_equal,
              "cache_structure_validated": True,
              "numeric_policy": "exploratory_cached_decode_drift" if allow_numeric_drift_research else "strict_equivalence",
              "research_accepted": bool(allow_numeric_drift_research and prefill_equal),
              "fresh_seconds": fresh_seconds, "cached_seconds": cached_seconds,
              "ple_history": "native cache conv_states[2] with native EOS boundary semantics"}
    reference._mid_lora_cache_receipt = report
    progress = getattr(reference, "_mid_lora_generation_progress", None)
    if progress is not None:
        progress({"native_cache_gate": report})
    if allow_numeric_drift_research and not prefill_equal:
        raise AssertionError("Research cache gate requires bitwise-equal fresh/cached prefill")
    if not passed and not report["research_accepted"]:
        raise AssertionError("Native cached/fresh logit or greedy-token disagreement")
    return report


def _cache_gate_revision(reference):
    """Bind validation reuse to this exact base/device/dtype and adapter tensors.

    Optimizer updates, copy_-based reset/load/roundtrip, tensor replacement and
    dtype/device changes invalidate naturally. Only validation is reused; never
    native KV/GDN/PLE state. Direct .data mutation is outside the training API.
    """
    return (reference.ple_identity.fingerprint(),
            str(reference.lm_head.weight.device), str(reference.lm_head.weight.dtype),
            tuple((name, id(parameter), parameter._version,
                   str(parameter.device), str(parameter.dtype))
                  for name, parameter in reference.named_parameters() if parameter.requires_grad),
            getattr(reference, "_mid_lora_generation_expert_backend", "reference"))


@torch.no_grad()
def generate_current_adapter(reference, question_input_ids, *, max_new_tokens,
                             stop_ids, total_tokens=32768, phase_boundary_ids=None,
                             phase_budget=None, continuation_budget=None):
    """Generate with an explicitly selected no-grad expert path, never backward."""
    from qwen_exo_booster.native_ple_checkpoint import generation_expert_backend
    backend = getattr(reference, "_mid_lora_generation_expert_backend", "reference")
    with generation_expert_backend(backend):
        return _generate_current_adapter(reference, question_input_ids, max_new_tokens=max_new_tokens,
            stop_ids=stop_ids, total_tokens=total_tokens, phase_boundary_ids=phase_boundary_ids,
            phase_budget=phase_budget, continuation_budget=continuation_budget)


def _generate_current_adapter(reference, question_input_ids, *, max_new_tokens,
                              stop_ids, total_tokens=32768, phase_boundary_ids=None,
                              phase_budget=None, continuation_budget=None):
    """Q-only greedy decoding, optionally continuing across a real Think closer.

    Budgets and marker IDs are structural inputs, never answer tokens or labels.
    One cache lives across the phase boundary, but never across calls/examples.
    No marker or separator is inserted. Research drift is explicit, not parity.
    """
    prefix = tuple(int(x) for x in question_input_ids)
    stop = tuple(int(x) for x in stop_ids)
    boundary = None if phase_boundary_ids is None else tuple(int(x) for x in phase_boundary_ids)
    if not prefix or not stop or max_new_tokens < 1 or len(prefix) + max_new_tokens > total_tokens:
        raise ValueError("Invalid causal generation budget or empty stop marker")
    if boundary is not None:
        if not boundary or phase_budget is None or continuation_budget is None or phase_budget < len(boundary) or continuation_budget < 1:
            raise ValueError("Invalid phase boundary or phase/continuation budget")
    elif phase_budget is not None or continuation_budget is not None:
        raise ValueError("Phase budgets require a phase boundary")
    if total_tokens > reference.config.max_position_embeddings:
        raise ValueError("Generation exceeds native context capacity")
    device = reference.model.embed_tokens.weight.device
    tokens = torch.empty((1, len(prefix) + max_new_tokens), dtype=torch.int64, device=device)
    tokens[0, :len(prefix)] = torch.tensor(prefix, dtype=torch.int64, device=device)
    generated = []
    _sync(device)
    started = time.perf_counter()
    terminated = False
    close_position = close_seconds = None
    reason = "context_limit" if len(prefix) + max_new_tokens == total_tokens else "generation_budget"
    progress = getattr(reference, "_mid_lora_generation_progress", None)
    reference.eval()
    mode = getattr(reference, "_mid_lora_decode_mode", "fresh")
    if mode not in {"fresh", "native-cache"}:
        raise ValueError("Unknown native generation mode")
    gate_policy = getattr(reference, "_mid_lora_cache_gate_policy", "each-prefix")
    if gate_policy not in {"each-prefix", "adapter-revision"}:
        raise ValueError("Unknown native cache gate policy")
    research = bool(getattr(reference, "_mid_lora_allow_numeric_drift_research", False))
    if research and mode != "native-cache":
        raise ValueError("Numeric drift research requires native-cache generation")
    gate_reused = False
    if mode == "native-cache":
        revision_key = _cache_gate_revision(reference)
        acceptance_key = (revision_key, research)
        receipt = getattr(reference, "_mid_lora_cache_receipt", None)
        gate_reused = (gate_policy == "adapter-revision"
                       and getattr(reference, "_mid_lora_cache_gate_key", None) == acceptance_key
                       and bool(receipt and (receipt.get("passed") or (research and receipt.get("research_accepted")))))
        if not gate_reused:
            # A failed gate must not leave old acceptance reusable.
            reference._mid_lora_cache_gate_key = None
            reference._mid_lora_cache_receipt = None
            gate = verify_native_generation_cache(reference, probe_ids=prefix,
                allow_numeric_drift_research=research)
            gate.update(policy=gate_policy,
                        validation_scope="every_actual_prefix" if gate_policy == "each-prefix"
                            else "representative_actual_prefix_per_adapter_revision",
                        base_identity=revision_key[0], device=revision_key[1], dtype=revision_key[2],
                        adapter_tensor_versions=[{"name": name, "version": version,
                            "device": device_name, "dtype": dtype_name}
                            for name, _identity, version, device_name, dtype_name in revision_key[3]])
            reference._mid_lora_cache_gate_key = acceptance_key
    cache = None
    with _cached_native_context(reference) if mode == "native-cache" else nullcontext():
        for offset in range(max_new_tokens):
            if mode == "native-cache":
                current = tokens[:, :len(prefix)] if cache is None else tokens[:, len(prefix) + offset - 1:len(prefix) + offset]
                result = reference.model(input_ids=current, past_key_values=cache,
                                         use_cache=True, output_router_logits=False)
                cache, hidden = result.past_key_values, result.last_hidden_state
                del result
                if cache is None:
                    raise RuntimeError("Native cache disappeared during generation")
                if hidden.shape != (1, current.shape[1], reference.lm_head.weight.shape[1]):
                    raise RuntimeError("Native generation hidden-state shape mismatch")
                _validate_native_cache(reference, cache, len(prefix) + offset)
            else:
                hidden = reference.forward_hidden(tokens[:, :len(prefix) + offset], ple_mode="off")
            _finite_native_state(hidden, path="generation_hidden")
            logits = reference.lm_head(hidden[:, -1, :])
            if not bool(torch.isfinite(logits).all()):
                raise FloatingPointError("Nonfinite current-adapter generation logits")
            token = int(logits.argmax(-1).item())
            tokens[0, len(prefix) + offset] = token
            generated.append(token)
            del logits, hidden
            if progress is not None and len(generated) % 32 == 0:
                progress({"elapsed_seconds": time.perf_counter() - started,
                          "generated_count": len(generated), "prefix_tokens": len(prefix)})
            if boundary is not None and close_position is None and tuple(generated[-len(boundary):]) == boundary:
                close_position = len(generated)  # exclusive offset through the sampled closer
                close_seconds = time.perf_counter() - started
            if len(generated) >= len(stop) and tuple(generated[-len(stop):]) == stop:
                terminated = True
                reason = "stop_before_phase_close" if boundary is not None and close_position is None else "stop_marker"
                break
            if boundary is not None:
                if close_position is None and len(generated) >= phase_budget:
                    reason = "phase_budget_without_closer"
                    break
                if close_position is not None and len(generated) - close_position >= continuation_budget:
                    reason = "continuation_budget"
                    break
    if cache is not None:
        _finite_native_state(cache)
    _sync(device)
    return {"token_ids": generated, "terminated": terminated,
            "seconds": time.perf_counter() - started, "termination_reason": reason,
            "phase_close_position": close_position, "phase_close_seconds": close_seconds,
            "phase_budget": phase_budget, "continuation_budget": continuation_budget,
            "continuation_tokens": None if close_position is None else len(generated) - close_position,
            "decode_mode": mode, "sampling": "greedy",
            "generation_expert_backend": getattr(reference, "_mid_lora_generation_expert_backend", "reference"),
            "cache_gate_policy": gate_policy,
            "native_cache_gate": ({**reference._mid_lora_cache_receipt, "reused": gate_reused}
                                  if mode == "native-cache" else None)}


def _backward(reference, suffix, input_ids, labels, *, clip_norm, chunk_size=128):
    suffix.zero_grad(set_to_none=True)
    state = collect_prefix(reference, input_ids, first_layer=suffix.config.first_layer)
    _assert_only_adapter(reference, suffix)
    loss = suffix.loss(state, labels, chunk_size=chunk_size)
    if not bool(torch.isfinite(loss)):
        raise FloatingPointError("Nonfinite answer-only loss")
    loss.backward()
    parameters = list(suffix.adapter_parameters())
    if any(p.grad is None or not bool(torch.isfinite(p.grad).all()) for p in parameters):
        raise FloatingPointError("Missing or nonfinite LoRA gradient")
    norm = torch.nn.utils.clip_grad_norm_(parameters, clip_norm, error_if_nonfinite=True)
    _assert_only_adapter(reference, suffix)
    return {"loss": float(loss.detach()), "gradient_norm_before_clip": float(norm),
            "gradient_abs_max_after_clip": max(float(p.grad.abs().max()) for p in parameters),
            "supervised_tokens": int((labels[:, 1:] != -100).sum()),
            "sequence_tokens": input_ids.shape[1]}


def _roundtrip(suffix, path):
    suffix.save_adapter(path)
    before = {name: _tensor_hash(p) for name, p in suffix._adapter_tensors().items()}
    with torch.no_grad():
        for p in suffix.adapter_parameters():
            p.zero_()
    suffix.load_adapter(path)
    after = {name: _tensor_hash(p) for name, p in suffix._adapter_tensors().items()}
    if before != after:
        raise AssertionError("Adapter save/reload changed values")
    return {"exact": True, "sha256_by_tensor": after}


def verify_tiny_persistent_mid_lora(*, verify_cache=False) -> dict:
    """CPU-only native parity and answer-gradient-through-Think proof; no optimizer.

    Positions 2..4 stand for masked Think, 6..7 for answer. Hooking the actual
    adapted shared-down activation proves loss gradients reach earlier positions,
    not merely the last answer position or the adapter's parameter tensor.
    """
    if torch.cuda.is_initialized():
        raise RuntimeError("Tiny verification requires an uninitialized CUDA runtime")
    config = MidLoraConfig(first_layer=1, last_layer=1, rank=2, alpha=4.)
    with _tiny_native_ple_fixture() as (root, reference, ids, history, _delta):
        reference.requires_grad_(False)
        reference.gradient_checkpointing_disable()
        reference.eval()
        with torch.no_grad():
            original = reference.forward_hidden(ids, ple_history=history, ple_mode="off")
        suffix = build_persistent_mid_lora(reference, config)
        try:
            frozen = frozen_fingerprint(reference)
            state = collect_prefix(reference, ids, first_layer=1, ple_history=history)
            with torch.no_grad():
                cached = suffix.forward_hidden(state)
            torch.testing.assert_close(original, cached, rtol=1e-6, atol=1e-6)
            labels = torch.full_like(ids, -100)
            labels[:, 6:] = ids[:, 6:]
            captured = []

            def retain_activation(_module, _args, output):
                if output.requires_grad:
                    output.retain_grad()
                    captured.append(output)

            handle = suffix.layers[0].mlp.shared_expert.down_proj.register_forward_hook(retain_activation)
            try:
                loss = suffix.loss(state, labels, chunk_size=3)
                loss.backward()
            finally:
                handle.remove()
            gradients = [x.grad for x in captured if x.grad is not None]
            think_gradient = max((float(g.reshape(-1, g.shape[-1])[2:5].abs().max())
                                  for g in gradients), default=0.)
            if not math.isfinite(think_gradient) or think_gradient <= 0:
                raise AssertionError("Answer loss failed to reach masked earlier Think activations")
            _assert_only_adapter(reference, suffix)
            parameters = list(suffix.adapter_parameters())
            if any(p.grad is None or not bool(torch.isfinite(p.grad).all()) for p in parameters):
                raise AssertionError("Persistent adapter gradients missing/nonfinite")
            # A second collection must not freeze adapters or change their objects.
            second = collect_prefix(reference, ids, first_layer=1, ple_history=history)
            _assert_only_adapter(reference, suffix)
            with torch.no_grad():
                for parameter in parameters[1::2]:
                    parameter.copy_(torch.arange(parameter.numel()).reshape_as(parameter).sin() * .01)
                full_adapted = reference.forward_hidden(ids, ple_history=history, ple_mode="off")
                cached_adapted = suffix.forward_hidden(second)
            torch.testing.assert_close(full_adapted, cached_adapted, rtol=1e-6, atol=1e-6)
            if torch.equal(full_adapted, original):
                raise AssertionError("Current adapter is not visible to full native forward")
            roundtrip = _roundtrip(suffix, root / "persistent-adapter.safetensors")
            cache_parity = verify_native_generation_cache(reference) if verify_cache else {"passed": False, "reason": "not requested"}
            if frozen != frozen_fingerprint(reference):
                raise AssertionError("Frozen base changed during persistent proof")
            suffix.close()
            try:
                suffix.forward_hidden(second)
            except ValueError:
                pass
            else:
                raise AssertionError("Closed persistent suffix remained usable")
            # Borrower close must not close the reference's checkpoint or PLE.
            with torch.no_grad():
                after_close = reference.forward_hidden(ids, ple_history=history, ple_mode="off")
            torch.testing.assert_close(after_close, full_adapted, rtol=0, atol=0)
            return {"device": "cpu", "optimizer_steps": 0,
                    "cache_full_forward_max_difference": float((cached_adapted - full_adapted).abs().max()),
                    "masked_think_activation_gradient_abs_max": think_gradient,
                    "answer_only_labels": labels.tolist(), "frozen_weights_unchanged": True,
                    "repeat_collection_preserves_trainability": True,
                    "current_adapter_visible_in_full_forward": True,
                    "adapter_roundtrip": roundtrip, "borrowed_resources_closed_once": True,
                    "incremental_native_cache": cache_parity, "gpu_validated": False}
        finally:
            suffix.close()


def _routing_probe(reference, question_ids, *, max_tokens):
    if max_tokens <= 0:
        return {"measured": False, "reason": "routing probe disabled"}
    if len(question_ids) > max_tokens:
        return {"measured": False, "reason": "complete causal question exceeds short-probe limit"}
    counts, route_hashes, handles = {}, {}, []
    for index, layer in enumerate(reference.model.layers):
        def record(module, args, kwargs, layer_index=index):
            routes = args[1] if len(args) > 1 else kwargs["top_k_index"]
            counts[str(layer_index)] = torch.bincount(routes.reshape(-1).long(),
                                                     minlength=module.num_experts).cpu().tolist()
            route_hashes[str(layer_index)] = _tensor_hash(routes)
        handles.append(layer.mlp.experts.register_forward_pre_hook(record, with_kwargs=True))
    try:
        with torch.no_grad():
            reference.forward_hidden(_ids(question_ids, reference), ple_mode="off")
    finally:
        for handle in handles:
            handle.remove()
    return {"measured": True, "question_tokens": len(question_ids), "expert_assignment_counts": counts,
            "token_route_sha256": route_hashes,
            "meaning": "native selected-route counts, not semantic reasoning evidence"}


def _text_metrics(tokenizer, actual, expected, *, terminated):
    """Lexical/structural diagnostics only; never an executed-action success score.

    Action matching drops envelope IDs/caller metadata, not keys inside tool
    inputs. Exact JSON matching remains available as a separate stricter metric.
    """
    text, gold = tokenizer.decode(actual), tokenizer.decode(expected)
    calls = re.findall(r"<tool_call>\s*(.*?)\s*</tool_call>", text, re.S)
    gold_calls = re.findall(r"<tool_call>\s*(.*?)\s*</tool_call>", gold, re.S)

    def reject_constant(value):
        raise ValueError(f"Non-JSON constant: {value}")

    def decode_json(value):
        return json.loads(value, parse_constant=reject_constant)

    def parse(values):
        try:
            return [decode_json(value) for value in values]
        except (ValueError, TypeError):
            return None

    def normalize(values):
        if values is None:
            return None, "invalid_json"
        normalized = []
        for value in values:
            if not isinstance(value, dict):
                return None, "invalid_tool_schema"
            action = value.get("function", value)
            if not isinstance(action, dict) or not isinstance(action.get("name"), str) or not action["name"]:
                return None, "invalid_tool_schema"
            if "arguments" not in action and "input" not in action:
                return None, "missing_tool_arguments"
            arguments = action.get("arguments", action.get("input"))
            if isinstance(arguments, str):
                try:
                    arguments = decode_json(arguments)
                except (ValueError, TypeError):
                    return None, "invalid_arguments_json"
            if not isinstance(arguments, dict):
                return None, "invalid_arguments_schema"
            if "arguments" in action and "input" in action and action["input"] != arguments:
                return None, "conflicting_tool_arguments"
            normalized.append({"name": action["name"], "arguments": arguments})
        return normalized, None

    parsed, gold_parsed = parse(calls), parse(gold_calls)
    normalized, normalization_error = normalize(parsed)
    gold_normalized, gold_normalization_error = normalize(gold_parsed)
    openings, closings = text.count("<tool_call>"), text.count("</tool_call>")
    incomplete_tool = openings != closings or openings != len(calls)
    no_tool = openings == 0 and closings == 0
    if incomplete_tool:
        tool_status = "unterminated_or_malformed_tool_markup"
    elif no_tool:
        tool_status = "no_tool"
    elif normalization_error:
        tool_status = normalization_error
    else:
        tool_status = "valid_tool_calls"
    grams = [tuple(actual[i:i + 4]) for i in range(max(0, len(actual) - 3))]
    return {"token_exact_match": list(actual) == list(expected), "text_exact_match": text == gold,
            "generated_tokens": len(actual), "expected_tokens": len(expected),
            "repeated_4gram_fraction": 0. if not grams else 1. - len(set(grams)) / len(grams),
            "output_terminated": bool(terminated), "output_unterminated": not terminated,
            "no_tool_output": no_tool, "tool_markup_incomplete": incomplete_tool,
            "tool_call_status": tool_status, "tool_call_invalid_json": bool(calls) and parsed is None,
            "gold_has_tool_call": bool(gold_calls), "generated_tool_call_count": len(calls),
            "tool_call_json_valid": bool(calls) and parsed is not None and not incomplete_tool,
            "tool_call_exact_match": bool(gold_calls) and gold_parsed is not None and parsed == gold_parsed and not incomplete_tool,
            "tool_action_normalized_match": bool(gold_calls) and gold_normalized is not None
                and normalized == gold_normalized and not incomplete_tool,
            "tool_action_normalization_error": normalization_error,
            "gold_tool_action_normalization_error": gold_normalization_error,
            "semantic_action_success_measured": False}


def _answer_evaluation_view(assembled, example):
    input_ids = tuple(assembled["input_ids"])
    answer_suffix = tuple(example.answer_suffix_ids)
    if (not answer_suffix or len(input_ids) < len(answer_suffix)
            or input_ids[-len(answer_suffix):] != answer_suffix):
        raise ValueError("answer_suffix_boundary_mismatch")
    suffix_start = len(input_ids) - len(answer_suffix)
    separator_size = example.provenance.get("answer_separator_tokens")
    if type(separator_size) is not int or not 0 <= separator_size < len(answer_suffix):
        raise ValueError("answer_separator_boundary_invalid")
    labels = [-100] * suffix_start + list(example.answer_suffix_labels)
    if len(labels) != len(input_ids):
        raise ValueError("answer_suffix_labels_length_mismatch")
    return {"input_ids": list(input_ids), "labels": labels}, suffix_start + separator_size

@torch.no_grad()
def _answer_nll(reference, suffix, assembled):
    input_ids, labels = _ids(assembled["input_ids"], reference), _ids(assembled["labels"], reference)
    state = collect_prefix(reference, input_ids, first_layer=suffix.config.first_layer)
    loss = suffix.loss(state, labels)
    if not bool(torch.isfinite(loss)):
        raise FloatingPointError("Nonfinite heldout answer NLL")
    return float(loss)


def _think_replay_nll(reference, suffix, example, donor, *, revision, fresh_nll, ordinary_nll):
    """Evaluate captured heldout Think under current weights; never generate/train."""
    from qwen_exo_booster.native_mid_lora_self_think import assemble_self_think
    generation = donor.get("think_generation")
    metadata = {"evaluated_adapter_revision": revision,
                "think_source_adapter_revision": donor.get("adapter_revision"),
                "think_source_sample_hash": donor.get("provenance", {}).get("sample_hash"),
                "target_sample_hash": example.provenance.get("sample_hash"),
                "donor_same_task_group": donor.get("provenance", {}).get("task_group") == example.provenance.get("task_group"),
                "new_generation_calls": 0, "teacher_forced_answer_only": True}
    if not generation or not generation.get("terminated"):
        return {**metadata, "excluded": "baseline_think_missing_or_unterminated"}
    if "excluded" in donor.get("self_think", {}):
        return {**metadata, "excluded": "baseline_think_invalid: " + donor["self_think"]["excluded"]}
    try:
        assembled = assemble_self_think(example, generation["token_ids"],
            adapter_revision=donor["adapter_revision"], generation_seconds=generation["seconds"])
    except ValueError as error:
        return {**metadata, "excluded": str(error)}
    nll = _answer_nll(reference, suffix, assembled)
    return {**metadata, "answer_nll": nll, "think_tokens": len(generation["token_ids"]),
            "nll_minus_current_fresh_think": None if fresh_nll is None else nll - fresh_nll,
            "nll_minus_empty_think": None if ordinary_nll is None else nll - ordinary_nll,
            "semantic_reasoning_proof": False}


def _evaluate(reference, suffix, examples, tokenizer, *, max_answer_tokens, routing_probe_tokens,
              revision, emit, baseline_rows=None, baseline_think_replay=False,
              cyclic_think_replay=False, precomputed_thinks=None, precomputed_source_sha256=None):
    from qwen_exo_booster.native_mid_lora_self_think import assemble_self_think, generation_plan
    rows = []
    if baseline_rows is not None:
        if len(baseline_rows) != len(examples) or any(
                row.get("provenance", {}).get("sample_hash") != example.provenance.get("sample_hash")
                for row, example in zip(baseline_rows, examples)):
            raise ValueError("Heldout baseline Think replay identity/order mismatch")
    end = tuple(tokenizer.encode("<|im_end|>", add_special_tokens=False))
    separator = tuple(tokenizer.encode("\n\n", add_special_tokens=False))
    for index, example in enumerate(examples):
        row = {"index": index, "provenance": example.provenance, "adapter_revision": revision}
        row["routing"] = _routing_probe(reference, example.question_input_ids, max_tokens=routing_probe_tokens)
        for policy in ("ordinary", "self_think"):
            format_diagnostics = {}
            if policy == "ordinary":
                assembled = {"input_ids": example.ordinary_input_ids, "labels": example.ordinary_labels}
            elif precomputed_thinks is not None:
                from qwen_exo_booster.native_precomputed_think import validate_record
                cached = precomputed_thinks[example.provenance["sample_hash"]]
                assembled = validate_record(cached, example, prepared_data_sha256=precomputed_source_sha256)
                row["think_generation"] = {"token_ids": cached["token_ids"], "terminated": True,
                    "seconds": cached["seconds"], "teacher_fingerprint": cached.get("teacher_fingerprint"),
                    "generation_policy": cached.get("origin", "fixed_precomputed_teacher_no_sampling_gradient")}
                format_diagnostics = {"continuous_rollout": False, "fixed_teacher_think": True}
            else:
                plan = generation_plan(example)
                generation = generate_current_adapter(reference, plan.question_input_ids,
                    max_new_tokens=min(plan.allowed_think_tokens + max_answer_tokens,
                                       example.total_tokens - len(plan.question_input_ids)),
                    stop_ids=end, total_tokens=example.total_tokens,
                    phase_boundary_ids=plan.think_close_ids,
                    phase_budget=plan.allowed_think_tokens, continuation_budget=max_answer_tokens)
                close = generation["phase_close_position"]
                row["think_generation"] = {**generation,
                    "token_ids": generation["token_ids"] if close is None else generation["token_ids"][:close],
                    "terminated": close is not None,
                    "seconds": generation["seconds"] if close is None else generation["phase_close_seconds"],
                    "termination_reason": generation["termination_reason"] if close is None else "phase_closed"}
                if close is None:
                    row[policy] = {"excluded": "think_missing_closer", "generation": generation}
                    continue
                try:
                    assembled = assemble_self_think(example, row["think_generation"]["token_ids"],
                        adapter_revision=revision, generation_seconds=row["think_generation"]["seconds"])
                except ValueError as error:
                    row[policy] = {"excluded": str(error), "generation": generation}
                    continue
            # Keep the assembled training labels intact; evaluation masks Think separately.
            nll_assembled, answer_start = _answer_evaluation_view(assembled, example)
            nll = _answer_nll(reference, suffix, nll_assembled)
            expected = list(assembled["input_ids"][answer_start:])
            for position in range(len(expected)):
                if tuple(expected[position:position + len(end)]) == end:
                    expected = expected[:position]
                    break
            if policy == "ordinary" or precomputed_thinks is not None:
                question = tuple(assembled["input_ids"][:answer_start])
                budget = min(max_answer_tokens, example.total_tokens - len(question))
                if budget < 1:
                    row[policy] = {"answer_nll": nll, "excluded": "answer_context_limit"}
                    continue
                generation = generate_current_adapter(reference, question,
                    max_new_tokens=budget, stop_ids=end, total_tokens=example.total_tokens)
                answer_tokens = generation["token_ids"]
            else:
                # Reuse the sampled continuation: no re-prefill, no gold separator injection.
                answer_tokens = generation["token_ids"][generation["phase_close_position"]:]
                separator_present = bool(separator) and tuple(answer_tokens[:len(separator)]) == separator
                format_diagnostics = {"continuous_rollout": True,
                    "answer_phase_tokens_including_separator": len(answer_tokens),
                    "native_separator_present": separator_present,
                    "native_separator_removed_for_metrics": separator_present,
                    "native_separator_expected_ids": list(separator)}
                if separator_present:
                    answer_tokens = answer_tokens[len(separator):]
            actual = answer_tokens[:-len(end)] if generation["terminated"] else answer_tokens
            row[policy] = {"answer_nll": nll, "generation": generation,
                           "format_diagnostics": format_diagnostics,
                           **_text_metrics(tokenizer, actual, expected, terminated=generation["terminated"])}
        if baseline_rows is not None and (baseline_think_replay or cyclic_think_replay):
            fresh_nll = row.get("self_think", {}).get("answer_nll")
            ordinary_nll = row.get("ordinary", {}).get("answer_nll")
            counterfactuals = {}
            if baseline_think_replay:
                counterfactuals["frozen_base_same_question_think"] = _think_replay_nll(
                    reference, suffix, example, baseline_rows[index], revision=revision,
                    fresh_nll=fresh_nll, ordinary_nll=ordinary_nll)
            if cyclic_think_replay:
                if len(baseline_rows) < 2:
                    counterfactuals["frozen_base_cyclic_other_question_think"] = {"excluded": "requires_two_distinct_heldout_examples"}
                else:
                    donor = baseline_rows[(index + 1) % len(baseline_rows)]
                    if donor["provenance"].get("sample_hash") == example.provenance.get("sample_hash"):
                        counterfactuals["frozen_base_cyclic_other_question_think"] = {"excluded": "cyclic_donor_is_same_example"}
                    else:
                        counterfactuals["frozen_base_cyclic_other_question_think"] = _think_replay_nll(
                            reference, suffix, example, donor, revision=revision,
                            fresh_nll=fresh_nll, ordinary_nll=ordinary_nll)
            row["think_counterfactuals"] = counterfactuals
        rows.append(row)
        emit("heldout", row)
    return rows


def verify_tiny_continuous_generation() -> dict:
    """CPU real-native decoding proof, with boundaries chosen from a real sample.

    This checks phase accounting, not semantic Think quality. No sampler mocks,
    forced tokens, optimizer, GPU, or real-model numerical-equivalence claim.
    """
    if torch.cuda.is_initialized():
        raise RuntimeError("Tiny generation verification requires uninitialized CUDA")
    with _tiny_native_ple_fixture() as (_root, reference, ids, _history, _delta):
        reference.requires_grad_(False)
        reference.gradient_checkpointing_disable()
        reference._mid_lora_decode_mode = "native-cache"
        reference._mid_lora_cache_gate_policy = "adapter-revision"
        prefix = ids[0].tolist()
        total = len(prefix) + 6
        # A seven-token marker cannot terminate a six-token real sample.
        stop = (0,) * 7
        plain = generate_current_adapter(reference, prefix, max_new_tokens=6,
            stop_ids=stop, total_tokens=total)
        strict = plain["native_cache_gate"]
        if not strict["passed"] or strict["research_accepted"] or strict["numeric_policy"] != "strict_equivalence":
            raise AssertionError("Default cache policy is not strict")
        boundary = (plain["token_ids"][0],)
        phased = generate_current_adapter(reference, prefix, max_new_tokens=6,
            stop_ids=stop, total_tokens=total, phase_boundary_ids=boundary,
            phase_budget=1, continuation_budget=2)
        if (phased["token_ids"] != plain["token_ids"][:3] or phased["phase_close_position"] != 1
                or phased["continuation_tokens"] != 2 or phased["termination_reason"] != "continuation_budget"
                or phased["phase_close_seconds"] is None):
            raise AssertionError("Continuous phase boundary changed native sampling/accounting")
        absent = next(token for token in range(reference.config.vocab_size) if token not in plain["token_ids"][:2])
        unclosed = generate_current_adapter(reference, prefix, max_new_tokens=6,
            stop_ids=stop, total_tokens=total, phase_boundary_ids=(absent,),
            phase_budget=2, continuation_budget=4)
        if len(unclosed["token_ids"]) != 2 or unclosed["termination_reason"] != "phase_budget_without_closer":
            raise AssertionError("Unclosed phase consumed continuation budget")
        early_stop = generate_current_adapter(reference, prefix, max_new_tokens=6,
            stop_ids=boundary, total_tokens=total, phase_boundary_ids=(absent,),
            phase_budget=2, continuation_budget=4)
        if not early_stop["terminated"] or early_stop["termination_reason"] != "stop_before_phase_close":
            raise AssertionError("Pre-closer stop marker crossed message boundary")
        limited = generate_current_adapter(reference, prefix, max_new_tokens=2,
            stop_ids=stop, total_tokens=len(prefix) + 2, phase_boundary_ids=boundary,
            phase_budget=1, continuation_budget=4)
        if limited["termination_reason"] != "context_limit" or limited["continuation_tokens"] != 1:
            raise AssertionError("Continuation did not respect remaining context")
        reference._mid_lora_allow_numeric_drift_research = True
        research = generate_current_adapter(reference, prefix, max_new_tokens=1,
            stop_ids=stop, total_tokens=total)["native_cache_gate"]
        if (research["reused"] or not research["research_accepted"]
                or not research["prefill_bitwise_equal"] or not research["finite_states_and_logits"]
                or research["numeric_policy"] != "exploratory_cached_decode_drift"):
            raise AssertionError("Research policy reused strict receipt or lost hard gates")
        return {"device": "cpu", "optimizer_steps": 0, "real_native_generation": True,
                "inclusive_phase_boundary": True, "continuous_prefix_tokens_equal": True,
                "unclosed_phase_stops": True, "preclose_stop_stops": True,
                "remaining_context_respected": True, "policy_change_invalidates_receipt": True,
                "strict_cache_gate": strict, "research_cache_gate": research,
                "gpu_validated": False, "real_model_drift_acceptance_validated": False}


def _validate_training_arms(arms):
 try:
  normalized = tuple(arms)
 except TypeError:
  raise ValueError("arms must start with self_think_sft") from None
 if normalized not in {("self_think_sft",), ("self_think_sft", "ordinary_sft")}:
  raise ValueError("arms must be self_think_sft or self_think_sft then ordinary_sft")
 return normalized

def run_experiment(*, profile, output, mode, authorize_gpu=False, authorize_training=False,
                   data=None, lengths=(256, 4096, 8192, 32768), steps=8, heldout_limit=4,
                   learning_rate=1e-5, clip_norm=1., seed=193, max_answer_tokens=128,
                   routing_probe_tokens=512, config=MidLoraConfig(), device="cuda:0",
                   generation_mode="fresh", baseline_think_replay=True,
                   cyclic_think_replay=True, cache_gate_policy="each-prefix", cpu_validation_receipt=None,
                   reference_dtype="bfloat16", fp32_linear_accumulation=False,
                   allow_numeric_drift_research=False, arms=("self_think_sft", "ordinary_sft"),
                   initial_adapter=None, initial_optimizer_steps=0, start_sample_index=0,
                   resume_training_state=None, transfer_dataset=False, generation_expert_backend="reference",
                   precomputed_think=None, precomputed_think_source=None):
    """Capacity backward-only or three-snapshot controlled optimizer experiment.

    Outputs are private, append-only evidence. Capacity inputs are explicitly
    synthetic valid token IDs, not a corpus or quality evaluation. No optimizer
    exists in capacity mode. Training samples share one deterministic order across
    arms; every self-Think sample regenerates with the latest adapter revision.
    """
    if generation_expert_backend not in {"reference", "nvfp4-gemv"}:
        raise ValueError("Unsupported generation expert backend")
    if generation_expert_backend != "reference" and not fp32_linear_accumulation:
        raise ValueError("NVFP4 generation GEMV requires the FP32-accumulation reference")
    if mode not in {"capacity", "experiment"}:
        raise ValueError("mode must be capacity or experiment")
    if generation_mode not in {"fresh", "native-cache"}:
        raise ValueError("generation_mode must be fresh or native-cache")
    if allow_numeric_drift_research and (mode != "experiment" or generation_mode != "native-cache"):
        raise ValueError("Numeric drift research is allowed only for experiment + native-cache")
    arms = _validate_training_arms(arms)
    if initial_adapter is not None and "self_think_sft" not in arms:
        raise ValueError("initial_adapter can only warm-start self_think_sft")
    if type(initial_optimizer_steps) is not int or initial_optimizer_steps < 0:
        raise ValueError("initial_optimizer_steps must be nonnegative")
    if type(start_sample_index) is not int or start_sample_index < 0:
        raise ValueError("start_sample_index must be nonnegative")
    if start_sample_index and initial_adapter is None:
        raise ValueError("start_sample_index requires a warm-start adapter")
    if cache_gate_policy not in {"each-prefix", "adapter-revision"}:
        raise ValueError("cache_gate_policy must be each-prefix or adapter-revision")
    if torch.device(device).type == "cuda" and not authorize_gpu:
        raise PermissionError("GPU allocation requires --authorize-gpu")
    if mode == "experiment" and not authorize_training:
        raise PermissionError("Optimizer updates require --authorize-training")
    if not math.isfinite(learning_rate) or not 0 < learning_rate <= 1e-3:
        raise ValueError("Learning rate must be finite, positive and <=1e-3")
    if not math.isfinite(clip_norm) or clip_norm <= 0 or steps < 1 or heldout_limit < 0:
        raise ValueError("Invalid clipping, steps or heldout limit")
    if max_answer_tokens < 1:
        raise ValueError("Answer generation budget must be positive")
    lengths = tuple(int(x) for x in lengths)
    if mode == "capacity" and (not lengths or tuple(sorted(set(lengths))) != lengths
                               or any(x < 8 or x > 32768 for x in lengths)):
        raise ValueError("Capacity lengths must be strictly increasing between 8 and 32768")
    resume_state = None
    if transfer_dataset and resume_training_state is None:
        raise ValueError("Dataset transfer requires a saved optimizer companion")
    if resume_training_state is not None:
        if mode != "experiment" or tuple(arms) != ("self_think_sft",) or initial_adapter is None or data is None:
            raise ValueError("Full-state resume requires one self_think_sft arm, data and initial adapter")
        with Path(data).open("rb") as source:
            data_sha256 = hashlib.file_digest(source, "sha256").hexdigest()
        resume_state = _load_training_state(resume_training_state, adapter_path=initial_adapter,
            optimizer_step=initial_optimizer_steps, next_sample_index=start_sample_index,
            transfer_dataset=transfer_dataset,
            expected_training={"prepared_sha256": data_sha256, "profile": str(Path(profile).resolve()),
                "config": asdict(config), "seed": seed, "learning_rate": learning_rate, "clip_norm": clip_norm,
                "generation_mode": generation_mode, "cache_gate_policy": cache_gate_policy,
                "allow_numeric_drift_research": allow_numeric_drift_research,
                "reference_dtype": reference_dtype, "fp32_linear_accumulation": fp32_linear_accumulation})
    precomputed = None
    precomputed_source_sha256 = precomputed_sha256 = None
    if (precomputed_think is None) != (precomputed_think_source is None):
        raise ValueError("Precomputed Think requires both cache and original prepared source")
    if precomputed_think is not None:
        if mode != "experiment" or data is None:
            raise ValueError("Precomputed Think requires experiment mode and prepared data")
        from qwen_exo_booster.native_mid_lora_self_think import load_prepared_examples, file_sha256
        from qwen_exo_booster.native_precomputed_think import load_precomputed_thinks
        precomputed_source_sha256 = file_sha256(precomputed_think_source)
        precomputed_sha256 = file_sha256(precomputed_think)
        originals = {x.provenance["sample_hash"]: x for x in load_prepared_examples(precomputed_think_source)}
        selected = load_prepared_examples(data)
        if any(originals.get(x.provenance["sample_hash"]) != x for x in selected):
            raise ValueError("Precomputed subset differs from original prepared examples")
        precomputed = load_precomputed_thinks(precomputed_think, selected,
            prepared_data_sha256=precomputed_source_sha256)
        del originals, selected
    if resume_state is not None and not transfer_dataset:
        if resume_state["training"].get("precomputed_think_sha256") != precomputed_sha256:
            raise ValueError("Changing fixed Think policy requires explicit dataset transfer")
    if data is not None and precomputed is None:
        with Path(data).open(encoding="utf-8") as source:
            if any(json.loads(line).get("schema") == 2 for line in source if line.strip()):
                raise ValueError("Recovered dual-context samples require precomputed full-context Think")
    output = Path(output).resolve()
    repo = Path(__file__).resolve().parents[2]
    if output.is_relative_to(repo):
        raise ValueError("Private experiment output must be outside the source tree")
    output.mkdir(parents=True, exist_ok=False)
    if reference_dtype not in {"bfloat16", "float32"}:
        raise ValueError("reference_dtype must be bfloat16 or float32")
    started = time.perf_counter()
    report = {"mode": mode, "config": asdict(config), "optimizer_steps": 0,
              "seed": seed, "learning_rate": learning_rate, "clip_norm": clip_norm,
              "weight_reference": "NVFP4 decoded W4A16, not serving W4A4",
              "decode_mode": generation_mode,
              "cache_gate_policy": cache_gate_policy,
              "generation_expert_backend": generation_expert_backend,
              "precomputed_think_sha256": precomputed_sha256,
              "think_policy": "fixed_precomputed_teacher_no_sampling_gradient" if precomputed is not None else "fresh_current_adapter",
              "numeric_policy": "exploratory_cached_decode_drift" if allow_numeric_drift_research else "strict_equivalence",
              "allow_numeric_drift_research": bool(allow_numeric_drift_research),
              "native_cache_gates": [],
              "cache_gate_validation_scope": "every_actual_prefix" if cache_gate_policy == "each-prefix"
                  else "representative_actual_prefix_per_adapter_revision",
              "incremental_native_cache_validated": False,
              "limitations": ["NLL and exact/format metrics do not prove semantic reasoning or tool success",
                              "No serving deployment or online environment mutation",
                              "Frozen mapped expert/PLE files are read-only; full file-byte hashing is not performed",
                              "Fresh-prefix Think decoding can cost a full native model pass per token"]}
    report["limitations"].append("Training and cached generation share deterministic chunked QSA selection. Equal-score ties use lower block index, not native varying-width TopK's unspecified tie ordering. Adapter-revision policy validates one representative actual causal prefix per exact adapter tensor revision, not every subsequent prefix; native KV/GDN/PLE state is fresh per generation.")
    report["limitations"].append("Think admissibility/budget reserves the complete gold suffix length; no gold content reaches sampling. " +
        ("Fixed teacher Think is reused; it is not refreshed after learner updates, and evaluation generates actions from that fixed context."
         if precomputed is not None else "Self-Think heldout uses one continuous rollout; its answer budget includes any sampled separator."))
    if allow_numeric_drift_research:
        report["limitations"].append("Exploratory cached before/after policy only: failed fresh/cached numerical equivalence is retained as passed=false. Neither gradient replay equivalence nor fresh-policy equivalence nor online benefit is established.")
    previous_reduction = torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
    previous_tf32 = torch.backends.cuda.matmul.allow_tf32
    if reference_dtype == "float32":
        torch.backends.cuda.matmul.allow_tf32 = False
    if torch.device(device).type == "cuda":
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    report["bf16_reduced_precision_reduction"] = torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
    def emit(event, payload):
        with (output / "events.jsonl").open("a", encoding="utf-8") as target:
            target.write(json.dumps({"event": event, **payload}, ensure_ascii=False, allow_nan=False) + "\n")
            target.flush()
    def generation_progress(payload):
        if "native_cache_gate" in payload:
            report["native_cache_gates"].append(payload["native_cache_gate"])
        emit("generation_progress", payload)
    reference = suffix = None
    try:
        if generation_mode == "native-cache" and mode == "experiment":
            if cpu_validation_receipt is None:
                raise ValueError("Native-cache GPU experiments require an isolated CPU validation receipt")
            raw_receipt = Path(cpu_validation_receipt).read_bytes()
            cpu_proof = json.loads(raw_receipt)
            proof = cpu_proof.get("persistent", {})
            cache_proof = proof.get("incremental_native_cache", {})
            if (proof.get("device") != "cpu" or proof.get("optimizer_steps") != 0
                    or proof.get("gpu_validated") is not False or not cache_proof.get("passed")
                    or not cache_proof.get("greedy_tokens_equal")
                    or not proof.get("frozen_weights_unchanged")
                    or not proof.get("repeat_collection_preserves_trainability")):
                raise ValueError("Invalid isolated CPU validation receipt")
            report["tiny_cache_gate"] = cache_proof
            report["cpu_validation_receipt_sha256"] = hashlib.sha256(raw_receipt).hexdigest()
        torch.manual_seed(seed)
        reference = load_mid_lora_reference(profile, device=device,
            dtype=getattr(torch, reference_dtype) if torch.device(device).type == "cuda" else torch.float32)
        report["reference_activation_dtype"] = str(reference.lm_head.weight.dtype)
        report["weight_reference"] = "Frozen original NVFP4 experts decoded to " + report["reference_activation_dtype"] + "; not serving W4A4"
        if fp32_linear_accumulation:
            from qwen_exo_booster.native_mid_lora_precision import install_frozen_fp32_accumulation
            report["stable_linear_policy"] = install_frozen_fp32_accumulation(reference)
        suffix = build_persistent_mid_lora(reference, config)
        reference._mid_lora_decode_mode = generation_mode
        reference._mid_lora_generation_expert_backend = generation_expert_backend
        reference._mid_lora_cache_gate_policy = cache_gate_policy
        reference._mid_lora_allow_numeric_drift_research = bool(allow_numeric_drift_research)
        reference._mid_lora_generation_progress = generation_progress
        _assert_only_adapter(reference, suffix)
        report["trainable_parameters"] = sum(p.numel() for p in suffix.adapter_parameters())
        report["native_fast_paths"] = getattr(reference, "native_fast_path_report", {})
        base_hashes = frozen_fingerprint(reference)
        report["frozen_initial_sha256"] = base_hashes
        # Only the tiny adapter is copied; all arms share identical immutable base storage.
        pristine = {name: p.detach().clone() for name, p in suffix._adapter_tensors().items()}
        def reset(load_initial=False):
            suffix.zero_grad(set_to_none=True)
            with torch.no_grad():
                for name, parameter in suffix._adapter_tensors().items():
                    parameter.copy_(pristine[name])
            if load_initial:
                if initial_adapter is None:
                    raise ValueError("warm-start adapter is required")
                suffix.load_adapter(initial_adapter)
            torch.manual_seed(seed)
        if mode == "capacity":
            report["capacity"] = []
            for length in lengths:
                if length > reference.config.max_position_embeddings:
                    raise ValueError("Capacity length exceeds model context")
                reset()
                if torch.device(device).type == "cuda":
                    torch.cuda.reset_peak_memory_stats(device)
                _sync(device)
                begin = time.perf_counter()
                ids = (torch.arange(length, device=device) % (reference.config.vocab_size - 1)).reshape(1, -1).long()
                labels = torch.full_like(ids, -100)
                labels[:, length * 3 // 4:] = ids[:, length * 3 // 4:]
                metrics = _backward(reference, suffix, ids, labels, clip_norm=clip_norm)
                _sync(device)
                metrics.update(seconds=time.perf_counter() - begin, memory=_memory(device),
                               synthetic_input=True, optimizer_steps=0)
                report["capacity"].append(metrics)
                emit("capacity", metrics)
                del ids, labels
            report["adapter_roundtrip"] = _roundtrip(suffix, output / "capacity-adapter.safetensors")
        else:
            from transformers import AutoTokenizer
            from qwen_exo_booster.native_mid_lora_self_think import (
                load_prepared_examples, assemble_self_think, generation_plan, validate_prepared_profile,
            )
            if data is None:
                raise ValueError("Prepared Claude conversation data is required")
            with Path(data).open("rb") as source:
                prepared_sha256 = hashlib.file_digest(source, "sha256").hexdigest()
            train = load_prepared_examples(data, split="train")
            heldout = load_prepared_examples(data, split="heldout")[:heldout_limit]
            if not train or (heldout_limit > 0 and not heldout):
                raise ValueError("Nonempty train and requested heldout splits are required")
            report["heldout_evaluation_enabled"] = heldout_limit > 0
            validate_prepared_profile(train + heldout, profile)
            tokenizer = AutoTokenizer.from_pretrained(profile, local_files_only=True, trust_remote_code=False)
            report["baseline"] = _evaluate(reference, suffix, heldout, tokenizer,
                max_answer_tokens=max_answer_tokens, routing_probe_tokens=routing_probe_tokens,
                revision="frozen-base-zero-adapter", emit=emit,
                precomputed_thinks=precomputed, precomputed_source_sha256=precomputed_source_sha256)
            report["heldout_counterfactual_policy"] = {
                "baseline_think_replay": baseline_think_replay and precomputed is None,
                "cyclic_other_heldout_think_replay": cyclic_think_replay and precomputed is None,
                "source": "fixed teacher cache" if precomputed is not None else "captured frozen-base heldout generation only",
                "extra_generation_calls": 0,
                "interpretation": "conditional answer NLL, not semantic reasoning or executed-tool success"}
            report["arms"] = {}
            checkpoint_directory = output / "checkpoints"
            checkpoint_directory.mkdir()
            order = torch.randperm(len(train), generator=torch.Generator().manual_seed(seed)).tolist()
            accepted_indices = []
            # Generate/train first, then replay the identical accepted examples
            # from pristine initialization for the ordinary control. Rejected
            # Think never silently removes examples from just one training arm.
            for arm in arms:
                reset(load_initial=(arm == "self_think_sft" and initial_adapter is not None))
                optimizer = torch.optim.AdamW(list(suffix.adapter_parameters()), lr=learning_rate,
                                              betas=(.9, .999), eps=1e-8, weight_decay=0., foreach=False)
                restored_optimizer_steps = 0
                if resume_state is not None:
                    restored_optimizer_steps = _restore_training_state(resume_state, optimizer,
                        [name + ending for name in suffix.target_names for ending in (".lora_A", ".lora_B")])
                    emit("training_state_restored", {"path": str(resume_training_state),
                        "optimizer_step": initial_optimizer_steps, "optimizer_session_step": restored_optimizer_steps,
                        "next_sample_index": start_sample_index, "adapter_sha256": resume_state["adapter_sha256"],
                        "dataset_transfer": transfer_dataset,
                        "previous_data_sha256": resume_state["training"]["prepared_sha256"],
                        "current_data_sha256": prepared_sha256})
                if arm == "self_think_sft":
                    accepted_indices = []
                    if resume_state is not None and not transfer_dataset:
                        accepted_indices = list(resume_state["progress"]["accepted_example_indices"])
                sample_indices = [order[i % len(order)] for i in range(start_sample_index, start_sample_index + steps)] if arm == "self_think_sft" else list(accepted_indices)
                arm_report = {"optimizer_steps": initial_optimizer_steps if arm == "self_think_sft" and initial_adapter is not None else 0, "attempted_examples": len(sample_indices), "samples": [],
                              "optimizer_steps_this_run": 0, "start_sample_index": start_sample_index if arm == "self_think_sft" else 0,
                              "fresh_pristine_adapter": initial_adapter is None, "warm_start_adapter": initial_adapter}
                report["arms"][arm] = arm_report
                for step, example_index in enumerate(sample_indices):
                    example = train[example_index]
                    revision = f"{arm}:step-{arm_report['optimizer_steps']}"
                    if torch.device(device).type == "cuda":
                        torch.cuda.reset_peak_memory_stats(device)
                    _sync(device)
                    begin = time.perf_counter()
                    sample = {"sample_index": start_sample_index + step if arm == "self_think_sft" else step, "adapter_revision_before": revision,
                              "provenance": example.provenance}
                    if arm == "ordinary_sft":
                        assembled = {"input_ids": example.ordinary_input_ids, "labels": example.ordinary_labels}
                    elif precomputed is not None:
                        from qwen_exo_booster.native_precomputed_think import validate_record
                        cached = precomputed[example.provenance["sample_hash"]]
                        assembled = validate_record(cached, example, prepared_data_sha256=precomputed_source_sha256)
                        sample["precomputed_think"] = {"token_ids_sha256": cached["token_ids_sha256"],
                            "generated_tokens": cached["generated_count"], "teacher_fingerprint": cached.get("teacher_fingerprint"),
                            "supervision": cached.get("supervision", "answer_only"),
                            "reasoning_origin": cached.get("origin", "fixed_teacher"),
                            "generation_seconds_saved": cached["seconds"], "learner_generated_think": False}
                    else:
                        plan = generation_plan(example)
                        generation = generate_current_adapter(reference, plan.question_input_ids,
                            max_new_tokens=plan.allowed_think_tokens, stop_ids=plan.think_close_ids,
                            total_tokens=example.total_tokens)
                        sample["think_generation"] = generation
                        try:
                            assembled = assemble_self_think(example, generation["token_ids"],
                                adapter_revision=revision, generation_seconds=generation["seconds"])
                        except ValueError as error:
                            sample["excluded"] = str(error)
                            arm_report["samples"].append(sample)
                            emit("training_sample", {"arm": arm, **sample})
                            continue
                    metrics = _backward(reference, suffix, _ids(assembled["input_ids"], reference),
                                        _ids(assembled["labels"], reference), clip_norm=clip_norm)
                    optimizer.step()
                    if any(not bool(torch.isfinite(p).all()) for p in suffix.adapter_parameters()):
                        raise FloatingPointError("Nonfinite adapter after optimizer step")
                    arm_report["optimizer_steps"] += 1
                    arm_report["optimizer_steps_this_run"] += 1
                    report["optimizer_steps"] += 1
                    checkpoint_path = checkpoint_directory / f"{arm}-step-{arm_report['optimizer_steps']:06d}.safetensors"
                    pending_path = checkpoint_path.with_suffix(".safetensors.partial")
                    if checkpoint_path.exists() or pending_path.exists():
                        raise FileExistsError(f"Refusing to overwrite step checkpoint: {checkpoint_path}")
                    suffix.save_adapter(pending_path)
                    with pending_path.open("rb") as adapter_file:
                        os.fsync(adapter_file.fileno())
                    pending_path.replace(checkpoint_path)
                    training_state = _save_training_state(
                        checkpoint_path.with_suffix(".training.pt"), optimizer=optimizer,
                        adapter_path=checkpoint_path,
                        parameter_names=[name + suffix_name for name in suffix.target_names
                                         for suffix_name in (".lora_A", ".lora_B")],
                        progress={"arm": arm, "optimizer_step": arm_report["optimizer_steps"],
                                  "optimizer_session_step": restored_optimizer_steps + arm_report["optimizer_steps_this_run"],
                                  "next_sample_index": sample["sample_index"] + 1,
                                  "accepted_example_indices": accepted_indices + [example_index] if arm == "self_think_sft" else list(accepted_indices)},
                        training={"prepared_data": str(Path(data).resolve()), "prepared_sha256": prepared_sha256,
                                  "profile": str(Path(profile).resolve()), "config": asdict(config),
                                  "seed": seed, "learning_rate": learning_rate, "clip_norm": clip_norm,
                                  "generation_mode": generation_mode, "cache_gate_policy": cache_gate_policy,
                                  "generation_expert_backend": generation_expert_backend,
                                  "precomputed_think_sha256": precomputed_sha256,
                                  "precomputed_think_source_sha256": precomputed_source_sha256,
                                  "allow_numeric_drift_research": allow_numeric_drift_research,
                                  "reference_dtype": reference_dtype, "fp32_linear_accumulation": fp32_linear_accumulation,
                                  "initial_adapter": str(initial_adapter) if initial_adapter is not None else None,
                                  "initial_optimizer_steps": initial_optimizer_steps,
                                  "dataset_transfer": transfer_dataset,
                                  "parent_training_state": str(resume_training_state) if resume_training_state is not None else None,
                                  "previous_data_sha256": resume_state["training"]["prepared_sha256"] if resume_state is not None else None,
                                  "start_sample_index": start_sample_index})
                    sample["adapter_checkpoint"] = str(checkpoint_path)
                    emit("adapter_checkpoint", {
                        "arm": arm, "optimizer_step": arm_report["optimizer_steps"],
                        "total_optimizer_steps": report["optimizer_steps"],
                        "sample_index": sample["sample_index"], "next_sample_index": sample["sample_index"] + 1,
                        "optimizer_session_step": restored_optimizer_steps + arm_report["optimizer_steps_this_run"],
                        "adapter_revision": f"{arm}:step-{arm_report['optimizer_steps']}",
                        "path": str(checkpoint_path), "prepared_data": str(Path(data).resolve()),
                        "source": example.provenance, "adapter_only": True,
                        "optimizer_state_saved": True, "training_state": training_state})
                    if arm == "self_think_sft":
                        accepted_indices.append(example_index)
                    _sync(device)
                    sample.update(metrics, seconds=time.perf_counter() - begin, memory=_memory(device),
                                  adapter_revision_after=f"{arm}:step-{arm_report['optimizer_steps']}")
                    arm_report["samples"].append(sample)
                    emit("training_sample", {"arm": arm, **sample})
                arm_report["optimizer_state_steps"] = [int(state["step"].item()) for state in optimizer.state.values()]
                if any(value != restored_optimizer_steps + arm_report["optimizer_steps_this_run"] for value in arm_report["optimizer_state_steps"]):
                    raise AssertionError("AdamW per-parameter update counters disagree")
                del optimizer
                if arm_report["optimizer_steps_this_run"] == 0:
                    raise RuntimeError(f"{arm}: no valid optimizer examples; refusing a fake trained arm")
                arm_report["adapter_roundtrip"] = _roundtrip(suffix, output / f"{arm}.safetensors")
                arm_report["heldout"] = _evaluate(reference, suffix, heldout, tokenizer,
                    max_answer_tokens=max_answer_tokens, routing_probe_tokens=routing_probe_tokens,
                    revision=f"{arm}:step-{arm_report['optimizer_steps']}", emit=emit,
                    baseline_rows=report["baseline"], baseline_think_replay=baseline_think_replay and precomputed is None,
                    cyclic_think_replay=cyclic_think_replay and precomputed is None,
                    precomputed_thinks=precomputed, precomputed_source_sha256=precomputed_source_sha256)
                if frozen_fingerprint(reference) != base_hashes:
                    raise AssertionError("Frozen base changed during optimizer arm")
                arm_report["frozen_weights_unchanged"] = True
            report["matched_training_example_indices"] = accepted_indices
            report["comparison_warning"] = "Both arms use identical accepted examples/order/update counts; selection is conditioned on self-Think termination, and compute/token budgets differ"
        _assert_only_adapter(reference, suffix)
        if frozen_fingerprint(reference) != base_hashes:
            raise AssertionError("Frozen native tensors changed")
        report["frozen_weights_unchanged"] = True
        report["status"] = "completed"
    except BaseException as error:
        report["status"] = "failed"
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        report["seconds"] = time.perf_counter() - started
        report["memory"] = _memory(device)
        if reference is not None:
            report["last_native_cache_gate"] = getattr(reference, "_mid_lora_cache_receipt", None)
            report["incremental_native_cache_validated"] = bool(report["native_cache_gates"]) and all(
                gate["passed"] for gate in report["native_cache_gates"])
            report["incremental_native_cache_research_accepted"] = bool(report["native_cache_gates"]) and all(
                gate.get("research_accepted", False) for gate in report["native_cache_gates"])
        if suffix is not None:
            suffix.close()
        if reference is not None:
            reference.close()
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = previous_reduction
        torch.backends.cuda.matmul.allow_tf32 = previous_tf32
        with (output / "receipt.json").open("x", encoding="utf-8") as target:
            json.dump(report, target, indent=2, ensure_ascii=False, allow_nan=False)
            target.write("\n")
    return report
