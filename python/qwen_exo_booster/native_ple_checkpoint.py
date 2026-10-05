"""Read-only mixed Flash-Next weights and activation-differentiable frozen MoE.

This is a W4A16 training reference: packed checkpoint weights are dequantized,
while activations remain ordinary floating point. It is not bit-exact to the
serving W4A4 activation quantizer. No dense expert weights survive a linear
forward or backward, and no checkpoint tensor is a trainable parameter.
"""
from __future__ import annotations

import json
import math
import mmap
import struct
import sys
from collections import OrderedDict
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F


_DTYPES = {
    "BOOL": torch.bool, "U8": torch.uint8, "I8": torch.int8,
    "I16": torch.int16, "I32": torch.int32, "I64": torch.int64,
    "F16": torch.float16, "BF16": torch.bfloat16,
    "F32": torch.float32, "F64": torch.float64,
    "F8_E4M3": torch.float8_e4m3fn, "F8_E5M2": torch.float8_e5m2,
}
_FLOAT_DTYPES = {torch.float16, torch.bfloat16, torch.float32, torch.float64}
_FP8_DTYPES = {torch.float8_e4m3fn, torch.float8_e5m2}
_E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
         -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0)


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def _load_json(data):
    return json.loads(data, object_pairs_hook=_unique_object)


def _signature(stat):
    return stat.st_size, stat.st_mtime_ns, stat.st_dev, stat.st_ino


def _positive_scale(scale, name):
    if not scale.is_floating_point():
        raise ValueError(f"{name}: scale must be floating point")
    value = scale.float()
    if not bool(torch.isfinite(value).all()) or not bool((value > 0).all()):
        raise ValueError(f"{name}: scales must be finite and strictly positive")
    return value


_NVFP4_CUDA_KERNEL = None


def _nvfp4_cuda(weight, blocks, global_scale, shape, dtype):
    """Decode on the current CUDA stream; the optional fused kernel saves scratch."""
    global _NVFP4_CUDA_KERNEL, tl
    if _NVFP4_CUDA_KERNEL is None:
        try:
            import triton
            import triton.language as tl
        except ImportError:
            _NVFP4_CUDA_KERNEL = False
        else:
            @triton.jit
            def decode(P, S, G, O, N: tl.constexpr, BLOCK: tl.constexpr):
                i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
                packed = tl.load(P + i // 2, i < N, 0).to(tl.int32)
                code = (packed >> ((i % 2) * 4)) & 15
                magnitude = code & 7
                value = tl.where(magnitude < 4, magnitude.to(tl.float32) * 0.5,
                                 tl.where(magnitude == 4, 2.0,
                                          tl.where(magnitude == 5, 3.0,
                                                   tl.where(magnitude == 6, 4.0, 6.0))))
                # Set the IEEE sign bit rather than subtracting from zero.
                value = (value.to(tl.int32, bitcast=True) | ((code >> 3) << 31)).to(tl.float32, bitcast=True)
                scale = tl.load(S + i // 16, i < N, 1.0).to(tl.float32)
                scale = scale * tl.load(G).to(tl.float32)
                tl.store(O + i, value * scale, i < N)
            _NVFP4_CUDA_KERNEL = decode
    if _NVFP4_CUDA_KERNEL is not False:
        output = torch.empty(shape, dtype=dtype, device=weight.device)
        count = math.prod(shape)
        _NVFP4_CUDA_KERNEL[((count + 255) // 256,)](
            weight, blocks, global_scale, output, count, 256,
            enable_fp_fusion=False,
        )
        return output
    scales = blocks.float() * global_scale.float().reshape(())
    codes = torch.stack((weight & 15, weight >> 4), dim=-1).reshape(shape)
    lut = torch.tensor(_E2M1, dtype=torch.float32, device=weight.device)
    decoded = lut[codes.long()]
    decoded.reshape(shape[0], -1, 16).mul_(scales.unsqueeze(-1))
    return decoded.to(dtype=dtype)


class IndexedCheckpoint:
    """Strict index/header validation with lazy read-only shard mappings.

    ``read`` copies only the requested byte range to independent CPU storage.
    CUDA decoding uses one checkpoint-wide packed-only LRU (256 MiB by default),
    shared by all layers. Decoded matrices are never cached. Scale contracts are
    validated once per immutable projection, optionally ahead of training via
    ``validate_linear``. Closing or changing the CUDA device clears the LRU.
    """

    def __init__(self, root: str | Path, *, packed_cache_bytes: int = 256 * 1024**2):
        if sys.byteorder != "little":
            raise ValueError("Safetensors reference requires a little-endian host")
        if type(packed_cache_bytes) is not int or packed_cache_bytes < 0:
            raise ValueError("Packed device cache bytes must be a nonnegative integer")
        self.packed_cache_bytes = packed_cache_bytes
        self._packed_cache = OrderedDict()
        self._packed_bytes = 0
        self._packed_device = None
        self._validated_linears = {}
        self.root = Path(root).resolve(strict=True)
        index_path = self.root / "model.safetensors.index.json"
        if not index_path.resolve(strict=True).is_relative_to(self.root):
            raise ValueError("Checkpoint index escapes profile")
        index = _load_json(index_path.read_bytes())
        weight_map = index.get("weight_map") if isinstance(index, dict) else None
        if not isinstance(weight_map, dict) or not weight_map:
            raise ValueError("Checkpoint index lacks a nonempty weight_map")
        self._entries = {}
        self._shards = {}
        self._maps = {}
        self._closed = False
        for name, filename in weight_map.items():
            if not isinstance(name, str) or not name or not isinstance(filename, str):
                raise ValueError("Invalid indexed tensor name or shard filename")
            relative = Path(filename)
            if (relative.is_absolute() or ".." in relative.parts
                    or relative.suffix != ".safetensors"):
                raise ValueError(f"Unsafe checkpoint shard: {filename}")
            path = (self.root / relative).resolve(strict=True)
            if not path.is_relative_to(self.root) or not path.is_file():
                raise ValueError(f"Checkpoint shard escapes profile: {filename}")
            if path not in self._shards:
                self._shards[path] = self._header(path)
            header, base, signature = self._shards[path]
            if name not in header:
                raise ValueError(f"Indexed tensor is missing from shard: {name}")
            self._entries[name] = (path, header[name], base, signature)

    @staticmethod
    def _header(path):
        with path.open("rb") as source:
            stat = path.stat()
            length = source.read(8)
            if len(length) != 8:
                raise ValueError(f"Truncated safetensors header: {path.name}")
            size = struct.unpack("<Q", length)[0]
            if not 0 < size <= 128 * 1024**2 or 8 + size > stat.st_size:
                raise ValueError(f"Invalid safetensors header size: {path.name}")
            header = _load_json(source.read(size))
        if not isinstance(header, dict):
            raise ValueError("Safetensors header must be an object")
        base = 8 + size
        tensors = {}
        ranges = []
        for name, entry in header.items():
            if name == "__metadata__":
                if not isinstance(entry, dict) or not all(
                    isinstance(k, str) and isinstance(v, str) for k, v in entry.items()
                ):
                    raise ValueError("Invalid safetensors string metadata")
                continue
            if not name or not isinstance(entry, dict):
                raise ValueError("Invalid safetensors tensor entry")
            dtype, shape, offsets = entry.get("dtype"), entry.get("shape"), entry.get("data_offsets")
            if not isinstance(dtype, str) or dtype not in _DTYPES:
                raise ValueError(f"Unsupported safetensors dtype: {dtype}")
            if not isinstance(shape, list) or any(type(n) is not int or n < 0 for n in shape):
                raise ValueError(f"Invalid tensor shape: {name}")
            if (not isinstance(offsets, list) or len(offsets) != 2
                    or any(type(n) is not int for n in offsets)):
                raise ValueError(f"Invalid tensor offsets: {name}")
            start, end = offsets
            nbytes = math.prod(shape) * torch.empty((), dtype=_DTYPES[dtype]).element_size()
            if not 0 <= start <= end <= stat.st_size - base or end - start != nbytes:
                raise ValueError(f"Invalid or incomplete tensor bytes: {name}")
            tensors[name] = {"dtype": dtype, "shape": shape, "data_offsets": offsets}
            ranges.append((start, end))
        cursor = 0
        for start, end in sorted(ranges):
            if start != cursor:
                raise ValueError(f"Overlapping or unindexed shard bytes: {path.name}")
            cursor = end
        if cursor != stat.st_size - base:
            raise ValueError(f"Incomplete safetensors byte coverage: {path.name}")
        return tensors, base, _signature(stat)

    def _entry(self, name):
        if self._closed:
            raise RuntimeError("Checkpoint reader is closed")
        return self._entries[name]

    def tensor_names(self):
        if self._closed:
            raise RuntimeError("Checkpoint reader is closed")
        return self._entries.keys()

    def metadata(self, name):
        path, entry, _, _ = self._entry(name)
        return {"dtype": entry["dtype"], "shape": list(entry["shape"]),
                "data_offsets": list(entry["data_offsets"]),
                "shard": str(path.relative_to(self.root))}

    def read(self, name, rows: slice | None = None):
        path, entry, base, signature = self._entry(name)
        if _signature(path.stat()) != signature:
            raise ValueError(f"Checkpoint shard changed after validation: {path.name}")
        shape = list(entry["shape"])
        start, end = entry["data_offsets"]
        dtype = _DTYPES[entry["dtype"]]
        if rows is not None:
            if not isinstance(rows, slice) or not shape or rows.step not in (None, 1):
                raise ValueError("Row reads require a unit-step slice of a nonscalar tensor")
            first, last, _ = rows.indices(shape[0])
            last = max(first, last)
            stride = math.prod(shape[1:]) * torch.empty((), dtype=dtype).element_size()
            end = start + last * stride
            start += first * stride
            shape[0] = last - first
        if start == end:
            return torch.empty(shape, dtype=dtype)
        if path not in self._maps:
            with path.open("rb") as source:
                mapping = mmap.mmap(source.fileno(), 0, access=mmap.ACCESS_READ)
            if _signature(path.stat()) != signature:
                mapping.close()
                raise ValueError(f"Checkpoint shard changed while mapping: {path.name}")
            self._maps[path] = mapping
        # One selected-range copy; no shared writable view of the checkpoint.
        view = memoryview(self._maps[path])[base + start : base + end]
        try:
            storage = bytearray(view)
        finally:
            view.release()
        return torch.frombuffer(storage, dtype=dtype).reshape(shape)

    def linear_shape(self, prefix):
        entry = self.metadata(prefix + ".weight")
        shape = entry["shape"]
        if len(shape) != 2 or min(shape) <= 0:
            raise ValueError(f"{prefix}: linear weight must be a nonempty matrix")
        if entry["dtype"] == "U8":
            shape[1] *= 2
        return tuple(shape)

    def _decode_linear_cpu(self, prefix, device, dtype):
        """Independent original CPU reference for codec and activation parity."""
        if dtype not in _FLOAT_DTYPES:
            raise ValueError("Decoded linear weights require a floating activation dtype")
        shape = self.linear_shape(prefix)
        weight = self.read(prefix + ".weight")
        names = self._entries
        block_name, global_name = prefix + ".weight_scale", prefix + ".weight_scale_2"
        inverse_name, input_name = prefix + ".weight_scale_inv", prefix + ".input_scale"
        if input_name in names:
            input_scale = self.read(input_name)
            if input_scale.numel() != 1:
                raise ValueError(f"{prefix}: input scale must be scalar")
            _positive_scale(input_scale, input_name)
        if weight.dtype == torch.uint8:
            if shape[1] % 16 or inverse_name in names:
                raise ValueError(f"{prefix}: invalid NVFP4 block geometry or ambiguous scales")
            blocks = self.read(block_name)
            global_scale = self.read(global_name)
            if blocks.dtype != torch.float8_e4m3fn or tuple(blocks.shape) != (shape[0], shape[1] // 16):
                raise ValueError(f"{prefix}: NVFP4 requires E4M3 scales per 16 columns")
            if global_scale.dtype != torch.float32 or global_scale.numel() != 1:
                raise ValueError(f"{prefix}: NVFP4 global scale must be scalar FP32")
            scales = _positive_scale(blocks, block_name)
            scales.mul_(_positive_scale(global_scale, global_name).reshape(()))
            lut = torch.tensor(_E2M1, dtype=torch.float32)
            decoded = torch.empty(shape, dtype=torch.float32)
            # First K column occupies the LOW nibble; retain both signed zeros.
            decoded[:, 0::2] = lut[(weight & 15).long()]
            decoded[:, 1::2] = lut[(weight >> 4).long()]
            decoded.reshape(shape[0], -1, 16).mul_(scales.unsqueeze(-1))
        elif weight.dtype in _FP8_DTYPES:
            if block_name in names or global_name in names:
                raise ValueError(f"{prefix}: ambiguous FP8 scale convention")
            scales = self.read(inverse_name)
            expected = ((shape[0] + 127) // 128, (shape[1] + 127) // 128)
            if tuple(scales.shape) != expected:
                raise ValueError(f"{prefix}: FP8 scales require ceil(N/128) by ceil(K/128)")
            scales = _positive_scale(scales, inverse_name)
            decoded = weight.float()
            for row in range(expected[0]):
                for col in range(expected[1]):
                    decoded[row * 128 : (row + 1) * 128, col * 128 : (col + 1) * 128].mul_(scales[row, col])
        elif weight.dtype in _FLOAT_DTYPES:
            if any(name in names for name in (block_name, global_name, inverse_name)):
                raise ValueError(f"{prefix}: unquantized matrix has unexpected weight scales")
            decoded = weight
        else:
            raise ValueError(f"{prefix}: unsupported linear weight dtype {weight.dtype}")
        if not bool(torch.isfinite(decoded).all()):
            raise ValueError(f"{prefix}: nonfinite decoded linear weights")
        return decoded.to(device=device, dtype=dtype).detach()

    def validate_linear(self, prefix):
        """Validate immutable scale contracts once, without decoding normal NVFP4."""
        self._entry(prefix + ".weight")
        if prefix in self._validated_linears:
            contract = self._validated_linears[prefix]
            self._check_linear_shards(contract)
            return contract[0]
        shape = self.linear_shape(prefix)
        names = self._entries
        weight_name = prefix + ".weight"
        kind = _DTYPES[names[weight_name][1]["dtype"]]
        block_name, global_name = prefix + ".weight_scale", prefix + ".weight_scale_2"
        inverse_name, input_name = prefix + ".weight_scale_inv", prefix + ".input_scale"
        tensor_names = [weight_name]
        checked_names = [weight_name]
        if input_name in names:
            input_scale = self.read(input_name)
            if input_scale.numel() != 1:
                raise ValueError(f"{prefix}: input scale must be scalar")
            _positive_scale(input_scale, input_name)
            checked_names.append(input_name)
        if kind == torch.uint8:
            if shape[1] % 16 or inverse_name in names:
                raise ValueError(f"{prefix}: invalid NVFP4 block geometry or ambiguous scales")
            blocks, global_scale = self.read(block_name), self.read(global_name)
            if blocks.dtype != torch.float8_e4m3fn or tuple(blocks.shape) != (shape[0], shape[1] // 16):
                raise ValueError(f"{prefix}: NVFP4 requires E4M3 scales per 16 columns")
            if global_scale.dtype != torch.float32 or global_scale.numel() != 1:
                raise ValueError(f"{prefix}: NVFP4 global scale must be scalar FP32")
            scales = _positive_scale(blocks, block_name)
            scales.mul_(_positive_scale(global_scale, global_name).reshape(()))
            # An analytic bound avoids an expensive CPU nibble decode at startup.
            # Unusual near-overflow checkpoints still get the exact old check.
            if float(scales.max()) > torch.finfo(torch.float32).max / 6:
                self._decode_linear_cpu(prefix, "cpu", torch.float32)
            tensor_names.extend((block_name, global_name))
        elif kind in _FP8_DTYPES:
            if block_name in names or global_name in names:
                raise ValueError(f"{prefix}: ambiguous FP8 scale convention")
            scales = self.read(inverse_name)
            expected = ((shape[0] + 127) // 128, (shape[1] + 127) // 128)
            if tuple(scales.shape) != expected:
                raise ValueError(f"{prefix}: FP8 scales require ceil(N/128) by ceil(K/128)")
            scales = _positive_scale(scales, inverse_name)
            weight = self.read(weight_name).float()
            if not bool(torch.isfinite(weight).all()):
                raise ValueError(f"{prefix}: nonfinite decoded linear weights")
            if float(weight.abs().max()) * float(scales.max()) > torch.finfo(torch.float32).max:
                self._decode_linear_cpu(prefix, "cpu", torch.float32)
            tensor_names.append(inverse_name)
        elif kind in _FLOAT_DTYPES:
            if any(name in names for name in (block_name, global_name, inverse_name)):
                raise ValueError(f"{prefix}: unquantized matrix has unexpected weight scales")
            if not bool(torch.isfinite(self.read(weight_name)).all()):
                raise ValueError(f"{prefix}: nonfinite decoded linear weights")
        else:
            raise ValueError(f"{prefix}: unsupported linear weight dtype {kind}")
        checked_names.extend(tensor_names[1:])
        shards = tuple({names[name][0]: names[name][3] for name in checked_names}.items())
        contract = (shape, kind, tuple(tensor_names), shards)
        self._check_linear_shards(contract)
        self._validated_linears[prefix] = contract
        return shape

    @staticmethod
    def _check_linear_shards(contract):
        for path, signature in contract[3]:
            if _signature(path.stat()) != signature:
                raise ValueError(f"Checkpoint shard changed after validation: {path.name}")

    def clear_device_cache(self):
        """Release packed references; record_stream keeps in-flight users safe."""
        self._packed_cache.clear()
        self._packed_bytes = 0
        self._packed_device = None

    def device_cache_info(self):
        return {"bytes": self._packed_bytes, "limit_bytes": self.packed_cache_bytes,
                "entries": len(self._packed_cache), "device": str(self._packed_device)}

    def _cuda_tensors(self, prefix, device):
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("Frozen checkpoint decoding does not support CUDA graph capture")
        if device != self._packed_device:
            self.clear_device_cache()
            self._packed_device = device
        stream = torch.cuda.current_stream(device)
        if prefix in self._packed_cache:
            tensors, ready, _ = self._packed_cache[prefix]
            self._packed_cache.move_to_end(prefix)
            stream.wait_event(ready)
        else:
            contract = self._validated_linears[prefix]
            names = contract[2]
            nbytes = sum(self._entries[name][1]["data_offsets"][1]
                         - self._entries[name][1]["data_offsets"][0] for name in names)
            while self._packed_cache and self._packed_bytes + nbytes > self.packed_cache_bytes:
                self._packed_bytes -= self._packed_cache.popitem(last=False)[1][2]
            tensors = tuple(self.read(name).to(device=device).contiguous() for name in names)
            if contract[1] not in _FLOAT_DTYPES and nbytes <= self.packed_cache_bytes:
                ready = torch.cuda.Event()
                ready.record(stream)
                self._packed_cache[prefix] = (tensors, ready, nbytes)
                self._packed_bytes += nbytes
        for tensor in tensors:
            tensor.record_stream(stream)
        return tensors

    def decode_linear(self, prefix, device, dtype):
        """Decode ephemeral W4A16 weights on the requested execution device."""
        if dtype not in _FLOAT_DTYPES:
            raise ValueError("Decoded linear weights require a floating activation dtype")
        device = torch.device(device)
        if device.type != "cuda":
            return self._decode_linear_cpu(prefix, device, dtype)
        if device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
        self.validate_linear(prefix)
        shape, kind, _, _ = self._validated_linears[prefix]
        with torch.cuda.device(device), torch.no_grad():
            tensors = self._cuda_tensors(prefix, device)
            if kind == torch.uint8:
                return _nvfp4_cuda(*tensors, shape, dtype).detach()
            if kind in _FP8_DTYPES:
                weight, scales = tensors
                rows = torch.arange(shape[0], device=device) // 128
                cols = torch.arange(shape[1], device=device) // 128
                # Advanced indexing works for noncontiguous/ragged 128x128 grids.
                decoded = weight.float() * scales.float()[rows[:, None], cols[None, :]]
                return decoded.to(dtype=dtype).detach()
            # Never retain a decoded floating matrix in the packed-only LRU.
            return tensors[0].to(dtype=dtype).detach()

    def close(self):
        self.clear_device_cache()
        self._validated_linears.clear()
        for mapping in self._maps.values():
            mapping.close()
        self._maps.clear()
        self._closed = True

    def __enter__(self):
        if self._closed:
            raise RuntimeError("Checkpoint reader is closed")
        return self

    def __exit__(self, *exc):
        self.close()


class _FrozenLinear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs, checkpoint, prefix):
        weight = checkpoint.decode_linear(prefix, inputs.device, inputs.dtype)
        ctx.checkpoint, ctx.prefix = checkpoint, prefix
        ctx.input_dtype, ctx.input_device = inputs.dtype, inputs.device
        return F.linear(inputs, weight)

    @staticmethod
    def backward(ctx, grad_output):
        if not ctx.needs_input_grad[0]:
            return None, None, None
        weight = ctx.checkpoint.decode_linear(ctx.prefix, ctx.input_device, ctx.input_dtype)
        return grad_output.matmul(weight), None, None


def frozen_linear(inputs, checkpoint: IndexedCheckpoint, prefix: str):
    """Linear with exact dequantized-reference activation gradients, no weight grads."""
    if inputs.requires_grad and torch.is_grad_enabled():
        return _FrozenLinear.apply(inputs, checkpoint, prefix)
    weight = checkpoint.decode_linear(prefix, inputs.device, inputs.dtype)
    return F.linear(inputs, weight)


def validate_cuda_linear_parity(checkpoint, prefix, *, device="cuda", dtype=torch.bfloat16):
    """Opt-in, content-blind real-projection smoke; returns only numeric metrics.

    The oracle is original CPU dequantization followed by the same CUDA linear
    arithmetic. This checks W4A16 codec/fwd/dx, not W4A4 serving equivalence.
    It allocates two ephemeral decoded projections and four synthetic inputs.
    """
    device = torch.device(device)
    if device.type != "cuda":
        raise ValueError("CUDA parity requires a CUDA device")
    reference = checkpoint.decode_linear(prefix, "cpu", dtype).to(device)
    actual = checkpoint.decode_linear(prefix, device, dtype)
    assert actual.dtype == dtype and not actual.requires_grad
    torch.testing.assert_close(actual, reference, rtol=0, atol=0)
    assert torch.equal(actual.signbit(), reference.signbit())
    generator = torch.Generator(device="cpu").manual_seed(1729)
    inputs = torch.randn(4, reference.shape[1], generator=generator).to(device=device, dtype=dtype).requires_grad_()
    upstream = torch.randn(4, reference.shape[0], generator=generator).to(device=device, dtype=dtype)
    saved = []
    with torch.autograd.graph.saved_tensors_hooks(lambda t: saved.append(t) or t, lambda t: t):
        output = frozen_linear(inputs, checkpoint, prefix)
    assert saved == [], "Frozen linear retained a dense activation/weight tensor"
    expected = F.linear(inputs.detach(), reference)
    output.backward(upstream)
    expected_grad = upstream @ reference
    torch.testing.assert_close(output, expected, rtol=0, atol=0)
    torch.testing.assert_close(inputs.grad, expected_grad, rtol=0, atol=0)
    cache = checkpoint.device_cache_info()
    assert cache["bytes"] <= cache["limit_bytes"]
    return {"shape": list(reference.shape), "dtype": str(dtype),
            "decode_max_abs_error": float((actual - reference).abs().max()),
            "forward_max_abs_error": float((output - expected).abs().max()),
            "dx_max_abs_error": float((inputs.grad - expected_grad).abs().max()),
            "packed_cache": cache}


class FrozenCheckpointExperts(nn.Module):
    """HF eager experts API, preserving every top-K route and router derivative."""

    def __init__(self, checkpoint, prefix, num_experts, hidden_size, intermediate_size, activation="silu"):
        super().__init__()
        if activation != "silu":
            raise ValueError("Native Flash-Next expert reference requires silu")
        if min(num_experts, hidden_size, intermediate_size) <= 0:
            raise ValueError("Expert dimensions must be positive")
        self.checkpoint, self.prefix = checkpoint, prefix
        self.num_experts, self.hidden_size = num_experts, hidden_size
        self.intermediate_size = intermediate_size
        for expert in range(num_experts):
            root = f"{prefix}.{expert}"
            for part, expected in (("gate_proj", (intermediate_size, hidden_size)),
                                   ("up_proj", (intermediate_size, hidden_size)),
                                   ("down_proj", (hidden_size, intermediate_size))):
                if checkpoint.linear_shape(f"{root}.{part}") != expected:
                    raise ValueError(f"{root}.{part}: checkpoint expert geometry mismatch")

    def forward(self, hidden_states, top_k_index, top_k_weights):
        if hidden_states.ndim != 2 or hidden_states.shape[1] != self.hidden_size:
            raise ValueError("Expert inputs require [tokens, hidden_size]")
        if (top_k_index.ndim != 2 or top_k_index.shape != top_k_weights.shape
                or top_k_index.shape[0] != hidden_states.shape[0]):
            raise ValueError("Expert routes and weights require matching [tokens, top_k]")
        if (top_k_index.dtype not in (torch.int32, torch.int64)
                or top_k_index.device != hidden_states.device
                or top_k_weights.device != hidden_states.device):
            raise ValueError("Expert indices must be integer and route tensors must share the input device")
        if not top_k_weights.is_floating_point():
            raise ValueError("Router weights must be floating point")
        if top_k_index.numel() and (bool((top_k_index < 0).any()) or bool((top_k_index >= self.num_experts).any())):
            raise ValueError("Expert route is outside checkpoint expert range")
        # This zero-valued path also keeps empty batches connected to autograd.
        output = hidden_states * 0 + top_k_weights.sum().to(hidden_states.dtype) * 0
        # One route sort and one small count transfer replace per-expert where()
        # synchronization. Stable order preserves repeated routes and accumulation.
        routes = top_k_index.reshape(-1).long()
        order = torch.argsort(routes, stable=True)
        counts = torch.bincount(routes, minlength=self.num_experts).tolist()
        offset = 0
        for expert, count in enumerate(counts):
            if not count:
                continue
            selected = order[offset : offset + count]
            offset += count
            tokens = selected // top_k_index.shape[1]
            slots = selected % top_k_index.shape[1]
            inputs = hidden_states.index_select(0, tokens)
            root = f"{self.prefix}.{expert}"
            gate = frozen_linear(inputs, self.checkpoint, root + ".gate_proj")
            up = frozen_linear(inputs, self.checkpoint, root + ".up_proj")
            values = frozen_linear(F.silu(gate) * up, self.checkpoint, root + ".down_proj")
            values = values * top_k_weights[tokens, slots].to(values.dtype).unsqueeze(-1)
            output.index_add_(0, tokens, values)
        return output
