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


class IndexedCheckpoint:
    """Strict index/header validation with lazy read-only shard mappings.

    ``read`` copies only the requested byte range to independent CPU storage.
    Metadata and mappings are cached, never decoded tensors. ``close`` releases
    all mappings; tensors returned by ``read`` remain valid after closing.
    """

    def __init__(self, root: str | Path):
        if sys.byteorder != "little":
            raise ValueError("Safetensors reference requires a little-endian host")
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

    def decode_linear(self, prefix, device, dtype):
        """Decode one ephemeral frozen matrix, never activation-quantizing."""
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

    def close(self):
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
        for expert in torch.unique(top_k_index).tolist():
            tokens, slots = torch.where(top_k_index == expert)
            inputs = hidden_states.index_select(0, tokens)
            root = f"{self.prefix}.{expert}"
            gate = frozen_linear(inputs, self.checkpoint, root + ".gate_proj")
            up = frozen_linear(inputs, self.checkpoint, root + ".up_proj")
            values = frozen_linear(F.silu(gate) * up, self.checkpoint, root + ".down_proj")
            values = values * top_k_weights[tokens, slots].to(values.dtype).unsqueeze(-1)
            output.index_add_(0, tokens, values)
        return output
