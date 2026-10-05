"""CPU codec checks and opt-in CUDA parity for the frozen checkpoint reference."""
import json
import struct

import pytest
import torch
from torch.nn import functional as F

from qwen_exo_booster.native_ple_checkpoint import (
    FrozenCheckpointExperts,
    IndexedCheckpoint,
    frozen_linear,
    validate_cuda_linear_parity,
)


_SAFETENSOR_DTYPES = {
    torch.uint8: "U8", torch.float8_e4m3fn: "F8_E4M3",
    torch.float8_e5m2: "F8_E5M2", torch.bfloat16: "BF16",
    torch.float16: "F16", torch.float32: "F32", torch.float64: "F64",
}


def _checkpoint(root, tensors, **reader_options):
    header, body, offset = {}, [], 0
    for name, tensor in tensors.items():
        tensor = tensor.detach().contiguous()
        raw = tensor.reshape(-1).view(torch.uint8).numpy().tobytes()
        header[name] = {"dtype": _SAFETENSOR_DTYPES[tensor.dtype],
                        "shape": list(tensor.shape),
                        "data_offsets": [offset, offset + len(raw)]}
        body.append(raw)
        offset += len(raw)
    data = json.dumps(header, separators=(",", ":")).encode()
    data += b" " * (-len(data) % 8)
    (root / "weights.safetensors").write_bytes(struct.pack("<Q", len(data)) + data + b"".join(body))
    (root / "model.safetensors.index.json").write_text(json.dumps({
        "weight_map": {name: "weights.safetensors" for name in tensors}
    }), encoding="utf-8")
    return IndexedCheckpoint(root, **reader_options)


def _nvfp4(prefix="linear", *, scale=2.0, global_scale=0.25):
    # Every E2M1 code appears exactly once; opposite nibbles differ, including +/-0.
    return {
        prefix + ".weight": torch.tensor([[0x10, 0x32, 0x54, 0x76,
                                           0x98, 0xBA, 0xDC, 0xFE]], dtype=torch.uint8),
        prefix + ".weight_scale": torch.tensor([[scale]], dtype=torch.float8_e4m3fn),
        prefix + ".weight_scale_2": torch.tensor(global_scale, dtype=torch.float32),
        # Nonunit activation calibration must NOT multiply decoded W4A16 weights.
        prefix + ".input_scale": torch.tensor(17.0, dtype=torch.float32),
    }


def test_nvfp4_signed_zero_nibble_order_and_both_scale_factors(tmp_path):
    with _checkpoint(tmp_path, _nvfp4()) as checkpoint:
        actual = checkpoint.decode_linear("linear", "cpu", torch.float32)
    expected = torch.tensor([[0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0,
                              -0.0, -0.25, -0.5, -0.75, -1.0, -1.5, -2.0, -3.0]])
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert not actual[0, 0].signbit() and actual[0, 8].signbit()


def test_nvfp4_distinct_column_blocks_and_rows(tmp_path):
    tensors = _nvfp4()
    tensors["linear.weight"] = tensors["linear.weight"].repeat(2, 2)
    tensors["linear.weight_scale"] = torch.tensor([[1.0, 2.0], [4.0, 8.0]], dtype=torch.float8_e4m3fn)
    tensors["linear.weight_scale_2"] = torch.tensor(0.5)
    with _checkpoint(tmp_path, tensors) as checkpoint:
        actual = checkpoint.decode_linear("linear", "cpu", torch.float64)
    codes = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
                          -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0], dtype=torch.float64)
    expected = torch.stack((torch.cat((codes * 0.5, codes)),
                            torch.cat((codes * 2, codes * 4))))
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.float8_e4m3fn, torch.float8_e5m2])
def test_fp8_ragged_128_blocks_apply_inverse_named_dequant_scale(tmp_path, dtype):
    values = torch.ones(129, 131, dtype=torch.float32)
    values[0, 0], values[128, 130] = -2.0, -4.0
    tensors = {"fp8.weight": values.to(dtype),
               "fp8.weight_scale_inv": torch.tensor([[0.5, 2.0], [4.0, 8.0]], dtype=torch.bfloat16)}
    with _checkpoint(tmp_path, tensors) as checkpoint:
        actual = checkpoint.decode_linear("fp8", "cpu", torch.float32)
    expected = torch.empty_like(values)
    expected[:128, :128] = values[:128, :128] * 0.5
    expected[:128, 128:] = values[:128, 128:] * 2.0
    expected[128:, :128] = values[128:, :128] * 4.0
    expected[128:, 128:] = values[128:, 128:] * 8.0
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32, torch.float64])
def test_plain_weight_row_reads_are_independent_and_survive_close(tmp_path, dtype):
    source = torch.arange(15, dtype=torch.float32).reshape(5, 3).to(dtype)
    checkpoint = _checkpoint(tmp_path, {"plain.weight": source})
    selected = checkpoint.read("plain.weight", rows=slice(1, 4))
    selected[0, 0] = -100
    torch.testing.assert_close(checkpoint.read("plain.weight"), source)
    empty = checkpoint.read("plain.weight", rows=slice(4, 2))
    assert empty.shape == (0, 3)
    torch.testing.assert_close(checkpoint.decode_linear("plain", "cpu", torch.float64), source.double())
    checkpoint.close()
    assert selected[0, 0] == -100
    with pytest.raises(RuntimeError, match="closed"):
        checkpoint.read("plain.weight")


@pytest.mark.parametrize("scale_key,value", [
    ("weight_scale", 0.0), ("weight_scale", -1.0),
    ("weight_scale", float("nan")), ("weight_scale_2", 0.0),
    ("weight_scale_2", float("inf")), ("input_scale", -2.0),
])
def test_invalid_quantization_scales_fail_closed(tmp_path, scale_key, value):
    tensors = _nvfp4()
    tensors["linear." + scale_key].fill_(value)
    with _checkpoint(tmp_path, tensors) as checkpoint:
        with pytest.raises(ValueError, match="finite and strictly positive"):
            checkpoint.decode_linear("linear", "cpu", torch.float32)


def test_scale_geometry_and_missing_scale_rejected(tmp_path):
    tensors = _nvfp4()
    tensors["linear.weight_scale"] = torch.ones(1, 2, dtype=torch.float8_e4m3fn)
    with _checkpoint(tmp_path, tensors) as checkpoint:
        with pytest.raises(ValueError, match="per 16 columns"):
            checkpoint.decode_linear("linear", "cpu", torch.float32)
    del tensors["linear.weight_scale_2"]
    with _checkpoint(tmp_path, tensors) as checkpoint:
        with pytest.raises(KeyError):
            checkpoint.decode_linear("linear", "cpu", torch.float32)


def test_frozen_linear_backward_redecodes_without_saving_dense_weights(tmp_path):
    with _checkpoint(tmp_path, _nvfp4()) as checkpoint:
        weight = checkpoint.decode_linear("linear", "cpu", torch.float64)
        inputs = torch.linspace(-1, 1, 32, dtype=torch.float64).reshape(2, 16).requires_grad_()
        saved = []
        with torch.autograd.graph.saved_tensors_hooks(lambda t: saved.append(t) or t, lambda t: t):
            output = frozen_linear(inputs, checkpoint, "linear")
        assert saved == []
        torch.testing.assert_close(output, F.linear(inputs, weight), rtol=0, atol=0)
        upstream = torch.tensor([[0.25], [-0.5]], dtype=torch.float64)
        output.backward(upstream)
        torch.testing.assert_close(inputs.grad, upstream @ weight, rtol=0, atol=0)
        assert not weight.requires_grad
        no_grad_output = frozen_linear(inputs.detach(), checkpoint, "linear")
        assert not no_grad_output.requires_grad


def test_experts_keep_all_ten_routes_repeats_and_router_gradients(tmp_path):
    tensors, weights = {}, {}
    for expert in range(3):
        parts = {
            "gate_proj": torch.tensor([[0.2, -0.1], [0.3, 0.4], [-0.2, 0.5]], dtype=torch.float64) + expert * 0.05,
            "up_proj": torch.tensor([[0.5, 0.1], [-0.4, 0.2], [0.3, -0.2]], dtype=torch.float64) - expert * 0.03,
            "down_proj": torch.tensor([[0.2, -0.1, 0.4], [0.3, 0.5, -0.2]], dtype=torch.float64) + expert * 0.02,
        }
        weights[expert] = parts
        tensors.update({f"experts.{expert}.{part}.weight": weight for part, weight in parts.items()})
    with _checkpoint(tmp_path, tensors) as checkpoint:
        experts = FrozenCheckpointExperts(checkpoint, "experts", 3, 2, 3)
        inputs = torch.tensor([[0.4, -0.8], [-0.2, 0.6]], dtype=torch.float64, requires_grad=True)
        routes = torch.tensor([[0, 1, 0, 2, 2, 1, 0, 1, 2, 0], [2, 2, 1, 0, 0, 1, 2, 0, 1, 2]])
        routing = torch.linspace(0.01, 0.2, 20, dtype=torch.float64).reshape(2, 10).requires_grad_()
        actual = experts(inputs, routes, routing)
        ref_inputs = inputs.detach().clone().requires_grad_()
        ref_routing = routing.detach().clone().requires_grad_()
        rows = []
        for token in range(2):
            row = torch.zeros(2, dtype=torch.float64)
            for slot in range(10):
                part = weights[routes[token, slot].item()]
                x = ref_inputs[token]
                value = F.linear(F.silu(F.linear(x, part["gate_proj"])) * F.linear(x, part["up_proj"]), part["down_proj"])
                row = row + value * ref_routing[token, slot]
            rows.append(row)
        expected = torch.stack(rows)
        torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
        upstream = torch.tensor([[1.0, -0.4], [0.2, 0.7]], dtype=torch.float64)
        actual.backward(upstream)
        expected.backward(upstream)
        torch.testing.assert_close(inputs.grad, ref_inputs.grad, rtol=1e-12, atol=1e-12)
        torch.testing.assert_close(routing.grad, ref_routing.grad, rtol=1e-12, atol=1e-12)
        assert list(experts.parameters()) == []
        empty = inputs.detach()[:0].requires_grad_()
        empty_routes = routes[:0]
        empty_weights = routing.detach()[:0].requires_grad_()
        experts(empty, empty_routes, empty_weights).sum().backward()
        assert empty.grad.shape == (0, 2) and empty_weights.grad.shape == (0, 10)


@pytest.mark.parametrize("corruption", ["truncated", "trailing", "overlap", "missing", "escaping", "duplicate"])
def test_corrupt_or_escaping_checkpoint_rejected(tmp_path, corruption):
    checkpoint = _checkpoint(tmp_path, {"a.weight": torch.ones(2, 2), "b.weight": torch.zeros(2, 2)})
    checkpoint.close()
    shard = tmp_path / "weights.safetensors"
    raw = shard.read_bytes()
    size = struct.unpack("<Q", raw[:8])[0]
    header = json.loads(raw[8:8 + size])
    index = tmp_path / "model.safetensors.index.json"
    if corruption == "truncated":
        shard.write_bytes(raw[:-1])
    elif corruption == "trailing":
        shard.write_bytes(raw + b"\0")
    elif corruption == "overlap":
        header["b.weight"]["data_offsets"] = header["a.weight"]["data_offsets"]
        encoded = json.dumps(header).encode()
        shard.write_bytes(struct.pack("<Q", len(encoded)) + encoded + raw[8 + size:])
    elif corruption == "missing":
        index.write_text(json.dumps({"weight_map": {"absent.weight": "weights.safetensors"}}), encoding="utf-8")
    elif corruption == "escaping":
        index.write_text(json.dumps({"weight_map": {"a.weight": "../escape.safetensors"}}), encoding="utf-8")
    else:
        index.write_text('{"weight_map":{"a.weight":"weights.safetensors","a.weight":"weights.safetensors"}}', encoding="utf-8")
    with pytest.raises(ValueError):
        IndexedCheckpoint(tmp_path)


def test_changed_shard_cannot_be_read_from_stale_metadata(tmp_path):
    with _checkpoint(tmp_path, {"a.weight": torch.ones(2, 2)}) as checkpoint:
        shard = tmp_path / "weights.safetensors"
        with shard.open("ab") as source:
            source.write(b"\0")
        with pytest.raises(ValueError, match="changed after validation"):
            checkpoint.read("a.weight")


def test_cached_contract_still_rejects_modified_shard(tmp_path):
    with _checkpoint(tmp_path, _nvfp4()) as checkpoint:
        assert checkpoint.validate_linear("linear") == (1, 16)
        shard = tmp_path / "weights.safetensors"
        with shard.open("ab") as source:
            source.write(b"\0")
        with pytest.raises(ValueError, match="changed after validation"):
            checkpoint.validate_linear("linear")


@pytest.mark.parametrize("scale_key,value", [
    ("weight_scale", 0.0), ("weight_scale", float("nan")),
    ("weight_scale_2", float("inf")), ("input_scale", -2.0),
])
def test_startup_contract_rejects_invalid_scales(tmp_path, scale_key, value):
    tensors = _nvfp4()
    tensors["linear." + scale_key].fill_(value)
    with _checkpoint(tmp_path, tensors) as checkpoint:
        with pytest.raises(ValueError, match="finite and strictly positive"):
            checkpoint.validate_linear("linear")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA codec validation")
@pytest.mark.parametrize("codec", ["nvfp4", "fp8_e4m3", "fp8_e5m2", "bf16"])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_cuda_codec_forward_and_activation_backward(tmp_path, codec, dtype):
    if codec == "nvfp4":
        tensors = _nvfp4(global_scale=0.13)
        tensors["linear.weight"] = tensors["linear.weight"].repeat(129, 4)
        scales = torch.tensor([[0.5, 1.25, 2.75, 16.0]], dtype=torch.float8_e4m3fn)
        tensors["linear.weight_scale"] = scales.repeat(129, 1)
    else:
        # Noncontiguous source plus ragged final row/column blocks.
        values = torch.linspace(-3, 3, 129 * 131).reshape(131, 129).T
        weight_dtype = {"fp8_e4m3": torch.float8_e4m3fn,
                        "fp8_e5m2": torch.float8_e5m2,
                        "bf16": torch.bfloat16}[codec]
        tensors = {"linear.weight": values.to(weight_dtype)}
        if codec != "bf16":
            tensors["linear.weight_scale_inv"] = torch.tensor([[0.5, 2.0], [4.0, 8.0]]).T
    with _checkpoint(tmp_path, tensors) as checkpoint:
        result = validate_cuda_linear_parity(checkpoint, "linear", dtype=dtype)
        assert result["decode_max_abs_error"] == 0
        assert result["forward_max_abs_error"] == 0
        assert result["dx_max_abs_error"] == 0
        if codec == "bf16":
            assert checkpoint.device_cache_info()["bytes"] == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA codec validation")
def test_cuda_tensor_decoder_retains_signed_zero_and_scale_order(tmp_path, monkeypatch):
    import qwen_exo_booster.native_ple_checkpoint as codec
    monkeypatch.setattr(codec, "_NVFP4_CUDA_KERNEL", False)
    with _checkpoint(tmp_path, _nvfp4(global_scale=0.13)) as checkpoint:
        validate_cuda_linear_parity(checkpoint, "linear", dtype=torch.float32)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA cache validation")
@pytest.mark.parametrize("cache_bytes", [0, 12, 26])
def test_cuda_eviction_and_cross_stream_backward_redecode(tmp_path, cache_bytes):
    tensors = {}
    for i in range(3):
        tensors.update(_nvfp4(f"linear{i}", global_scale=0.25 * (i + 1)))
    checkpoint = _checkpoint(tmp_path, tensors, packed_cache_bytes=cache_bytes)
    try:
        inputs = torch.arange(16, device="cuda", dtype=torch.float32).reshape(1, 16).requires_grad_()
        reference = checkpoint.decode_linear("linear0", "cpu", torch.float32).cuda()
        first = torch.cuda.Stream()
        first.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(first):
            output = frozen_linear(inputs, checkpoint, "linear0")
        torch.cuda.current_stream().wait_stream(first)
        # Consume a cache hit on another stream, then force its eviction.
        torch.testing.assert_close(checkpoint.decode_linear("linear0", "cuda", torch.float32), reference)
        for prefix in ("linear1", "linear2"):
            validate_cuda_linear_parity(checkpoint, prefix, dtype=torch.float32)
            assert checkpoint.device_cache_info()["bytes"] <= cache_bytes
        output.sum().backward()
        torch.testing.assert_close(output, F.linear(inputs.detach(), reference), rtol=0, atol=0)
        torch.testing.assert_close(inputs.grad, reference, rtol=0, atol=0)
        assert checkpoint.device_cache_info()["bytes"] <= cache_bytes
        checkpoint.clear_device_cache()
        assert checkpoint.device_cache_info()["bytes"] == 0
        torch.testing.assert_close(checkpoint.decode_linear("linear0", "cuda", torch.float32), reference)
    finally:
        checkpoint.close()
    assert checkpoint.device_cache_info()["bytes"] == 0
    with pytest.raises(RuntimeError, match="closed"):
        checkpoint.decode_linear("linear0", "cuda", torch.float32)
