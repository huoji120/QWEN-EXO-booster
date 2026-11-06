"""Synthetic generation-only NVFP4 expert parity; no checkpoints or serving imports."""
import math

import pytest
import torch
from torch.nn import functional as F

from qwen_exo_booster.native_mid_lora_decode import grouped_nvfp4_decode


# Independent literal E2M1 table: byte low nibble precedes high nibble.
_CODES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
          -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0)
_CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")


def _decode(triplet):
    packed, blocks, global_scale = (t.detach().cpu() for t in triplet)
    indices = torch.stack((packed & 15, packed >> 4), dim=-1).reshape(packed.shape[0], -1)
    values = torch.tensor(_CODES, dtype=torch.float32)[indices.long()]
    scales = (blocks.float() * global_scale.float()).repeat_interleave(16, dim=1)
    return (values * scales).to(torch.bfloat16)


def _reference(hidden, routes, projections):
    hidden, routes = hidden.detach().cpu(), routes.detach().cpu()
    output = torch.zeros_like(hidden)
    for slot, gate, up, down in projections:
        gate_value = F.linear(hidden.float(), _decode(gate).float()).bfloat16()
        up_value = F.linear(hidden.float(), _decode(up).float()).bfloat16()
        intermediate = (F.silu(gate_value) * up_value).bfloat16()
        value = F.linear(intermediate.float(), _decode(down).float()).bfloat16()
        contribution = (value * routes[:, slot:slot + 1].bfloat16()).bfloat16()
        output = (output + contribution).bfloat16()
    return output


def _triplet(rows, columns, generator):
    packed = torch.randint(0, 256, (rows, columns // 2), generator=generator, dtype=torch.uint8)
    choices = torch.tensor([0.5, 0.75, 1.25, 2.0], dtype=torch.float32)
    blocks = choices[torch.randint(0, 4, (rows, columns // 16), generator=generator)]
    # Non-power-of-two FP32 global scale exercises BF16 decode rounding.
    return (packed, blocks.to(torch.float8_e4m3fn), torch.tensor(0.73 / math.sqrt(columns)))


def _case(hidden_size=32, intermediate_size=16):
    generator = torch.Generator().manual_seed(89173 + hidden_size)
    hidden = (torch.randn(1, hidden_size, generator=generator) * 0.25).bfloat16()
    routes = torch.tensor([[0.1713, 0.6137, 0.2150]], dtype=torch.float32)
    projections = [
        (slot, _triplet(intermediate_size, hidden_size, generator),
         _triplet(intermediate_size, hidden_size, generator),
         _triplet(hidden_size, intermediate_size, generator))
        for slot in (2, 0, 1)
    ]
    return hidden, routes, projections


def _cuda_case(case):
    hidden, routes, projections = case
    return hidden.cuda(), routes.cuda(), [
        (slot, *(tuple(t.cuda() for t in triplet) for triplet in parts))
        for slot, *parts in projections
    ]


def _assert_numerical(actual, expected):
    assert actual.dtype == torch.bfloat16
    assert actual.is_cuda and not actual.requires_grad
    assert torch.isfinite(actual).all()
    # Three rounded BF16 projections plus sequential weighted additions permit
    # a small accumulated rounding difference, not codec/scale/routing drift.
    torch.testing.assert_close(actual.cpu(), expected, rtol=0.02, atol=0.002)




@_CUDA
@pytest.mark.parametrize("hidden_size,intermediate_size", [(32, 16), (80, 48), (2560, 640)])
def test_grouped_decode_matches_ordered_bf16_reference_without_mutation(hidden_size, intermediate_size):
    case = _case(hidden_size, intermediate_size)
    expected = _reference(*case)
    hidden, routes, projections = _cuda_case(case)
    # Only packed tensors reside on CUDA. The largest case holds < 10 MiB of
    # weights; dense oracle decoding stays on CPU, far below 512 MiB GPU use.
    with torch.no_grad():
        actual = grouped_nvfp4_decode(hidden, routes, projections)
    assert actual.shape == (1, hidden_size)
    _assert_numerical(actual, expected)
    assert torch.equal(hidden.cpu(), case[0])
    assert torch.equal(routes.cpu(), case[1])
    for gpu, cpu in zip(projections, case[2]):
        for gpu_part, cpu_part in zip(gpu[1:], cpu[1:]):
            for actual_tensor, original in zip(gpu_part, cpu_part):
                # Compare bytes, including FP8 scales and signed zero payloads.
                assert torch.equal(actual_tensor.cpu().reshape(-1).view(torch.uint8),
                                   original.reshape(-1).view(torch.uint8))


@_CUDA
def test_every_nibble_affects_public_output():
    # Isolate each E2M1 code in up rows; gate and down are fixed. Final
    # addition can erase -0, so signed-zero storage is checked above instead.
    packed = torch.zeros(16, 8, dtype=torch.uint8)
    packed[:, 0] = torch.arange(16, dtype=torch.uint8)
    blocks = torch.full((16, 1), 1.5).to(torch.float8_e4m3fn)
    gate = torch.zeros_like(packed)
    gate[:, 0] = 2
    down = torch.zeros_like(packed)
    for row in range(16):
        down[row, row // 2] = 2 << (4 * (row % 2))
    scale = torch.tensor(0.5)
    projections = [(0, (gate, blocks, scale), (packed, blocks, scale), (down, blocks, scale))]
    hidden = torch.zeros(1, 16, dtype=torch.bfloat16)
    hidden[0, 0] = 1
    case = hidden, torch.tensor([[0.375]]), projections
    with torch.no_grad():
        actual = grouped_nvfp4_decode(*_cuda_case(case))
    _assert_numerical(actual, _reference(*case))
    assert torch.count_nonzero(actual) == 14


@_CUDA
def test_changed_hidden_and_route_slots_change_output_and_match_reference():
    case = _case()
    hidden, routes, projections = _cuda_case(case)
    with torch.no_grad():
        baseline = grouped_nvfp4_decode(hidden, routes, projections)
        changed_hidden = grouped_nvfp4_decode(-hidden, routes, projections)
        changed_routes = grouped_nvfp4_decode(hidden, routes.flip(1).contiguous(), projections)
        zero = grouped_nvfp4_decode(hidden, torch.zeros_like(routes), projections)
    _assert_numerical(changed_hidden, _reference(-case[0], case[1], case[2]))
    _assert_numerical(changed_routes, _reference(case[0], case[1].flip(1), case[2]))
    assert not torch.equal(baseline, changed_hidden)
    assert not torch.equal(baseline, changed_routes)
    assert torch.count_nonzero(zero) == 0
    # Relabel route slots without changing expert order or their weights.
    reordered = [(2 - slot, *parts) for slot, *parts in projections]
    with torch.no_grad():
        equivalent = grouped_nvfp4_decode(hidden, routes.flip(1).contiguous(), reordered)
    torch.testing.assert_close(equivalent, baseline, rtol=0, atol=0)


@_CUDA
@pytest.mark.parametrize("requires_grad", [False, True])
def test_grad_enabled_decode_is_rejected_even_for_detached_inputs(requires_grad):
    hidden, routes, projections = _cuda_case(_case())
    hidden.requires_grad_(requires_grad)
    routes.requires_grad_(requires_grad)
    with torch.enable_grad(), pytest.raises((ValueError, TypeError, RuntimeError)):
        grouped_nvfp4_decode(hidden, routes, projections)


@_CUDA
@pytest.mark.parametrize("invalid", [
    "hidden_dtype", "hidden_batch", "hidden_stride", "hidden_cpu",
    "routes_dtype", "routes_shape", "packed_dtype", "packed_shape",
    "blocks_dtype", "blocks_shape", "global_dtype", "global_shape",
])
def test_invalid_tensor_contract_is_rejected(invalid):
    hidden, routes, projections = _cuda_case(_case())
    slot, gate, up, down = projections[0]
    packed, blocks, scale = gate
    if invalid == "hidden_dtype":
        hidden = hidden.float()
    elif invalid == "hidden_batch":
        hidden = hidden.repeat(2, 1)
    elif invalid == "hidden_stride":
        hidden = hidden.repeat_interleave(2, dim=1)[:, ::2]
    elif invalid == "hidden_cpu":
        hidden = hidden.cpu()
    elif invalid == "routes_dtype":
        routes = routes.to(torch.int32)
    elif invalid == "routes_shape":
        routes = routes.flatten()
    elif invalid == "packed_dtype":
        packed = packed.to(torch.int16)
    elif invalid == "packed_shape":
        packed = packed[:, :-1].contiguous()
    elif invalid == "blocks_dtype":
        blocks = blocks.float()
    elif invalid == "blocks_shape":
        blocks = blocks[:, :1].contiguous()
    elif invalid == "global_dtype":
        scale = scale.bfloat16()
    elif invalid == "global_shape":
        scale = scale.repeat(2)
    projections[0] = slot, (packed, blocks, scale), up, down
    with torch.no_grad(), pytest.raises((ValueError, TypeError, RuntimeError)):
        grouped_nvfp4_decode(hidden, routes, projections)


@_CUDA
def test_invalid_slots_cannot_read_outside_route_weights():
    hidden, routes, projections = _cuda_case(_case())
    for slots in ((0, 0, 2), (-1, 1, 2), (0, 1, 3)):
        changed = [(slot, *entry[1:]) for slot, entry in zip(slots, projections)]
        with torch.no_grad(), pytest.raises(ValueError):
            grouped_nvfp4_decode(hidden, routes, changed)


@_CUDA
def test_checkpoint_dispatch_preserves_gradients_and_context(tmp_path):
    from qwen_exo_booster.native_ple_checkpoint import FrozenCheckpointExperts, generation_expert_backend
    from test_native_ple_checkpoint import _checkpoint

    cpu_case = _case(2560, 640)
    tensors = {}
    for expert, entry in enumerate(cpu_case[2]):
        for name, triplet in zip(("gate_proj", "up_proj", "down_proj"), entry[1:]):
            for ending, tensor in zip(("weight", "weight_scale", "weight_scale_2"), triplet):
                tensors[f"experts.{expert}.{name}.{ending}"] = tensor
    with _checkpoint(tmp_path, tensors) as checkpoint:
        checkpoint.fp32_accumulation = True
        experts = FrozenCheckpointExperts(checkpoint, "experts", 3, 2560, 640)
        hidden = cpu_case[0].cuda()
        routes = torch.tensor([[1, 2, 0]], dtype=torch.int64, device="cuda")
        weights = cpu_case[1].cuda()
        with torch.no_grad():
            baseline = experts(hidden, routes, weights)
            with generation_expert_backend("nvfp4-gemv"):
                accelerated = experts(hidden, routes, weights)
            restored = experts(hidden, routes, weights)
        torch.testing.assert_close(accelerated, baseline, rtol=0.02, atol=0.002)
        torch.testing.assert_close(restored, baseline, rtol=0, atol=0)
        # A no-grad generation context may never bypass autograd if a caller
        # explicitly enables gradients; compare real input and route gradients.
        gradients = []
        for backend in ("reference", "nvfp4-gemv"):
            x = hidden.detach().clone().requires_grad_()
            w = weights.detach().clone().requires_grad_()
            with torch.enable_grad(), generation_expert_backend(backend):
                out = experts(x, routes, w)
                out.float().square().sum().backward()
            gradients.append((out.detach(), x.grad, w.grad))
        for actual, expected in zip(gradients[1], gradients[0]):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        with pytest.raises(RuntimeError, match="interrupt"):
            with generation_expert_backend("nvfp4-gemv"):
                raise RuntimeError("interrupt")
        with torch.no_grad():
            torch.testing.assert_close(experts(hidden, routes, weights), baseline, rtol=0, atol=0)
