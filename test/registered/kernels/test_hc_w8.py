import pytest
import torch

from sglang.kernels.ops.elementwise.hc_w8 import (
    hc_combine_mix_w8,
    hc_combine_w8,
    hc_mix_fused_w8,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=60, stage="base-b-kernel-unit", runner_config="1-gpu-large")

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.cuda.get_device_capability() != (12, 0)
    or torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count < 188,
    reason="Requires the SM120 PRO6000 resident-grid geometry",
)

HC, H, K, R = 4, 2560, 10240, 320


def _w8(weight, group):
    grouped = weight.float().reshape(weight.shape[0], -1, group)
    scale = (grouped.abs().amax(-1) / 127).clamp_min(1e-30).to(torch.bfloat16)
    q = torch.round(grouped / scale.float().unsqueeze(-1)).clamp(-127, 127).to(torch.int8)
    return q.reshape_as(weight).contiguous(), scale.contiguous()


def _inputs(rows):
    x = torch.randn(rows, K, device="cuda", dtype=torch.bfloat16)
    y = torch.randn(rows, H, device="cuda", dtype=torch.bfloat16)
    inj_prev = torch.randn(rows, HC, device="cuda", dtype=torch.float32)
    norm = torch.randn(K, device="cuda", dtype=torch.bfloat16)
    wd, wd_s = _w8(torch.randn(R, K, device="cuda", dtype=torch.bfloat16) * 0.02, 128)
    wu, wu_s = _w8(torch.randn(K, R, device="cuda", dtype=torch.bfloat16) * 0.02, 64)
    winj = torch.randn(HC, K, device="cuda", dtype=torch.bfloat16) * 0.02
    return x, y, inj_prev, norm, wd, wd_s, winj, wu, wu_s


def _reference_mix(x, norm, wd, wd_s, winj, wu, wu_s, eps):
    rows = x.shape[0]
    xn = x.float().reshape(rows, HC, H)
    xn = xn * torch.rsqrt(xn.square().mean(-1, keepdim=True) + eps)
    xn = (xn * (1 + norm.float().reshape(HC, H))).reshape(rows, K).to(torch.bfloat16)
    down_w = (wd.float() * wd_s.float().repeat_interleave(128, 1)).to(torch.bfloat16).float()
    down = torch.einsum("mk,rk->mr", xn.float(), down_w)
    t = torch.nn.functional.silu(down / HC).to(torch.bfloat16)
    up_w = (wu.float() * wu_s.float().repeat_interleave(64, 1)).to(torch.bfloat16).float()
    up = torch.einsum("mr,kr->mk", t.float(), up_w)
    out = (torch.sigmoid(up.reshape(rows, HC, H)) * xn.reshape(rows, HC, H)).mean(1)
    inj = torch.einsum("mk,ck->mc", xn.float(), winj.float())
    return out.to(torch.bfloat16), inj, xn


@pytest.mark.parametrize("rows", [1, 2, 4, 8])
def test_hc_w8_mix_matches_reference(rows):
    vals = _inputs(rows)
    out, inj, normed = hc_mix_fused_w8(*vals[:1], vals[3], vals[4], vals[5], vals[6], vals[7], vals[8], 1e-6)
    ref = _reference_mix(vals[0], vals[3], vals[4], vals[5], vals[6], vals[7], vals[8], 1e-6)
    torch.testing.assert_close(out, ref[0], atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(inj, ref[1], atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(normed, ref[2], atol=2e-2, rtol=2e-2)


def test_hc_w8_rejects_prefill_rows():
    vals = _inputs(9)
    with pytest.raises(RuntimeError, match="1..8"):
        hc_mix_fused_w8(vals[0], vals[3], vals[4], vals[5], vals[6], vals[7], vals[8], 1e-6)


def test_hc_w8_combine_mix_boundary_and_reuse():
    vals = _inputs(2)
    x, y, ip, norm, wd, wd_s, winj, wu, wu_s = vals
    one = hc_combine_mix_w8(y, x, ip, norm, wd, wd_s, winj, wu, wu_s, 1e-6)
    two_x = x.clone()
    gates = 2 * torch.sigmoid(ip / HC)
    two_x.copy_((x.reshape(2, HC, H) + y[:, None] * gates[..., None]).reshape(2, K).to(torch.bfloat16))
    two = hc_mix_fused_w8(two_x, norm, wd, wd_s, winj, wu, wu_s, 1e-6)
    torch.testing.assert_close(one[0], two[0], atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(one[1], two[1], atol=2e-2, rtol=2e-2)
    assert torch.equal(one[3], two_x)

def test_hc_w8_padded_rows_are_zeroed():
    vals = list(_inputs(8))
    vals[0][3:].zero_()
    out, inj, normed = hc_mix_fused_w8(
        vals[0], vals[3], vals[4], vals[5], vals[6], vals[7], vals[8], 1e-6
    )
    assert torch.count_nonzero(out[3:]) == 0
    assert torch.count_nonzero(inj[3:]) == 0
    assert torch.count_nonzero(normed[3:]) == 0


def _combine_inputs(rows, seed=0):
    generator = torch.Generator(device="cuda").manual_seed(20261021 + seed)
    x = torch.randn(rows, K, device="cuda", dtype=torch.bfloat16, generator=generator)
    y = torch.randn(rows, H, device="cuda", dtype=torch.bfloat16, generator=generator)
    ip = torch.randn(rows, HC, device="cuda", generator=generator) * 12
    # Cancellation makes an accidental fused multiply-add observable even
    # after BF16 conversion. Also cover sigmoid saturation and its midpoint.
    y[:, :16] = 1.0078125
    x.reshape(rows, HC, H)[:, :, :16] = -1.0078125
    ip[0] = torch.tensor(
        [2.0 ** (-20 + seed), -1000.0 + seed, float(seed), 1000.0 - seed],
        device="cuda",
    )
    return y, x, ip


def _reference_combine(y, x, ip):
    # Original marked-smallrow expression: do not compile or rewrite its
    # separate FP32 sigmoid, multiplication and addition.
    gates = 2.0 * torch.sigmoid(ip / HC)
    return (
        x.unflatten(-1, (HC, H)) + y.unsqueeze(-2) * gates.unsqueeze(-1)
    ).flatten(-2).to(torch.bfloat16)


@pytest.mark.parametrize("rows", range(1, 9))
def test_hc_w8_standalone_combine_exact(rows):
    y, x, ip = _combine_inputs(rows)
    x_before, y_before, ip_before = x.clone(), y.clone(), ip.clone()
    out = hc_combine_w8(y, x, ip)
    expected = _reference_combine(y, x, ip)
    assert torch.equal(out.view(torch.int16), expected.view(torch.int16))
    assert out.shape == (rows, K) and out.dtype == torch.bfloat16
    assert out.is_contiguous() and out.data_ptr() != x.data_ptr()
    assert torch.equal(x, x_before)
    assert torch.equal(y, y_before)
    assert torch.equal(ip, ip_before)


@pytest.mark.parametrize("rows,active_rows", [(4, 3), (8, 3), (8, 5)])
def test_hc_w8_standalone_combine_padding(rows, active_rows):
    y, x, ip = _combine_inputs(rows)
    x[active_rows:].zero_()
    y[active_rows:].zero_()
    # Deliberately keep nonzero injection dots on padded rows.
    out = hc_combine_w8(y, x, ip)
    expected = _reference_combine(y, x, ip)
    assert torch.equal(out.view(torch.int16), expected.view(torch.int16))
    assert torch.count_nonzero(out[active_rows:]) == 0


@pytest.mark.parametrize("rows", range(1, 9))
def test_hc_w8_standalone_combine_graph_changed_inputs(rows):
    y, x, ip = _combine_inputs(rows)
    hc_combine_w8(y, x, ip)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = hc_combine_w8(y, x, ip)
    output_ptr = out.data_ptr()
    previous = None
    for seed in (1, 2, 3):
        new_y, new_x, new_ip = _combine_inputs(rows, seed)
        y.copy_(new_y)
        x.copy_(new_x)
        ip.copy_(new_ip)
        graph.replay()
        expected = _reference_combine(y, x, ip)
        assert torch.equal(out.view(torch.int16), expected.view(torch.int16))
        assert out.data_ptr() == output_ptr
        if previous is not None:
            assert not torch.equal(out, previous)
        previous = out.clone()


def test_hc_w8_standalone_combine_rejects_prefill():
    y, x, ip = _combine_inputs(9)
    with pytest.raises(RuntimeError, match="1..8"):
        hc_combine_w8(y, x, ip)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
