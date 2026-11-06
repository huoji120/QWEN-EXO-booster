"""Independent numerical/graph coverage for selected NVFP4 QSA kernels."""

import unittest

import torch

from sglang.srt.layers.attention.qsa.nvfp4 import (
    compact_nvfp4_indices,
    gather_nvfp4_rows,
    gather_nvfp4_selected,
    sparse_gqa_nvfp4_paged,
)
from sglang.srt.layers.attention.qsa.sparse_attn import (
    sparse_gqa_fwd_interface_triton_ck,
)
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=60, stage="base-b", runner_config="1-gpu-small")


def _strided(data, *, device="cuda", pinned=False):
    storage = torch.empty(
        (*data.shape[:-1], data.shape[-1] * 2), dtype=data.dtype,
        device=device, pin_memory=pinned,
    )
    view = storage[..., ::2]
    view.copy_(data)
    return view


def _decode(data, scales, global_scale, dtype):
    """CPU reference: literal E2M1 LUT and arithmetic E4M3FN byte decoding.

    Deliberately does not call any production FP4/FP8 conversion or quantizer.
    """
    data, scales = data.cpu().long(), scales.cpu().long()
    codes = torch.stack((data & 15, data >> 4), dim=-1).flatten(-2)
    lut = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6,
                        -0., -.5, -1, -1.5, -2, -3, -4, -6])
    exponent, mantissa = (scales >> 3) & 15, scales & 7
    sf = torch.where(
        exponent == 0, mantissa.float() * (2.0 ** -9),
        (1.0 + mantissa.float() / 8) * torch.pow(2.0, exponent.float() - 7),
    )
    sf = torch.where((scales & 128) != 0, -sf, sf)
    sf = sf.repeat_interleave(16, dim=-1)
    return (lut[codes] * sf * global_scale).to(dtype)


def _banks(gpu_rows, *, rows=96, heads=2, dim=256):
    generator = torch.Generator().manual_seed(4821)
    data = [torch.randint(0, 256, (rows, heads, dim // 2), dtype=torch.uint8,
                          generator=generator) for _ in range(2)]
    # Includes zero, subnormal and normal E4M3 scales, independently per K/V.
    scale_codes = torch.tensor([0, 1, 7, 8, 24, 40, 48, 56, 60, 64], dtype=torch.uint8)
    scales = [scale_codes[torch.randint(0, scale_codes.numel(),
                                      (rows, heads, dim // 16), generator=generator)]
              for _ in range(2)]
    full = (*data, *scales)
    gpu = tuple(_strided(t[:gpu_rows]) for t in full)
    host = tuple(_strided(t[gpu_rows:], device="cpu", pinned=True) for t in full)
    return gpu, host


def _kwargs(host):
    return dict(k_scale=.73, v_scale=1.37, host_k=host[0], host_v=host[1],
                host_ks=host[2], host_vs=host[3])


def _decoded(gpu, host, dtype):
    full = [torch.cat((g.cpu(), h.cpu())) for g, h in zip(gpu, host)]
    return (_decode(full[0], full[2], .73, dtype),
            _decode(full[1], full[3], 1.37, dtype))


def _slot_reference(gpu, host, slots, dtype, rows):
    k, v = _decoded(gpu, host, dtype)
    outputs = [torch.zeros((rows, *bank.shape[1:]), dtype=dtype) for bank in (k, v)]
    for row, slot in enumerate(slots.cpu().tolist()):
        if 0 <= slot < k.shape[0]:
            outputs[0][row] = k[slot]
            outputs[1][row] = v[slot]
    return outputs


def _selected_slots(req_table, reqs, indices, lengths, cu, rows):
    req_table, reqs, indices, lengths, cu = [x.cpu() for x in (req_table, reqs, indices, lengths, cu)]
    slots = torch.full((rows,), -1, dtype=torch.int64)
    for batch, req in enumerate(reqs.tolist()):
        if not 0 <= req < req_table.shape[0]:
            continue
        start, end = int(cu[batch]), int(cu[batch + 1])
        for col in range(min(end - start, indices.shape[1])):
            position = int(indices[batch, col])
            if 0 <= position < min(int(lengths[batch]), req_table.shape[1]):
                slots[start + col] = req_table[req, position]
    return slots


def _attention_reference(q, gpu, host, indices, cu_q, cu_k, lengths, locations):
    k, v = _decoded(gpu, host, q.dtype)
    q = q.cpu().float()
    indices, cu_q, cu_k, lengths, locations = [
        t.cpu() for t in (indices, cu_q, cu_k, lengths, locations)
    ]
    output = torch.zeros_like(q)
    group_size = q.shape[1] // k.shape[1]
    for batch in range(lengths.numel()):
        q_start, q_end = int(cu_q[batch]), int(cu_q[batch + 1])
        for row in range(q_start, q_end):
            visible = int(lengths[batch]) - (q_end - q_start) + row - q_start + 1
            for head in range(q.shape[1]):
                group = head // group_size
                candidates = indices[row, group] if indices.ndim == 3 else indices[row]
                slots = []
                for position in candidates.tolist():
                    logical = int(cu_k[batch]) + position
                    if 0 <= position < visible and 0 <= logical < locations.numel():
                        slot = int(locations[logical])
                        if 0 <= slot < k.shape[0]:
                            slots.append(slot)
                if slots:
                    keys, values = k[slots, group].float(), v[slots, group].float()
                    scores = (keys @ q[row, head]) * q.shape[-1] ** -.5
                    output[row, head] = scores.softmax(0) @ values
    return output


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class TestQSANVFP4(CustomTestCase):
    def test_mtp_tail_compaction_replays_causal_lengths(self):
        indices = torch.full((2, 2051), -1, dtype=torch.int32, device="cuda")
        indices[:, :4] = torch.tensor([3, 0, 5, 1], device="cuda")
        indices[:, -3:] = torch.tensor([6, 7, 8], device="cuda")
        lengths = torch.tensor([9, 5], dtype=torch.int32, device="cuda")
        output = torch.empty_like(indices)
        compact_nvfp4_indices(indices, lengths, output)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            compact_nvfp4_indices(indices, lengths, output)
        for limits in ((9, 5), (0, 8), (7, 9)):
            lengths.copy_(torch.tensor(limits, device="cuda"))
            graph.replay()
            for row, limit in enumerate(limits):
                valid = indices[row][(indices[row] >= 0) & (indices[row] < limit)]
                expected = torch.full_like(indices[row], -1)
                expected[:valid.numel()] = valid
                torch.testing.assert_close(output[row], expected, rtol=0, atol=0)

    def test_rows_all_nibbles_scales_strides_and_host_boundary(self):
        for gpu_rows in (0, 37, 96):
            for dtype in (torch.float16, torch.bfloat16, torch.float32):
                with self.subTest(gpu_rows=gpu_rows, dtype=dtype):
                    gpu, host = _banks(gpu_rows)
                    slots = _strided(torch.tensor([0, 36, 37, 95, -1, 96, 2**33], dtype=torch.int64))
                    outputs = [_strided(torch.full((10, 2, 256), float("nan"), dtype=dtype)) for _ in range(2)]
                    gather_nvfp4_rows(*gpu, slots, *outputs, **_kwargs(host))
                    expected = _slot_reference(gpu, host, slots, dtype, 10)
                    for actual, reference in zip(outputs, expected):
                        torch.testing.assert_close(actual.cpu(), reference, rtol=0, atol=0)

    def test_rows_varying_prefix_bounds_and_reused_output_tail(self):
        gpu, host = _banks(37)
        all_slots = _strided(torch.arange(95, -1, -1, dtype=torch.int64))
        all_slots[3:7] = torch.tensor([-1, 96, 2**33, 37], device="cuda")
        for dtype in (torch.float16, torch.bfloat16):
            outputs = [torch.empty((100, 2, 256), device="cuda", dtype=dtype) for _ in range(2)]
            # Exercise empty/unit/aligned/unaligned prefixes and a shrinking
            # prefix after growth so stale staging rows cannot survive.
            for count in (0, 1, 15, 16, 17, 31, 32, 33, 95, 96, 1, 0):
                with self.subTest(dtype=dtype, count=count):
                    slots = all_slots[:count]
                    for output in outputs:
                        output.fill_(float("nan"))
                    gather_nvfp4_rows(*gpu, slots, *outputs, **_kwargs(host))
                    expected = _slot_reference(gpu, host, slots, dtype, 100)
                    for actual, reference in zip(outputs, expected):
                        torch.testing.assert_close(actual.cpu(), reference, rtol=0, atol=0)

    def test_selected_packed_and_page_padded_plans(self):
        gpu, host = _banks(37)
        req_table = _strided(torch.stack((torch.arange(32), torch.arange(32) + 32,
                                          torch.arange(32) + 64)).to(torch.int64))
        reqs = _strided(torch.tensor([2, 0, 1, -1], dtype=torch.int32))
        lengths = _strided(torch.tensor([8, 0, 12, 3], dtype=torch.int32))
        indices = _strided(torch.tensor([[0, 7, -1, 8, 31, -1],
                                         [0, -1, -1, -1, -1, -1],
                                         [4, 5, 0, 11, 12, -1],
                                         [0, 1, 2, -1, -1, -1]], dtype=torch.int32))
        for boundaries in ([0, 4, 4, 10, 13], [0, 8, 8, 16, 24]):
            with self.subTest(boundaries=boundaries):
                cu = _strided(torch.tensor(boundaries, dtype=torch.int32))
                outputs = [_strided(torch.full((28, 2, 256), float("nan"), dtype=torch.bfloat16)) for _ in range(2)]
                gather_nvfp4_selected(*gpu, req_table, reqs, indices, lengths, cu, *outputs, **_kwargs(host))
                slots = _selected_slots(req_table, reqs, indices, lengths, cu, 28)
                expected = _slot_reference(gpu, host, slots, torch.bfloat16, 28)
                for actual, reference in zip(outputs, expected):
                    torch.testing.assert_close(actual.cpu(), reference, rtol=0, atol=0)

    def test_gather_graph_replay_changes_data_scales_and_mapping(self):
        gpu, host = _banks(37)
        slots = torch.tensor([36, 37, 2, 95, -1], device="cuda")
        req_table = torch.arange(96, device="cuda", dtype=torch.int32).reshape(3, 32)
        reqs = torch.tensor([1, 2], device="cuda", dtype=torch.int32)
        indices = torch.tensor([[4, 5, 0, -1], [0, 1, 2, -1]], device="cuda", dtype=torch.int32)
        lengths = torch.tensor([12, 10], device="cuda", dtype=torch.int32)
        cu = torch.tensor([0, 4, 8], device="cuda", dtype=torch.int32)
        row_outputs = [torch.empty((7, 2, 256), device="cuda", dtype=torch.bfloat16) for _ in range(2)]
        selected_outputs = [torch.empty((12, 2, 256), device="cuda", dtype=torch.bfloat16) for _ in range(2)]

        def invoke():
            gather_nvfp4_rows(*gpu, slots, *row_outputs, **_kwargs(host))
            gather_nvfp4_selected(*gpu, req_table, reqs, indices, lengths, cu,
                                  *selected_outputs, **_kwargs(host))

        invoke()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            invoke()
        for changed in (False, True):
            if changed:
                # Prior replay/reference synchronizes before any pinned-host write.
                gpu[0].bitwise_xor_(0x5A)
                host[1].bitwise_xor_(0xA5)
                gpu[2].fill_(56)
                host[3].fill_(60)
                slots.copy_(torch.tensor([37, 36, -1, 96, 94], device="cuda"))
                reqs.copy_(torch.tensor([2, 1], device="cuda", dtype=torch.int32))
                indices.copy_(torch.tensor([[5, 2, 1, -1], [4, 5, 6, -1]], device="cuda", dtype=torch.int32))
                lengths.copy_(torch.tensor([3, 7], device="cuda", dtype=torch.int32))
                cu.copy_(torch.tensor([0, 8, 12], device="cuda", dtype=torch.int32))
            graph.replay()
            torch.cuda.synchronize()
            row_expected = _slot_reference(gpu, host, slots, torch.bfloat16, 7)
            selected_slots = _selected_slots(req_table, reqs, indices, lengths, cu, 12)
            selected_expected = _slot_reference(gpu, host, selected_slots, torch.bfloat16, 12)
            for actual, expected in zip(row_outputs + selected_outputs, row_expected + selected_expected):
                torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)

    def _attention_case(self, *, gpu_rows, dtype, query_lengths, per_head, graph_replay=False):
        gpu, host = _banks(gpu_rows)
        total = sum(query_lengths)
        q = _strided(torch.randn(total, 8, 256, dtype=dtype) * .2)
        cu_q = _strided(torch.tensor([0, query_lengths[0], total], dtype=torch.int32))
        cu_k = _strided(torch.tensor([0, 40, 80], dtype=torch.int32))
        lengths = _strided(torch.tensor([40, 40], dtype=torch.int32))
        locations = _strided(torch.cat((torch.arange(40) + 20, torch.arange(40) + 50)).to(torch.int64))
        locations[5] = -1
        locations[40 + 7] = 96
        shape = (total, 2, 65) if per_head else (total, 65)
        selected = torch.full(shape, -1, dtype=torch.int32)
        # More than a single 32-column tile; invalid slots and future positions
        # are interleaved rather than silently compacted by the test reference.
        selected[..., :40] = torch.arange(39, -1, -1)
        selected[..., 40] = 10000
        if per_head:
            selected[:, 1, :40] = torch.arange(40)
        selected[-1].fill_(-1)  # entirely empty attention row must be zero
        indices = _strided(selected)

        def invoke():
            return sparse_gqa_nvfp4_paged(q, *gpu, indices, cu_q, cu_k, lengths,
                                          256 ** -.5, locations, max_q=max(query_lengths), **_kwargs(host))

        output = invoke()
        graph = None
        if graph_replay:
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                output = invoke()
        for changed in ((False, True) if graph_replay else (False,)):
            if changed:
                q.mul_(.7)
                gpu[1].bitwise_xor_(0x81)
                host[0].bitwise_xor_(0x18)
                gpu[3].fill_(56)
                host[2].fill_(48)
                indices[..., :40] = torch.arange(40, device="cuda", dtype=torch.int32)
                indices[-1].fill_(-1)
                locations.copy_(locations.flip(0))
                lengths.copy_(torch.tensor([37, 39], device="cuda", dtype=torch.int32))
            if graph is not None:
                graph.replay()
            torch.cuda.synchronize()
            reference = _attention_reference(q, gpu, host, indices, cu_q, cu_k, lengths, locations)
            torch.testing.assert_close(output.float().cpu(), reference,
                                       atol=.015 if dtype == torch.bfloat16 else .003,
                                       rtol=.02 if dtype == torch.bfloat16 else .004)
            self.assertEqual(torch.count_nonzero(output[-1]).item(), 0)

    def test_chunked_prefill_head256_against_pytorch(self):
        for gpu_rows in (0, 37, 96):
            for dtype in (torch.float16, torch.bfloat16):
                with self.subTest(gpu_rows=gpu_rows, dtype=dtype):
                    self._attention_case(gpu_rows=gpu_rows, dtype=dtype,
                                         query_lengths=(7, 3), per_head=True)

    def test_paged_varying_location_bounds_against_pytorch(self):
        gpu, host = _banks(37)
        cu_q = torch.tensor([0, 4], dtype=torch.int32, device="cuda")
        cu_k = torch.tensor([0, 96], dtype=torch.int32, device="cuda")
        lengths = torch.tensor([96], dtype=torch.int32, device="cuda")
        all_locations = _strided(torch.arange(95, -1, -1, dtype=torch.int64))
        all_locations[3:6] = torch.tensor([-1, 96, 2**33], device="cuda")
        indices = torch.arange(97, dtype=torch.int32, device="cuda").repeat(4, 1)
        indices[-1].fill_(-1)
        for dtype in (torch.float16, torch.bfloat16):
            q = _strided(torch.randn(4, 8, 256, dtype=dtype) * .2)
            for count in (0, 1, 15, 16, 17, 31, 32, 33, 95, 96, 1, 0):
                with self.subTest(dtype=dtype, count=count):
                    locations = all_locations[:count]
                    output = sparse_gqa_nvfp4_paged(
                        q, *gpu, indices, cu_q, cu_k, lengths, 256 ** -.5,
                        locations, max_q=4, **_kwargs(host),
                    )
                    reference = _attention_reference(
                        q, gpu, host, indices, cu_q, cu_k, lengths, locations,
                    )
                    torch.testing.assert_close(
                        output.float().cpu(), reference,
                        atol=.015 if dtype == torch.bfloat16 else .003,
                        rtol=.02 if dtype == torch.bfloat16 else .004,
                    )
                    self.assertEqual(torch.count_nonzero(output[-1]).item(), 0)

    def test_decoded_prefill_causal_guard_and_empty_tiles(self):
        gpu, host = _banks(96)
        cu_q = torch.tensor([0, 4], dtype=torch.int32, device="cuda")
        cu_k = torch.tensor([0, 96], dtype=torch.int32, device="cuda")
        lengths = torch.tensor([96], dtype=torch.int32, device="cuda")
        locations = torch.arange(96, dtype=torch.int64, device="cuda")
        indices = torch.full((4, 65), -1, dtype=torch.int32, device="cuda")
        # An empty initial tile precedes valid, future and out-of-bounds rows.
        indices[:, 32:41] = torch.tensor(
            [0, 36, 37, 92, 93, 94, 95, 96, 10000], device="cuda",
        )
        indices[-1].fill_(-1)
        for dtype in (torch.float16, torch.bfloat16):
            with self.subTest(dtype=dtype):
                q = _strided(torch.randn(4, 8, 256, dtype=dtype) * .2)
                k, v = [bank.cuda() for bank in _decoded(gpu, host, dtype)]
                output = sparse_gqa_fwd_interface_triton_ck(
                    q, k, v, indices, cu_q, cu_k, lengths, 256 ** -.5,
                    max_q=4, causal_guard=True,
                )
                reference = _attention_reference(
                    q, gpu, host, indices, cu_q, cu_k, lengths, locations,
                )
                torch.testing.assert_close(
                    output.float().cpu(), reference,
                    atol=.015 if dtype == torch.bfloat16 else .003,
                    rtol=.02 if dtype == torch.bfloat16 else .004,
                )
                self.assertEqual(torch.count_nonzero(output[-1]).item(), 0)

    def test_mtp_and_decode_graph_rows_against_pytorch(self):
        for query_lengths in ((1, 1), (4, 4)):
            for per_head in (False, True):
                with self.subTest(query_lengths=query_lengths, per_head=per_head):
                    self._attention_case(gpu_rows=37, dtype=torch.bfloat16,
                                         query_lengths=query_lengths, per_head=per_head,
                                         graph_replay=True)

    def test_rejects_missing_host_scales_and_device_global_scale(self):
        gpu, host = _banks(37)
        slots = torch.tensor([0], device="cuda")
        outputs = [torch.empty((1, 2, 256), device="cuda", dtype=torch.bfloat16) for _ in range(2)]
        kwargs = _kwargs(host)
        kwargs["host_ks"] = None
        with self.assertRaisesRegex(ValueError, "supplied together"):
            gather_nvfp4_rows(*gpu, slots, *outputs, **kwargs)
        kwargs = _kwargs(host)
        kwargs["k_scale"] = torch.tensor(.73, device="cuda")
        with self.assertRaisesRegex(ValueError, "CPU numbers"):
            gather_nvfp4_rows(*gpu, slots, *outputs, **kwargs)


if __name__ == "__main__":
    unittest.main()
