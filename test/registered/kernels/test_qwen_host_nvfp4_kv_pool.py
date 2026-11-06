"""Behavioral CUDA coverage for split-residency QSA NVFP4 KV storage."""

import unittest
from types import SimpleNamespace

import torch

from sglang.srt.layers.quantization.fp4_kv_cache_quant_method import NVFP4KVCacheMethod
from sglang.srt.layers.quantization.kvfp4_tensor import NVFP4KVQuantizeUtil
from sglang.srt.mem_cache.qwen_host_kv_pool import QwenHostKVPool
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=45, stage="base-b", runner_config="1-gpu-small")


def decode_bytes(packed, scales, global_scale):
    """Independent E2M1 reference; no production gather/dequantizer involved."""
    table = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6,
                          -0., -.5, -1, -1.5, -2, -3, -4, -6])
    codes = torch.stack((packed & 15, packed >> 4), dim=-1).long()
    values = table[codes].flatten(-2)
    blocks = scales.view(torch.float8_e4m3fn).float().repeat_interleave(16, dim=-1)
    return values * blocks * global_scale


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class TestQwenHostNVFP4KVPool(CustomTestCase):
    def setUp(self):
        super().setUp()
        self.pools = []
        self.addCleanup(self._clear_pools)

    def _clear_pools(self):
        for pool in self.pools:
            pool._clear_buffers()
        self.pools.clear()

    def make_pool(self, gpu_size, *, size=64, layers=2):
        quant = NVFP4KVCacheMethod(10, "cuda", attention_backend="qsa")
        # Deliberately unrelated global/compact ids and independent K/V scales.
        quant.k_scales_float[7:9] = [.25, 2.0]
        quant.v_scales_float[7:9] = [1.5, .5]
        quant.k_scales_gpu.copy_(torch.tensor(quant.k_scales_float, device="cuda"))
        quant.v_scales_gpu.copy_(torch.tensor(quant.v_scales_float, device="cuda"))
        pool = QwenHostKVPool(
            size=size, gpu_size=gpu_size, page_size=4, dtype=torch.uint8,
            head_num=2, head_dim=64, v_head_dim=64, layer_num=layers,
            device="cuda", enable_memory_saver=False, enable_alt_stream=False,
            enable_kv_cache_copy=True, kv_cache_layout="nhd",
            start_layer=3, end_layer=3 + layers - 1, quant_method=quant,
        )
        self.pools.append(pool)
        return pool

    def write_rows(self, pool, slots):
        if torch.cuda.get_device_capability()[0] not in (10, 12):
            self.skipTest("Native NVFP4 writer requires SM100/SM120")
        references = []
        for local in range(pool.layer_num):
            payloads = []
            inputs = []
            for dim, scale in ((64, pool.quant_method.k_scales_gpu[7 + local:8 + local]),
                               (64, pool.quant_method.v_scales_gpu[7 + local:8 + local])):
                source = torch.arange(slots.numel() * 2 * dim, device="cuda").reshape(-1, 2, dim)
                source = ((source.remainder(127).float() - 63) / 9 + local).to(torch.bfloat16)
                inputs.append(source)
                packed, sf, _ = NVFP4KVQuantizeUtil.quantize(source, scale)
                payloads.append((packed.view(torch.uint8).cpu(), sf.view(torch.uint8).cpu()))
            pool.set_kv_buffer(SimpleNamespace(layer_id=7 + local), slots, *inputs,
                               layer_id_override=3 + local)
            references.append([payloads[0][0], payloads[1][0], payloads[0][1], payloads[1][1]])
        return references

    def assert_snapshot(self, snapshot, expected):
        for layer, reference in zip(snapshot, expected):
            self.assertTrue(all(len(chunk) == 4 for chunk in layer))
            for kind in range(4):
                actual = torch.cat([chunk[kind] for chunk in layer])
                torch.testing.assert_close(actual, reference[kind], atol=0, rtol=0)

    def test_nonunit_scale_roundtrip_and_snapshot_cycles(self):
        for gpu_size in (0, 256, 520):
            with self.subTest(gpu_size=gpu_size):
                pool = self.make_pool(gpu_size, size=520)
                pool.cpu_offloading_chunk_size = 127
                source = torch.arange(4, 517, device="cuda")
                references = self.write_rows(pool, source)
                self.assert_snapshot(pool.get_cpu_copy(source), references)
                for local in range(pool.layer_num):
                    actual = pool.get_kv_tokens(7 + local, source, torch.float32,
                                               layer_id_override=3 + local)
                    for kind, scale in enumerate(pool.quant_method.get_bmm_scales(7 + local)):
                        expected = decode_bytes(references[local][kind], references[local][kind + 2], scale)
                        torch.testing.assert_close(actual[kind].cpu(), expected, atol=0, rtol=0)
                # Cycles span the host/GPU boundary and CPU snapshot chunk size.
                target = source.roll(257)
                pool.move_kv_cache(target, source)
                self.assert_snapshot(pool.get_cpu_copy(target), references)
                snapshot = pool.get_cpu_copy(target)
                replacement = source.flip(0)
                pool.load_cpu_copy(snapshot, replacement)
                self.assert_snapshot(pool.get_cpu_copy(replacement), references)
                # Duplicate snapshot slots are copied rather than aliased.
                selected = replacement[torch.tensor([0, 2, 0, 3], device="cuda")]
                duplicate = pool.get_cpu_copy(selected)
                expected = [[bank[[0, 2, 0, 3]] for bank in layer] for layer in references]
                self.assert_snapshot(duplicate, expected)
                retained = duplicate[0][0][2][2].clone()
                duplicate[0][0][2][0].zero_()
                torch.testing.assert_close(duplicate[0][0][2][2], retained, atol=0, rtol=0)

    def test_gathers_replay_changed_slots_and_page_padding(self):
        for gpu_size in (0, 32, 64):
            with self.subTest(gpu_size=gpu_size):
                pool = self.make_pool(gpu_size, layers=1)
                written = torch.tensor([4, 35, 36, 67], device="cuda")
                raw = self.write_rows(pool, written)[0]
                reference = [decode_bytes(raw[k], raw[k + 2], scale).cuda()
                             for k, scale in enumerate(pool.quant_method.get_bmm_scales(7))]
                slots = torch.tensor([36, -1, 4, 68, 35], device="cuda")
                mapping = torch.tensor([[4, 35, 36, 67], [67, 36, -3, 68]], device="cuda")
                reqs = torch.tensor([0, 1], device="cuda", dtype=torch.int32)
                indices = torch.tensor([[2, -1, 0], [0, 1, 3]], device="cuda", dtype=torch.int32)
                lengths = torch.tensor([4, 4], device="cuda", dtype=torch.int32)
                cu = torch.tensor([0, 4, 8], device="cuda", dtype=torch.int32)
                outputs = [torch.empty((10, 2, 64), device="cuda", dtype=torch.float32) for _ in range(2)]
                selected = [torch.empty_like(t) for t in outputs]

                def run():
                    pool.gather_into(0, slots, *outputs, layer_id=7)
                    pool.gather_selected_into(0, mapping, reqs, indices, lengths, cu,
                                              *selected, layer_id=7)

                stream = torch.cuda.Stream()
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    for _ in range(3):
                        run()
                torch.cuda.current_stream().wait_stream(stream)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    run()
                pointers = [t.data_ptr() for t in outputs + selected]
                for slot_values, request_order, prefix in (
                    ([36, -1, 4, 68, 35], [0, 1], [0, 4, 8]),
                    ([67, 35, -4, 36, 4], [1, 0], [0, 0, 4]),
                    ([68, 67, 36, 4, -1], [1, 0], [0, 0, 0]),
                ):
                    slots.copy_(torch.tensor(slot_values, device="cuda"))
                    reqs.copy_(torch.tensor(request_order, device="cuda"))
                    cu.copy_(torch.tensor(prefix, device="cuda"))
                    graph.replay()
                    read_slots = [-1] * 10
                    for row in range(2):
                        for col in range(prefix[row + 1] - prefix[row]):
                            pos = int(indices[row, col]) if col < 3 else -1
                            if 0 <= pos < 4:
                                read_slots[prefix[row] + col] = int(mapping[request_order[row], pos])
                    for gathered, plan in ((outputs, slot_values), (selected, read_slots)):
                        for kind in range(2):
                            expected = torch.zeros_like(gathered[kind])
                            for row, slot in enumerate(plan):
                                if slot in [4, 35, 36, 67]:
                                    expected[row] = reference[kind][[4, 35, 36, 67].index(slot)]
                            torch.testing.assert_close(gathered[kind], expected, atol=0, rtol=0)
                    self.assertEqual([t.data_ptr() for t in outputs + selected], pointers)
                with self.assertRaisesRegex(ValueError, "global layer_id"):
                    pool.gather_into(0, slots, *outputs)

    def test_raw_bytes_snapshot_cycle_and_graph_without_native_quantizer(self):
        # These are real packed E2M1/FP8 bytes, not mocked quantizer outputs.
        # This path also runs on SM89: only the native writer needs Blackwell.
        for gpu_size in (0, 256, 520):
            with self.subTest(gpu_size=gpu_size):
                pool = self.make_pool(gpu_size, size=520)
                pool.cpu_offloading_chunk_size = 127
                source = torch.arange(4, 517, device="cuda")
                references, snapshot = [], []
                for local in range(pool.layer_num):
                    packed = torch.arange(513 * 2 * 32).reshape(513, 2, 32)
                    scales = torch.arange(513 * 2 * 4).reshape(513, 2, 4)
                    payloads = [
                        (packed + 31 * local).remainder(256).to(torch.uint8),
                        (packed * 7 + 11 + local).remainder(256).to(torch.uint8),
                        (scales.remainder(7).float() / 4).to(torch.float8_e4m3fn).view(torch.uint8),
                        (scales.remainder(5).float() / 2).to(torch.float8_e4m3fn).view(torch.uint8),
                    ]
                    references.append(payloads)
                    snapshot.append([[p[start:start + 127].clone() for p in payloads]
                                     for start in range(0, 513, 127)])
                pool.load_cpu_copy(snapshot, source)
                self.assert_snapshot(pool.get_cpu_copy(source), references)
                target = source.roll(257)
                pool.move_kv_cache(target, source)
                self.assert_snapshot(pool.get_cpu_copy(target), references)
                slots = torch.empty(5, device="cuda", dtype=torch.int64)
                outputs = [torch.empty((8, 2, 64), device="cuda", dtype=torch.float32)
                           for _ in range(2)]
                slots.copy_(target[:5])
                stream = torch.cuda.Stream()
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    for _ in range(3):
                        pool.gather_into(1, slots, *outputs, layer_id=8)
                torch.cuda.current_stream().wait_stream(stream)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    pool.gather_into(1, slots, *outputs, layer_id=8)
                for order in ([0, 256, -1, 512, 257], [512, -1, 0, 128, 1]):
                    slots.copy_(torch.tensor([int(target[i]) if i >= 0 else -1 for i in order],
                                             device="cuda"))
                    graph.replay()
                    for kind, scale in enumerate(pool.quant_method.get_bmm_scales(8)):
                        decoded = decode_bytes(references[1][kind], references[1][kind + 2], scale)
                        expected = torch.zeros((8, 2, 64), dtype=torch.float32)
                        for row, index in enumerate(order):
                            if index >= 0:
                                expected[row] = decoded[index]
                        torch.testing.assert_close(outputs[kind].cpu(), expected, atol=0, rtol=0)
                restored = pool.get_cpu_copy(target)
                pool.load_cpu_copy(restored, source.flip(0))
                self.assert_snapshot(pool.get_cpu_copy(source.flip(0)), references)

    def test_capacity_bytes_and_no_logical_workspace(self):
        for gpu_size in (0, 32768, 65536):
            with self.subTest(gpu_size=gpu_size):
                torch.empty(1, device="cuda")
                allocated_before = torch.cuda.memory_allocated()
                pool = self.make_pool(gpu_size, size=65536, layers=1)
                allocated_bytes = torch.cuda.memory_allocated() - allocated_before
                gpu_rows = gpu_size + 4 if gpu_size else 0
                host_rows = 65540 - gpu_rows
                row_bytes = 2 * (64 // 2 + 64 // 2 + 64 // 16 + 64 // 16)
                self.assertEqual(pool.host_bytes, host_rows * row_bytes)
                self.assertEqual(pool.get_kv_size_bytes(),
                                 (gpu_rows * 2 * (32 + 4), gpu_rows * 2 * (32 + 4)))
                host, gpu = pool._raw_banks()
                self.assertEqual(sum(t.nbytes for bank in host for t in bank), pool.host_bytes)
                self.assertEqual(sum(t.nbytes for bank in gpu for t in bank), gpu_rows * row_bytes)
                self.assertGreaterEqual(allocated_bytes, gpu_rows * row_bytes)
                self.assertLess(allocated_bytes, gpu_rows * row_bytes + 1024**2)
                self.assertIsNone(pool.dq_k_buffer)
                self.assertIsNone(pool.dq_v_buffer)
                # The finalizer retains all data/scale tensors from both banks.
                held = pool._host_lifetime.peek()[2][1]
                self.assertEqual({id(t) for t in held}, {id(t) for bank in host + gpu for t in bank})
                for getter in (pool.get_key_buffer, pool.get_value_buffer, pool.get_raw_kv_buffer):
                    with self.assertRaises(NotImplementedError):
                        getter(3)
                with self.assertRaises(NotImplementedError):
                    pool.get_contiguous_buf_infos()
                with self.assertRaises(NotImplementedError):
                    pool.set_kv_buffer_prefix_valid()
                dummy = pool.get_kv_tokens(7, torch.arange(4, device="cuda"), torch.float32,
                                           layer_id_override=3)
                for value in dummy:
                    self.assertEqual(torch.count_nonzero(value).item(), 0)


if __name__ == "__main__":
    unittest.main()
