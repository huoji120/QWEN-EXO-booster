import json
import hashlib
import os
import struct
import tempfile
import unittest
from pathlib import Path

import torch
from safetensors.torch import save_file

from sglang.srt.environ import envs
from sglang.srt.models.qwen4_ple_nvme import NVMePLEEmbedding, PLEFileReader, PLEManifest, PLEShard


def load_checkpoint_scale(table, scale):
    from sglang.srt.models.qwen4_exp import Qwen4ExpForConditionalGeneration

    model = Qwen4ExpForConditionalGeneration.__new__(Qwen4ExpForConditionalGeneration)
    torch.nn.Module.__init__(model)
    model.model = torch.nn.Module()
    model.model.ple = torch.nn.Module()
    model.model.ple.ple_embedding = torch.nn.Module()
    model.model.ple.ple_embedding.ngram_embedding = table
    name = "model.ple.ple_embedding.ngram_embedding.weight_scale"
    model._load_qwen4_exp_ple_buffer(
        name, torch.tensor([scale], dtype=torch.bfloat16), dict(model.named_buffers()), set()
    )


class TestNativePLEDisk(unittest.TestCase):
    def test_loader_releases_replaced_parameters_without_cyclic_gc(self):
        import gc
        import weakref
        from types import SimpleNamespace
        from sglang.srt.models.qwen4_exp import Qwen4ExpForConditionalGeneration

        model = Qwen4ExpForConditionalGeneration.__new__(Qwen4ExpForConditionalGeneration)
        torch.nn.Module.__init__(model)
        model.config = SimpleNamespace(num_experts=None, tie_word_embeddings=False)
        model.weight = torch.nn.Parameter(torch.zeros(2), requires_grad=False)
        original = weakref.ref(model.weight)
        enabled = gc.isenabled()
        gc.disable()
        try:
            loaded = model.load_weights([("weight", torch.ones(2))])
            self.assertIn("weight", loaded)
            torch.testing.assert_close(model.weight, torch.ones(2))
            model.weight = torch.nn.Parameter(torch.zeros(2), requires_grad=False)
            self.assertIsNone(original())
        finally:
            if enabled:
                gc.enable()

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.prefix = "model.language_model.layers.1.ple.ple_embedding.ngram_embedding"
        self.first = torch.tensor([[1., 2., 3., 4.], [5., 6., 7., 8.]], dtype=torch.float8_e4m3fn)
        self.last = torch.tensor([[9., 10., 11., 12.]], dtype=torch.float8_e4m3fn)
        # Physical order in a combined shard is not necessarily numerical order.
        save_file({f"{self.prefix}.shard_1.weight": self.last,
                   f"{self.prefix}.shard_0.weight": self.first}, str(self.root / "weights.safetensors"))
        self.index = {"weight_map": {
            f"{self.prefix}.shard_0.weight": "weights.safetensors",
            f"{self.prefix}.shard_1.weight": "weights.safetensors",
        }}
        self.write_index()

    def write_index(self):
        (self.root / "model.safetensors.index.json").write_text(json.dumps(self.index))

    def test_lookup_preserves_order_duplicates_and_short_final_shard(self):
        with envs.SGLANG_QWEN4_PLE_NVME_BACKEND.override("mmap"):
            table = NVMePLEEmbedding(self.root, num_embeddings=3, embedding_dim=4, expected_shards=2)
            self.addCleanup(table.close)
            ids = torch.tensor([[2, 0], [1, 2]])
            actual = table(ids)
            expected = torch.tensor([[[9., 10., 11., 12.], [1., 2., 3., 4.]],
                                     [[5., 6., 7., 8.], [9., 10., 11., 12.]]], dtype=torch.bfloat16)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_pending_gathers_keep_independent_decoded_rows(self):
        with envs.SGLANG_QWEN4_PLE_NVME_BACKEND.override("mmap"):
            table = NVMePLEEmbedding(self.root, num_embeddings=3, embedding_dim=4, expected_shards=2)
            self.addCleanup(table.close)
            first = table.start_gather(torch.tensor([2, 0]))
            second = table.start_gather(torch.tensor([1]))
            actual_second = table.finish_gather(second, torch.device("cpu"))
            actual_first = table.finish_gather(first, torch.device("cpu"))
            torch.testing.assert_close(actual_second, self.first[1:].to(torch.bfloat16), rtol=0, atol=0)
            torch.testing.assert_close(actual_first, torch.cat([self.last.to(torch.bfloat16), self.first[:1].to(torch.bfloat16)]), rtol=0, atol=0)
            empty = table(torch.empty((2, 0), dtype=torch.int64))
            self.assertEqual(empty.shape, (2, 0, 4))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_pinned_staging_waits_for_copies_on_different_streams_and_close(self):
        with envs.SGLANG_QWEN4_PLE_NVME_BACKEND.override("mmap"):
            table = NVMePLEEmbedding(self.root, num_embeddings=3, embedding_dim=4, expected_shards=2)
            self.addCleanup(table.close)
            device = torch.device("cuda", torch.cuda.current_device())
            first_stream, second_stream = torch.cuda.Stream(), torch.cuda.Stream()
            first_pending = table.start_gather(torch.tensor([2, 0]))
            second_pending = table.start_gather(torch.tensor([1]))
            with torch.cuda.stream(first_stream):
                torch.cuda._sleep(50_000_000)
            first = table.finish_gather(first_pending, device, stream=first_stream)
            second = table.finish_gather(second_pending, device, stream=second_stream)
            table.close()
            torch.testing.assert_close(first.cpu(), torch.cat([self.last.to(torch.bfloat16), self.first[:1].to(torch.bfloat16)]), rtol=0, atol=0)
            torch.testing.assert_close(second.cpu(), self.first[1:].to(torch.bfloat16), rtol=0, atol=0)

    def test_original_fp8_checkpoint_retains_global_scale(self):
        with envs.SGLANG_QWEN4_PLE_NVME_BACKEND.override("mmap"):
            table = NVMePLEEmbedding(self.root, num_embeddings=3, embedding_dim=4, expected_shards=2)
            self.addCleanup(table.close)
            load_checkpoint_scale(table, 0.25)
            actual = table(torch.tensor([0])) * table.weight_scale
            expected = torch.tensor([[0.25, 0.5, 0.75, 1.0]], dtype=torch.bfloat16)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_native_bf16_snapshot_uses_two_byte_rows(self):
        values = torch.tensor([[0.125, -2.0, 7.0, 32.0]], dtype=torch.bfloat16)
        save_file({f"{self.prefix}.shard_0.weight": values}, str(self.root / "bf16.safetensors"))
        self.index["weight_map"] = {f"{self.prefix}.shard_0.weight": "bf16.safetensors"}
        self.write_index()
        with envs.SGLANG_QWEN4_PLE_NVME_BACKEND.override("mmap"):
            table = NVMePLEEmbedding(self.root, num_embeddings=1, embedding_dim=4, expected_shards=1)
            self.addCleanup(table.close)
            load_checkpoint_scale(table, 0.00019931793212890625)
            torch.testing.assert_close(table(torch.tensor([0])) * table.weight_scale, values, rtol=0, atol=0)

    def test_rejects_truncated_tensor_payload_before_lookup(self):
        path = self.root / "weights.safetensors"
        path.write_bytes(path.read_bytes()[:-1])
        with self.assertRaisesRegex(ValueError, "incomplete native FP8"):
            PLEManifest.from_snapshot(self.root, expected_shards=2)

    def test_rejects_checkpoint_path_escape(self):
        self.index["weight_map"][f"{self.prefix}.shard_1.weight"] = "../weights.safetensors"
        self.write_index()
        with self.assertRaisesRegex(ValueError, "escapes"):
            PLEManifest.from_snapshot(self.root, expected_shards=2)

    def test_rejects_out_of_range_rows_not_padding_to_zero(self):
        manifest = PLEManifest.from_snapshot(self.root, expected_shards=2)
        for row in (-1, 3):
            with self.assertRaises(IndexError):
                manifest.locate(row)


class TestPLEFileReader(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.backends = ["mmap"] + (["pread"] if hasattr(os, "pread") else [])

    def _fixture(self, storage, scale_mode):
        dtype = {"bf16": torch.bfloat16, "f16": torch.float16, "fp8_e4m3": torch.float8_e4m3fn}[storage]
        values = torch.tensor([[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]], dtype=dtype)
        raw = values.view(torch.uint8).numpy().tobytes()
        row_bytes = 3 * values.element_size()
        path = self.root / "rows.bin"
        payload = bytearray(b"header!")
        shards = [None, None]
        # Reversed physical order, shared file, independent data/scale offsets.
        for index, start, end in ((1, 2, 3), (0, 0, 2)):
            offset = len(payload)
            payload.extend(raw[start * row_bytes:end * row_bytes])
            payload.extend(b"gap")
            scale_offset, scale_row_bytes = -1, 0
            if scale_mode == "row" or scale_mode == "global" or (scale_mode == "mixed" and index == 0):
                scale_offset, scale_row_bytes = len(payload), 4
                scale_values = [2., 3., 4.][start:end] if scale_mode == "row" else [float(index + 2)]
                payload.extend(struct.pack("<" + "f" * len(scale_values), *scale_values))
            shards[index] = PLEShard(path, start, end, offset, row_bytes, scale_offset, scale_row_bytes)
            payload.extend(b"tail")
        path.write_bytes(payload)
        manifest = PLEManifest("fixture", 3, 3, 2, tuple(shards), storage=storage,
                               scale_is_per_row=scale_mode == "row")
        scales = {"row": [2., 3., 4.], "global": [2., 2., 3.], "mixed": [2., 2., 1.]}.get(scale_mode)
        return manifest, raw, scales

    def test_raw_bytes_and_scales_preserve_order_across_storage_and_backends(self):
        ids = [2, 0, 1, 2, 1, 0]
        for storage in ("bf16", "f16", "fp8_e4m3"):
            for scale_mode in ("none", "row", "global", "mixed"):
                manifest, raw, scales = self._fixture(storage, scale_mode)
                row_bytes = manifest.shards[0].row_bytes
                expected = b"".join(raw[row * row_bytes:(row + 1) * row_bytes] for row in ids)
                expected_scales = b"" if scales is None else struct.pack("<6f", *(scales[row] for row in ids))
                for backend in self.backends:
                    with self.subTest(storage=storage, scale_mode=scale_mode, backend=backend):
                        reader = PLEFileReader(manifest, backend)
                        try:
                            self.assertEqual(reader.read_rows(ids), (expected, expected_scales))
                            self.assertEqual(reader.read_rows([]), (b"", b""))
                            for invalid in (-1, 3, 1 << 80):
                                with self.assertRaises(IndexError):
                                    reader.read_rows([0, invalid])
                        finally:
                            reader.close()

    def test_small_and_vectorized_mmap_paths_preserve_identical_records(self):
        for count in (1, 8, 32, 128, 129, 256):
            ids = ([2, 0, 1, 2] * ((count + 3) // 4))[:count]
            for storage in ("bf16", "f16", "fp8_e4m3"):
                for scale_mode in ("none", "row", "global", "mixed"):
                    manifest, raw, scales = self._fixture(storage, scale_mode)
                    width = manifest.shards[0].row_bytes
                    expected = b"".join(raw[row * width:(row + 1) * width] for row in ids)
                    expected_scales = b"" if scales is None else struct.pack(
                        f"<{count}f", *(scales[row] for row in ids)
                    )
                    reader = PLEFileReader(manifest, "mmap")
                    try:
                        with self.subTest(count=count, storage=storage, scale_mode=scale_mode):
                            self.assertEqual(reader.read_rows(ids), (expected, expected_scales))
                            for invalid in (-1, 3, 1 << 80):
                                with self.assertRaises(IndexError):
                                    reader.read_rows(ids[:-1] + [invalid])
                    finally:
                        reader.close()

    def test_mmap_chunk_boundary_preserves_duplicates_and_scale_alignment(self):
        manifest, raw, scales = self._fixture("bf16", "row")
        ids = [2, 0, 1, 2] * 400_000
        expected = b"".join(raw[row * 6:(row + 1) * 6] for row in [2, 0, 1, 2]) * 400_000
        expected_scales = struct.pack("<4f", *(scales[row] for row in [2, 0, 1, 2])) * 400_000
        reader = PLEFileReader(manifest, "mmap")
        try:
            self.assertEqual(reader.read_rows(ids), (expected, expected_scales))
        finally:
            reader.close()

    def test_truncated_rows_and_scales_raise_and_release_mapping_views(self):
        for backend in self.backends:
            for scale in (False, True):
                with self.subTest(backend=backend, scale=scale):
                    path = self.root / "short.bin"
                    path.write_bytes(b"\x01\x02\x03" + (b"\x00\x00" if scale else b""))
                    shard = PLEShard(path, 0, 1, 0, 3 if scale else 4,
                                     scale_offset=3 if scale else -1,
                                     scale_row_bytes=4 if scale else 0)
                    manifest = PLEManifest("short", 3, 1, 1, (shard,), scale_is_per_row=True)
                    reader = PLEFileReader(manifest, backend)
                    try:
                        with self.assertRaisesRegex(OSError, "incomplete PLE row"):
                            reader.read_rows([0])
                    finally:
                        reader.close()



class TestNativePLEMarkerFormats(unittest.TestCase):
    def _write_marker(self, storage, rows, dim=3, scale=None):
        root = Path(self.directory.name) / storage
        root.mkdir()
        descriptors = []
        for index, values in enumerate(rows):
            path = root / f"shard_{index}.safetensors"
            tensors = {"data": values}
            if scale is not None:
                tensors["scale"] = scale[index]
            save_file(tensors, str(path))
            descriptors.append({
                "file": path.name,
                "rows": values.shape[0],
                "size": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "data_tensor": "data",
                **({"scale_tensor": "scale"} if scale is not None else {}),
            })
        marker = {
            "schema": 1,
            "storage": storage,
            "root": str(root),
            "num_shards": len(rows),
            "rows_per_shard": rows[0].shape[0],
            "num_embeddings": sum(item.shape[0] for item in rows),
            "embedding_dim": dim,
            "global_scale": 1.0,
            "hash": {
                "vocab_size": 17,
                "eos_id": 2,
                "ngram_size": 3,
                "heads_per_order": 2,
                "multipliers": [3, 5, 7],
                "head_sizes": [11, 13],
                "head_offsets": [0, 11],
            },
            "shards": descriptors,
        }
        (root / "native-ple.json").write_text(json.dumps(marker))
        return root

    def _lookup(self, root, expected):
        with envs.SGLANG_QWEN4_PLE_NVME_BACKEND.override("mmap"):
            table = NVMePLEEmbedding(root, num_embeddings=expected.shape[0],
                                     embedding_dim=expected.shape[-1], expected_shards=2)
            self.addCleanup(table.close)
            ids = torch.tensor([2, 0, 1, 2, 1])
            actual = table(ids)
            load_checkpoint_scale(table, 0.00019931793212890625)
            actual = actual * table.weight_scale
        torch.testing.assert_close(actual, expected.index_select(0, ids), rtol=0, atol=0)

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)

    def test_fp8_per_row_decodes_scale_and_deduplicates_rows(self):
        rows = [
            torch.tensor([[1., 2., 3.], [4., 5., 6.]], dtype=torch.float8_e4m3fn),
            torch.tensor([[7., 8., 9.]], dtype=torch.float8_e4m3fn),
        ]
        scales = [torch.tensor([2., 3.]), torch.tensor([4.])]
        root = self._write_marker("fp8_e4m3_rowscale", rows, scale=scales)
        expected = torch.cat([rows[0].float() * scales[0].view(-1, 1),
                              rows[1].float() * scales[1].view(-1, 1)]).to(torch.bfloat16)
        self._lookup(root, expected)

    def _fp8_cuda_fixture(self):
        rows = [
            torch.tensor([[0., -0., 1., -2., 448., -448., 0.001953125],
                          [2., 4., 8., 16., 32., 64., 128.]], dtype=torch.float8_e4m3fn),
            torch.tensor([[3., 6., 9., 12., 15., 18., 21.]], dtype=torch.float8_e4m3fn),
        ]
        scales = [torch.tensor([0.75, -2.]), torch.tensor([2.0**-20])]
        root = self._write_marker("fp8_e4m3_rowscale", rows, dim=7, scale=scales)
        with envs.SGLANG_QWEN4_PLE_NVME_BACKEND.override("mmap"):
            table = NVMePLEEmbedding(root, num_embeddings=3, embedding_dim=7, expected_shards=2)
        self.addCleanup(table.close)
        expected = torch.cat(rows).float().mul_(torch.cat(scales)[:, None]).to(torch.bfloat16)
        return table, expected


    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_fp8_cuda_threshold_and_reverse_cpu_strided_finishes(self):
        table, source = self._fp8_cuda_fixture()
        device = torch.device("cuda", torch.cuda.current_device())
        for count in (1023, 1024, 1025):
            ids = (torch.arange(count, device=device) % 3).flip(0)
            expected = source.index_select(0, ids.cpu())
            actual = table(ids)
            torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)

        ids = torch.arange(1025, device=device) % 3
        first = table.start_gather(ids)
        second = table.start_gather(ids.flip(0))
        cpu_out = torch.empty((1025, 14), dtype=torch.bfloat16)[:, ::2]
        table.finish_gather(second, torch.device("cpu"), out=cpu_out)
        torch.testing.assert_close(cpu_out, source.index_select(0, ids.flip(0).cpu()), rtol=0, atol=0)
        strided = torch.empty((1025, 14), dtype=torch.bfloat16, device=device)[:, ::2]
        table.finish_gather(first, device, out=strided)
        torch.testing.assert_close(strided.cpu(), source.index_select(0, ids.cpu()), rtol=0, atol=0)
        empty = table(torch.empty((2, 0), dtype=torch.long, device=device))
        self.assertEqual(empty.shape, (2, 0, 7))
        pending = table.start_gather(ids)
        for out in (torch.empty((1025, 6), dtype=torch.bfloat16, device=device),
                    torch.empty((1025, 7), dtype=torch.float32, device=device),
                    torch.empty((1025, 7), dtype=torch.bfloat16)):
            with self.assertRaisesRegex(ValueError, "invalid native PLE output"):
                table.finish_gather(pending, device, out=out)


    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_fp8_cuda_raw_leases_wait_across_streams_growth_and_close(self):
        table, source = self._fp8_cuda_fixture()
        device = torch.device("cuda", torch.cuda.current_device())
        first_ids = torch.arange(1024, device=device) % 3
        second_ids = (torch.arange(1025, device=device) + 1) % 3
        first_pending = table.start_gather(first_ids)
        second_pending = table.start_gather(second_ids)
        first_stream, second_stream = torch.cuda.Stream(), torch.cuda.Stream()
        with torch.cuda.stream(first_stream):
            torch.cuda._sleep(50_000_000)
        first = table.finish_gather(first_pending, device, stream=first_stream)
        second = table.finish_gather(second_pending, device, stream=second_stream)
        table.close()
        table.close()
        torch.testing.assert_close(first.cpu(), source.index_select(0, first_ids.cpu()), rtol=0, atol=0)
        torch.testing.assert_close(second.cpu(), source.index_select(0, second_ids.cpu()), rtol=0, atol=0)


    def test_bf16_uses_scalar_source_scale_and_two_byte_stride(self):
        rows = [torch.tensor([[1., 2., 3.], [4., 5., 6.]], dtype=torch.bfloat16),
                torch.tensor([[7., 8., 9.]], dtype=torch.bfloat16)]
        scales = [torch.tensor([2.]), torch.tensor([3.])]
        root = self._write_marker("bf16", rows, scale=scales)
        expected = torch.cat([rows[0].float() * 2, rows[1].float() * 3]).to(torch.bfloat16)
        self._lookup(root, expected)

    def test_f16_defaults_to_source_scale_one(self):
        rows = [torch.tensor([[1., 2., 3.], [4., 5., 6.]], dtype=torch.float16),
                torch.tensor([[7., 8., 9.]], dtype=torch.float16)]
        root = self._write_marker("f16", rows)
        expected = torch.cat(rows).to(torch.bfloat16)
        self._lookup(root, expected)

    def test_marker_rejects_global_scale_round_trip(self):
        rows = [torch.ones((2, 3), dtype=torch.bfloat16), torch.ones((1, 3), dtype=torch.bfloat16)]
        root = self._write_marker("bf16", rows)
        marker_path = root / "native-ple.json"
        marker = json.loads(marker_path.read_text())
        marker["global_scale"] = 0.00019931793212890625
        marker_path.write_text(json.dumps(marker))
        with self.assertRaisesRegex(ValueError, "global_scale"):
            PLEManifest.from_snapshot(root, expected_shards=2)

class TestPLEAcceptedBoundary(unittest.TestCase):
    def test_rejected_draft_tail_does_not_enter_live_or_tracked_state(self):
        from types import SimpleNamespace
        from sglang.srt.models.qwen4_exp import Qwen4ExpForConditionalGeneration
        from sglang.srt.mem_cache.ple_state_pool import ShortConvPool, NGramPool

        conv = ShortConvPool(
            size=5, state_shape=(8, 3), layer_ids=[1], dtype=torch.bfloat16,
            device="cpu", spec_state_size=2, speculative_num_draft_tokens=4,
        )
        ngram = NGramPool(
            size=5, context_len=2, eos_token_id=248044, device="cpu",
            spec_state_size=2, speculative_num_draft_tokens=4,
        )
        for row in range(2):
            for step in range(4):
                conv.intermediate_conv_state[:, row, step].fill_(row * 10 + step)
                ngram.intermediate_context[row, step] = torch.tensor(
                    [row * 10 + step, row * 10 + step + 1]
                )
        physical_slots = torch.tensor([0, 2, 1])
        pool = SimpleNamespace(
            short_conv_pool=conv, ngram_pool=ngram,
            get_mamba_indices=lambda requests: physical_slots.index_select(0, requests),
        )
        Qwen4ExpForConditionalGeneration.update_conv_state_after_mtp_verify(
            None, pool, torch.tensor([1, 2]), torch.tensor([0, 2]),
            torch.tensor([3, 4]), torch.tensor([-1, 1]),
        )
        torch.testing.assert_close(conv.conv_state[0, 2], torch.zeros(8, 3, dtype=torch.bfloat16))
        torch.testing.assert_close(conv.conv_state[0, 1], torch.full((8, 3), 12, dtype=torch.bfloat16))
        self.assertEqual(ngram.context[2].tolist(), [0, 1])
        self.assertEqual(ngram.context[1].tolist(), [12, 13])
        self.assertEqual(ngram.context[3].tolist(), [248044, 248044])
        self.assertEqual(ngram.context[4].tolist(), [11, 12])


if __name__ == "__main__":
    unittest.main()
