import json
import hashlib
import tempfile
import unittest
from pathlib import Path

import torch
from safetensors.torch import save_file

from sglang.srt.environ import envs
from sglang.srt.models.qwen4_ple_nvme import NVMePLEEmbedding, PLEManifest


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
