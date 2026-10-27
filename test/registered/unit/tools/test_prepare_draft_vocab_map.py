"""Actual offline CLI coverage with tiny JSON fixtures; no models or GPU access."""

import hashlib
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

SCRIPT = Path(__file__).resolve().parents[4] / "scripts/qwen_exo/prepare_draft_vocab_map.py"


class TestPrepareDraftVocabMap(CustomTestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.model = self.root / "model"
        self.model.mkdir()
        self.config_path = self.model / "config.json"
        self.tokenizer_path = self.model / "tokenizer.json"
        self.source = self.root / "ids.json"
        self.output = self.root / "map.json"
        self.config = {
            "model_type": "qwen4_exp",
            "bos_token_id": 10,
            "text_config": {
                "model_type": "qwen4_exp_text",
                "vocab_size": 16,
                "hidden_size": 4,
                "bos_token_id": 11,
                "eos_token_id": [12, 13],
            },
        }
        self.tokenizer = {
            "added_tokens": [
                {"id": 14, "special": True, "content": "<special>"},
                {"id": 15, "special": False, "content": "ordinary"},
            ]
        }
        self.write_json(self.config_path, self.config)
        self.write_json(self.tokenizer_path, self.tokenizer)
        self.write_json(self.source, [4, 2, 4, 0])

    @staticmethod
    def write_json(path, value):
        path.write_text(json.dumps(value, indent=1) + "\n", encoding="utf-8")

    def run_cli(self, *extra):
        return subprocess.run(
            [
                sys.executable,
                str(SCRIPT),
                "--model-path", str(self.model),
                "--token-ids-json", str(self.source),
                "--output", str(self.output),
                *map(str, extra),
            ],
            cwd=self.root,
            env={**os.environ, "CUDA_VISIBLE_DEVICES": "", "HF_HUB_OFFLINE": "1"},
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        )

    def assert_failed(self, result, message):
        self.assertNotEqual(result.returncode, 0, result.stdout)
        self.assertIn(message, result.stderr)
        self.assertNotIn("Traceback", result.stderr)

    def test_default_tokenizer_normalizes_union_and_preserves_all_sources(self):
        sources = (self.config_path, self.tokenizer_path, self.source)
        before = {path: path.read_bytes() for path in sources}
        result = self.run_cli(
            "--source-url", "https://example.invalid/reference.json",
            "--source-revision", "rev-123",
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        document = json.loads(self.output.read_text(encoding="utf-8"))
        self.assertEqual(document, {
            "schema": "qwen4-draft-vocab-v1",
            "model_config_sha256": hashlib.sha256(before[self.config_path]).hexdigest(),
            "tokenizer_sha256": hashlib.sha256(before[self.tokenizer_path]).hexdigest(),
            "vocab_size": 16,
            "hidden_size": 4,
            "token_ids": [0, 2, 4, 10, 11, 12, 13, 14],
            "provenance": {
                "source_sha256": hashlib.sha256(before[self.source]).hexdigest(),
                "source_url": "https://example.invalid/reference.json",
                "source_revision": "rev-123",
            },
        })
        self.assertEqual(before, {path: path.read_bytes() for path in sources})
        self.assertLess(self.output.stat().st_size, 8 * 1024 * 1024)

    def test_explicit_tokenizer_directory_or_file_and_top_level_config(self):
        self.write_json(self.config_path, self.config["text_config"])
        other = self.root / "tokenizer"
        other.mkdir()
        other_file = other / "tokenizer.json"
        self.write_json(other_file, {"added_tokens": [{"id": 9, "special": True}]})
        for tokenizer_arg in (other, other_file):
            with self.subTest(tokenizer_arg=tokenizer_arg):
                result = self.run_cli(
                    "--model-path", self.config_path,
                    "--tokenizer-path", tokenizer_arg,
                    "--overwrite",
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                document = json.loads(self.output.read_text(encoding="utf-8"))
                self.assertEqual(document["token_ids"], [0, 2, 4, 9, 11, 12, 13])
                self.assertEqual(document["tokenizer_sha256"], hashlib.sha256(other_file.read_bytes()).hexdigest())
                self.assertEqual(set(document["provenance"]), {"source_sha256"})

    def test_existing_output_requires_overwrite_and_failed_overwrite_preserves_it(self):
        self.output.write_bytes(b"existing-map")
        self.assert_failed(self.run_cli(), "--overwrite")
        self.assertEqual(self.output.read_bytes(), b"existing-map")
        self.write_json(self.source, [True])
        self.assert_failed(self.run_cli("--overwrite"), "Token IDs")
        self.assertEqual(self.output.read_bytes(), b"existing-map")
        self.write_json(self.source, [3])
        result = self.run_cli("--overwrite")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(self.output.read_bytes())["token_ids"], [3, 10, 11, 12, 13, 14])

    def test_overwrite_cannot_replace_any_input(self):
        for path in (self.config_path, self.tokenizer_path, self.source):
            with self.subTest(path=path):
                before = path.read_bytes()
                result = self.run_cli("--output", path, "--overwrite")
                self.assert_failed(result, "must not replace")
                self.assertEqual(path.read_bytes(), before)

    def test_bad_source_values_fail_without_output(self):
        for value in ([True], [False], [1.0], ["2"], [-1], [16], [None], [[2]], {"token_ids": [2]}, None):
            with self.subTest(value=value):
                self.write_json(self.source, value)
                result = self.run_cli()
                self.assertNotEqual(result.returncode, 0)
                self.assertFalse(self.output.exists())

    def test_empty_or_full_subset_including_required_union_fails(self):
        cases = [
            ([], {"added_tokens": []}, {}),
            (list(range(16)), self.tokenizer, self.config),
            (list(range(10)) + [15], self.tokenizer, self.config),
        ]
        for ids, tokenizer, config in cases:
            with self.subTest(ids=ids):
                effective_config = config or {
                    "model_type": "qwen4_exp_text", "vocab_size": 16, "hidden_size": 4,
                }
                self.write_json(self.config_path, effective_config)
                self.write_json(self.tokenizer_path, tokenizer)
                self.write_json(self.source, ids)
                self.assert_failed(self.run_cli(), "nonempty and smaller")
                self.assertFalse(self.output.exists())

    def test_wrong_model_type_and_invalid_dimensions_fail(self):
        changes = [
            ("model_type", "qwen3_5_text"),
            ("vocab_size", True), ("vocab_size", 0), ("vocab_size", "16"),
            ("hidden_size", False), ("hidden_size", -1), ("hidden_size", 4.0),
        ]
        for key, value in changes:
            with self.subTest(key=key, value=value):
                config = json.loads(json.dumps(self.config))
                config["text_config"][key] = value
                self.write_json(self.config_path, config)
                self.assertNotEqual(self.run_cli().returncode, 0)
                self.assertFalse(self.output.exists())

    def test_bad_required_specials_fail_instead_of_omitting_them(self):
        for value in (True, -1, 16, "14", None):
            with self.subTest(tokenizer_id=value):
                self.write_json(self.tokenizer_path, {"added_tokens": [{"id": value, "special": True}]})
                self.assert_failed(self.run_cli(), "Token IDs")
                self.assertFalse(self.output.exists())
        self.write_json(self.tokenizer_path, self.tokenizer)
        for value in (True, -1, 16, [12, False], "12"):
            with self.subTest(eos_id=value):
                self.config["text_config"]["eos_token_id"] = value
                self.write_json(self.config_path, self.config)
                self.assert_failed(self.run_cli(), "Token IDs")
                self.assertFalse(self.output.exists())

    def test_malformed_json_and_missing_files_fail_cleanly(self):
        self.source.write_bytes(b"[1,")
        self.assert_failed(self.run_cli(), "error:")
        self.assertFalse(self.output.exists())
        self.source.unlink()
        self.assert_failed(self.run_cli(), "error:")
        self.assertFalse(self.output.exists())

    def test_duplicate_config_keys_are_rejected(self):
        self.config_path.write_bytes(b'{"model_type":"other","model_type":"qwen4_exp_text"}')
        self.assert_failed(self.run_cli(), "Duplicate JSON key")
        self.assertFalse(self.output.exists())

    def test_oversized_source_is_rejected(self):
        self.source.write_bytes(b"[1]" + b" " * (8 * 1024 * 1024))
        self.assert_failed(self.run_cli(), "8 MiB")
        self.assertFalse(self.output.exists())


if __name__ == "__main__":
    unittest.main()
