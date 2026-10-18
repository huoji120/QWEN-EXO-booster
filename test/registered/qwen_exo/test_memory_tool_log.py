import tempfile
import unittest
from pathlib import Path
from unittest import mock

from qwen_exo_booster import memory_tool_log
from qwen_exo_booster.memory_tool_log import MemoryToolLog


def call(index, output="x"):
    return {
        "id": f"mcp_{index}",
        "name": "qwen_exo_memory_search",
        "arguments": '{"pattern": "nginx"}',
        "ok": True,
        "output": output,
        "duration_ms": 1.5,
    }


class MemoryToolLogTestCase(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.path = Path(self.tempdir.name) / "calls.sqlite3"

    def tearDown(self):
        self.tempdir.cleanup()

    def test_full_text_per_request_and_summaries_without_output(self):
        log = MemoryToolLog(self.path)
        log.record("resp_a", 0, [call(1, "first hit"), call(2, "second hit")])
        log.record("resp_b", 0, [call(3)])
        log.record("resp_a", 1, [call(4, "read output")])
        calls = log.for_request("resp_a")
        self.assertEqual(
            [(c["call_id"], c["round"], c["output"]) for c in calls],
            [("mcp_1", 0, "first hit"), ("mcp_2", 0, "second hit"), ("mcp_4", 1, "read output")],
        )
        recent = log.recent(2)
        self.assertEqual([c["call_id"] for c in recent], ["mcp_4", "mcp_3"])
        self.assertNotIn("output", recent[0])
        self.assertEqual(recent[0]["output_chars"], len("read output"))

    def test_retention_drops_oldest_by_rows_and_bytes(self):
        with mock.patch.object(memory_tool_log, "_PRUNE_EVERY", 1):
            log = MemoryToolLog(self.path, max_rows=3, max_bytes=10_000)
            for index in range(5):
                log.record(f"resp_{index}", 0, [call(index)])
            self.assertEqual(
                [c["call_id"] for c in log.recent()], ["mcp_4", "mcp_3", "mcp_2"]
            )
            # Each row is ~20 bytes of arguments plus its output.
            sized = MemoryToolLog(Path(self.tempdir.name) / "sized.sqlite3", max_rows=100, max_bytes=250)
            for index in range(4):
                sized.record(f"resp_{index}", 0, [call(index, "o" * 80)])
            self.assertEqual([c["call_id"] for c in sized.recent()], ["mcp_3", "mcp_2"])


if __name__ == "__main__":
    unittest.main()
