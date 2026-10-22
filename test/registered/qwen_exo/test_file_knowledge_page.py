"""The knowledge console pages large corpora instead of shipping every file.

Workers still call ``metadata()`` for the full index. The browser uses
``directory_summary`` (no file rows) and ``page_files`` (one SQL page), including
when every file sits in the corpus root.
"""

from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from qwen_exo_booster.file_knowledge import FileKnowledgeIndex


def _write_corpus(root: Path, count: int, *, subdirectory: str = "") -> None:
    target = root / subdirectory if subdirectory else root
    target.mkdir(parents=True, exist_ok=True)
    for index in range(count):
        (target / f"note-{index:05d}.md").write_text(
            f"# Note {index}\nflat corpus body\n", encoding="utf-8"
        )


class TestFileKnowledgePage(unittest.TestCase):
    def test_flat_root_pages_without_dropping_the_full_index(self):
        with TemporaryDirectory() as temp:
            root = Path(temp) / "knowledge"
            _write_corpus(root, 250)
            _write_corpus(root, 3, subdirectory="nested")
            (root / "work-memory").mkdir()
            (root / "work-memory" / "today.md").write_text("# Today\nwork\n", encoding="utf-8")
            index = FileKnowledgeIndex(root, Path(temp) / "state")
            index.refresh()

            # Dream and summary workers keep the unpaged index.
            self.assertEqual(len(index.metadata()), 254)

            first = index.page_files(folder="", nested=True, area="imported", limit=100, offset=0)
            self.assertEqual(first["total"], 253)
            self.assertEqual(len(first["files"]), 100)
            self.assertEqual(first["offset"], 0)
            self.assertTrue(all("content" not in row for row in first["files"]))
            self.assertTrue(all("summary" not in (row["summary"] or {}) for row in first["files"]))

            last = index.page_files(folder="", nested=True, area="imported", limit=100, offset=200)
            self.assertEqual(last["total"], 253)
            self.assertEqual(len(last["files"]), 53)
            self.assertTrue(
                {row["relative_path"] for row in first["files"]}.isdisjoint(
                    {row["relative_path"] for row in last["files"]}
                )
            )

            direct = index.page_files(folder="", nested=False, area="imported", limit=200, offset=0)
            self.assertEqual(direct["total"], 250)
            self.assertTrue(all("/" not in row["relative_path"] for row in direct["files"]))

            nested = index.page_files(folder="nested", nested=False, area="all", limit=50, offset=0)
            self.assertEqual(
                {row["relative_path"] for row in nested["files"]},
                {"nested/note-00000.md", "nested/note-00001.md", "nested/note-00002.md"},
            )

            titled = index.page_files(query="Note 12", area="imported", limit=50, offset=0)
            self.assertGreaterEqual(titled["total"], 1)
            self.assertTrue(any(row["relative_path"].endswith("note-00012.md") for row in titled["files"]))

            summary = index.directory_summary()
            self.assertNotIn("files", summary)
            self.assertEqual(summary["counts"][""]["files"], 254)
            self.assertEqual(summary["areas"]["imported"]["files"], 253)
            self.assertEqual(summary["areas"]["work"]["files"], 1)
            self.assertEqual(summary["counts"]["nested"]["files"], 3)
            self.assertIn("nested", summary["directories"])

    def test_directory_summary_stays_small_and_page_actions_still_work(self):
        with TemporaryDirectory() as temp:
            root = Path(temp) / "knowledge"
            _write_corpus(root, 120)
            index = FileKnowledgeIndex(root, Path(temp) / "state")
            index.refresh()

            summary = index.directory_summary()
            self.assertNotIn("files", summary)
            self.assertEqual(summary["counts"][""]["files"], 120)
            self.assertLess(len(json.dumps(summary)), 64_000)

            page = index.page_files(folder="", nested=True, area="imported", limit=40, offset=80)
            self.assertEqual(page["total"], 120)
            self.assertEqual(len(page["files"]), 40)
            self.assertEqual(page["offset"], 80)

            opened = index.read(page["files"][0]["relative_path"], offset=0, limit=64)
            self.assertIn("flat corpus body", opened["text"])

            renamed = index.move(page["files"][0]["relative_path"], "renamed-page.md")
            self.assertEqual(renamed["to"], "renamed-page.md")
            self.assertEqual(index.page_files(query="renamed-page", limit=10)["total"], 1)
            self.assertEqual(len(index.metadata()), 120)


if __name__ == "__main__":
    unittest.main()
