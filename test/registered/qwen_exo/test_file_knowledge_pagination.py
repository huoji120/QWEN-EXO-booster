import sqlite3
import tempfile
import unittest
from contextlib import closing
from pathlib import Path
from unittest.mock import patch

from qwen_exo_booster.file_knowledge import FileKnowledgeError, FileKnowledgeIndex


class TestFileKnowledgePagination(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tempdir.name) / "knowledge"
        self.index = FileKnowledgeIndex(self.root, Path(self.tempdir.name) / "state")

    def tearDown(self):
        self.tempdir.cleanup()

    def _files(self, paths):
        for relative in paths:
            path = self.root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"text")
        self.index.refresh()

    def _paths(self, **kwargs):
        return [row["relative_path"] for row in self.index.page_files(**kwargs)["files"]]

    def _metadata_copy(self, source, destination):
        # Model Linux-only case distinctions without creating aliases on NTFS.
        with closing(sqlite3.connect(self.index.database_path)) as db, db:
            db.execute(
                "INSERT INTO knowledge_files "
                "SELECT ?, sha256, byte_count, mtime_ns, encoding, line_count, "
                "indexed_ns, schema_version, kind, title FROM knowledge_files "
                "WHERE relative_path = ?",
                (destination, source),
            )

    def test_distinct_bounded_pages_and_complete_worker_metadata(self):
        paths = [f"docs/{number:03d}.txt" for number in range(205)]
        self._files(reversed(paths))
        pages = [self.index.page_files(offset=offset) for offset in (0, 100, 200)]
        self.assertEqual([len(page["files"]) for page in pages], [100, 100, 5])
        self.assertEqual(
            [row["relative_path"] for page in pages for row in page["files"]], paths
        )
        for page, offset in zip(pages, (0, 100, 200)):
            self.assertEqual(set(page), {"files", "total", "total_bytes", "offset", "limit"})
            self.assertEqual(
                (page["total"], page["total_bytes"], page["offset"], page["limit"]),
                (205, 820, offset, 100),
            )
        self.assertEqual(len(self.index.metadata()), 205)
        self.assertEqual(len(self.index.page_files(limit=200)["files"]), 200)
        for offset in (205, 206, 10**100):
            page = self.index.page_files(offset=offset)
            self.assertEqual(page["files"], [])
            self.assertEqual((page["total"], page["offset"]), (205, offset))

    def test_page_validation(self):
        for name, values in {
            "limit": [0, -1, 201, True, False, 1.5, "100", None],
            "offset": [-1, True, False, 1.5, "0", None],
            "nested": [0, 1, "true", None],
            "q": [None, True, 123, "x" * 257],
            "area": ["WORK", "unknown", None, []],
            "folder": [None, True, "../docs", "/docs", "docs/", "docs//sub", "C:/docs", "docs\\sub"],
        }.items():
            for value in values:
                with self.subTest(name=name, value=value):
                    with self.assertRaises(FileKnowledgeError):
                        self.index.page_files(**{name: value})
        self.assertEqual(self.index.page_files(q="x" * 256)["total"], 0)

    def test_exact_folder_membership_root_direct_and_slash_boundaries(self):
        self._files(["root.txt", "docs/a.txt", "docs/deep/b.txt", "docs-extra/c.txt"])
        self.assertEqual(self._paths(folder="docs"), ["docs/a.txt", "docs/deep/b.txt"])
        self.assertEqual(self._paths(folder="docs", nested=False), ["docs/a.txt"])
        self.assertEqual(self._paths(nested=False), ["root.txt"])
        self.assertEqual(len(self._paths()), 4)
        self.assertEqual(self._paths(folder="Docs"), [])
        self.assertEqual(self._paths(folder="doc"), [])
        self.assertEqual(self._paths(folder="missing"), [])
        self.assertEqual(self._paths(folder="docs/a.txt"), [])
        self.assertEqual(self._paths(q="docs/"), ["docs/a.txt", "docs/deep/b.txt"])

    def test_work_area_is_exact_case_and_boundary_safe(self):
        self._files(["work-memory/a.txt", "work-memory-extra/b.txt", "root.txt"])
        self._metadata_copy("work-memory/a.txt", "Work-Memory/a.txt")
        self._metadata_copy("root.txt", "work-memory")
        self.assertEqual(self._paths(area="work"), ["work-memory/a.txt"])
        self.assertEqual(
            self._paths(area="imported"),
            ["Work-Memory/a.txt", "root.txt", "work-memory", "work-memory-extra/b.txt"],
        )
        self.assertEqual(self._paths(folder="Work-Memory"), ["Work-Memory/a.txt"])
        self.assertEqual(self._paths(folder="Work-Memory", area="work"), [])
        self.assertEqual(self._paths(folder="work-memory", area="imported"), [])
        tree = self.index.directory_summary()
        self.assertEqual(tree["areas"], {
            "imported": {"files": 4, "bytes": 16},
            "work": {"files": 1, "bytes": 4},
        })
        self.assertEqual(tree["counts"]["Work-Memory"], {"files": 1, "bytes": 4})
        self.assertEqual(tree["counts"]["work-memory"], {"files": 1, "bytes": 4})

    def test_unicode_case_insensitive_paths_and_titles(self):
        self._files(["ÉTUDE.md", "plain.md", "summary.md"])
        with closing(sqlite3.connect(self.index.database_path)) as db, db:
            db.execute("UPDATE knowledge_files SET title = ? WHERE relative_path = ?", ("Über Straße", "plain.md"))
        summary_row = next(row for row in self.index.metadata() if row["relative_path"] == "summary.md")
        self.index.store_summary(
            "summary.md", sha256=summary_row["sha256"], summary="# CAFÉ notes",
            sampled=False, origin="manual",
        )
        for q in ("ÉTUDE", "étude"):
            self.assertEqual(self._paths(q=q), ["ÉTUDE.md"])
        for q in ("ÜBER", "über", "STRASSE", "straße"):
            self.assertEqual(self._paths(q=q), ["plain.md"])
        for q in ("CAFÉ", "café"):
            page = self.index.page_files(q=q)
            self.assertEqual([row["relative_path"] for row in page["files"]], ["summary.md"])
            self.assertEqual((page["total"], page["total_bytes"]), (1, 4))

    def test_literal_search_and_folder_wildcards(self):
        self._files(["100%_real/a.md", "100xxreal/b.md", "under_score.md", "underXscore.md", "apostrophe's.md"])
        self.assertEqual(self._paths(q="%"), ["100%_real/a.md"])
        self.assertEqual(self._paths(q="_"), ["100%_real/a.md", "under_score.md"])
        self.assertEqual(self._paths(q="%' OR 1=1 --"), [])
        self.assertEqual(self._paths(q="'"), ["apostrophe's.md"])
        self.assertEqual(self._paths(folder="100%_real"), ["100%_real/a.md"])
        self.assertEqual(self._paths(q="100%_real/"), ["100%_real/a.md"])

    def test_tree_unions_empty_hidden_and_indexed_parent_directories(self):
        self._files(["root.txt", ".notes/deep/a.txt", "docs/a.txt", "docs/sub/b.txt"])
        (self.root / "empty" / "child").mkdir(parents=True)
        (self.root / ".empty").mkdir()
        self._metadata_copy("root.txt", ".indexed-only/sub/file.txt")
        tree = self.index.directory_summary()
        self.assertEqual(set(tree), {"directories", "counts", "areas"})
        self.assertEqual(tree["directories"], [
            ".empty", ".indexed-only", ".indexed-only/sub", ".notes", ".notes/deep",
            "docs", "docs/sub", "empty", "empty/child",
        ])
        self.assertEqual(tree["counts"][""], {"files": 5, "bytes": 20})
        self.assertEqual(tree["counts"]["docs"], {"files": 2, "bytes": 8})
        for directory in (".notes", ".notes/deep", ".indexed-only", ".indexed-only/sub"):
            self.assertEqual(tree["counts"][directory], {"files": 1, "bytes": 4})
        for directory in ("empty", "empty/child", ".empty"):
            self.assertEqual(tree["counts"][directory], {"files": 0, "bytes": 0})
        self.assertEqual(tree["areas"]["imported"], {"files": 5, "bytes": 20})
        self.assertEqual(tree["areas"]["work"], {"files": 0, "bytes": 0})

    def test_summary_shape_and_staleness_are_preserved(self):
        self._files(["a.md", "b.md"])
        row = self.index.metadata()[0]
        stored = self.index.store_summary(
            "a.md", sha256=row["sha256"], summary="# My title\nDetails",
            sampled=True, origin="manual",
        )
        page = self.index.page_files(limit=1)
        self.assertEqual(page["files"][0]["summary"], {
            "title": "My title", "origin": "manual", "sampled": True,
            "updated_ns": stored["updated_ns"], "stale": False,
        })
        self.assertEqual(page["files"][0], self.index.metadata()[0])
        self.assertIsNone(self.index.page_files(offset=1)["files"][0]["summary"])
        self.index.upsert_bytes("a.md", b"changed bytes")
        self.assertTrue(self.index.page_files(limit=1)["files"][0]["summary"]["stale"])
        self.assertEqual(len(self.index.metadata()), 2)

    def test_pages_never_scan_or_read_source_files(self):
        self._files([f"docs/{number:03d}.txt" for number in range(105)])
        (self.root / "not-indexed.txt").write_bytes(b"new")
        with patch.object(self.index, "refresh", side_effect=AssertionError("refresh")), \
             patch("os.walk", side_effect=AssertionError("walk")), \
             patch.object(Path, "read_bytes", side_effect=AssertionError("read")):
            page = self.index.page_files(offset=100)
            self.assertEqual(page["total"], 105)
            self.assertEqual(len(page["files"]), 5)
        self.index.refresh()
        self.assertEqual(self.index.page_files()["total"], 106)

    def test_later_page_read_move_delete_and_out_of_range_offset(self):
        self._files([f"docs/{number:03d}.txt" for number in range(103)])
        last = self.index.page_files(offset=100)
        selected = last["files"][0]["relative_path"]
        self.assertEqual(selected, "docs/100.txt")
        self.assertEqual(self.index.read(selected)["text"], "text")
        self.index.move(selected, "moved/100.txt")
        self.assertEqual(self._paths(folder="moved"), ["moved/100.txt"])
        self.index.delete("moved/100.txt")
        self.index.delete("docs/101.txt")
        self.index.delete("docs/102.txt")
        last = self.index.page_files(offset=100)
        self.assertEqual((last["files"], last["total"], last["offset"]), ([], 100, 100))
        self.assertEqual(len(self.index.page_files(offset=0)["files"]), 100)
        self.assertEqual(self.index.directory_summary()["counts"][""], {"files": 100, "bytes": 400})

    def test_folder_mutations_do_not_touch_other_case_or_prefix(self):
        self._files(["docs/a.txt", "docs/sub/b.txt", "docs-extra/c.txt"])
        self._metadata_copy("docs/a.txt", "Docs/a.txt")
        result = self.index.move("docs", "renamed")
        self.assertEqual(result["files"], 2)
        self.assertEqual(self._paths(folder="Docs"), ["Docs/a.txt"])
        self.assertEqual(self._paths(folder="renamed"), ["renamed/a.txt", "renamed/sub/b.txt"])
        self._metadata_copy("renamed/a.txt", "Renamed/a.txt")
        self.assertEqual(self.index.delete_directory("renamed"), 2)
        self.assertEqual(self._paths(), ["Docs/a.txt", "Renamed/a.txt", "docs-extra/c.txt"])

    def test_page_rows_and_totals_share_one_read_snapshot(self):
        self._files(["a.txt", "b.txt"])
        with closing(sqlite3.connect(self.index.database_path)) as db, db:
            db.execute("PRAGMA journal_mode = WAL")
        connect = self.index._connect
        deleted = False

        def connect_with_concurrent_delete():
            connection = connect()

            def before_statement(sql):
                nonlocal deleted
                if "SELECT f.*" in sql and not deleted:
                    with closing(sqlite3.connect(self.index.database_path)) as writer, writer:
                        writer.execute("DELETE FROM knowledge_files WHERE relative_path = 'b.txt'")
                    deleted = True

            connection.set_trace_callback(before_statement)
            return connection

        with patch.object(self.index, "_connect", side_effect=connect_with_concurrent_delete):
            page = self.index.page_files()
        self.assertTrue(deleted)
        self.assertEqual((page["total"], page["total_bytes"]), (2, 8))
        self.assertEqual([row["relative_path"] for row in page["files"]], ["a.txt", "b.txt"])
        self.assertEqual(self.index.page_files()["total"], 1)


if __name__ == "__main__":
    unittest.main()
