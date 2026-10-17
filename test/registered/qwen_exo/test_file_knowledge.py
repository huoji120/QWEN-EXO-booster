import hashlib
import os
import sqlite3
import tempfile
import unittest
from pathlib import Path

from qwen_exo_booster.file_knowledge import (
    FileKnowledgeIndex,
    InvalidEncodingError,
    InvalidPathError,
    ReadLimitError,
    StaleFileError,
)


class TestFileKnowledgeIndex(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tempdir.name) / "knowledge"
        self.state = Path(self.tempdir.name) / "state"
        self.index = FileKnowledgeIndex(self.root, self.state)

    def tearDown(self):
        self.tempdir.cleanup()

    def test_metadata_only_schema_and_utf8_bytes(self):
        raw = "ASCII\n中文 😀\n".encode("utf-8")
        (self.root / "docs").mkdir()
        (self.root / "docs/a.md").write_bytes(raw)
        self.index.refresh()

        with sqlite3.connect(self.index.database_path) as db:
            columns = [row[1] for row in db.execute("PRAGMA table_info(knowledge_files)")]
            row = db.execute("SELECT * FROM knowledge_files").fetchone()
        db.close()
        self.assertEqual(
            columns,
            [
                "relative_path",
                "sha256",
                "byte_count",
                "mtime_ns",
                "encoding",
                "line_count",
                "indexed_ns",
                "schema_version",
            ],
        )
        self.assertEqual(row[1], hashlib.sha256(raw).hexdigest())
        self.assertEqual(row[2], len(raw))
        self.assertNotIn(raw, repr(row).encode())

    def test_search_literal_regex_glob_context_and_byte_offsets(self):
        raw = "before\n中文目标\nafter\n".encode("utf-8")
        (self.root / "docs").mkdir()
        (self.root / "docs/a.md").write_bytes(raw)
        (self.root / "other.txt").write_text("中文目标", encoding="utf-8")

        result = self.index.search(
            "中文目标",
            path_glob="docs/*.md",
            context_before=1,
            context_after=1,
        )
        match = result["results"][0]
        line_start = len("before\n".encode())
        match_start = line_start
        match_end = match_start + len("中文目标".encode())
        self.assertEqual(
            (match["path"], match["line"], match["line_start_offset"]),
            ("docs/a.md", 2, line_start),
        )
        self.assertEqual((match["match_start_offset"], match["match_end_offset"]), (match_start, match_end))
        self.assertEqual((match["before"], match["after"]), (["before"], ["after"]))
        regex = self.index.search(r"中文.+", regex=True, path_glob="docs/**")
        self.assertEqual(regex["count"], 1)
        self.assertEqual(
            self.index.search("中文目标", path_glob="docs/*.md", case_sensitive=True)["count"], 1
        )
        self.assertEqual(
            self.index.search("中文目标", path_glob="other.txt", case_sensitive=True)["count"], 1
        )
        with self.assertRaises(ValueError):
            self.index.search("[", regex=True)

    def test_read_boundaries_limit_and_stale_mutation(self):
        raw = "A中文B".encode("utf-8")
        path = self.root / "a.txt"
        path.write_bytes(raw)
        self.index.refresh()
        self.assertEqual(self.index.read("a.txt", offset=1, limit=3)["text"], "中")
        with self.assertRaises(ValueError):
            self.index.read("a.txt", offset=2, limit=1)
        with self.assertRaises(ReadLimitError):
            self.index.read("a.txt", limit=1_048_577)
        path.write_bytes("changed".encode())
        with self.assertRaises(StaleFileError):
            self.index.read("a.txt")

    def test_invalid_utf8_is_metadata_only_and_unavailable(self):
        path = self.root / "bad.bin"
        path.write_bytes(b"ok\xff")
        self.index.refresh()
        row = self.index.metadata()[0]
        self.assertEqual(row["encoding"], "invalid-utf-8")
        self.assertEqual(self.index.search("ok")["count"], 0)
        with self.assertRaises(InvalidEncodingError):
            self.index.read("bad.bin")

    def test_traversal_and_symlink_are_rejected(self):
        with self.assertRaises(InvalidPathError):
            self.index.read("../outside")
        outside = Path(self.tempdir.name) / "outside.txt"
        outside.write_text("outside", encoding="utf-8")
        link = self.root / "link.txt"
        try:
            link.symlink_to(outside)
        except (OSError, NotImplementedError):
            self.skipTest("symlinks are unavailable on this platform")
        self.index.refresh()
        with self.assertRaises(InvalidPathError):
            self.index.read("link.txt")
        self.assertEqual(self.index.metadata(), [])

    def test_incremental_refresh_reuses_unchanged_metadata_and_removes_deleted(self):
        path = self.root / "a.md"
        path.write_text("one", encoding="utf-8")
        self.index.refresh()
        first = self.index.metadata()[0]
        self.index.refresh()
        second = self.index.metadata()[0]
        self.assertEqual(first["indexed_ns"], second["indexed_ns"])
        path.write_text("two", encoding="utf-8")
        # 10 ms, not 1 ns: coarser filesystems (NTFS: 100 ns) round 1 ns away.
        bumped = second["mtime_ns"] + 10_000_000
        os.utime(path, ns=(bumped, bumped))
        self.index.refresh()
        third = self.index.metadata()[0]
        self.assertNotEqual(first["sha256"], third["sha256"])
        path.unlink()
        self.index.refresh()
        self.assertEqual(self.index.metadata(), [])

    def test_atomic_raw_upsert_and_delete(self):
        raw = b"raw\x00bytes\xff"
        metadata = self.index.upsert_bytes("nested/raw.bin", raw)
        self.assertEqual((self.root / "nested/raw.bin").read_bytes(), raw)
        self.assertEqual(metadata["sha256"], hashlib.sha256(raw).hexdigest())
        self.index.delete("nested/raw.bin")
        self.assertFalse((self.root / "nested/raw.bin").exists())
        self.assertEqual(self.index.metadata(), [])
        # The admin API maps this to 404 instead of reporting a phantom delete.
        with self.assertRaises(FileNotFoundError):
            self.index.delete("nested/raw.bin")

    def test_zip_archive_unpacks_text_members_and_rejects_the_rest(self):
        """A knowledge ZIP becomes one file per text member; traversal names,
        macOS metadata, binaries and oversized members never reach the root."""
        import io
        import zipfile
        from unittest import mock

        from qwen_exo_booster import file_knowledge
        from qwen_exo_booster.file_knowledge import FileKnowledgeError

        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
            archive.writestr("a.md", "alpha\n")
            archive.writestr("b.txt", "中文 beta\n")
            archive.writestr("../escape.md", "nope")
            archive.writestr("__MACOSX/._a.md", "meta")
            archive.writestr(".hidden", "nope")
            archive.writestr("bin.dat", b"\x00\x01\x02")
            archive.writestr("big.log", "x" * 1025)
            archive.writestr("folder/", "")
        with mock.patch.object(file_knowledge, "MAX_UPLOAD_BYTES", 1024):
            imported, skipped = self.index.import_archive(
                buffer.getvalue(), directory="pack/"
            )
        self.assertEqual(
            [row["relative_path"] for row in imported], ["pack/a.md", "pack/b.txt"]
        )
        self.assertEqual(
            sorted(item["path"] for item in skipped), ["pack/big.log", "pack/bin.dat"]
        )
        self.assertEqual(
            [row["relative_path"] for row in self.index.metadata()],
            ["pack/a.md", "pack/b.txt"],
        )
        with self.assertRaises(FileKnowledgeError):
            self.index.import_archive(b"not a zip")

    def test_grep_wildcard_alternation_literals_and_numbered_context(self):
        """Model-facing search follows grep habits: ``*`` spans text within a
        line, ``|`` is OR, and regex metacharacters stay literal."""
        (self.root / "docs").mkdir()
        (self.root / "docs/deploy.md").write_text(
            "intro\nrestart the qwen service\nport (8800) via nginx\nend\n",
            encoding="utf-8",
        )
        (self.root / "notes.txt").write_text("NGINX reload\n", encoding="utf-8")

        wildcard = self.index.grep("restart*service", context=1)
        self.assertEqual(
            [(hit["path"], hit["line"]) for hit in wildcard["hits"]],
            [("docs/deploy.md", 2)],
        )
        self.assertEqual(
            wildcard["hits"][0]["context"],
            [{"line": 1, "text": "intro"}, {"line": 3, "text": "port (8800) via nginx"}],
        )
        either = self.index.grep("(8800)|reload")
        self.assertEqual(
            [(hit["path"], hit["line"]) for hit in either["hits"]],
            [("docs/deploy.md", 3), ("notes.txt", 1)],
        )
        self.assertEqual(either["file_count"], 2)
        self.assertEqual(self.index.grep("nginx", path="docs/*.md")["count"], 1)
        self.assertEqual(self.index.grep("nginx", path="*.txt")["count"], 1)
        self.assertEqual(self.index.grep("nginx", path="docs")["count"], 1)
        self.assertEqual(self.index.grep(".*")["count"], 0)
        limited = self.index.grep("e", max_results=2)
        self.assertEqual((limited["count"], limited["truncated"]), (2, True))
        with self.assertRaises(ValueError):
            self.index.grep("**")

    def test_read_lines_offset_limit_boundaries(self):
        (self.root / "a.md").write_text(
            "".join(f"line {n}\n" for n in range(1, 6)), encoding="utf-8"
        )
        page = self.index.read_lines("a.md", offset=2, limit=2)
        self.assertEqual(
            [(item["line"], item["text"]) for item in page["lines"]],
            [(2, "line 2"), (3, "line 3")],
        )
        self.assertEqual((page["end_line"], page["total_lines"], page["eof"]), (3, 5, False))
        tail = self.index.read_lines("a.md", offset=5, limit=200)
        self.assertEqual((tail["end_line"], tail["eof"]), (5, True))
        with self.assertRaises(ValueError):
            self.index.read_lines("a.md", offset=6)
        with self.assertRaises(ValueError):
            self.index.read_lines("a.md", offset=0)
        with self.assertRaises(ReadLimitError):
            self.index.read_lines("a.md", limit=2001)

    def test_unchanged_stat_skips_rehash_but_reads_still_verify(self):
        """refresh() trusts an unchanged (mtime, size); the digest check on
        read still catches a same-size in-place rewrite."""
        path = self.root / "a.md"
        path.write_text("one", encoding="utf-8")
        self.index.refresh()
        stat = path.stat()
        path.write_text("two", encoding="utf-8")
        os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
        self.assertEqual(self.index.refresh()["reused"], 1)
        with self.assertRaises(StaleFileError):
            self.index.read_lines("a.md")
        # Search skips the stale file instead of failing the whole query.
        self.assertEqual(self.index.grep("two")["count"], 0)


if __name__ == "__main__":
    unittest.main()
