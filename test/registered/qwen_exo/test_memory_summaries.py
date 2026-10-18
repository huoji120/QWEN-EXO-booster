import asyncio
from types import SimpleNamespace

from qwen_exo_booster import memory_summaries
from qwen_exo_booster.file_knowledge import FileKnowledgeIndex
from qwen_exo_booster.memory_summaries import MemorySummaryService, plan_chunks


def test_chunk_plan_reads_small_files_whole_and_samples_huge_ones():
    assert plan_chunks(10) == ([(0, 10)], False)
    whole, sampled = plan_chunks(memory_summaries.CHUNK_CHARS * 3)
    assert len(whole) == 3 and not sampled and whole[-1][1] == memory_summaries.CHUNK_CHARS * 3
    length = memory_summaries.CHUNK_CHARS * 100
    chunks, sampled = plan_chunks(length)
    assert sampled and len(chunks) == memory_summaries.MAX_CHUNKS
    assert chunks[0][0] == 0 and chunks[-1][1] == length
    assert all(left[0] < right[0] for left, right in zip(chunks, chunks[1:]))


class _Tokenizer:
    def apply_chat_template(self, messages, **_kwargs):
        return messages[-1]["content"]


class _Runner:
    def __init__(self):
        self.prompts = []

    async def run_batch(self, jobs, prompts, _sampling):
        self.prompts.extend(prompts)
        return tuple(SimpleNamespace(text="标题：生成的总结\n要点") for _ in prompts)


def _service(tmp_path):
    index = FileKnowledgeIndex(tmp_path / "kb", tmp_path / "state")
    runner = _Runner()

    async def quiet():
        return None

    service = MemorySummaryService(
        runner=runner,
        tokenizer=_Tokenizer(),
        index=index,
        telemetry=SimpleNamespace(emit=lambda *args, **kwargs: None),
        wait_until_quiet=quiet,
    )
    return index, runner, service


def test_summaries_are_stored_in_the_index_and_never_overwrite_manual_edits(tmp_path, monkeypatch):
    index, runner, service = _service(tmp_path)
    (index.root / "small.md").write_text("tiny\n", encoding="utf-8")
    (index.root / "notes.md").write_text("note line\n" * 600, encoding="utf-8")
    index.refresh()
    assert service.missing_paths() == ["notes.md"]

    stored = asyncio.run(service.summarize("notes.md"))
    assert (stored["title"], stored["origin"], stored["sampled"]) == ("生成的总结", "auto", 0)
    assert len(runner.prompts) == 1 and service.missing_paths() == []
    assert (index.root / "notes.md").read_text(encoding="utf-8") == "note line\n" * 600

    (index.root / "notes.md").write_text("changed line\n" * 600, encoding="utf-8")
    index.refresh()
    assert service.missing_paths() == ["notes.md"]  # stale automatic summary
    sha256, _text = index.read_text("notes.md")
    index.store_summary("notes.md", sha256="old", summary="标题：手写", sampled=False, origin="manual")
    assert service.missing_paths() == []
    assert asyncio.run(service.summarize("notes.md"))["title"] == "手写"
    assert asyncio.run(service.summarize("notes.md", force=True))["origin"] == "auto"

    monkeypatch.setattr(memory_summaries, "CHUNK_CHARS", 1000)
    monkeypatch.setattr(memory_summaries, "MAX_CHUNKS", 3)
    runner.prompts.clear()
    sampled = asyncio.run(service.summarize("notes.md", force=True))
    assert len(runner.prompts) == 4 and sampled["sampled"] == 1
    assert "均匀采样" in sampled["summary"]
