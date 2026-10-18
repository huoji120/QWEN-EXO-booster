import asyncio
from types import SimpleNamespace

from qwen_exo_booster.file_knowledge import FileKnowledgeIndex
from qwen_exo_booster.memory_summaries import HEAD_BYTES, MemorySummaryService, head_text


def test_head_text_cuts_at_the_byte_budget_on_a_character_boundary():
    assert head_text("short") == ("short", False)
    head, cut = head_text("汉" * HEAD_BYTES)
    assert cut and len(head.encode("utf-8")) <= HEAD_BYTES
    assert head == "汉" * (HEAD_BYTES // 3)


class _Tokenizer:
    def apply_chat_template(self, messages, **_kwargs):
        return messages[-1]["content"]


class _Runner:
    def __init__(self):
        self.prompts = []

    async def run_batch(self, jobs, prompts, _sampling):
        self.prompts.extend(prompts)
        return tuple(SimpleNamespace(text="标题：生成的总结\n要点") for _ in prompts)

    async def finish_parent(self, _parent_request_id):
        return None


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


def test_summaries_read_only_the_head_and_never_overwrite_manual_edits(tmp_path):
    index, runner, service = _service(tmp_path)
    (index.root / "small.md").write_text("tiny\n", encoding="utf-8")
    (index.root / "notes.md").write_text("note line\n" * 600, encoding="utf-8")
    (index.root / "huge.log").write_text("x" * (HEAD_BYTES * 50) + "TAIL_MARKER\n", encoding="utf-8")
    index.refresh()
    assert sorted(service.missing_paths()) == ["huge.log", "notes.md"]

    stored = asyncio.run(service.summarize("notes.md"))
    assert (stored["title"], stored["origin"], stored["sampled"]) == ("生成的总结", "auto", 0)
    huge = asyncio.run(service.summarize("huge.log"))
    assert huge["sampled"] == 1 and "只根据文件开头" in huge["summary"]
    assert len(runner.prompts) == 2 and "TAIL_MARKER" not in runner.prompts[1]
    assert len(runner.prompts[1]) < HEAD_BYTES * 2
    assert (index.root / "notes.md").read_text(encoding="utf-8") == "note line\n" * 600

    (index.root / "notes.md").write_text("changed line\n" * 600, encoding="utf-8")
    index.refresh()
    assert service.missing_paths() == ["notes.md"]  # stale automatic summary
    index.store_summary("notes.md", sha256="old", summary="标题：手写", sampled=False, origin="manual")
    assert service.missing_paths() == []
    assert asyncio.run(service.summarize("notes.md"))["title"] == "手写"
    assert asyncio.run(service.summarize("notes.md", force=True))["origin"] == "auto"
