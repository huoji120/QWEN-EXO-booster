"""Daily work memory: the model's write tool, secret refusal, trash, and dream."""

import asyncio
import json
from types import SimpleNamespace

import pytest
from qwen_exo_booster.contracts import InternalJobType
from qwen_exo_booster.file_knowledge import FileKnowledgeError, FileKnowledgeIndex
from qwen_exo_booster.memory_dream import MemoryDreamService, parse_action, seconds_until_midnight
from qwen_exo_booster.memory_tools import MEMORY_WRITE_TOOL, MemoryToolExecutor
from qwen_exo_booster.task_scheduling import current_internal_work_class
from qwen_exo_booster.work_memory import (
    WorkMemoryTrash,
    find_secrets,
    list_notes,
    parse_note,
    work_path,
    write_note,
)


@pytest.fixture
def index(tmp_path):
    return FileKnowledgeIndex(tmp_path / "kb", tmp_path / "state")


@pytest.mark.parametrize(
    "text",
    [
        "key: sk-" + "a1B2" * 8,
        "-----BEGIN OPENSSH PRIVATE KEY-----\nabc",
        "aws AKIA" + "ABCDEFGHIJKLMNOP",
        "token = ghp_" + "x" * 36,
        "password: hunter2hunter2",
        "db: postgres://admin:s3cretpw@db.internal:5432/app",
        "Authorization: Bearer " + "abc.def-ghi" * 4,
    ],
)
def test_secrets_are_detected(text):
    assert find_secrets(text)


@pytest.mark.parametrize(
    "text",
    [
        "password: <REDACTED>",
        "api_key = ${API_KEY}",
        "token: xxxxxxxxxx",
        "the deploy key lives in the team vault under ci/deploy",
        "use flag{abc} format; sk- prefix keys are OpenAI style",
    ],
)
def test_placeholders_and_prose_are_allowed(text):
    assert find_secrets(text) == []


def test_write_tool_creates_appends_and_refuses_secrets(index):
    tools = MemoryToolExecutor(index)
    ok, text = tools.run_sync(
        MEMORY_WRITE_TOOL,
        {"path": "ctf/web/jdwp", "title": "JDWP attach", "content": "jdb -attach works after port-forward"},
    )
    assert ok and "work-memory/ctf/web/jdwp.md" in text
    ok, text = tools.run_sync(
        MEMORY_WRITE_TOOL,
        {"path": "ctf/web/jdwp.md", "title": "again", "content": "duplicate"},
    )
    assert not ok and "already exists" in text
    ok, _ = tools.run_sync(
        MEMORY_WRITE_TOOL,
        {"path": "ctf/web/jdwp.md", "title": "JDWP", "content": "also check the thread suspend flag", "mode": "append"},
    )
    assert ok
    fields, body = parse_note((index.root / "work-memory/ctf/web/jdwp.md").read_text(encoding="utf-8"))
    assert fields["title"] == "JDWP attach" and fields["source"] == "model"
    assert "port-forward" in body and "thread suspend" in body
    assert fields["created"] <= fields["updated"]
    ok, text = tools.run_sync(
        MEMORY_WRITE_TOOL,
        {"path": "secrets.md", "title": "creds", "content": "password=CorrectHorse99"},
    )
    assert not ok and "secrets" in text and not (index.root / "work-memory/secrets.md").exists()
    hits = index.grep("thread suspend")
    assert hits["files"][0]["kind"] == "work" and hits["files"][0]["title"] == "JDWP attach"
    with pytest.raises(FileKnowledgeError):
        work_path("../escape.md")


def test_model_corrects_its_own_notes_but_never_imported_memory(index):
    """A model that finds its note wrong can edit/replace/delete it; every old
    version stays restorable, and imported files stay read-only."""
    tools = MemoryToolExecutor(index)
    write_note(index, "ops/port.md", title="port", content="The admin API listens on 8080.\nUse curl.")
    (index.root / "imported.md").write_text("imported fact\n", encoding="utf-8")
    index.refresh()
    note = index.root / "work-memory/ops/port.md"

    ok, text = tools.run_sync(
        MEMORY_WRITE_TOOL, {"path": "ops/port.md", "mode": "edit", "old_text": "8080", "new_text": "30000"}
    )
    assert ok, text
    fields, body = parse_note(note.read_text(encoding="utf-8"))
    assert "listens on 30000" in body and "8080" not in body and fields["title"] == "port"
    ok, text = tools.run_sync(
        MEMORY_WRITE_TOOL, {"path": "ops/port.md", "mode": "edit", "old_text": "9999", "new_text": "x"}
    )
    assert not ok and "not found" in text
    ok, text = tools.run_sync(
        MEMORY_WRITE_TOOL,
        {"path": "ops/port.md", "mode": "edit", "old_text": "Use curl.", "new_text": "token=Abcdefgh12345"},
    )
    assert not ok and "secrets" in text and "30000" in note.read_text(encoding="utf-8")
    ok, _ = tools.run_sync(
        MEMORY_WRITE_TOOL,
        {
            "path": "work-memory/ops/port.md",
            "mode": "replace",
            "title": "admin port",
            "content": "# admin port\n\nPort 30000.",  # models often repeat the heading
        },
    )
    assert ok and parse_note(note.read_text(encoding="utf-8"))[1].strip() == "# admin port\n\nPort 30000."
    for path, mode in (("imported.md", "delete"), ("missing.md", "edit")):
        ok, text = tools.run_sync(MEMORY_WRITE_TOOL, {"path": path, "mode": mode, "old_text": "a", "new_text": "b"})
        assert not ok and ("read-only" in text if path == "imported.md" else "does not exist" in text)
    assert (index.root / "imported.md").exists()
    ok, _ = tools.run_sync(MEMORY_WRITE_TOOL, {"path": "ops/port.md", "mode": "delete"})
    assert ok and not note.exists()

    trash = WorkMemoryTrash(index)
    versions = sorted(entry["batch"] for entry in trash.entries())
    assert len(versions) == 3  # before edit, before replace, the deleted note
    trash.restore(versions[0], "work-memory/ops/port.md")
    assert "8080" in note.read_text(encoding="utf-8")
    # Restoring over an existing note keeps the current version in the trash.
    trash.restore(versions[1], "work-memory/ops/port.md")
    assert "listens on 30000" in note.read_text(encoding="utf-8")
    assert any(entry["batch"].startswith("replaced-") for entry in trash.entries())


def test_trash_keeps_deleted_notes_restorable(index):
    write_note(index, "a/one.md", title="one", content="first note")
    write_note(index, "a/two.md", title="two", content="second note")
    trash = WorkMemoryTrash(index)
    assert trash.discard("a/one", batch="dream-1") == "work-memory/a/one.md"
    assert trash.discard("a", batch="dream-2") == "work-memory/a"
    assert [note["path"] for note in list_notes(index)] == []
    entries = {(entry["batch"], entry["path"]) for entry in trash.entries()}
    assert entries == {("dream-1", "work-memory/a/one.md"), ("dream-2", "work-memory/a/two.md")}
    assert trash.restore("dream-1", "work-memory/a/one.md") == "work-memory/a/one.md"
    assert [note["path"] for note in list_notes(index)] == ["work-memory/a/one.md"]
    assert not (trash.root / "dream-1").exists() and (trash.root / "dream-2").is_dir()


def test_parse_action_and_schedule_helpers():
    assert parse_action('思考…\n```json\n{"action": "done", "summary": "ok"}\n```') == {
        "action": "done",
        "summary": "ok",
    }
    with pytest.raises(ValueError):
        parse_action("no json here")
    assert 0 < seconds_until_midnight() <= 86400
    assert current_internal_work_class(InternalJobType.MEMORY_DREAM) == "maintenance"
    assert current_internal_work_class(InternalJobType.MEMORY_SUMMARY) == "maintenance"


class _ScriptedRunner:
    """Mimics InternalJobRunner's cumulative per-parent token reserve."""

    def __init__(self, replies, max_tokens_per_parent=8000):
        self.replies = list(replies)
        self.prompts = []
        self.max_tokens_per_parent = max_tokens_per_parent
        self.reserved = {}

    async def run_batch(self, jobs, prompts, _sampling):
        for job in jobs:
            total = self.reserved.get(job.parent_request_id, 0) + job.token_budget
            if total > self.max_tokens_per_parent:
                raise RuntimeError("Internal jobs exceed the parent cumulative token reserve")
            self.reserved[job.parent_request_id] = total
        self.prompts.extend(prompts)
        return (SimpleNamespace(text=self.replies.pop(0)),)

    async def finish_parent(self, parent_request_id):
        self.reserved.pop(parent_request_id, None)


class _Tokenizer:
    def apply_chat_template(self, messages, **_kwargs):
        return json.dumps(messages, ensure_ascii=False)


def _dream(index, replies):
    async def quiet():
        return None

    runner = _ScriptedRunner(replies)
    service = MemoryDreamService(
        runner=runner,
        tokenizer=_Tokenizer(),
        index=index,
        telemetry=SimpleNamespace(emit=lambda *args, **kwargs: None),
        wait_until_quiet=quiet,
    )
    return service, runner


def test_dream_agent_merges_duplicates_and_logs_its_changes(index):
    """Each step's generation must release its token reserve: a fixed parent
    id once accumulated budget across steps until the runner refused the
    fifth step (prod dream run #1 died with a ContractViolation)."""
    write_note(index, "inbox/nginx-a.md", title="nginx reload", content="reload nginx after editing upstream")
    write_note(index, "inbox/nginx-b.md", title="nginx reload again", content="nginx -s reload applies upstream edits")
    (index.root / "imported.md").write_text("user notes\n", encoding="utf-8")
    replies = [
        json.dumps({"action": "list", "scope": "today"}),
        json.dumps({"action": "search", "pattern": "nginx"}),
        "not json at all",
        json.dumps(
            {
                "action": "write",
                "path": "ops/nginx/reload.md",
                "title": "nginx reload",
                "content": "Run `nginx -s reload` after editing the upstream config.",
            }
        ),
        json.dumps({"action": "delete", "path": "inbox/nginx-a.md"}),
        json.dumps({"action": "delete", "path": "inbox/nginx-b.md"}),
        json.dumps({"action": "delete", "path": "imported.md"}),
        json.dumps({"action": "delete", "path": "inbox"}),
        json.dumps({"action": "done", "summary": "合并了 nginx 笔记"}),
    ]
    service, runner = _dream(index, replies)
    result = asyncio.run(service.run("manual"))
    assert result["status"] == "completed", service.log.run(result["id"])["summary"]
    assert runner.reserved == {}
    assert [note["path"] for note in list_notes(index)] == ["work-memory/ops/nginx/reload.md"]
    assert (index.root / "imported.md").exists()
    run = service.log.run(result["id"])
    assert run["summary"] == "合并了 nginx 笔记"
    assert [(item["action"], item["ok"]) for item in run["actions"]] == [
        ("write", True),
        ("delete", True),
        ("delete", True),
        ("delete", False),
        ("delete", True),
    ]
    assert "nginx-a.md" in runner.prompts[1] and "不是合法的 JSON" in runner.prompts[3]
    assert {entry["path"] for entry in service.trash.entries()} == {
        "work-memory/inbox/nginx-a.md",
        "work-memory/inbox/nginx-b.md",
    }


def test_dream_with_no_work_memory_finishes_without_calling_the_model(index):
    service, runner = _dream(index, [])
    assert asyncio.run(service.run("nightly"))["status"] == "completed"
    assert runner.prompts == []
