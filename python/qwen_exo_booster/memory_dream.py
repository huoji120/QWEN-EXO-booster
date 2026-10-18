"""Nightly "dream": an agent that tidies the daily work memory.

At local midnight (or on demand) the served model is run as an agent over
``work-memory/``: it lists today's and older notes separately, searches and
reads them (imported memory is read-only reference), and merges duplicates,
resolves conflicts, rewrites, moves, creates or deletes notes and folders one
at a time until it says it is done. Each turn is one JSON action; deletions go
to the work-memory trash, secrets are refused like any other write, and every
run is logged with its actions.
"""

from __future__ import annotations

import asyncio
import json
import logging
import sqlite3
import time
from contextlib import closing
from typing import Any, Awaitable, Callable

from qwen_exo_booster.contracts import InternalJobType
from qwen_exo_booster.file_knowledge import FileKnowledgeError, FileKnowledgeIndex
from qwen_exo_booster.memory_summaries import generate_internal_text
from qwen_exo_booster.memory_tools import _format_read, _format_search
from qwen_exo_booster.work_memory import (
    WORK_ROOT,
    WorkMemoryTrash,
    list_notes,
    work_directory,
    work_path,
    write_note,
)

logger = logging.getLogger(__name__)

MAX_STEPS = 150
MAX_BAD_TURNS = 3
TURN_TOKENS = 3000
RESULT_CHARS = 8000
# Older turns are dropped (kept as a one-line log) beyond this much history.
HISTORY_CHARS = 120_000
_LIST_LIMIT = 200

_SYSTEM = """你是 QWEN-EXO 的记忆整理 agent（夜间 dream）。你要整理“日常工作记忆”：\
work-memory/ 下由模型在以往会话里写的 Markdown 笔记。用户导入的其他记忆只能读，不能改。

目标：让工作记忆以后更容易搜到、更可靠：
- 合并重复或高度相似的笔记（把内容合并进一篇，再删掉多余的）；
- 处理互相冲突的说法：保留证据更充分、更新的结论，并在笔记里注明曾经的不同说法；
- 删除没有复用价值的内容（流水账、纯过程记录、已经过时且被取代的结论）；
- 把笔记放进合理的文件夹，名字要能看出主题；
- 不要编造事实，不要写入任何密钥、token、密码、私钥（会被拒绝）。

一条一条整理：先看今天新写的，再结合历史笔记处理重复和冲突。每次只输出一个 JSON \
对象（不要别的文字），格式是 {"action": ..., 其它参数}。可用动作：
- {"action":"list","scope":"today|history|all","path":"可选，文件夹"}  列出笔记（路径、标题、大小、修改时间）
- {"action":"tree"}  列出工作记忆的文件夹结构
- {"action":"search","pattern":"关键词，支持 * | -排除","scope":"work|all","mode":"content|files"}
- {"action":"read","path":"...","offset":1,"limit":200}
- {"action":"write","path":"work-memory 下的路径","title":"标题","content":"Markdown 正文"}  新建或整篇改写
- {"action":"move","from":"...","to":"..."}  移动或重命名笔记/文件夹
- {"action":"delete","path":"..."}  删除笔记或文件夹（进回收站）
- {"action":"mkdir","path":"..."}
- {"action":"done","summary":"这次做了什么"}  认为不再需要整理时结束
路径都在 work-memory/ 之下，可以省略前缀。"""


class DreamRunLog:
    def __init__(self, path):
        self.path = path
        with closing(sqlite3.connect(self.path)) as connection, connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS dream_runs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    trigger TEXT NOT NULL,
                    started_ns INTEGER NOT NULL,
                    finished_ns INTEGER,
                    status TEXT NOT NULL,
                    steps INTEGER NOT NULL DEFAULT 0,
                    summary TEXT NOT NULL DEFAULT '',
                    actions TEXT NOT NULL DEFAULT '[]'
                )
                """
            )

    def start(self, trigger: str) -> int:
        with closing(sqlite3.connect(self.path)) as connection, connection:
            return int(
                connection.execute(
                    "INSERT INTO dream_runs(trigger, started_ns, status) VALUES (?, ?, 'running')",
                    (trigger, time.time_ns()),
                ).lastrowid
            )

    def finish(self, run_id: int, *, status: str, steps: int, summary: str, actions: list[dict[str, Any]]) -> None:
        with closing(sqlite3.connect(self.path)) as connection, connection:
            connection.execute(
                "UPDATE dream_runs SET finished_ns=?, status=?, steps=?, summary=?, actions=? WHERE id=?",
                (time.time_ns(), status, steps, summary[:4000], json.dumps(actions, ensure_ascii=False), run_id),
            )

    def runs(self, limit: int = 20) -> list[dict[str, Any]]:
        with closing(sqlite3.connect(self.path)) as connection:
            connection.row_factory = sqlite3.Row
            rows = connection.execute(
                "SELECT id, trigger, started_ns, finished_ns, status, steps, summary, actions "
                "FROM dream_runs ORDER BY id DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [
            {**{key: row[key] for key in row.keys() if key != "actions"}, "action_count": len(json.loads(row["actions"]))}
            for row in rows
        ]

    def run(self, run_id: int) -> dict[str, Any] | None:
        with closing(sqlite3.connect(self.path)) as connection:
            connection.row_factory = sqlite3.Row
            row = connection.execute("SELECT * FROM dream_runs WHERE id = ?", (run_id,)).fetchone()
        if row is None:
            return None
        return {**{key: row[key] for key in row.keys()}, "actions": json.loads(row["actions"])}

    def last_completed_start_ns(self) -> int | None:
        with closing(sqlite3.connect(self.path)) as connection:
            row = connection.execute(
                "SELECT max(started_ns) FROM dream_runs WHERE status = 'completed'"
            ).fetchone()
        return int(row[0]) if row and row[0] is not None else None


def parse_action(text: str) -> dict[str, Any]:
    """The first JSON object in a model turn (code fences tolerated)."""
    start = text.find("{")
    if start < 0:
        raise ValueError("no JSON object")
    action, _end = json.JSONDecoder().raw_decode(text[start:])
    if not isinstance(action, dict) or not isinstance(action.get("action"), str):
        raise ValueError("object has no action")
    return action


def seconds_until_midnight(now: float | None = None) -> float:
    now = time.time() if now is None else now
    local = time.localtime(now)
    midnight = time.mktime((local.tm_year, local.tm_mon, local.tm_mday + 1, 0, 0, 0, 0, 0, -1))
    return max(1.0, midnight - now)


class MemoryDreamService:
    def __init__(
        self,
        *,
        runner: Any,
        tokenizer: Any,
        index: FileKnowledgeIndex,
        telemetry: Any,
        wait_until_quiet: Callable[[], Awaitable[None]],
    ):
        self.runner = runner
        self.tokenizer = tokenizer
        self.index = index
        self.telemetry = telemetry
        self.wait_until_quiet = wait_until_quiet
        self.trash = WorkMemoryTrash(index)
        self.log = DreamRunLog(index.state_directory / "memory-dream.sqlite3")
        self._task: asyncio.Task[None] | None = None
        self._current: dict[str, Any] | None = None
        # Notes modified at or after this are "today" for the running pass.
        self._since_ns: int | None = None

    # --- control ---------------------------------------------------------------------

    def status(self) -> dict[str, Any]:
        return {
            "running": self._task is not None and not self._task.done(),
            "current": dict(self._current) if self._current else None,
            "next_run_in_seconds": int(seconds_until_midnight()),
            "runs": self.log.runs(),
        }

    def request_run(self, trigger: str = "manual") -> dict[str, Any]:
        if self._task is not None and not self._task.done():
            raise RuntimeError("a dream run is already in progress")
        self._task = asyncio.create_task(self.run(trigger), name="qwen-exo-memory-dream")
        return self.status()

    async def schedule_forever(self) -> None:
        while True:
            await asyncio.sleep(seconds_until_midnight())
            try:
                self.request_run("nightly")
                await self._task
            except RuntimeError:
                continue
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("nightly memory dream failed")

    def cancel(self) -> None:
        if self._task is not None and not self._task.done():
            self._task.cancel()

    # --- agent loop ------------------------------------------------------------------

    async def run(self, trigger: str) -> dict[str, Any]:
        self.index.refresh()
        notes = list_notes(self.index)
        run_id = self.log.start(trigger)
        actions: list[dict[str, Any]] = []
        if not notes:
            self.log.finish(run_id, status="completed", steps=0, summary="没有工作记忆，无需整理", actions=[])
            return {"id": run_id, "status": "completed"}
        since = self.log.last_completed_start_ns() or int((time.time() - 86400) * 1e9)
        self._since_ns = since
        today = [note for note in notes if note["mtime_ns"] >= since]
        messages = [
            {"role": "system", "content": _SYSTEM},
            {
                "role": "user",
                "content": (
                    f"开始整理。工作记忆共 {len(notes)} 篇，其中今天（上次整理之后）新增或修改的 "
                    f"{len(today)} 篇，历史 {len(notes) - len(today)} 篇。\n今天的笔记：\n"
                    + "\n".join(f"- {note['path']}  {note['title']}" for note in today[:_LIST_LIMIT])
                    + "\n先从今天的笔记开始，一条一条整理。只输出一个 JSON 动作。"
                ),
            },
        ]
        summary, status, bad, step = "", "stopped", 0, 0
        batch = f"dream-{run_id}"
        self._current = {"run_id": run_id, "step": 0, "last_action": None}
        try:
            for step in range(1, MAX_STEPS + 1):
                await self.wait_until_quiet()
                self._current["step"] = step
                reply = await generate_internal_text(
                    self.runner,
                    job_type=InternalJobType.MEMORY_DREAM,
                    label="memory-dream",
                    prompt=self._render(messages, actions),
                    token_budget=TURN_TOKENS,
                )
                messages.append({"role": "assistant", "content": reply})
                try:
                    action = parse_action(reply)
                except ValueError as exc:
                    bad += 1
                    if bad >= MAX_BAD_TURNS:
                        summary = "连续多次没有输出合法的 JSON 动作，提前结束"
                        break
                    messages.append({"role": "user", "content": f"上一条不是合法的 JSON 动作（{exc}）。只输出一个 JSON 对象。"})
                    continue
                bad = 0
                if action["action"] == "done":
                    summary, status = str(action.get("summary") or ""), "completed"
                    break
                ok, result = await asyncio.to_thread(self._execute, action, batch)
                record = {
                    "step": step,
                    "action": action["action"],
                    "target": str(action.get("path") or action.get("from") or action.get("pattern") or ""),
                    "to": str(action.get("to") or ""),
                    "ok": ok,
                    "message": result[:200] if not ok else "",
                }
                if action["action"] not in {"list", "tree", "search", "read"}:
                    actions.append(record)
                self._current["last_action"] = record
                messages.append({"role": "user", "content": ("结果：" if ok else "失败：") + result[:RESULT_CHARS]})
            else:
                summary = f"达到 {MAX_STEPS} 步上限，下次继续"
        except asyncio.CancelledError:
            self.log.finish(run_id, status="cancelled", steps=step, summary="被取消", actions=actions)
            raise
        except Exception as exc:
            logger.exception("memory dream run failed")
            self.log.finish(run_id, status="failed", steps=step, summary=f"{type(exc).__name__}: {exc}"[:400], actions=actions)
            return {"id": run_id, "status": "failed"}
        finally:
            self._current = None
        self.log.finish(run_id, status=status, steps=step, summary=summary, actions=actions)
        self.telemetry.emit("runtime", "memory_dream.finished", {"status": status, "steps": step, "changes": len(actions)})
        return {"id": run_id, "status": status}

    def _render(self, messages: list[dict[str, str]], actions: list[dict[str, Any]]) -> str:
        """Chat prompt, dropping the oldest turns (kept as an action log) once
        the history grows past ``HISTORY_CHARS``."""
        head, tail = messages[:2], messages[2:]
        while tail and sum(len(message["content"]) for message in tail) > HISTORY_CHARS:
            tail = tail[2:]
        if len(tail) < len(messages) - 2 and actions:
            done = "\n".join(
                f"- {item['action']} {item['target']}{' -> ' + item['to'] if item['to'] else ''}"
                for item in actions[-60:]
            )
            head = [head[0], {"role": "user", "content": head[1]["content"] + "\n\n（更早的步骤已省略）已经完成的改动：\n" + done}]
        while tail and tail[0]["role"] != "assistant":
            tail = tail[1:]
        return self.tokenizer.apply_chat_template(
            head + tail, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )

    def _readable(self, path: str) -> str:
        """A work note given without the ``work-memory/`` prefix, or any
        other (read-only) memory path as given."""
        path = path.strip().lstrip("/")
        if path.startswith(WORK_ROOT + "/"):
            return path
        candidate = work_path(path)
        return candidate if self.index.path_of(candidate).exists() else path

    def _execute(self, action: dict[str, Any], batch: str) -> tuple[bool, str]:
        name = action["action"]
        try:
            if name == "list":
                scope = str(action.get("scope") or "all")
                since = self._since_ns
                notes = list_notes(
                    self.index,
                    since_ns=since if scope == "today" else None,
                    until_ns=since if scope == "history" else None,
                )
                folder = work_directory(action["path"]) + "/" if action.get("path") else ""
                notes = [note for note in notes if note["path"].startswith(folder)]
                lines = [f"{note['path']}  [{note['modified']}, {note['bytes']}B]  {note['title']}" for note in notes[:_LIST_LIMIT]]
                more = f"\n… 还有 {len(notes) - _LIST_LIMIT} 篇" if len(notes) > _LIST_LIMIT else ""
                return True, (f"{len(notes)} 篇：\n" + "\n".join(lines) + more) if notes else "没有笔记"
            if name == "tree":
                folders = [path for path in self.index.directories() if path == WORK_ROOT or path.startswith(WORK_ROOT + "/")]
                counts = {folder: 0 for folder in folders}
                for note in list_notes(self.index):
                    parent = note["path"].rsplit("/", 1)[0]
                    counts[parent] = counts.get(parent, 0) + 1
                return True, "\n".join(f"{folder}/  ({counts.get(folder, 0)} 篇)" for folder in sorted(counts)) or "空"
            if name == "search":
                result = self.index.grep(
                    str(action.get("pattern") or ""),
                    path=None if action.get("scope") == "all" else WORK_ROOT,
                    mode="files" if action.get("mode") == "files" else "content",
                    per_file=5,
                    max_results=80,
                )
                return True, _format_search(str(action.get("pattern")), result)
            if name == "read":
                return True, _format_read(
                    self.index.read_lines(
                        self._readable(str(action.get("path") or "")),
                        offset=int(action.get("offset") or 1),
                        limit=int(action.get("limit") or 200),
                    )
                )
            if name == "write":
                note = work_path(str(action.get("path") or ""))
                if self.index.path_of(note).is_file():
                    self.trash.keep_copy(note, batch=batch)
                result = write_note(
                    self.index,
                    str(action.get("path") or ""),
                    title=str(action.get("title") or ""),
                    content=str(action.get("content") or ""),
                    mode="replace",
                    source="dream",
                )
                return True, f"已写入 {result['path']}（{result['bytes']} 字节）"
            if name == "move":
                source, destination = str(action.get("from") or ""), str(action.get("to") or "")
                if self.index.path_of(work_directory(source)).is_dir():
                    source, destination = work_directory(source), work_directory(destination)
                else:
                    source, destination = work_path(source), work_path(destination)
                moved = self.index.move(source, destination)
                return True, f"已移动 {moved['from']} -> {moved['to']}"
            if name == "delete":
                return True, f"已删除（进回收站）{self.trash.discard(str(action.get('path') or ''), batch=batch)}"
            if name == "mkdir":
                return True, f"已创建 {self.index.make_directory(work_directory(str(action.get('path') or '')))}"
            return False, f"未知动作 {name}"
        except FileNotFoundError as exc:
            return False, f"不存在：{exc}"
        except (FileKnowledgeError, ValueError, OSError, TypeError) as exc:
            return False, str(exc)


__all__ = ["MemoryDreamService", "parse_action", "seconds_until_midnight"]
