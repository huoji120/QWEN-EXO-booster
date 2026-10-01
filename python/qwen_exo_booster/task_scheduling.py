"""Event-loop-local work classification inherited by spawned internal tasks."""

from __future__ import annotations

from contextlib import AbstractContextManager, AsyncContextDecorator
from contextvars import ContextVar

_WORK_CLASSES = frozenset({"foreground", "reflection", "maintenance"})
_internal_work_class: ContextVar[str | None] = ContextVar(
    "qwen_exo_internal_work_class", default=None
)


class internal_task_scope(AbstractContextManager, AsyncContextDecorator):
    """Scope an async operation and its children without changing its caller.

    Background descendants cannot promote themselves into the foreground lane.
    Explicit background-to-background changes support detached maintenance, such
    as a GDN refresh spawned when a reflection is published.
    """

    def __init__(self, work_class: str):
        if work_class not in _WORK_CLASSES:
            raise ValueError(f"Unknown internal work class: {work_class}")
        self.work_class = work_class

    def _recreate_cm(self):
        return type(self)(self.work_class)

    def __enter__(self):
        inherited = _internal_work_class.get()
        work_class = self.work_class
        if inherited in {"reflection", "maintenance"} and work_class == "foreground":
            work_class = inherited
        self._token = _internal_work_class.set(work_class)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        _internal_work_class.reset(self._token)
        return False

    async def __aenter__(self):
        return self.__enter__()

    async def __aexit__(self, exc_type, exc_value, traceback):
        return self.__exit__(exc_type, exc_value, traceback)


def current_internal_work_class(job_type) -> str:
    inherited = _internal_work_class.get()
    if inherited is not None:
        return inherited
    if getattr(job_type, "value", job_type) == "reflection_memory":
        return "reflection"
    return "foreground"
