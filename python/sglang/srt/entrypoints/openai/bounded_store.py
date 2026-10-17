"""Bounded in-memory stores for stored Responses (``store=true``).

Gateways that chain turns with ``previous_response_id`` force ``store=true`` on
every request, so an unbounded dict grows without limit. Entries are evicted
least-recently-used first once a count or byte budget is exceeded, and expire
after an idle TTL. Eviction is safe: a chained turn whose parent is gone gets
404 ``previous_response_not_found`` and the client resends full history.
"""

from __future__ import annotations

import time
from collections import OrderedDict
from collections.abc import Callable, Iterator, MutableMapping
from typing import Generic, TypeVar

V = TypeVar("V")


class BoundedStore(MutableMapping[str, V], Generic[V]):
    def __init__(
        self,
        *,
        max_entries: int,
        idle_ttl_seconds: float,
        max_bytes: int | None = None,
        sizer: Callable[[V], int] | None = None,
        clock: Callable[[], float] = time.monotonic,
    ):
        self._entries: OrderedDict[str, tuple[V, float, int]] = OrderedDict()
        self._max_entries = max_entries
        self._idle_ttl = idle_ttl_seconds
        self._max_bytes = max_bytes
        self._sizer = sizer
        self._clock = clock
        self._bytes = 0

    def _expired(self, touched: float, now: float) -> bool:
        return now - touched > self._idle_ttl

    def __getitem__(self, key: str) -> V:
        value, touched, size = self._entries[key]
        now = self._clock()
        if self._expired(touched, now):
            del self[key]
            raise KeyError(key)
        self._entries[key] = (value, now, size)
        self._entries.move_to_end(key)
        return value

    def __setitem__(self, key: str, value: V) -> None:
        if key in self._entries:
            del self[key]
        size = self._sizer(value) if self._sizer is not None else 0
        self._entries[key] = (value, self._clock(), size)
        self._bytes += size
        self._evict()

    def __delitem__(self, key: str) -> None:
        _value, _touched, size = self._entries.pop(key)
        self._bytes -= size

    def __contains__(self, key: object) -> bool:
        try:
            self[key]  # type: ignore[index]
        except KeyError:
            return False
        return True

    def __iter__(self) -> Iterator[str]:
        return iter(list(self._entries))

    def __len__(self) -> int:
        return len(self._entries)

    @property
    def total_bytes(self) -> int:
        return self._bytes

    def _evict(self) -> None:
        now = self._clock()
        while self._entries:
            key, (_value, touched, _size) = next(iter(self._entries.items()))
            over_budget = len(self._entries) > self._max_entries or (
                self._max_bytes is not None
                and self._bytes > self._max_bytes
                and len(self._entries) > 1
            )
            if not over_budget and not self._expired(touched, now):
                break
            del self[key]
