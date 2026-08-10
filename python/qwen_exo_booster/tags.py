from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Any

_MAX_TAGS = 16
_MAX_TAG_LENGTH = 32
_CONTROL_CHARACTERS = re.compile(r"[\x00-\x1f\x7f]")


class TagValidationError(ValueError):
    pass


def normalize_tags(value: Any, *, required: bool = False) -> tuple[str, ...]:
    if value is None:
        values: Iterable[Any] = ()
    elif isinstance(value, str):
        text = value.strip()
        if text.startswith("[") and text.endswith("]"):
            text = text[1:-1]
        values = text.split(",") if text else ()
    elif isinstance(value, (list, tuple, set)):
        values = value
    else:
        raise TagValidationError("标签必须是字符串数组")

    normalized: list[str] = []
    seen: set[str] = set()
    for raw_tag in values:
        if not isinstance(raw_tag, str):
            raise TagValidationError("每个标签都必须是字符串")
        tag = raw_tag.strip().strip("\"'")
        if not tag:
            continue
        if _CONTROL_CHARACTERS.search(tag):
            raise TagValidationError("标签不能包含控制字符")
        if len(tag) > _MAX_TAG_LENGTH:
            raise TagValidationError(f"单个标签不能超过 {_MAX_TAG_LENGTH} 个字符")
        identity = tag.casefold()
        if identity in seen:
            continue
        seen.add(identity)
        normalized.append(tag)
        if len(normalized) > _MAX_TAGS:
            raise TagValidationError(f"标签数量不能超过 {_MAX_TAGS} 个")

    if required and not normalized:
        raise TagValidationError("至少填写一个标签")
    return tuple(normalized)
