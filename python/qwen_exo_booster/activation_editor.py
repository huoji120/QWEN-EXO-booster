from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any, Sequence

import torch

_EDITOR_NAME = re.compile(r"^[a-z0-9][a-z0-9._-]{0,95}$")
_EDITOR_SCHEMA = 1


class ActivationEditorError(RuntimeError):
    pass


def _normalize_editor_strength(raw: object) -> float | None:
    if raw is None:
        return None
    try:
        strength = float(raw)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(strength) or not 0 < strength <= 4.0:
        return None
    return strength


def parse_activation_editor_spec(raw: object) -> dict[str, object] | None:
    if isinstance(raw, str):
        raw = {"editor": raw}
    if not isinstance(raw, dict):
        return None
    mode = str(raw.get("mode") or "active").strip().lower()
    if mode != "active":
        return None
    name = str(raw.get("editor") or raw.get("name") or "").strip().lower()
    if not _EDITOR_NAME.fullmatch(name):
        return None
    spec: dict[str, object] = {"mode": "active", "editor": name}
    if "strength" in raw:
        strength = _normalize_editor_strength(raw.get("strength"))
        if strength is None:
            return None
        spec["strength"] = strength
    if "tail_offset" in raw:
        try:
            tail_offset = int(raw.get("tail_offset"))
        except (TypeError, ValueError):
            return None
        if not 0 <= tail_offset <= 100000:
            return None
        spec["tail_offset"] = tail_offset
    return spec


def resolve_default_activation_editor_spec(
    active_spec: dict[str, object] | None,
    env_name: str | None,
    *,
    enabled: bool,
    fallback_strength: float,
) -> dict[str, object] | None:
    active_name = str((active_spec or {}).get("editor") or "").strip().lower()
    configured_name = str(env_name or "").strip().lower()
    name = active_name or configured_name
    if not _EDITOR_NAME.fullmatch(name):
        return None
    if not active_name and not enabled and not configured_name:
        return None
    return {
        "mode": "active",
        "editor": name,
        "strength": _normalize_editor_strength(fallback_strength) or 1.0,
    }


class ActivationEditor:
    """Low-rank hidden-state editor: h' = h + R^T (W h + b - R h)."""

    def __init__(self, payload: dict[str, Any], device: torch.device):
        if int(payload.get("schema") or 0) != _EDITOR_SCHEMA:
            raise ActivationEditorError("activation editor schema is unsupported")
        self.layer = int(payload["layer"])
        self.window = int(payload["window"])
        self.rank = int(payload["rank"])
        self.hidden_size = int(payload["hidden_size"])
        state = payload["state_dict"]
        self.projection = state["projection"].to(device=device, dtype=torch.float32)
        self.transform = state["transform"].to(device=device, dtype=torch.float32)
        self.bias = state["bias"].to(device=device, dtype=torch.float32)

    def apply(self, hidden: torch.Tensor, strength: float = 1.0) -> torch.Tensor:
        source = hidden.float()
        base = source @ self.projection.T
        target = source @ self.transform.T + self.bias
        delta = (target - base) @ self.projection
        return (source + delta * float(strength)).to(hidden.dtype)


class ActivationEditorStore:
    def __init__(self, root: Path | str):
        self.root = Path(root)
        self._cache: dict[tuple[str, float, str], ActivationEditor | None] = {}

    def editor(self, name: str, device: torch.device) -> ActivationEditor | None:
        if not _EDITOR_NAME.fullmatch(name):
            return None
        path = self.root / f"{name}.editor.pt"
        try:
            modified = path.stat().st_mtime
        except OSError:
            return None
        key = (name, modified, str(device))
        if key not in self._cache:
            try:
                payload = torch.load(path, map_location="cpu", weights_only=True)
                self._cache[key] = ActivationEditor(payload, device)
            except Exception:
                self._cache[key] = None
        return self._cache[key]


def apply_activation_editor(
    store: ActivationEditorStore,
    hidden_states: torch.Tensor,
    specs: Sequence[dict[str, object] | None],
    extend_seq_lens: Sequence[int],
    final_prefill: Sequence[bool] | None,
    *,
    layer_index: int,
) -> torch.Tensor | None:
    batch_size = len(specs)
    if not specs or not any(spec and spec.get("mode") == "active" for spec in specs):
        return None
    final_flags = tuple(final_prefill or (True,) * batch_size)
    edited: torch.Tensor | None = None
    offset = 0
    for request_index, (spec, length) in enumerate(zip(specs, extend_seq_lens)):
        length = int(length)
        end = offset + length
        if (
            spec
            and spec.get("mode") == "active"
            and request_index < len(final_flags)
            and final_flags[request_index]
            and length > 0
        ):
            editor = store.editor(str(spec.get("editor") or ""), hidden_states.device)
            if editor is not None and editor.layer == layer_index:
                try:
                    tail_offset = int(spec.get("tail_offset") or 0)
                    strength = float(spec.get("strength") or 1.0)
                except (TypeError, ValueError):
                    tail_offset = 0
                    strength = 1.0
                window_end = min(end, max(offset, end - tail_offset))
                start = max(offset, window_end - editor.window)
                if edited is None:
                    edited = hidden_states.clone()
                edited[start:window_end] = editor.apply(
                    hidden_states[start:window_end], strength
                )
        offset = end
    return edited
