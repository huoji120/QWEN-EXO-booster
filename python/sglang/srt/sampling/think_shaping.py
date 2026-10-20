"""Per-request logit shaping while a request is still thinking.

Two knobs, both carried in ``custom_params[QWEN_EXO_THINK_SHAPING_KEY]``:

* **suppression** - after ``suppress_after`` thinking tokens, the
  "let me reconsider" tokens (Wait / Hmm / 等等 ...) get a negative bias, which
  trims long self-verification loops;
* **soft landing** - from ``landing_start`` thinking tokens the ``</think>``
  token gets a bias ramping linearly up to ``landing_bias`` at
  ``landing_end`` (the reasoning budget), so the model closes its thinking at
  a natural point before the hard budget cut.

The bias is recomputed for every forward from each request's own output, so
it works inside one scheduler request (no extra round trip) for normal decode
and for speculative verify. Under overlap scheduling ``output_ids`` may lag by
one step; that only delays the bias by a token, and right after ``</think>``
the model gives a second ``</think>`` far less probability than any landing
bias could make up for.
"""

from __future__ import annotations

from typing import Any, Iterable, Optional

import msgspec
import torch

QWEN_EXO_THINK_SHAPING_KEY = "qwen_exo_think_shaping"


class ThinkShapingConfig(msgspec.Struct, frozen=True):
    end_token_id: int
    suppress_after: int = 0
    suppress_token_ids: tuple[int, ...] = ()
    suppress_strength: float = 0.0
    landing_start: int = 0
    landing_end: int = 0
    landing_bias: float = 0.0

    def to_params(self) -> dict[str, Any]:
        """custom_params payload. Values must be scalars to survive the
        msgpack IPC (``CustomParamValue``), so token ids travel as "1,2,3"."""
        params = msgspec.to_builtins(self)
        params["suppress_token_ids"] = ",".join(str(i) for i in self.suppress_token_ids)
        return params

    @classmethod
    def from_params(cls, params: Any) -> Optional["ThinkShapingConfig"]:
        if not params:
            return None
        params = dict(params)
        ids = params.get("suppress_token_ids") or ""
        params["suppress_token_ids"] = tuple(int(i) for i in str(ids).split(",") if i)
        return msgspec.convert(params, cls)

    def row_bias(self, thinking_tokens: int) -> dict[int, float]:
        """Token -> bias for a request that has thought ``thinking_tokens``."""
        bias: dict[int, float] = {}
        if (
            self.suppress_after > 0
            and self.suppress_strength > 0
            and thinking_tokens >= self.suppress_after
        ):
            for token_id in self.suppress_token_ids:
                bias[token_id] = -self.suppress_strength
        if (
            self.landing_bias > 0
            and self.landing_end > self.landing_start
            and thinking_tokens >= self.landing_start
        ):
            progress = (thinking_tokens - self.landing_start) / (
                self.landing_end - self.landing_start
            )
            bias[self.end_token_id] = self.landing_bias * min(1.0, progress)
        return bias


class ThinkShapingState:
    """Per-request incremental scan: has ``</think>`` appeared yet?"""

    __slots__ = ("config", "scanned", "closed")

    def __init__(self, config: ThinkShapingConfig):
        self.config = config
        self.scanned = 0
        self.closed = False

    def thinking_tokens(self, output_ids: list[int]) -> Optional[int]:
        """Thinking tokens so far, or ``None`` once the thinking has ended."""
        if not self.closed:
            end = self.config.end_token_id
            for token in output_ids[self.scanned :]:
                if token == end:
                    self.closed = True
                    break
            self.scanned = len(output_ids)
        return None if self.closed else len(output_ids)


class ThinkShapingBias(msgspec.Struct):
    """Bias for one forward: ``bias[row, j]`` is added to ``token_ids[j]``."""

    token_ids: torch.Tensor
    bias: torch.Tensor

    def apply(self, logits: torch.Tensor) -> None:
        """``logits``: [batch, vocab]."""
        columns = logits[:, self.token_ids]
        logits[:, self.token_ids] = columns + self.bias.to(columns.dtype)

    def apply_repeated(self, logits: torch.Tensor, repeats: int) -> None:
        """``logits``: [batch * repeats, vocab], rows grouped per request
        (EAGLE / n-gram verify)."""
        columns = logits[:, self.token_ids]
        bias = self.bias.repeat_interleave(int(repeats), dim=0)
        logits[:, self.token_ids] = columns + bias.to(columns.dtype)

    def apply_per_position(self, logits_3d: torch.Tensor) -> None:
        """``logits_3d``: [batch, positions, vocab] (speculative verify)."""
        columns = logits_3d[:, :, self.token_ids]
        logits_3d[:, :, self.token_ids] = columns + self.bias[:, None, :].to(
            columns.dtype
        )


def _state_for(req: Any) -> Optional[ThinkShapingState]:
    state = req.qwen_exo_think_shaping
    if state is None:
        custom = req.sampling_params.custom_params
        config = ThinkShapingConfig.from_params(
            custom.get(QWEN_EXO_THINK_SHAPING_KEY) if custom else None
        )
        if config is None:
            return None
        state = ThinkShapingState(config)
        req.qwen_exo_think_shaping = state
    return state


def build_think_shaping(
    reqs: Iterable[Any], *, device: torch.device | str
) -> Optional[ThinkShapingBias]:
    """The shaping bias for this forward's rows, or ``None`` when no row needs it."""
    rows: list[dict[int, float]] = []
    any_bias = False
    for req in reqs:
        state = _state_for(req)
        thinking = None if state is None else state.thinking_tokens(req.output_ids)
        row = {} if thinking is None else state.config.row_bias(thinking)
        any_bias = any_bias or bool(row)
        rows.append(row)
    if not any_bias:
        return None
    token_ids = sorted({token for row in rows for token in row})
    column = {token: index for index, token in enumerate(token_ids)}
    bias = torch.zeros((len(rows), len(token_ids)), dtype=torch.float32)
    for index, row in enumerate(rows):
        for token, value in row.items():
            bias[index, column[token]] = value
    return ThinkShapingBias(
        token_ids=torch.tensor(token_ids, dtype=torch.int64).to(device, non_blocking=True),
        bias=bias.to(device, non_blocking=True),
    )


__all__ = [
    "QWEN_EXO_THINK_SHAPING_KEY",
    "ThinkShapingBias",
    "ThinkShapingConfig",
    "ThinkShapingState",
    "build_think_shaping",
]
