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
    # Keep one-shot pinned staging alive until its asynchronous copy completes.
    _staging: Any = None

    def apply(self, logits: torch.Tensor) -> None:
        """``logits``: [batch, vocab]."""
        columns = logits[:, self.token_ids]
        logits[:, self.token_ids] = columns + self.bias.to(columns.dtype)

    def apply_repeated(self, logits: torch.Tensor, repeats: int) -> None:
        """``logits``: [batch * repeats, vocab], rows grouped per request
        (EAGLE / n-gram verify)."""
        self.apply_per_position(logits.view(self.bias.shape[0], int(repeats), -1))

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


class _ThinkShapingStaging:
    def __init__(self, rows: int, columns: int, *, pinned: bool):
        self.host = torch.empty((rows, columns), dtype=torch.float32, pin_memory=pinned)
        self.values = self.host.numpy()
        self.copied = torch.cuda.Event() if pinned else None
        self.pending = False

    def wait(self) -> None:
        # Protect DMA's pinned source, not only the lifetime of its Tensor object.
        if self.pending:
            self.copied.synchronize()
            self.pending = False


class ThinkShapingWorkspace:
    """Scheduler-owned layouts and two bounded pinned CPU staging slots.

    Device bias snapshots remain independent: a delayed sampler or an overlap
    forward can retain one after the next forward has prepared its bias.
    """

    def __init__(self, device: torch.device | str):
        self.device = torch.device(device)
        self._signature = None
        self._token_ids = None
        self._tokens = ()
        self._row_layouts = ()
        self._staging = []
        self._slot = 0
        self._row_capacity = 0

    def _layout(self, signature) -> None:
        if signature == self._signature:
            return
        tokens = set()
        for config in signature:
            if config is None:
                continue
            if config.suppress_after > 0 and config.suppress_strength > 0:
                tokens.update(config.suppress_token_ids)
            if config.landing_bias > 0 and config.landing_end > config.landing_start:
                tokens.add(config.end_token_id)
        token_ids = tuple(sorted(tokens))
        columns = {token: index for index, token in enumerate(token_ids)}
        self._row_layouts = tuple(
            (
                tuple(columns[token] for token in set(config.suppress_token_ids) if token in columns),
                columns.get(config.end_token_id),
            ) if config is not None else ((), None)
            for config in signature
        )
        if self._token_ids is None or token_ids != self._tokens:
            for staging in self._staging:
                staging.wait()
            self._staging = []
            self._tokens = token_ids
            self._token_ids = torch.tensor(token_ids, dtype=torch.int64, device=self.device)
        self._signature = signature

    def build(self, reqs: Iterable[Any]) -> Optional[ThinkShapingBias]:
        rows = []
        any_bias = False
        for req in reqs:
            state = _state_for(req)
            config = None if state is None else state.config
            thinking = None if state is None else state.thinking_tokens(req.output_ids)
            suppress = (
                thinking is not None and config.suppress_after > 0
                and config.suppress_strength > 0 and thinking >= config.suppress_after
                and bool(config.suppress_token_ids)
            )
            landing = (
                thinking is not None and config.landing_bias > 0
                and config.landing_end > config.landing_start
                and thinking >= config.landing_start
            )
            any_bias = any_bias or suppress or landing
            rows.append((config, thinking, suppress, landing))
        if not any_bias:
            return None
        self._layout(tuple(row[0] for row in rows))
        count = len(rows)
        if not self._staging or count > self._row_capacity:
            for staging in self._staging:
                staging.wait()
            self._row_capacity = max(self._row_capacity, 1 << (count - 1).bit_length())
            self._staging = [
                _ThinkShapingStaging(
                    self._row_capacity, len(self._tokens), pinned=self.device.type == "cuda"
                ) for _ in range(2)
            ]
            self._slot = 0
        staging = self._staging[self._slot]
        self._slot = (self._slot + 1) % len(self._staging)
        staging.wait()
        values = staging.values[:count]
        values.fill(0)
        for index, ((config, thinking, suppress, landing), (suppress_columns, end_column)) in enumerate(
            zip(rows, self._row_layouts)
        ):
            if suppress:
                for column in suppress_columns:
                    values[index, column] = -config.suppress_strength
            if landing:
                progress = (thinking - config.landing_start) / (config.landing_end - config.landing_start)
                # Landing overrides suppression if both target the end token.
                values[index, end_column] = config.landing_bias * min(1.0, progress)
        host = staging.host[:count]
        if self.device.type == "cpu":
            bias = host.clone()
        else:
            bias = host.to(self.device, non_blocking=staging.copied is not None)
        if staging.copied is not None:
            staging.copied.record(torch.cuda.current_stream(self.device))
            staging.pending = True
        return ThinkShapingBias(self._token_ids, bias, staging)


def build_think_shaping(
    reqs: Iterable[Any], *, device: torch.device | str,
    workspace: Optional[ThinkShapingWorkspace] = None,
) -> Optional[ThinkShapingBias]:
    """Build an independent forward snapshot, optionally reusing its staging."""
    if workspace is None:
        workspace = ThinkShapingWorkspace(device)
    return workspace.build(reqs)


__all__ = [
    "QWEN_EXO_THINK_SHAPING_KEY",
    "ThinkShapingBias",
    "ThinkShapingConfig",
    "ThinkShapingState",
    "ThinkShapingWorkspace",
    "build_think_shaping",
]
