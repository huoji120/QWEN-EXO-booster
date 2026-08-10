from __future__ import annotations

from collections import OrderedDict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import mlx.core as mx
import torch
from mlx_lm.models.base import create_attention_mask, create_ssm_mask

from qwen_exo_booster.latent_transplant import (
    LATENT_TRANSPLANT_APPLIED_KEY,
    LATENT_TRANSPLANT_CAPTURE_COUNT_KEY,
    LATENT_TRANSPLANT_CAPTURE_LAYERS_KEY,
    LATENT_TRANSPLANT_CAPTURE_TRAJECTORY_CHUNKS_KEY,
    LATENT_TRANSPLANT_CAPTURE_TRAJECTORY_COUNT_KEY,
    LATENT_TRANSPLANT_CAPTURE_TRAJECTORY_VECTOR_KEY,
    LATENT_TRANSPLANT_CAPTURE_VECTOR_KEY,
    LATENT_TRANSPLANT_STRENGTH_KEY,
    LatentArtifactStore,
    parse_latent_transplant_spec,
    select_latent_layers,
)
from qwen_exo_booster.score_bias import (
    SCORE_BIAS_KERNEL_MAX_BLOCKS,
    SCORE_BIAS_SKETCH_DIMENSIONS,
)
from sglang.srt.hardware_backend.mlx.kv_cache.attention_contract import (
    get_head_dim,
    get_num_heads,
    get_num_kv_heads,
)
from sglang.srt.runtime_context import get_server_args


def _normalize(value: mx.array, axis: int = -1) -> mx.array:
    source = value.astype(mx.float32)
    norm = mx.sqrt(mx.sum(source * source, axis=axis, keepdims=True))
    return source / mx.maximum(norm, mx.array(1e-12, dtype=source.dtype))


def _cosine(left: mx.array, right: mx.array) -> mx.array:
    return mx.sum(_normalize(left) * _normalize(right), axis=-1)


def _compress(value: mx.array, dimensions: int) -> mx.array:
    width = int(value.shape[-1])
    source = value.astype(mx.float32)
    if width == dimensions:
        return _normalize(source)
    if width % dimensions:
        indices = mx.floor(
            mx.arange(dimensions, dtype=mx.float32) * width / dimensions
        ).astype(mx.int32)
        return _normalize(source[..., indices])
    return _normalize(source.reshape(*source.shape[:-1], dimensions, -1).mean(-1))


def _nan(dtype=mx.float32) -> mx.array:
    return mx.array(float("nan"), dtype=dtype)


@dataclass(frozen=True, slots=True)
class MlxQwenExoRequestMetadata:
    rid: str
    observe: bool
    memory_span: tuple[int, int, str] | None
    trajectory_spans: tuple[tuple[int, int], ...] | None
    user_query_spans: tuple[tuple[int, int], ...] | None
    persisted_user_queries: tuple[tuple[float, ...], ...] | None
    score_bias_blocks: tuple[dict[str, object], ...] | None
    score_bias_phase: int
    latent_transplant: dict[str, object] | None
    activation_editor: dict[str, object] | None


@dataclass(slots=True)
class _ScoreState:
    starts: mx.array
    ends: mx.array
    scores: mx.array
    keys: mx.array
    valid: mx.array
    user_queries: mx.array
    user_valid: mx.array
    shortlist: mx.array
    shortlist_relevance: mx.array
    phase: int


@dataclass(slots=True)
class _LatentCaptureState:
    layers: tuple[int, ...]
    counts: list[int] = field(default_factory=list)
    vectors: list[mx.array] = field(default_factory=list)


@dataclass(slots=True)
class _Editor:
    layer: int
    window: int
    hidden_size: int
    projection: mx.array
    transform: mx.array
    bias: mx.array


class _EditorStore:
    def __init__(self, root: Path):
        self.root = root
        self._cache: dict[tuple[str, int], _Editor | None] = {}

    def get(self, name: str) -> _Editor | None:
        path = self.root / f"{name}.editor.pt"
        try:
            modified = path.stat().st_mtime_ns
        except OSError:
            return None
        key = (name, modified)
        if key in self._cache:
            return self._cache[key]
        try:
            payload = torch.load(path, map_location="cpu", weights_only=True)
            if int(payload.get("schema") or 0) != 1:
                raise ValueError("unsupported editor schema")
            state = payload["state_dict"]
            editor = _Editor(
                layer=int(payload["layer"]),
                window=int(payload["window"]),
                hidden_size=int(payload["hidden_size"]),
                projection=mx.array(state["projection"].float().tolist()),
                transform=mx.array(state["transform"].float().tolist()),
                bias=mx.array(state["bias"].float().tolist()),
            )
        except Exception:
            editor = None
        self._cache = {
            cached_key: cached
            for cached_key, cached in self._cache.items()
            if cached_key[0] != name
        }
        self._cache[key] = editor
        return editor


@dataclass(slots=True)
class MlxQwenExoForwardContext:
    runtime: MlxQwenExoRuntime
    metadata: tuple[MlxQwenExoRequestMetadata, ...]
    mode: str
    prefix_lens: tuple[int, ...]
    extend_lens: tuple[int, ...]
    final_prefill: tuple[bool, ...]
    info: dict[str, mx.array] = field(default_factory=dict)
    latent_captured: list[tuple[int, mx.array]] = field(default_factory=list)

    @property
    def is_prefill(self) -> bool:
        return self.mode == "prefill"

    def observes_attention_layer(self, layer_idx: int) -> bool:
        return (
            self.runtime.observer_enabled
            and layer_idx == self.runtime.target_attention_layer
        )

    def observe_attention(
        self,
        q: mx.array,
        k: mx.array,
        *,
        layer_idx: int,
    ) -> None:
        self.runtime.observe_attention(self, q, k, layer_idx=layer_idx)

    def attention_bias(self, *, key_length: int, layer_idx: int) -> mx.array | None:
        return self.runtime.attention_bias(
            self, key_length=key_length, layer_idx=layer_idx
        )

    def before_layer(self, layer_idx: int, hidden_states: mx.array) -> mx.array:
        return self.runtime.before_layer(self, layer_idx, hidden_states)

    def after_layer(self, layer_idx: int, hidden_states: mx.array) -> mx.array:
        return self.runtime.after_layer(self, layer_idx, hidden_states)


class MlxQwenExoRuntime:
    """MLX-native QWEN-EXO observer, Score Bias, and hidden-state controls."""

    def __init__(self, model_runner: Any):
        self.model_runner = model_runner
        self.server_args = get_server_args()
        self.enabled = bool(getattr(self.server_args, "enable_qwen_exo", False))
        self.observer_enabled = (
            self.enabled
            and str(getattr(self.server_args, "qwen_exo_observer_mode", "off")) != "off"
        )
        self.score_bias_mode = str(
            getattr(self.server_args, "qwen_exo_score_bias_mode", "off")
        )
        self.score_bias_enabled = self.enabled and self.score_bias_mode != "off"
        self.sketch_dimensions = SCORE_BIAS_SKETCH_DIMENSIONS
        self.max_blocks = min(
            SCORE_BIAS_KERNEL_MAX_BLOCKS,
            max(1, int(getattr(self.server_args, "qwen_exo_score_bias_max_blocks", 8))),
        )
        self.selected_blocks = min(
            self.max_blocks,
            max(
                1,
                int(
                    getattr(self.server_args, "qwen_exo_score_bias_selected_blocks", 2)
                ),
            ),
        )
        self.query_window = min(
            16,
            max(
                1,
                int(getattr(self.server_args, "qwen_exo_score_bias_query_window", 8)),
            ),
        )
        self.min_relevance = float(
            getattr(self.server_args, "qwen_exo_score_bias_min_relevance", 0.0)
        )
        self.relevance_margin = float(
            getattr(self.server_args, "qwen_exo_score_bias_relevance_margin", 0.005)
        )
        self.attention_layers = tuple(
            model_runner._cache_layout.attention_layer_indices
        )
        self.target_attention_layer = (
            self.attention_layers[-1] if self.attention_layers else -1
        )
        target = (
            model_runner._attention_module_for_layer(self.target_attention_layer)
            if self.target_attention_layer >= 0
            else None
        )
        self.num_heads = int(get_num_heads(target) or 0)
        self.num_kv_heads = int(get_num_kv_heads(target) or 0)
        self.head_dim = int(get_head_dim(target) or 0)

        state_root = Path(self.server_args.qwen_exo_state_dir).expanduser()
        self.editor_store = _EditorStore(state_root / "activation-editors")
        hidden_size = int(model_runner._model_embed.weight.shape[-1])
        layer_types = tuple(
            "attention" if index in self.attention_layers else "linear_attention"
            for index in range(model_runner._cache_layout.num_layers)
        )
        self.latent_layers = select_latent_layers(layer_types)
        self.latent_store = LatentArtifactStore(
            state_root / "latent-transplant" / "artifacts",
            hidden_size=hidden_size,
            target_layers=self.latent_layers,
        )
        self._latent_vectors: dict[tuple[str, int, str], mx.array | None] = {}

        self._previous_q: OrderedDict[str, mx.array] = OrderedDict()
        self._q_history: OrderedDict[str, deque[mx.array]] = OrderedDict()
        self._memory_anchors: OrderedDict[str, tuple[mx.array, int]] = OrderedDict()
        self._prefill_key_sums: OrderedDict[str, tuple[mx.array, int]] = OrderedDict()
        self._trajectory_sums: OrderedDict[tuple[str, int], tuple[mx.array, int]] = (
            OrderedDict()
        )
        self._user_query_sums: OrderedDict[tuple[str, int], tuple[mx.array, int]] = (
            OrderedDict()
        )
        self._user_queries: OrderedDict[str, tuple[mx.array, ...]] = OrderedDict()
        self._score_states: OrderedDict[str, _ScoreState] = OrderedDict()
        self._latent_captures: OrderedDict[str, _LatentCaptureState] = OrderedDict()
        self.max_requests = max(
            1, int(getattr(self.server_args, "max_running_requests", 4096) or 4096)
        )

    @staticmethod
    def _metadata(req: Any) -> MlxQwenExoRequestMetadata:
        from sglang.srt.model_executor.forward_batch_info import (
            _qwen_exo_activation_editor,
            _qwen_exo_memory_span,
            _qwen_exo_persisted_user_queries,
            _qwen_exo_score_bias_blocks,
            _qwen_exo_score_bias_phase,
            _qwen_exo_should_observe,
            _qwen_exo_trajectory_spans,
            _qwen_exo_user_query_spans,
        )

        custom = dict(getattr(req.sampling_params, "custom_params", None) or {})
        return MlxQwenExoRequestMetadata(
            rid=str(req.rid),
            observe=_qwen_exo_should_observe(req),
            memory_span=_qwen_exo_memory_span(req),
            trajectory_spans=_qwen_exo_trajectory_spans(req),
            user_query_spans=_qwen_exo_user_query_spans(req),
            persisted_user_queries=_qwen_exo_persisted_user_queries(req),
            score_bias_blocks=_qwen_exo_score_bias_blocks(req),
            score_bias_phase=_qwen_exo_score_bias_phase(req),
            latent_transplant=parse_latent_transplant_spec(
                custom.get("qwen_exo_latent_transplant")
            ),
            activation_editor=_qwen_exo_activation_editor(req),
        )

    def prefill_context(
        self,
        req: Any | None,
        *,
        prefix_len: int,
        extend_len: int,
        final_prefill: bool,
    ) -> MlxQwenExoForwardContext | None:
        if not self.enabled or req is None:
            return None
        return MlxQwenExoForwardContext(
            runtime=self,
            metadata=(self._metadata(req),),
            mode="prefill",
            prefix_lens=(int(prefix_len),),
            extend_lens=(int(extend_len),),
            final_prefill=(bool(final_prefill),),
        )

    def decode_context(
        self, reqs: Sequence[Any] | None
    ) -> MlxQwenExoForwardContext | None:
        if not self.enabled or not reqs:
            return None
        return MlxQwenExoForwardContext(
            runtime=self,
            metadata=tuple(self._metadata(req) for req in reqs),
            mode="decode",
            prefix_lens=tuple(0 for _ in reqs),
            extend_lens=tuple(1 for _ in reqs),
            final_prefill=tuple(False for _ in reqs),
        )

    def forward_model(
        self,
        input_ids: mx.array,
        cache: list[Any],
        context: MlxQwenExoForwardContext,
    ) -> mx.array:
        runner = self.model_runner
        hidden_states = runner._model_embed(input_ids)
        first_attention = runner._cache_layout.first_attention_layer_index
        first_auxiliary = (
            runner._cache_layout.auxiliary_layer_indices[0]
            if runner._cache_layout.auxiliary_layer_indices
            else None
        )
        attention_mask = create_attention_mask(hidden_states, cache[first_attention])
        auxiliary_mask = (
            create_ssm_mask(hidden_states, cache[first_auxiliary])
            if first_auxiliary is not None
            else None
        )
        from sglang.srt.hardware_backend.mlx.kv_cache import clear_context, set_context

        set_context(context)
        try:
            for layer_idx, (layer, layer_cache) in enumerate(
                zip(runner._cache_layout.layers, cache)
            ):
                hidden_states = context.before_layer(layer_idx, hidden_states)
                mask = (
                    attention_mask
                    if runner._cache_layout.attention_attrs[layer_idx] is not None
                    else auxiliary_mask
                )
                hidden_states = layer(hidden_states, mask=mask, cache=layer_cache)
                hidden_states = context.after_layer(layer_idx, hidden_states)
        finally:
            clear_context()
        hidden_states = runner._model_norm(hidden_states)
        return runner._extract_logits(runner._model_lm_head(hidden_states))

    def observe_attention(
        self,
        context: MlxQwenExoForwardContext,
        q: mx.array,
        k: mx.array,
        *,
        layer_idx: int,
    ) -> None:
        if not self.observer_enabled or layer_idx != self.target_attention_layer:
            return
        if context.mode == "prefill":
            self._observe_prefill(context, q, k)
        else:
            self._observe_decode(context, q)

    def _register_persisted_queries(self, metadata: MlxQwenExoRequestMetadata) -> None:
        if metadata.rid in self._user_queries or not metadata.persisted_user_queries:
            return
        queries = tuple(
            _normalize(mx.array(row, dtype=mx.float32))
            for row in metadata.persisted_user_queries
            if len(row) == self.sketch_dimensions
        )
        if queries:
            self._user_queries[metadata.rid] = queries[-8:]

    def _observe_prefill(
        self,
        context: MlxQwenExoForwardContext,
        q: mx.array,
        k: mx.array,
    ) -> None:
        metadata = context.metadata[0]
        self._register_persisted_queries(metadata)
        if not metadata.observe or not context.extend_lens[0]:
            return
        prefix_len = context.prefix_lens[0]
        extend_len = min(context.extend_lens[0], int(q.shape[1]))
        final = context.final_prefill[0]
        q_tokens = q[0, :extend_len].astype(mx.float32)
        k_tokens = k[0, :extend_len].astype(mx.float32)

        key_sum = mx.sum(k_tokens, axis=(0, 1)) / max(self.num_kv_heads, 1)
        prior_key = self._prefill_key_sums.get(metadata.rid)
        total_key = key_sum if prior_key is None else prior_key[0] + key_sum
        total_key_count = extend_len if prior_key is None else prior_key[1] + extend_len
        self._prefill_key_sums[metadata.rid] = (total_key, total_key_count)

        self._capture_memory(metadata, k_tokens, prefix_len, extend_len)
        trajectory = self._capture_spans(
            metadata.rid,
            metadata.trajectory_spans,
            k_tokens,
            prefix_len,
            extend_len,
            self._trajectory_sums,
            self.num_kv_heads,
            final,
        )
        user_queries = self._capture_spans(
            metadata.rid,
            metadata.user_query_spans,
            q_tokens,
            prefix_len,
            extend_len,
            self._user_query_sums,
            self.num_heads,
            final,
        )
        if user_queries:
            previous = list(self._user_queries.get(metadata.rid, ()))
            for query in user_queries:
                if not any(
                    bool(mx.all(mx.abs(query - item) < 1e-6).item())
                    for item in previous
                ):
                    previous.append(query)
            self._user_queries[metadata.rid] = tuple(previous[-8:])

        if not final:
            return
        current = q_tokens[-1].mean(axis=0)
        q_norm = mx.sqrt(mx.mean(current * current))
        previous_q = self._previous_q.get(metadata.rid)
        q_drift = _nan() if previous_q is None else 1.0 - _cosine(current, previous_q)
        self._previous_q[metadata.rid] = current
        history = self._q_history.setdefault(metadata.rid, deque(maxlen=16))
        history.append(_compress(current, self.sketch_dimensions))

        memory_energy = _nan()
        if metadata.memory_span is not None:
            anchor = self._memory_anchors.get(metadata.memory_span[2])
            if anchor is not None:
                memory_energy = (
                    _cosine(current, anchor[0] / max(anchor[1], 1)) + 1
                ) * 0.5

        key_sketch = _compress(
            total_key / max(total_key_count, 1), self.sketch_dimensions
        )
        self._prefill_key_sums.pop(metadata.rid, None)
        context.info.update(
            {
                "qwen_exo_q_norm": q_norm.reshape(1),
                "qwen_exo_q_drift": q_drift.reshape(1),
                "qwen_exo_memory_energy": memory_energy.reshape(1),
                "qwen_exo_q_sketch": _compress(current, self.sketch_dimensions)[None],
                "qwen_exo_k_sketch": key_sketch[None],
            }
        )
        if trajectory:
            context.info["qwen_exo_trajectory_k_sketch"] = mx.stack(trajectory)[None]
        if self._user_queries.get(metadata.rid):
            context.info["qwen_exo_user_query_sketch"] = mx.stack(
                self._user_queries[metadata.rid]
            )[None]
        self._prepare_score_state(context, metadata)
        self._trim_request_state()

    def _capture_memory(
        self,
        metadata: MlxQwenExoRequestMetadata,
        k_tokens: mx.array,
        prefix_len: int,
        extend_len: int,
    ) -> None:
        if metadata.memory_span is None:
            return
        start, length, key = metadata.memory_span
        overlap_start = max(start, prefix_len)
        overlap_end = min(start + length, prefix_len + extend_len)
        if overlap_start >= overlap_end:
            return
        local_start = overlap_start - prefix_len
        local_end = overlap_end - prefix_len
        segment = k_tokens[local_start:local_end]
        count = overlap_end - overlap_start
        anchor_sum = mx.sum(segment, axis=(0, 1)) / max(self.num_kv_heads, 1)
        prior = self._memory_anchors.get(key)
        self._memory_anchors[key] = (
            anchor_sum if prior is None else prior[0] + anchor_sum,
            count if prior is None else prior[1] + count,
        )
        self._memory_anchors.move_to_end(key)
        while len(self._memory_anchors) > 512:
            self._memory_anchors.popitem(last=False)

    def _capture_spans(
        self,
        rid: str,
        spans: tuple[tuple[int, int], ...] | None,
        values: mx.array,
        prefix_len: int,
        extend_len: int,
        accumulators: OrderedDict[tuple[str, int], tuple[mx.array, int]],
        heads: int,
        final: bool,
    ) -> tuple[mx.array, ...]:
        completed: list[mx.array] = []
        for span_index, span in enumerate(spans or ()):
            span_start, span_end = int(span[0]), int(span[1])
            overlap_start = max(span_start, prefix_len)
            overlap_end = min(span_end, prefix_len + extend_len)
            key = (rid, span_index)
            if overlap_start < overlap_end:
                segment = values[overlap_start - prefix_len : overlap_end - prefix_len]
                chunk = mx.sum(segment, axis=(0, 1)) / max(heads, 1)
                prior = accumulators.get(key)
                accumulators[key] = (
                    chunk if prior is None else prior[0] + chunk,
                    overlap_end - overlap_start
                    if prior is None
                    else prior[1] + overlap_end - overlap_start,
                )
            if final:
                captured = accumulators.pop(key, None)
                if captured is not None and captured[1] == span_end - span_start:
                    completed.append(
                        _compress(
                            captured[0] / max(captured[1], 1), self.sketch_dimensions
                        )
                    )
        return tuple(completed)

    def _prepare_score_state(
        self,
        context: MlxQwenExoForwardContext,
        metadata: MlxQwenExoRequestMetadata,
    ) -> None:
        blocks = tuple(metadata.score_bias_blocks or ())[: self.max_blocks]
        block_count = len(blocks)
        starts = mx.zeros((self.max_blocks,), dtype=mx.float32)
        ends = mx.zeros((self.max_blocks,), dtype=mx.float32)
        scores = mx.zeros((self.max_blocks,), dtype=mx.float32)
        keys = mx.zeros((self.max_blocks, self.sketch_dimensions), dtype=mx.float32)
        valid = mx.zeros((self.max_blocks,), dtype=mx.bool_)
        for index, block in enumerate(blocks):
            starts[index] = float(block["start"])
            ends[index] = float(block["end"])
            scores[index] = float(block["score"])
            keys[index] = _normalize(mx.array(block["key_sketch"], dtype=mx.float32))
            valid[index] = True

        raw_queries = self._user_queries.get(metadata.rid, ())[-8:]
        user_queries = mx.zeros((8, self.sketch_dimensions), dtype=mx.float32)
        user_valid = mx.zeros((8,), dtype=mx.bool_)
        for index, query in enumerate(raw_queries):
            user_queries[index] = _normalize(query)
            user_valid[index] = True

        shortlist, relevance = self._shortlist(
            keys,
            valid,
            user_queries,
            user_valid,
        )
        if metadata.score_bias_phase <= 0:
            shortlist = mx.zeros_like(shortlist)
        state = _ScoreState(
            starts=starts,
            ends=ends,
            scores=scores,
            keys=keys,
            valid=valid,
            user_queries=user_queries,
            user_valid=user_valid,
            shortlist=shortlist,
            shortlist_relevance=relevance,
            phase=metadata.score_bias_phase,
        )
        self._score_states[metadata.rid] = state
        shortlist_count = mx.sum(shortlist).astype(mx.float32)
        shortlist_best = mx.where(
            shortlist_count > 0,
            mx.max(mx.where(shortlist, relevance, float("-inf"))),
            _nan(),
        )
        context.info.update(
            {
                "qwen_exo_score_bias_phase": mx.array(
                    [metadata.score_bias_phase], dtype=mx.float32
                ),
                "qwen_exo_score_bias_is_decode": mx.array([0.0]),
                "qwen_exo_score_bias_candidate_count": mx.array(
                    [block_count], dtype=mx.float32
                ),
                "qwen_exo_score_bias_user_query_count": mx.array(
                    [len(raw_queries)], dtype=mx.float32
                ),
                "qwen_exo_score_bias_shortlist_count": shortlist_count.reshape(1),
                "qwen_exo_score_bias_shortlist_max_relevance": shortlist_best.reshape(
                    1
                ),
                "qwen_exo_score_bias_selected_count": mx.array([0.0]),
                "qwen_exo_score_bias_max_relevance": mx.array([float("nan")]),
                "qwen_exo_score_bias_would_apply_max": mx.array([0.0]),
                "qwen_exo_score_bias_query_consensus": mx.array([0.0]),
            }
        )

    def _shortlist(
        self,
        keys: mx.array,
        valid: mx.array,
        queries: mx.array,
        query_valid: mx.array,
    ) -> tuple[mx.array, mx.array]:
        pair_scores = queries @ keys.T
        masked = mx.where(
            query_valid[:, None] & valid[None, :], pair_scores, float("-inf")
        )
        top_r = min(3, int(queries.shape[0]))
        sorted_scores = mx.sort(masked, axis=0)
        selected = sorted_scores[-top_r:]
        finite = mx.isfinite(selected)
        relevance = mx.sum(mx.where(finite, selected, 0), axis=0) / mx.maximum(
            mx.sum(finite, axis=0), 1
        )
        relevance = mx.where(mx.any(finite, axis=0), relevance, float("-inf"))
        order = mx.argsort(relevance)[::-1]
        limit = min(4, self.max_blocks)
        ranked = mx.zeros((self.max_blocks,), dtype=mx.bool_)
        ranked[order[:limit]] = True
        margin_ok = mx.array(True)
        if self.max_blocks > limit:
            margin_ok = (mx.sum(valid) <= limit) | (
                relevance[order[limit - 1]] - relevance[order[limit]]
                >= self.relevance_margin
            )
        winner = mx.argmax(masked, axis=-1)
        consensus = mx.stack(
            [
                mx.sum((winner == index) & query_valid)
                for index in range(self.max_blocks)
            ]
        )
        gate = (
            mx.any(query_valid)
            & mx.any(valid)
            & (relevance[order[0]] >= self.min_relevance)
            & margin_ok
        )
        shortlist = (
            ranked & valid & gate & (relevance >= self.min_relevance) & (consensus > 0)
        )
        return shortlist, relevance

    def _observe_decode(
        self,
        context: MlxQwenExoForwardContext,
        q: mx.array,
    ) -> None:
        q_norms: list[mx.array] = []
        q_drifts: list[mx.array] = []
        memory_energies: list[mx.array] = []
        sketches: list[mx.array] = []
        for row, metadata in enumerate(context.metadata):
            if not metadata.observe:
                q_norms.append(_nan())
                q_drifts.append(_nan())
                memory_energies.append(_nan())
                sketches.append(mx.full((self.sketch_dimensions,), float("nan")))
                continue
            current = q[row, :, -1, :].astype(mx.float32).mean(axis=0)
            previous = self._previous_q.get(metadata.rid)
            q_norms.append(mx.sqrt(mx.mean(current * current)))
            q_drifts.append(
                _nan() if previous is None else 1.0 - _cosine(current, previous)
            )
            self._previous_q[metadata.rid] = current
            sketch = _compress(current, self.sketch_dimensions)
            sketches.append(sketch)
            history = self._q_history.setdefault(metadata.rid, deque(maxlen=16))
            history.append(sketch)
            memory_energy = _nan()
            if metadata.memory_span is not None:
                anchor = self._memory_anchors.get(metadata.memory_span[2])
                if anchor is not None:
                    memory_energy = (
                        _cosine(current, anchor[0] / max(anchor[1], 1)) + 1
                    ) * 0.5
            memory_energies.append(memory_energy)
        context.info.update(
            {
                "qwen_exo_q_norm": mx.stack(q_norms),
                "qwen_exo_q_drift": mx.stack(q_drifts),
                "qwen_exo_memory_energy": mx.stack(memory_energies),
                "qwen_exo_q_sketch": mx.stack(sketches),
                "qwen_exo_k_sketch": mx.full(
                    (len(context.metadata), self.sketch_dimensions), float("nan")
                ),
            }
        )

    def attention_bias(
        self,
        context: MlxQwenExoForwardContext,
        *,
        key_length: int,
        layer_idx: int,
    ) -> mx.array | None:
        if (
            not self.score_bias_enabled
            or context.mode != "decode"
            or layer_idx != self.target_attention_layer
        ):
            return None
        masks: list[mx.array] = []
        telemetry: dict[str, list[mx.array]] = {
            "qwen_exo_score_bias_phase": [],
            "qwen_exo_score_bias_is_decode": [],
            "qwen_exo_score_bias_candidate_count": [],
            "qwen_exo_score_bias_user_query_count": [],
            "qwen_exo_score_bias_shortlist_count": [],
            "qwen_exo_score_bias_shortlist_max_relevance": [],
            "qwen_exo_score_bias_selected_count": [],
            "qwen_exo_score_bias_max_relevance": [],
            "qwen_exo_score_bias_would_apply_max": [],
            "qwen_exo_score_bias_query_consensus": [],
        }
        positions = mx.arange(key_length, dtype=mx.float32)
        for metadata in context.metadata:
            state = self._score_states.get(metadata.rid)
            history = tuple(self._q_history.get(metadata.rid, ()))
            if state is None or not history:
                masks.append(mx.zeros((key_length,), dtype=mx.float32))
                values = (
                    mx.array(float(metadata.score_bias_phase)),
                    mx.array(1.0),
                    mx.array(0.0),
                    mx.array(0.0),
                    mx.array(0.0),
                    _nan(),
                    mx.array(0.0),
                    _nan(),
                    mx.array(0.0),
                    mx.array(0.0),
                )
                for key, value in zip(telemetry, values):
                    telemetry[key].append(value)
                continue

            queries = mx.stack(history[-self.query_window :])
            pair_scores = queries @ state.keys.T
            valid_pairs = state.shortlist[None, :]
            masked = mx.where(valid_pairs, pair_scores, float("-inf"))
            top_r = min(3, int(queries.shape[0]))
            top = mx.sort(masked, axis=0)[-top_r:]
            finite = mx.isfinite(top)
            relevance = mx.sum(mx.where(finite, top, 0), axis=0) / mx.maximum(
                mx.sum(finite, axis=0), 1
            )
            relevance = mx.where(mx.any(finite, axis=0), relevance, float("-inf"))
            order = mx.argsort(relevance)[::-1]
            shortlist_count = mx.sum(state.shortlist)
            margin_ok = mx.array(True)
            if self.max_blocks > self.selected_blocks:
                margin_ok = (shortlist_count <= self.selected_blocks) | (
                    relevance[order[self.selected_blocks - 1]]
                    - relevance[order[self.selected_blocks]]
                    >= self.relevance_margin
                )
            winner = mx.argmax(masked, axis=-1)
            consensus = mx.stack(
                [mx.sum(winner == index) for index in range(self.max_blocks)]
            )
            selected = order[: self.selected_blocks]
            selected_relevance = relevance[selected]
            selected_consensus = consensus[selected]
            selected_valid = (
                (mx.arange(self.selected_blocks) < shortlist_count)
                & state.shortlist[selected]
                & (selected_relevance >= self.min_relevance)
                & (selected_consensus > 0)
                & margin_ok
                & (state.phase > 0)
            )
            effective = state.scores[selected] * mx.maximum(selected_relevance, 0)
            selected_valid &= effective > 0
            effective = mx.where(selected_valid, effective, 0)
            row_bias = mx.zeros((key_length,), dtype=mx.float32)
            for slot in range(self.selected_blocks):
                applies = (
                    selected_valid[slot]
                    & (positions >= state.starts[selected[slot]])
                    & (positions < state.ends[selected[slot]])
                    & (state.phase == 2)
                )
                row_bias = mx.maximum(
                    row_bias,
                    mx.where(applies, effective[slot], 0),
                )
            masks.append(row_bias)

            selected_count = mx.sum(selected_valid).astype(mx.float32)
            max_relevance = mx.where(
                selected_count > 0,
                mx.max(mx.where(selected_valid, selected_relevance, float("-inf"))),
                _nan(),
            )
            max_consensus = mx.max(mx.where(selected_valid, selected_consensus, 0))
            max_bias = mx.max(effective)
            shortlist_best = mx.where(
                shortlist_count > 0,
                mx.max(
                    mx.where(
                        state.shortlist,
                        state.shortlist_relevance,
                        float("-inf"),
                    )
                ),
                _nan(),
            )
            values = (
                mx.array(float(state.phase)),
                mx.array(1.0),
                mx.sum(state.valid).astype(mx.float32),
                mx.sum(state.user_valid).astype(mx.float32),
                shortlist_count.astype(mx.float32),
                shortlist_best,
                selected_count,
                max_relevance,
                max_bias,
                max_consensus.astype(mx.float32),
            )
            for key, value in zip(telemetry, values):
                telemetry[key].append(value)

        context.info.update(
            {key: mx.stack(values) for key, values in telemetry.items()}
        )
        return mx.stack(masks)[:, None, None, :]

    def _latent_vector(
        self, artifact: str, layer_idx: int, dtype: mx.Dtype
    ) -> mx.array | None:
        key = (artifact, int(layer_idx), str(dtype))
        if key in self._latent_vectors:
            return self._latent_vectors[key]
        value = self.latent_store.vector(
            artifact,
            layer_idx,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )
        restored = mx.array(value.tolist(), dtype=dtype) if value is not None else None
        self._latent_vectors[key] = restored
        return restored

    def before_layer(
        self,
        context: MlxQwenExoForwardContext,
        layer_idx: int,
        hidden_states: mx.array,
    ) -> mx.array:
        if context.mode != "prefill" or len(context.metadata) != 1:
            return hidden_states
        metadata = context.metadata[0]
        spec = metadata.latent_transplant
        if (
            not spec
            or spec.get("mode") != "active"
            or not context.final_prefill[0]
            or layer_idx not in self.latent_layers
        ):
            return hidden_states
        vector = self._latent_vector(
            str(spec.get("artifact") or ""), layer_idx, hidden_states.dtype
        )
        if vector is None:
            return hidden_states
        strength = min(0.5, max(0.0, float(spec.get("strength") or 0.0)))
        window = min(
            int(hidden_states.shape[1]),
            max(1, int(spec.get("token_window") or 1)),
        )
        if strength <= 0 or window < 1:
            return hidden_states
        edited = mx.array(hidden_states)
        edited[:, -window:, :] = edited[:, -window:, :] + vector * strength
        context.info[LATENT_TRANSPLANT_APPLIED_KEY] = mx.array([1.0])
        context.info[LATENT_TRANSPLANT_STRENGTH_KEY] = mx.array([strength])
        return edited

    def after_layer(
        self,
        context: MlxQwenExoForwardContext,
        layer_idx: int,
        hidden_states: mx.array,
    ) -> mx.array:
        if context.mode != "prefill" or len(context.metadata) != 1:
            return hidden_states
        metadata = context.metadata[0]
        latent = metadata.latent_transplant
        if (
            latent
            and latent.get("mode") == "capture"
            and layer_idx in self.latent_layers
        ):
            tail = int(latent.get("capture_tail_tokens") or 0)
            segment = hidden_states[0]
            if tail > 0:
                segment = segment[-min(tail, int(segment.shape[0])) :]
            context.latent_captured.append((layer_idx, segment.mean(axis=0)))
            if layer_idx == self.latent_layers[-1]:
                self._commit_latent_capture(context)

        spec = metadata.activation_editor
        if not spec or spec.get("mode") != "active" or not context.final_prefill[0]:
            return hidden_states
        editor = self.editor_store.get(str(spec.get("editor") or ""))
        if editor is None or editor.layer != layer_idx:
            return hidden_states
        if editor.hidden_size != int(hidden_states.shape[-1]):
            return hidden_states
        tail_offset = max(0, int(spec.get("tail_offset") or 0))
        end = max(0, int(hidden_states.shape[1]) - tail_offset)
        start = max(0, end - editor.window)
        if start >= end:
            return hidden_states
        strength = float(spec.get("strength") or 1.0)
        source = hidden_states[:, start:end, :].astype(mx.float32)
        base = source @ editor.projection.T
        target = source @ editor.transform.T + editor.bias
        delta = (target - base) @ editor.projection
        edited = mx.array(hidden_states)
        edited[:, start:end, :] = (source + delta * strength).astype(
            hidden_states.dtype
        )
        return edited

    def _commit_latent_capture(self, context: MlxQwenExoForwardContext) -> None:
        metadata = context.metadata[0]
        if not context.latent_captured:
            return
        layer_ids = tuple(layer for layer, _ in context.latent_captured)
        block = mx.stack([value for _, value in context.latent_captured])
        state = self._latent_captures.get(metadata.rid)
        if state is None:
            state = _LatentCaptureState(layers=layer_ids)
            self._latent_captures[metadata.rid] = state
        elif state.layers != layer_ids:
            self._latent_captures.pop(metadata.rid, None)
            return
        count = context.extend_lens[0]
        state.counts.append(count)
        state.vectors.append(block)
        context.latent_captured.clear()
        if not context.final_prefill[0]:
            return
        state = self._latent_captures.pop(metadata.rid)
        counts = mx.array(state.counts, dtype=mx.int32)
        trajectory = mx.stack(state.vectors)
        total = max(sum(state.counts), 1)
        weighted = (
            mx.stack(
                [vector * count for vector, count in zip(state.vectors, state.counts)]
            ).sum(axis=0)
            / total
        )
        context.info.update(
            {
                LATENT_TRANSPLANT_CAPTURE_VECTOR_KEY: weighted.reshape(1, -1),
                LATENT_TRANSPLANT_CAPTURE_COUNT_KEY: mx.array([total]),
                LATENT_TRANSPLANT_CAPTURE_LAYERS_KEY: mx.array(layer_ids)[None],
                LATENT_TRANSPLANT_CAPTURE_TRAJECTORY_VECTOR_KEY: trajectory.reshape(
                    1, -1
                ),
                LATENT_TRANSPLANT_CAPTURE_TRAJECTORY_COUNT_KEY: counts[None],
                LATENT_TRANSPLANT_CAPTURE_TRAJECTORY_CHUNKS_KEY: mx.array(
                    [len(state.vectors)]
                ),
            }
        )

    def register_memory_keys(self, memory_key: str, keys: mx.array) -> None:
        if not memory_key or keys.size == 0:
            return
        view = keys.reshape(-1, self.num_kv_heads, self.head_dim).astype(mx.float32)
        self._memory_anchors[str(memory_key)] = (
            mx.sum(view, axis=(0, 1)) / max(self.num_kv_heads, 1),
            int(view.shape[0]),
        )

    def forget_request(self, rid: str) -> None:
        request_id = str(rid)
        self._previous_q.pop(request_id, None)
        self._q_history.pop(request_id, None)
        self._prefill_key_sums.pop(request_id, None)
        self._user_queries.pop(request_id, None)
        self._score_states.pop(request_id, None)
        self._latent_captures.pop(request_id, None)
        for mapping in (self._trajectory_sums, self._user_query_sums):
            for key in tuple(mapping):
                if key[0] == request_id:
                    mapping.pop(key, None)

    def clear(self) -> None:
        self._previous_q.clear()
        self._q_history.clear()
        self._prefill_key_sums.clear()
        self._trajectory_sums.clear()
        self._user_query_sums.clear()
        self._user_queries.clear()
        self._score_states.clear()
        self._latent_captures.clear()

    def _trim_request_state(self) -> None:
        for mapping in (
            self._previous_q,
            self._q_history,
            self._score_states,
            self._user_queries,
        ):
            while len(mapping) > self.max_requests:
                mapping.popitem(last=False)
