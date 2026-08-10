from __future__ import annotations

from collections import OrderedDict, deque
from dataclasses import dataclass
from typing import Any, Callable

import torch

MemorySpan = tuple[int, int, str]
TrajectorySpan = tuple[int, int]


def inverse_qwen35_rope(
    value: torch.Tensor,
    positions: torch.Tensor,
    *,
    rotary: Any,
    head_dim: int,
) -> torch.Tensor:
    """Undo Qwen3.5 RoPE/MRoPE without changing the tensor's packed layout."""

    if positions.ndim not in {1, 2}:
        raise ValueError("Qwen3.5 observer positions must be one- or two-dimensional")
    token_count = int(positions.shape[-1])
    original_shape = value.shape
    if token_count < 1 or value.numel() % (token_count * int(head_dim)):
        raise ValueError("Qwen3.5 observer tensor does not match its position rows")
    view = value.reshape(token_count, -1, int(head_dim))
    rotary_dim = int(rotary.rotary_dim)
    rotated = view[..., :rotary_dim]
    passed = view[..., rotary_dim:]
    cos_sin_cache = rotary.cos_sin_cache
    position_ids = positions.to(device=cos_sin_cache.device, dtype=torch.long)
    cos_sin = cos_sin_cache[position_ids]
    cos, sin = cos_sin.chunk(2, dim=-1)
    if position_ids.ndim == 2:
        sections = tuple(int(item) for item in (rotary.mrope_section or ()))
        if len(sections) != position_ids.shape[0] or sum(sections) != cos.shape[-1]:
            raise ValueError("Qwen3.5 MRoPE sections do not match the rotary width")
        if rotary.mrope_interleaved:
            mixed_cos = cos[0].clone()
            mixed_sin = sin[0].clone()
            height_end = sections[1] * 3
            width_end = sections[2] * 3
            mixed_cos[..., 1:height_end:3] = cos[1, ..., 1:height_end:3]
            mixed_sin[..., 1:height_end:3] = sin[1, ..., 1:height_end:3]
            mixed_cos[..., 2:width_end:3] = cos[2, ..., 2:width_end:3]
            mixed_sin[..., 2:width_end:3] = sin[2, ..., 2:width_end:3]
            cos, sin = mixed_cos, mixed_sin
        else:
            cos = torch.cat(
                [part[axis] for axis, part in enumerate(cos.split(sections, dim=-1))],
                dim=-1,
            )
            sin = torch.cat(
                [part[axis] for axis, part in enumerate(sin.split(sections, dim=-1))],
                dim=-1,
            )
    cos = cos.to(device=view.device, dtype=view.dtype).unsqueeze(1)
    sin = sin.to(device=view.device, dtype=view.dtype).unsqueeze(1)
    if rotary.is_neox_style:
        first, second = torch.chunk(rotated, 2, dim=-1)
        raw = torch.cat(
            (first * cos + second * sin, second * cos - first * sin), dim=-1
        )
    else:
        first = rotated[..., ::2]
        second = rotated[..., 1::2]
        raw = torch.stack(
            (first * cos + second * sin, second * cos - first * sin), dim=-1
        ).flatten(-2)
    if passed.numel():
        raw = torch.cat((raw, passed), dim=-1)
    return raw.reshape(original_shape)


@dataclass(frozen=True, slots=True)
class AttentionBatchMetadata:
    is_decode: bool
    is_extend: bool
    contains_last_prefill_chunk: bool
    rids: tuple[str, ...]
    observe_mask: tuple[bool, ...]
    memory_spans: tuple[MemorySpan | None, ...]
    trajectory_spans: tuple[tuple[TrajectorySpan, ...] | None, ...] = ()
    user_query_spans: tuple[tuple[TrajectorySpan, ...] | None, ...] = ()
    anchor_spans: tuple[tuple[TrajectorySpan, ...] | None, ...] = ()
    persisted_user_queries: tuple[tuple[tuple[float, ...], ...] | None, ...] = ()
    full_query_capture: tuple[bool, ...] = ()
    final_prefill_mask: tuple[bool, ...] | None = None
    extend_lens: tuple[int, ...] | None = None
    prefix_lens: tuple[int, ...] | None = None


@dataclass(slots=True)
class _DecodeSlotState:
    observe: torch.Tensor
    previous_q: torch.Tensor
    previous_valid: torch.Tensor
    last_q_drift: torch.Tensor
    memory_anchor: torch.Tensor
    memory_valid: torch.Tensor
    history: torch.Tensor
    history_count: torch.Tensor
    history_cursor: torch.Tensor
    block_starts: torch.Tensor
    block_ends: torch.Tensor
    block_scores: torch.Tensor
    block_keys: torch.Tensor
    block_valid: torch.Tensor
    user_queries: torch.Tensor
    user_query_valid: torch.Tensor
    anchor_starts: torch.Tensor
    anchor_ends: torch.Tensor
    anchor_valid: torch.Tensor
    phases: torch.Tensor


class AttentionSignalTracker:
    """Computes request-stable Q drift and Q-to-memory-K alignment.

    Memory energy is an aggregate proxy: cosine alignment between the selected
    decode Q sketch and the centroid of K vectors over the exact external-memory
    token span. It is not a materialized softmax attention matrix.
    """

    def __init__(
        self,
        *,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        reduce_across_tp: Callable[[torch.Tensor], torch.Tensor] | None = None,
        total_num_heads: int | None = None,
        gather_heads_across_tp: Callable[[torch.Tensor], torch.Tensor] | None = None,
        max_requests: int = 4096,
        max_memory_anchors: int = 512,
        sketch_dimensions: int = 32,
        score_bias_max_blocks: int = 8,
        score_bias_selected_blocks: int = 2,
        score_bias_query_window: int = 8,
        score_bias_min_relevance: float = 0.0,
        score_bias_relevance_margin: float = 0.005,
        score_bias_anchor_bias: float = 0.0,
        score_bias_anchor_drift_threshold: float = 0.35,
        score_bias_anchor_max_blocks: int = 2,
    ):
        self.num_heads = int(num_heads)
        self.num_kv_heads = int(num_kv_heads)
        self.head_dim = int(head_dim)
        self.reduce_across_tp = reduce_across_tp
        self.total_num_heads = int(total_num_heads or num_heads)
        self.gather_heads_across_tp = gather_heads_across_tp
        if self.total_num_heads < self.num_heads:
            raise ValueError(
                "Total Q head count cannot be smaller than the local count"
            )
        self.max_requests = int(max_requests)
        self.max_memory_anchors = int(max_memory_anchors)
        if sketch_dimensions < 1:
            raise ValueError("Q sketch dimensions must be positive")
        self.sketch_dimensions = min(int(sketch_dimensions), self.head_dim)
        self.score_bias_max_blocks = max(1, int(score_bias_max_blocks))
        self.score_bias_anchor_bias = float(score_bias_anchor_bias)
        self.score_bias_anchor_drift_threshold = float(
            score_bias_anchor_drift_threshold
        )
        if self.score_bias_anchor_drift_threshold < 0:
            raise ValueError("Score Bias anchor drift threshold must be non-negative")
        if not 0 <= self.score_bias_anchor_bias <= 1:
            raise ValueError("Score Bias anchor bias must be within [0, 1]")
        requested_anchor_blocks = max(0, int(score_bias_anchor_max_blocks))
        self.score_bias_anchor_max_blocks = (
            min(self.score_bias_max_blocks - 1, requested_anchor_blocks)
            if self.score_bias_anchor_bias > 0
            else 0
        )
        self.score_bias_selected_blocks = min(
            self.score_bias_max_blocks - self.score_bias_anchor_max_blocks,
            max(1, int(score_bias_selected_blocks)),
        )
        self.score_bias_history_capacity = 16
        self.score_bias_query_window = min(
            self.score_bias_history_capacity, max(1, int(score_bias_query_window))
        )
        self.score_bias_min_relevance = float(score_bias_min_relevance)
        self.score_bias_relevance_margin = float(score_bias_relevance_margin)
        self._decode_slots: _DecodeSlotState | None = None
        self._q_sketches: OrderedDict[str, torch.Tensor] = OrderedDict()
        self._q_histories: OrderedDict[str, deque[torch.Tensor]] = OrderedDict()
        self._trajectory_keys: OrderedDict[
            str, tuple[tuple[tuple[float, ...], ...], torch.Tensor]
        ] = OrderedDict()
        self._memory_anchors: OrderedDict[str, tuple[torch.Tensor, int]] = OrderedDict()
        self._prefill_key_sums: OrderedDict[str, tuple[torch.Tensor, int]] = (
            OrderedDict()
        )
        self._trajectory_key_sums: OrderedDict[
            tuple[str, int], tuple[torch.Tensor, int]
        ] = OrderedDict()
        self._user_queries: OrderedDict[str, tuple[torch.Tensor, ...]] = OrderedDict()
        self._user_query_sums: OrderedDict[
            tuple[str, int], tuple[torch.Tensor, int]
        ] = OrderedDict()
        self._full_user_query_sums: OrderedDict[
            tuple[str, int], tuple[torch.Tensor, int]
        ] = OrderedDict()
        self._trajectory_shortlists: OrderedDict[
            str,
            tuple[
                tuple[tuple[tuple[float, ...], ...], tuple[tuple[float, ...], ...]],
                tuple[tuple[int, float, int], ...],
            ],
        ] = OrderedDict()

    def _ensure_decode_slots(self, device: torch.device) -> _DecodeSlotState:
        state = self._decode_slots
        if state is not None and state.observe.device == device:
            return state
        slot_count = self.max_requests + 1
        block_count = self.score_bias_max_blocks
        query_count = 8
        state = _DecodeSlotState(
            observe=torch.zeros(slot_count, dtype=torch.bool, device=device),
            previous_q=torch.zeros(
                (slot_count, self.head_dim), dtype=torch.float32, device=device
            ),
            previous_valid=torch.zeros(slot_count, dtype=torch.bool, device=device),
            last_q_drift=torch.full(
                (slot_count,), float("nan"), dtype=torch.float32, device=device
            ),
            memory_anchor=torch.zeros(
                (slot_count, self.head_dim), dtype=torch.float32, device=device
            ),
            memory_valid=torch.zeros(slot_count, dtype=torch.bool, device=device),
            history=torch.zeros(
                (
                    slot_count,
                    self.score_bias_history_capacity,
                    self.sketch_dimensions,
                ),
                dtype=torch.float32,
                device=device,
            ),
            history_count=torch.zeros(slot_count, dtype=torch.long, device=device),
            history_cursor=torch.zeros(slot_count, dtype=torch.long, device=device),
            block_starts=torch.zeros(
                (slot_count, block_count), dtype=torch.float32, device=device
            ),
            block_ends=torch.zeros(
                (slot_count, block_count), dtype=torch.float32, device=device
            ),
            block_scores=torch.zeros(
                (slot_count, block_count), dtype=torch.float32, device=device
            ),
            block_keys=torch.zeros(
                (slot_count, block_count, self.sketch_dimensions),
                dtype=torch.float32,
                device=device,
            ),
            block_valid=torch.zeros(
                (slot_count, block_count), dtype=torch.bool, device=device
            ),
            user_queries=torch.zeros(
                (slot_count, query_count, self.sketch_dimensions),
                dtype=torch.float32,
                device=device,
            ),
            user_query_valid=torch.zeros(
                (slot_count, query_count), dtype=torch.bool, device=device
            ),
            anchor_starts=torch.zeros(
                (slot_count, self.score_bias_anchor_max_blocks),
                dtype=torch.float32,
                device=device,
            ),
            anchor_ends=torch.zeros(
                (slot_count, self.score_bias_anchor_max_blocks),
                dtype=torch.float32,
                device=device,
            ),
            anchor_valid=torch.zeros(
                (slot_count, self.score_bias_anchor_max_blocks),
                dtype=torch.bool,
                device=device,
            ),
            phases=torch.zeros(slot_count, dtype=torch.long, device=device),
        )
        self._decode_slots = state
        return state

    def prepare_decode_slots(
        self,
        metadata: AttentionBatchMetadata,
        request_slots: torch.Tensor | None,
        score_bias_blocks: tuple[tuple[dict, ...] | None, ...] = (),
        score_bias_phases: tuple[int, ...] = (),
    ) -> None:
        """Publish final-prefill state into request-pool-indexed graph tensors."""

        if request_slots is None or not metadata.rids:
            return
        state = self._ensure_decode_slots(request_slots.device)
        final_mask = metadata.final_prefill_mask or tuple(
            metadata.contains_last_prefill_chunk for _ in metadata.rids
        )
        slots = request_slots[: len(metadata.rids)].detach().cpu().tolist()
        for row, rid in enumerate(metadata.rids):
            if row >= len(final_mask) or not final_mask[row]:
                continue
            slot = int(slots[row])
            if slot < 0 or slot >= self.max_requests:
                continue

            state.observe[slot] = False
            state.previous_q[slot].zero_()
            state.previous_valid[slot] = False
            state.memory_anchor[slot].zero_()
            state.memory_valid[slot] = False
            state.history[slot].zero_()
            state.history_count[slot] = 0
            state.history_cursor[slot] = 0
            state.last_q_drift[slot] = float("nan")
            state.block_starts[slot].zero_()
            state.block_ends[slot].zero_()
            state.block_scores[slot].zero_()
            state.block_keys[slot].zero_()
            state.block_valid[slot].zero_()
            state.anchor_starts[slot].zero_()
            state.anchor_ends[slot].zero_()
            state.anchor_valid[slot].zero_()
            state.user_queries[slot].zero_()
            state.user_query_valid[slot].zero_()
            state.phases[slot] = 0

            observed = row < len(metadata.observe_mask) and metadata.observe_mask[row]
            state.observe[slot] = observed
            if not observed:
                continue

            previous = self._q_sketches.get(rid)
            if previous is not None and previous.numel() == self.head_dim:
                state.previous_q[slot].copy_(previous.float())
                state.previous_valid[slot] = True

            history = tuple(self._q_histories.get(rid, ()))
            history = history[-self.score_bias_history_capacity :]
            if history:
                rows = torch.stack(tuple(value.float() for value in history)).to(
                    request_slots.device
                )
                state.history[slot, : len(history)].copy_(rows)
                state.history_count[slot] = len(history)
                state.history_cursor[slot] = (
                    len(history) % self.score_bias_history_capacity
                )

            span = (
                metadata.memory_spans[row] if row < len(metadata.memory_spans) else None
            )
            anchor_entry = (
                self._memory_anchors.get(span[2]) if span is not None else None
            )
            if anchor_entry is not None:
                anchor_sum, anchor_tokens = anchor_entry
                state.memory_anchor[slot].copy_(
                    (anchor_sum / max(anchor_tokens, 1)).float()
                )
                state.memory_valid[slot] = True

            blocks = score_bias_blocks[row] if row < len(score_bias_blocks) else None
            for block_index, block in enumerate(
                tuple(blocks or ())[: self.score_bias_max_blocks]
            ):
                key = torch.tensor(
                    tuple(block["key_sketch"]),
                    dtype=torch.float32,
                    device=request_slots.device,
                )
                if (
                    key.numel() != self.sketch_dimensions
                    or not torch.isfinite(key).all()
                ):
                    continue
                state.block_starts[slot, block_index] = float(block["start"])
                state.block_ends[slot, block_index] = float(block["end"])
                state.block_scores[slot, block_index] = float(block["score"])
                state.block_keys[slot, block_index].copy_(
                    torch.nn.functional.normalize(key, dim=0)
                )
                state.block_valid[slot, block_index] = True
            anchor_spans = (
                metadata.anchor_spans[row] if row < len(metadata.anchor_spans) else None
            )
            for anchor_index, span in enumerate(
                tuple(anchor_spans or ())[: self.score_bias_anchor_max_blocks]
            ):
                anchor_start, anchor_end = int(span[0]), int(span[1])
                if anchor_start < 0 or anchor_end <= anchor_start:
                    continue
                state.anchor_starts[slot, anchor_index] = float(anchor_start)
                state.anchor_ends[slot, anchor_index] = float(anchor_end)
                state.anchor_valid[slot, anchor_index] = True

            queries = tuple(self._user_queries.get(rid, ()))
            for query_index, query in enumerate(queries[-8:]):
                state.user_queries[slot, query_index].copy_(
                    torch.nn.functional.normalize(
                        query.to(request_slots.device).float(), dim=0
                    )
                )
                state.user_query_valid[slot, query_index] = True
            if row < len(score_bias_phases):
                state.phases[slot] = int(score_bias_phases[row])

    def _compress_rows(self, value: torch.Tensor) -> torch.Tensor:
        width = value.shape[-1]
        if width % self.sketch_dimensions == 0:
            compressed = value.reshape(
                *value.shape[:-1],
                self.sketch_dimensions,
                width // self.sketch_dimensions,
            ).mean(dim=-1)
        else:
            compressed = torch.nn.functional.adaptive_avg_pool1d(
                value.reshape(-1, 1, width), self.sketch_dimensions
            ).reshape(*value.shape[:-1], self.sketch_dimensions)
        return torch.nn.functional.normalize(compressed.float(), dim=-1)

    def observe_decode_slots(
        self,
        q: torch.Tensor,
        request_slots: torch.Tensor,
        row_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Observe decode Q with graph-resident state keyed by request-pool slot."""

        state = self._ensure_decode_slots(q.device)
        q_view = q.reshape(-1, self.num_heads, self.head_dim)
        row_count = q_view.shape[0]
        raw_slots = request_slots[:row_count].long()
        rows_enabled = row_mask[:row_count].bool()
        valid_slots = (raw_slots >= 0) & (raw_slots < self.max_requests)
        safe_slots = raw_slots.clamp(min=0, max=self.max_requests)
        configured = state.observe.index_select(0, safe_slots)
        active = rows_enabled & valid_slots & configured
        dummy_slot = torch.full_like(safe_slots, self.max_requests)
        slots = torch.where(active, safe_slots, dummy_slot)

        current = q_view.mean(dim=1, dtype=torch.float32)
        current = self._reduce(current)
        previous = state.previous_q.index_select(0, slots)
        previous_valid = state.previous_valid.index_select(0, slots)
        memory_anchor = state.memory_anchor.index_select(0, slots)
        memory_valid = state.memory_valid.index_select(0, slots)

        q_norms = current.square().mean(dim=-1).sqrt()
        q_drifts = 1 - torch.nn.functional.cosine_similarity(
            current, previous, dim=-1
        ).clamp(-1, 1)
        memory_energies = (
            torch.nn.functional.cosine_similarity(current, memory_anchor, dim=-1).clamp(
                -1, 1
            )
            + 1
        ) * 0.5
        nan_rows = torch.full_like(q_norms, float("nan"))
        q_norms = torch.where(active, q_norms, nan_rows)
        q_drifts = torch.where(active & previous_valid, q_drifts, nan_rows)
        memory_energies = torch.where(active & memory_valid, memory_energies, nan_rows)

        q_sketches = self._compress_rows(current)
        q_sketches = torch.where(
            active.unsqueeze(-1),
            q_sketches,
            torch.full_like(q_sketches, float("nan")),
        )
        k_sketches = torch.full_like(q_sketches, float("nan"))

        prior_q = state.previous_q.index_select(0, slots)
        state.previous_q.index_copy_(
            0, slots, torch.where(active.unsqueeze(-1), current, prior_q)
        )
        prior_valid = state.previous_valid.index_select(0, slots)
        state.previous_valid.index_copy_(
            0, slots, torch.where(active, torch.ones_like(active), prior_valid)
        )
        prior_drift = state.last_q_drift.index_select(0, slots)
        state.last_q_drift.index_copy_(
            0,
            slots,
            torch.where(active & previous_valid, q_drifts, prior_drift),
        )

        compressed = self._compress_rows(current)
        cursors = state.history_cursor.index_select(0, slots)
        counts = state.history_count.index_select(0, slots)
        flat_indices = slots * self.score_bias_history_capacity + cursors
        flat_history = state.history.view(-1, self.sketch_dimensions)
        prior_history = flat_history.index_select(0, flat_indices)
        flat_history.index_copy_(
            0,
            flat_indices,
            torch.where(active.unsqueeze(-1), compressed, prior_history),
        )
        next_cursors = torch.where(
            active,
            (cursors + 1) % self.score_bias_history_capacity,
            cursors,
        )
        next_counts = torch.where(
            active,
            torch.clamp(counts + 1, max=self.score_bias_history_capacity),
            counts,
        )
        state.history_cursor.index_copy_(0, slots, next_cursors)
        state.history_count.index_copy_(0, slots, next_counts)

        return {
            "qwen_exo_q_norm": q_norms,
            "qwen_exo_q_drift": q_drifts,
            "qwen_exo_memory_energy": memory_energies,
            "qwen_exo_q_sketch": q_sketches,
            "qwen_exo_k_sketch": k_sketches,
        }

    @staticmethod
    def _masked_top_mean(
        values: torch.Tensor,
        mask: torch.Tensor,
        *,
        dim: int,
        limit: int,
    ) -> torch.Tensor:
        masked = torch.where(mask, values, torch.full_like(values, float("-inf")))
        top = torch.topk(masked, k=limit, dim=dim, largest=True, sorted=False).values
        finite = torch.isfinite(top)
        total = torch.where(finite, top, torch.zeros_like(top)).sum(dim=dim)
        count = finite.sum(dim=dim)
        mean = total / count.clamp(min=1).to(total.dtype)
        return torch.where(count > 0, mean, torch.full_like(mean, float("-inf")))

    def score_bias_decode_slots(
        self,
        request_slots: torch.Tensor,
        row_mask: torch.Tensor,
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        """Select bounded trajectory blocks entirely inside decode CUDA graphs."""

        state = self._ensure_decode_slots(request_slots.device)
        row_count = row_mask.shape[0]
        raw_slots = request_slots[:row_count].long()
        rows_enabled = row_mask[:row_count].bool()
        valid_slots = (raw_slots >= 0) & (raw_slots < self.max_requests)
        safe_slots = raw_slots.clamp(min=0, max=self.max_requests)
        configured = state.observe.index_select(0, safe_slots)
        active = rows_enabled & valid_slots & configured
        slots = torch.where(
            active,
            safe_slots,
            torch.full_like(safe_slots, self.max_requests),
        )

        phases = state.phases.index_select(0, slots)
        score_active = active & (phases > 0)
        block_valid = state.block_valid.index_select(0, slots)
        block_keys = state.block_keys.index_select(0, slots)
        block_starts = state.block_starts.index_select(0, slots)
        block_ends = state.block_ends.index_select(0, slots)
        block_scores = state.block_scores.index_select(0, slots)
        anchor_starts = state.anchor_starts.index_select(0, slots)
        anchor_ends = state.anchor_ends.index_select(0, slots)
        anchor_valid = state.anchor_valid.index_select(0, slots)
        last_q_drift = state.last_q_drift.index_select(0, slots)
        user_queries = state.user_queries.index_select(0, slots)
        user_valid = state.user_query_valid.index_select(0, slots)
        candidate_count = block_valid.sum(dim=-1)
        user_count = user_valid.sum(dim=-1)

        user_scores = torch.matmul(user_queries, block_keys.transpose(1, 2))
        user_pair_valid = user_valid.unsqueeze(-1) & block_valid.unsqueeze(1)
        user_relevance = self._masked_top_mean(
            user_scores,
            user_pair_valid,
            dim=1,
            limit=min(3, user_queries.shape[1]),
        )
        user_winners = torch.argmax(
            torch.where(
                user_pair_valid,
                user_scores,
                torch.full_like(user_scores, float("-inf")),
            ),
            dim=-1,
        )
        user_consensus = (
            torch.nn.functional.one_hot(
                user_winners, num_classes=self.score_bias_max_blocks
            )
            * user_valid.unsqueeze(-1)
        ).sum(dim=1)

        shortlist_limit = min(4, self.score_bias_max_blocks)
        user_order = torch.argsort(user_relevance, dim=-1, descending=True)
        sorted_user_relevance = torch.gather(user_relevance, 1, user_order)
        shortlist_margin_ok = torch.ones_like(score_active)
        if self.score_bias_max_blocks > shortlist_limit:
            shortlist_margin_ok = (candidate_count <= shortlist_limit) | (
                sorted_user_relevance[:, shortlist_limit - 1]
                - sorted_user_relevance[:, shortlist_limit]
                >= self.score_bias_relevance_margin
            )
        shortlist_gate = (
            score_active
            & (candidate_count > 0)
            & (user_count > 0)
            & (sorted_user_relevance[:, 0] >= self.score_bias_min_relevance)
            & shortlist_margin_ok
        )
        shortlist_ranked = torch.zeros_like(block_valid)
        shortlist_ranked.scatter_(
            1,
            user_order[:, :shortlist_limit],
            torch.ones_like(user_order[:, :shortlist_limit], dtype=torch.bool),
        )
        shortlist = (
            shortlist_ranked
            & block_valid
            & shortlist_gate.unsqueeze(-1)
            & (user_relevance >= self.score_bias_min_relevance)
            & (user_consensus > 0)
        )
        shortlist_count = shortlist.sum(dim=-1)
        shortlist_best = (
            torch.where(
                shortlist,
                user_relevance,
                torch.full_like(user_relevance, float("-inf")),
            )
            .max(dim=-1)
            .values
        )
        shortlist_best = torch.where(
            shortlist_count > 0,
            shortlist_best,
            torch.full_like(shortlist_best, float("nan")),
        )

        history_count = state.history_count.index_select(0, slots)
        history_cursor = state.history_cursor.index_select(0, slots)
        offsets = torch.arange(
            self.score_bias_query_window,
            dtype=torch.long,
            device=request_slots.device,
        )
        history_positions = (
            history_cursor.unsqueeze(-1) - 1 - offsets.unsqueeze(0)
        ) % self.score_bias_history_capacity
        history = state.history[
            slots.unsqueeze(-1).expand_as(history_positions), history_positions
        ]
        history_valid = (
            offsets.unsqueeze(0) < history_count.unsqueeze(-1)
        ) & score_active.unsqueeze(-1)
        query_scores = torch.matmul(history, block_keys.transpose(1, 2))
        query_pair_valid = history_valid.unsqueeze(-1) & shortlist.unsqueeze(1)
        query_relevance = self._masked_top_mean(
            query_scores,
            query_pair_valid,
            dim=1,
            limit=min(3, self.score_bias_query_window),
        )
        query_winners = torch.argmax(
            torch.where(
                query_pair_valid,
                query_scores,
                torch.full_like(query_scores, float("-inf")),
            ),
            dim=-1,
        )
        query_consensus = (
            torch.nn.functional.one_hot(
                query_winners, num_classes=self.score_bias_max_blocks
            )
            * history_valid.unsqueeze(-1)
        ).sum(dim=1)

        query_order = torch.argsort(query_relevance, dim=-1, descending=True)
        sorted_query_relevance = torch.gather(query_relevance, 1, query_order)
        selected_limit = self.score_bias_selected_blocks
        selection_margin_ok = torch.ones_like(score_active)
        if self.score_bias_max_blocks > selected_limit:
            selection_margin_ok = (shortlist_count <= selected_limit) | (
                sorted_query_relevance[:, selected_limit - 1]
                - sorted_query_relevance[:, selected_limit]
                >= self.score_bias_relevance_margin
            )
        selection_gate = (
            score_active
            & (shortlist_count > 0)
            & (sorted_query_relevance[:, 0] >= self.score_bias_min_relevance)
            & selection_margin_ok
        )
        selected_indices = query_order[:, :selected_limit]
        selected_relevance = torch.gather(query_relevance, 1, selected_indices)
        selected_consensus = torch.gather(query_consensus, 1, selected_indices)
        selected_scores = torch.gather(block_scores, 1, selected_indices)
        selected_starts = torch.gather(block_starts, 1, selected_indices)
        selected_ends = torch.gather(block_ends, 1, selected_indices)
        selected_valid = (
            selection_gate.unsqueeze(-1)
            & (
                torch.arange(selected_limit, device=request_slots.device).unsqueeze(0)
                < shortlist_count.unsqueeze(-1)
            )
            & (selected_relevance >= self.score_bias_min_relevance)
            & (selected_consensus > 0)
        )
        effective_scores = selected_scores * selected_relevance.clamp(min=0)
        selected_valid = selected_valid & (effective_scores > 0)
        effective_scores = torch.where(
            selected_valid, effective_scores, torch.zeros_like(effective_scores)
        )
        selected_count = selected_valid.sum(dim=-1)
        max_relevance = (
            torch.where(
                selected_valid,
                selected_relevance,
                torch.full_like(selected_relevance, float("-inf")),
            )
            .max(dim=-1)
            .values
        )
        max_relevance = torch.where(
            selected_count > 0,
            max_relevance,
            torch.full_like(max_relevance, float("nan")),
        )
        max_consensus = (
            torch.where(
                selected_valid,
                selected_consensus,
                torch.zeros_like(selected_consensus),
            )
            .max(dim=-1)
            .values
        )
        max_bias = effective_scores.max(dim=-1).values

        phase_active = score_active & (phases == 2)
        anchor_apply = (
            anchor_valid
            & phase_active.unsqueeze(-1)
            & (last_q_drift >= self.score_bias_anchor_drift_threshold).unsqueeze(-1)
        )
        anchor_count = anchor_apply.sum(dim=-1).float()
        anchor_triplets = torch.stack(
            (
                torch.where(
                    anchor_apply, anchor_starts, torch.zeros_like(anchor_starts)
                ),
                torch.where(anchor_apply, anchor_ends, torch.zeros_like(anchor_ends)),
                torch.where(
                    anchor_apply,
                    torch.full_like(anchor_starts, self.score_bias_anchor_bias),
                    torch.zeros_like(anchor_starts),
                ),
            ),
            dim=-1,
        )
        apply = selected_valid & phase_active.unsqueeze(-1)
        triplets = torch.stack(
            (
                torch.where(apply, selected_starts, torch.zeros_like(selected_starts)),
                torch.where(apply, selected_ends, torch.zeros_like(selected_ends)),
                torch.where(
                    apply, effective_scores, torch.zeros_like(effective_scores)
                ),
            ),
            dim=-1,
        )
        aux = torch.zeros(
            (row_count, 1, 3 * self.score_bias_max_blocks),
            dtype=torch.float32,
            device=request_slots.device,
        )
        if self.score_bias_anchor_max_blocks:
            aux[:, 0, : 3 * self.score_bias_anchor_max_blocks].copy_(
                anchor_triplets.reshape(row_count, -1)
            )
        start_slot = 3 * self.score_bias_anchor_max_blocks
        end_slot = start_slot + 3 * self.score_bias_selected_blocks
        aux[:, 0, start_slot:end_slot].copy_(triplets.reshape(row_count, -1))

        info = {
            "qwen_exo_score_bias_phase": torch.where(
                rows_enabled, phases, torch.zeros_like(phases)
            ).float(),
            "qwen_exo_score_bias_is_decode": rows_enabled.float(),
            "qwen_exo_score_bias_candidate_count": candidate_count.float(),
            "qwen_exo_score_bias_user_query_count": user_count.float(),
            "qwen_exo_score_bias_shortlist_count": shortlist_count.float(),
            "qwen_exo_score_bias_shortlist_max_relevance": shortlist_best,
            "qwen_exo_score_bias_anchor_drift": last_q_drift,
            "qwen_exo_score_bias_anchor_count": anchor_count,
            "qwen_exo_score_bias_anchor_bias": torch.where(
                anchor_count > 0,
                torch.full_like(anchor_count, self.score_bias_anchor_bias),
                torch.zeros_like(anchor_count),
            ),
            "qwen_exo_score_bias_selected_count": selected_count.float(),
            "qwen_exo_score_bias_max_relevance": max_relevance,
            "qwen_exo_score_bias_would_apply_max": max_bias,
            "qwen_exo_score_bias_query_consensus": max_consensus.float(),
        }
        return info, aux

    def register_memory_keys(self, memory_key: str, keys: torch.Tensor) -> None:
        key = str(memory_key)
        if not key or keys.numel() == 0:
            return
        view = keys.reshape(-1, self.num_kv_heads, self.head_dim)
        token_count = int(view.shape[0])
        anchor_sum = view.sum(dim=(0, 1), dtype=torch.float32) / max(
            self.num_kv_heads, 1
        )
        anchor_sum = self._reduce(anchor_sum)
        self._memory_anchors[key] = (anchor_sum.detach(), token_count)
        self._memory_anchors.move_to_end(key)
        while len(self._memory_anchors) > self.max_memory_anchors:
            self._memory_anchors.popitem(last=False)

    def observe(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        metadata: AttentionBatchMetadata,
    ) -> dict[str, torch.Tensor] | None:
        if (
            not metadata.rids
            or not metadata.observe_mask
            or not any(metadata.observe_mask)
        ):
            return None
        self._register_persisted_user_queries(metadata)

        q_view = q.reshape(-1, self.num_heads, self.head_dim)
        selected_rows: list[int] = []
        q_indices: list[int] = []
        prefill_key_sketches: dict[int, torch.Tensor] = {}
        full_user_queries = None
        if metadata.is_decode:
            selected_rows = list(range(min(len(metadata.rids), q_view.shape[0])))
            q_indices = selected_rows[:]
        elif metadata.is_extend:
            self._capture_memory_keys(k, metadata)
            if not metadata.extend_lens:
                return None
            final_mask = metadata.final_prefill_mask or tuple(
                metadata.contains_last_prefill_chunk for _ in metadata.rids
            )
            trajectory_key_sketches = self._capture_trajectory_keys(
                k, metadata, final_mask
            )
            prefill_key_sketches = self._capture_prefill_keys(k, metadata, final_mask)
            user_query_sketches = self._capture_user_queries(q, metadata, final_mask)
            full_user_queries = self._capture_full_user_queries(q, metadata, final_mask)
            offset = 0
            for row, extend_len in enumerate(
                metadata.extend_lens[: len(metadata.rids)]
            ):
                if row < len(final_mask) and final_mask[row]:
                    selected_rows.append(row)
                    q_indices.append(offset + max(0, int(extend_len) - 1))
                offset += int(extend_len)
        else:
            return None
        if not q_indices or max(q_indices) >= q_view.shape[0]:
            return None

        index_tensor = torch.tensor(q_indices, dtype=torch.long, device=q.device)
        selected_sketches = q_view.index_select(0, index_tensor).mean(
            dim=1, dtype=torch.float32
        )
        selected_sketches = self._reduce(selected_sketches)
        row_count = len(metadata.rids)
        q_norms = selected_sketches.new_full((row_count,), float("nan"))
        q_drifts = selected_sketches.new_full((row_count,), float("nan"))
        memory_energies = selected_sketches.new_full((row_count,), float("nan"))

        for selected_index, row in enumerate(selected_rows):
            if row >= len(metadata.observe_mask) or not metadata.observe_mask[row]:
                continue
            rid = metadata.rids[row]
            current = selected_sketches[selected_index]
            q_norms[row] = current.square().mean().sqrt()
            previous = self._q_sketches.get(rid)
            if previous is not None:
                q_drifts[row] = 1 - torch.cosine_similarity(
                    current.unsqueeze(0), previous.unsqueeze(0), dim=-1
                )[0].clamp(-1, 1)
            self._q_sketches[rid] = current.detach().clone()
            history = self._q_histories.setdefault(rid, deque(maxlen=16))
            history.append(self._compress(current).detach().clone())
            self._q_histories.move_to_end(rid)
            self._q_sketches.move_to_end(rid)

            span = (
                metadata.memory_spans[row] if row < len(metadata.memory_spans) else None
            )
            if span is not None:
                anchor_entry = self._memory_anchors.get(span[2])
                if anchor_entry is not None:
                    anchor_sum, anchor_tokens = anchor_entry
                    anchor = anchor_sum / max(anchor_tokens, 1)
                    similarity = torch.cosine_similarity(
                        current.unsqueeze(0), anchor.unsqueeze(0), dim=-1
                    )[0].clamp(-1, 1)
                    memory_energies[row] = (similarity + 1) * 0.5

        while len(self._q_sketches) > self.max_requests:
            self._q_sketches.popitem(last=False)
        q_sketches = selected_sketches.new_full(
            (row_count, self.sketch_dimensions), float("nan")
        )
        k_sketches = selected_sketches.new_full(
            (row_count, self.sketch_dimensions), float("nan")
        )
        for selected_index, row in enumerate(selected_rows):
            if row >= len(metadata.observe_mask) or not metadata.observe_mask[row]:
                continue
            q_sketches[row] = self._compress(selected_sketches[selected_index])
            if row in prefill_key_sketches:
                k_sketches[row] = prefill_key_sketches[row]
        result = {
            "qwen_exo_q_norm": q_norms,
            "qwen_exo_q_drift": q_drifts,
            "qwen_exo_memory_energy": memory_energies,
            "qwen_exo_q_sketch": q_sketches,
            "qwen_exo_k_sketch": k_sketches,
        }
        if metadata.is_extend and trajectory_key_sketches is not None:
            result["qwen_exo_trajectory_k_sketch"] = trajectory_key_sketches
        if metadata.is_extend and user_query_sketches is not None:
            result["qwen_exo_user_query_sketch"] = user_query_sketches
        if metadata.is_extend and full_user_queries is not None:
            result["qwen_exo_user_query_full_heads"] = full_user_queries
        return result

    def _capture_prefill_keys(
        self,
        k: torch.Tensor,
        metadata: AttentionBatchMetadata,
        final_mask: tuple[bool, ...],
    ) -> dict[int, torch.Tensor]:
        if not metadata.extend_lens:
            return {}
        k_view = k.reshape(-1, self.num_kv_heads, self.head_dim)
        completed: dict[int, torch.Tensor] = {}
        offset = 0
        for row, extend_len in enumerate(metadata.extend_lens[: len(metadata.rids)]):
            extend_len = int(extend_len)
            if row >= len(metadata.observe_mask) or not metadata.observe_mask[row]:
                offset += extend_len
                continue
            segment = k_view[offset : offset + extend_len]
            offset += extend_len
            if not segment.numel():
                continue
            rid = metadata.rids[row]
            chunk_sum = segment.sum(dim=0, dtype=torch.float32).mean(dim=0)
            chunk_sum = self._reduce(chunk_sum)
            previous = self._prefill_key_sums.get(rid)
            if previous is None:
                total_sum = chunk_sum
                total_tokens = extend_len
            else:
                total_sum = previous[0] + chunk_sum
                total_tokens = previous[1] + extend_len
            self._prefill_key_sums[rid] = (total_sum.detach(), total_tokens)
            self._prefill_key_sums.move_to_end(rid)
            if row < len(final_mask) and final_mask[row]:
                completed[row] = self._compress(total_sum / max(total_tokens, 1))
                self._prefill_key_sums.pop(rid, None)
        while len(self._prefill_key_sums) > self.max_requests:
            self._prefill_key_sums.popitem(last=False)
        return completed

    def _capture_trajectory_keys(
        self,
        k: torch.Tensor,
        metadata: AttentionBatchMetadata,
        final_mask: tuple[bool, ...],
    ) -> torch.Tensor | None:
        if not metadata.extend_lens or metadata.prefix_lens is None:
            return None
        max_spans = max(
            (len(spans or ()) for spans in metadata.trajectory_spans), default=0
        )
        if max_spans < 1:
            return None
        row_count = len(metadata.rids)
        output = k.new_full(
            (row_count, max_spans, self.sketch_dimensions),
            float("nan"),
            dtype=torch.float32,
        )
        k_view = k.reshape(-1, self.num_kv_heads, self.head_dim)
        offset = 0
        for row, extend_len in enumerate(metadata.extend_lens[:row_count]):
            extend_len = int(extend_len)
            prefix_len = int(metadata.prefix_lens[row])
            spans = (
                metadata.trajectory_spans[row]
                if row < len(metadata.trajectory_spans)
                else None
            )
            for span_index, span in enumerate(spans or ()):
                span_start, span_end = (int(span[0]), int(span[1]))
                overlap_start = max(span_start, prefix_len)
                overlap_end = min(span_end, prefix_len + extend_len)
                key = (metadata.rids[row], span_index)
                if overlap_start < overlap_end:
                    local_start = offset + overlap_start - prefix_len
                    local_end = offset + overlap_end - prefix_len
                    segment = k_view[local_start:local_end]
                    chunk_sum = segment.sum(dim=(0, 1), dtype=torch.float32) / max(
                        self.num_kv_heads, 1
                    )
                    chunk_sum = self._reduce(chunk_sum)
                    previous = self._trajectory_key_sums.get(key)
                    if previous is None:
                        total_sum = chunk_sum
                        total_tokens = overlap_end - overlap_start
                    else:
                        total_sum = previous[0] + chunk_sum
                        total_tokens = previous[1] + overlap_end - overlap_start
                    self._trajectory_key_sums[key] = (
                        total_sum.detach(),
                        total_tokens,
                    )
                if row < len(final_mask) and final_mask[row]:
                    captured = self._trajectory_key_sums.pop(key, None)
                    if captured is not None and captured[1] == span_end - span_start:
                        output[row, span_index] = self._compress(
                            captured[0] / max(captured[1], 1)
                        )
            offset += extend_len
        while len(self._trajectory_key_sums) > self.max_requests * max_spans:
            self._trajectory_key_sums.popitem(last=False)
        return output

    def _register_persisted_user_queries(
        self, metadata: AttentionBatchMetadata
    ) -> None:
        for row, raw_queries in enumerate(metadata.persisted_user_queries):
            if not raw_queries or row >= len(metadata.rids):
                continue
            rid = metadata.rids[row]
            if rid in self._user_queries:
                continue
            queries = tuple(
                torch.nn.functional.normalize(
                    torch.tensor(query, dtype=torch.float32, device="cpu"), dim=0
                )
                for query in raw_queries
                if len(query) == self.sketch_dimensions
            )
            if queries:
                self._user_queries[rid] = queries
                self._user_queries.move_to_end(rid)

    def _capture_user_queries(
        self,
        q: torch.Tensor,
        metadata: AttentionBatchMetadata,
        final_mask: tuple[bool, ...],
    ) -> torch.Tensor | None:
        if not metadata.extend_lens or metadata.prefix_lens is None:
            return None
        max_span_count = max(
            (len(spans or ()) for spans in metadata.user_query_spans), default=0
        )
        max_persisted_count = max(
            (len(queries or ()) for queries in metadata.persisted_user_queries),
            default=0,
        )
        max_spans = min(8, max_span_count + max_persisted_count)
        if max_spans < 1:
            return None
        row_count = len(metadata.rids)
        output = q.new_full(
            (row_count, max_spans, self.sketch_dimensions),
            float("nan"),
            dtype=torch.float32,
        )
        q_view = q.reshape(-1, self.num_heads, self.head_dim)
        offset = 0
        for row, extend_len in enumerate(metadata.extend_lens[:row_count]):
            extend_len = int(extend_len)
            prefix_len = int(metadata.prefix_lens[row])
            spans = (
                metadata.user_query_spans[row]
                if row < len(metadata.user_query_spans)
                else None
            )
            completed: list[torch.Tensor] = []
            for span_index, span in enumerate(spans or ()):
                span_start, span_end = int(span[0]), int(span[1])
                overlap_start = max(span_start, prefix_len)
                overlap_end = min(span_end, prefix_len + extend_len)
                key = (metadata.rids[row], span_index)
                if overlap_start < overlap_end:
                    local_start = offset + overlap_start - prefix_len
                    local_end = offset + overlap_end - prefix_len
                    segment = q_view[local_start:local_end]
                    chunk_sum = segment.sum(dim=(0, 1), dtype=torch.float32) / max(
                        self.num_heads, 1
                    )
                    chunk_sum = self._reduce(chunk_sum)
                    previous = self._user_query_sums.get(key)
                    total_sum = (
                        chunk_sum if previous is None else previous[0] + chunk_sum
                    )
                    total_tokens = (
                        overlap_end - overlap_start
                        if previous is None
                        else previous[1] + overlap_end - overlap_start
                    )
                    self._user_query_sums[key] = (total_sum.detach(), total_tokens)
                if row < len(final_mask) and final_mask[row]:
                    captured = self._user_query_sums.pop(key, None)
                    if captured is not None and captured[1] == span_end - span_start:
                        sketch = self._compress(captured[0] / max(captured[1], 1))
                        output[row, span_index] = sketch
                        completed.append(sketch.detach().clone())
            if completed:
                rid = metadata.rids[row]
                merged = [
                    query.to(q.device) for query in self._user_queries.get(rid, ())
                ]
                for query in completed:
                    if not any(
                        torch.allclose(query, prior, rtol=1e-5, atol=1e-6)
                        for prior in merged
                    ):
                        merged.append(query)
                self._user_queries[rid] = tuple(merged[-max_spans:])
                self._user_queries.move_to_end(rid)
                self._trajectory_shortlists.pop(rid, None)
            if row < len(final_mask) and final_mask[row]:
                for query_index, query in enumerate(
                    self._user_queries.get(metadata.rids[row], ())[:max_spans]
                ):
                    output[row, query_index] = query.to(output.device)
            offset += extend_len
        while len(self._user_query_sums) > self.max_requests * max_spans:
            self._user_query_sums.popitem(last=False)
        while len(self._user_queries) > self.max_requests:
            stale, _queries = self._user_queries.popitem(last=False)
            self._trajectory_shortlists.pop(stale, None)
        return output

    def _capture_full_user_queries(
        self,
        q: torch.Tensor,
        metadata: AttentionBatchMetadata,
        final_mask: tuple[bool, ...],
    ) -> torch.Tensor | None:
        if (
            not metadata.extend_lens
            or metadata.prefix_lens is None
            or not any(metadata.full_query_capture)
        ):
            return None
        max_spans = min(
            8,
            max((len(spans or ()) for spans in metadata.user_query_spans), default=0),
        )
        if max_spans < 1:
            return None
        row_count = len(metadata.rids)
        output = q.new_full(
            (row_count, max_spans, self.total_num_heads, self.head_dim),
            float("nan"),
            dtype=torch.float32,
        )
        q_view = q.reshape(-1, self.num_heads, self.head_dim)
        offset = 0
        for row, extend_len in enumerate(metadata.extend_lens[:row_count]):
            extend_len = int(extend_len)
            if (
                row >= len(metadata.full_query_capture)
                or not metadata.full_query_capture[row]
            ):
                offset += extend_len
                continue
            prefix_len = int(metadata.prefix_lens[row])
            spans = (
                metadata.user_query_spans[row]
                if row < len(metadata.user_query_spans)
                else None
            )
            for span_index, span in enumerate((spans or ())[:max_spans]):
                span_start, span_end = int(span[0]), int(span[1])
                overlap_start = max(span_start, prefix_len)
                overlap_end = min(span_end, prefix_len + extend_len)
                key = (metadata.rids[row], span_index)
                if overlap_start < overlap_end:
                    local_start = offset + overlap_start - prefix_len
                    local_end = offset + overlap_end - prefix_len
                    chunk_sum = q_view[local_start:local_end].sum(
                        dim=0, dtype=torch.float32
                    )
                    previous = self._full_user_query_sums.get(key)
                    total_sum = (
                        chunk_sum if previous is None else previous[0] + chunk_sum
                    )
                    total_tokens = (
                        overlap_end - overlap_start
                        if previous is None
                        else previous[1] + overlap_end - overlap_start
                    )
                    self._full_user_query_sums[key] = (
                        total_sum.detach(),
                        total_tokens,
                    )
                if row < len(final_mask) and final_mask[row]:
                    captured = self._full_user_query_sums.pop(key, None)
                    if captured is not None and captured[1] == span_end - span_start:
                        heads = captured[0] / max(captured[1], 1)
                        if self.gather_heads_across_tp is not None:
                            heads = self.gather_heads_across_tp(heads.contiguous())
                        if tuple(heads.shape) != (
                            self.total_num_heads,
                            self.head_dim,
                        ):
                            raise RuntimeError(
                                "Full Q-head capture returned an invalid TP shape"
                            )
                        output[row, span_index].copy_(heads)
            offset += extend_len
        while len(self._full_user_query_sums) > self.max_requests * max_spans:
            self._full_user_query_sums.popitem(last=False)
        return output

    def user_query_count(self, request_id: str) -> int:
        return len(self._user_queries.get(str(request_id), ()))

    def shortlist_trajectory_keys(
        self,
        request_id: str,
        key_sketches: tuple[tuple[float, ...], ...],
        *,
        limit: int,
        min_score: float,
        margin: float,
    ) -> tuple[tuple[int, float, int], ...]:
        rid = str(request_id)
        queries = self._user_queries.get(rid)
        if not queries or not key_sketches or limit < 1:
            return ()
        signature = tuple(tuple(float(value) for value in row) for row in key_sketches)
        query_signature = tuple(
            tuple(float(value) for value in query.tolist()) for query in queries
        )
        cache_key = (signature, query_signature)
        cached = self._trajectory_shortlists.get(rid)
        if cached is not None and cached[0] == cache_key:
            self._trajectory_shortlists.move_to_end(rid)
            return cached[1]
        device = queries[-1].device
        keys = torch.nn.functional.normalize(
            torch.tensor(signature, dtype=torch.float32, device=device), dim=-1
        )
        query_tensor = torch.stack(tuple(query.to(device) for query in queries))
        query_tensor = torch.nn.functional.normalize(query_tensor.float(), dim=-1)
        scores = query_tensor @ keys.transpose(0, 1)
        top_r = min(3, int(scores.shape[0]))
        aggregate = torch.topk(
            scores, k=top_r, dim=0, largest=True, sorted=False
        ).values.mean(dim=0)
        order = torch.argsort(aggregate, descending=True)
        accepted = order[: min(int(limit), int(order.numel()))]
        result: tuple[tuple[int, float, int], ...] = ()
        if accepted.numel() and float(aggregate[accepted[0]].item()) >= min_score:
            enough_margin = True
            if order.numel() > accepted.numel():
                enough_margin = float(aggregate[accepted[-1]].item()) - float(
                    aggregate[order[accepted.numel()]].item()
                ) >= float(margin)
            if enough_margin:
                winners = torch.argmax(scores, dim=1)
                result = tuple(
                    (
                        int(index),
                        float(aggregate[index].item()),
                        int((winners == index).sum().item()),
                    )
                    for index in accepted.tolist()
                    if float(aggregate[index].item()) >= min_score
                    and int((winners == index).sum().item()) > 0
                )
        self._trajectory_shortlists[rid] = (cache_key, result)
        self._trajectory_shortlists.move_to_end(rid)
        while len(self._trajectory_shortlists) > self.max_requests:
            self._trajectory_shortlists.popitem(last=False)
        return result

    def rank_trajectory_keys(
        self,
        request_id: str,
        key_sketches: tuple[tuple[float, ...], ...],
        *,
        limit: int,
        query_window: int,
        min_score: float,
        margin: float,
        allowed_indices: tuple[int, ...] | None = None,
    ) -> tuple[tuple[int, float, int], ...]:
        history = self._q_histories.get(str(request_id))
        if not history or not key_sketches or limit < 1:
            return ()
        signature = tuple(tuple(float(value) for value in row) for row in key_sketches)
        cached = self._trajectory_keys.get(str(request_id))
        if cached is None or cached[0] != signature:
            keys = torch.nn.functional.normalize(
                torch.tensor(signature, dtype=torch.float32, device=history[-1].device),
                dim=-1,
            )
            self._trajectory_keys[str(request_id)] = (signature, keys)
        else:
            keys = cached[1]
        self._trajectory_keys.move_to_end(str(request_id))
        candidate_indices = tuple(
            index
            for index in (
                allowed_indices
                if allowed_indices is not None
                else tuple(range(len(key_sketches)))
            )
            if 0 <= index < len(key_sketches)
        )
        if not candidate_indices:
            return ()
        index_tensor = torch.tensor(
            candidate_indices, dtype=torch.long, device=keys.device
        )
        candidate_keys = keys.index_select(0, index_tensor)
        queries = torch.stack(tuple(history)[-max(1, int(query_window)) :])
        queries = torch.nn.functional.normalize(queries.float(), dim=-1)
        query_scores = queries @ candidate_keys.transpose(0, 1)
        top_r = min(3, int(query_scores.shape[0]))
        aggregate = torch.topk(
            query_scores, k=top_r, dim=0, largest=True, sorted=False
        ).values.mean(dim=0)
        order = torch.argsort(aggregate, descending=True)
        accepted = order[: min(int(limit), int(order.numel()))]
        if not accepted.numel() or float(aggregate[accepted[0]].item()) < min_score:
            return ()
        if order.numel() > accepted.numel():
            selected_floor = float(aggregate[accepted[-1]].item())
            rejected_ceiling = float(aggregate[order[accepted.numel()]].item())
            if selected_floor - rejected_ceiling < float(margin):
                return ()
        winners = torch.argmax(query_scores, dim=1)
        result: list[tuple[int, float, int]] = []
        for local_index in accepted.tolist():
            score = float(aggregate[local_index].item())
            consensus = int((winners == local_index).sum().item())
            if score >= min_score and consensus > 0:
                result.append((candidate_indices[local_index], score, consensus))
        while len(self._q_histories) > self.max_requests:
            stale, _history = self._q_histories.popitem(last=False)
            self._trajectory_keys.pop(stale, None)
            self._user_queries.pop(stale, None)
            self._trajectory_shortlists.pop(stale, None)
        return tuple(result)

    def _capture_memory_keys(
        self, k: torch.Tensor, metadata: AttentionBatchMetadata
    ) -> None:
        if (
            not metadata.is_extend
            or not metadata.memory_spans
            or not metadata.extend_lens
            or metadata.prefix_lens is None
        ):
            return
        k_view = k.reshape(-1, self.num_kv_heads, self.head_dim)
        offset = 0
        for index, extend_len in enumerate(
            metadata.extend_lens[: len(metadata.memory_spans)]
        ):
            extend_len = int(extend_len)
            span = metadata.memory_spans[index]
            if span is not None:
                memory_start, memory_length, memory_key = span
                prefix_len = int(metadata.prefix_lens[index])
                overlap_start = max(memory_start, prefix_len)
                overlap_end = min(memory_start + memory_length, prefix_len + extend_len)
                if overlap_start < overlap_end:
                    local_start = offset + overlap_start - prefix_len
                    local_end = offset + overlap_end - prefix_len
                    anchor = k_view[local_start:local_end].mean(
                        dim=(0, 1), dtype=torch.float32
                    )
                    anchor = self._reduce(anchor)
                    token_count = overlap_end - overlap_start
                    previous = self._memory_anchors.get(memory_key)
                    if previous is None:
                        anchor_sum = anchor * token_count
                        total_tokens = token_count
                    else:
                        anchor_sum = previous[0] + anchor * token_count
                        total_tokens = previous[1] + token_count
                    self._memory_anchors[memory_key] = (
                        anchor_sum.detach(),
                        total_tokens,
                    )
                    self._memory_anchors.move_to_end(memory_key)
            offset += extend_len
        while len(self._memory_anchors) > self.max_memory_anchors:
            self._memory_anchors.popitem(last=False)

    def _compress(self, value: torch.Tensor) -> torch.Tensor:
        width = value.shape[-1]
        if width % self.sketch_dimensions == 0:
            compressed = value.reshape(
                self.sketch_dimensions, width // self.sketch_dimensions
            ).mean(dim=-1)
        else:
            compressed = torch.nn.functional.adaptive_avg_pool1d(
                value.view(1, 1, -1), self.sketch_dimensions
            ).view(-1)
        return torch.nn.functional.normalize(compressed.float(), dim=0)

    def _reduce(self, value: torch.Tensor) -> torch.Tensor:
        return self.reduce_across_tp(value) if self.reduce_across_tp else value
