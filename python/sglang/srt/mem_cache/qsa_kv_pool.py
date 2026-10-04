"""KV pools carrying the QSA sparse-attention indexer caches.

``QSATokenToKVPool`` (compressed, Qwen4-Exp) adds the per-token BF16 index-key
state, its RoPE coordinates, and the paged compressed-K cache on top of the
hybrid full/linear KV pool. ``QwenDSATokenToKVPool`` (tokenwise,
Qwen3Next-DSA) adds only the flat per-token index-K cache.
"""

from __future__ import annotations

from typing import List, Optional

import torch
import triton
import triton.language as tl

from sglang.srt.mem_cache.memory_pool import GB, HybridLinearKVPool, MambaPool


def _index_k_bytes(*, kv_heads: int, head_dim: int, dtype: torch.dtype) -> int:
    return kv_heads * head_dim * dtype.itemsize


@triton.jit
def _qsa_verify_plan_kernel(
    requests, positions, group_ends, group_rows, row_slots, group_slots,
    staged_positions, num_rows, num_groups,
    RATIO: tl.constexpr, DRAFT: tl.constexpr, RING_SLOTS: tl.constexpr,
):
    row = tl.program_id(0)
    if row < num_rows:
        req = tl.load(requests + row).to(tl.int64)
        step = row % DRAFT
        slot = req * RATIO + step
        position = tl.load(positions + row)
        tl.store(row_slots + row, RING_SLOTS + slot)
        tl.store(staged_positions + slot, position)
    if row < num_groups:
        source_row = tl.load(group_rows + row).to(tl.int64)
        req = tl.load(requests + source_row).to(tl.int64)
        prefix = tl.load(positions + source_row) - source_row % DRAFT
        end = tl.load(group_ends + row)
        for offset in tl.static_range(RATIO):
            member = tl.maximum(end - (RATIO - 1 - offset), 0)
            candidate = (req > 0) & (member >= prefix)
            slot = tl.where(
                candidate,
                RING_SLOTS + req * RATIO + member - prefix,
                req * RATIO + member % RATIO,
            )
            tl.store(group_slots + row * RATIO + offset, slot.to(tl.int32))


@triton.jit
def _qsa_verify_commit_kernel(
    key_state, rope_state, staged_positions, requests, accept_lens, accept_index,
    index_row_stride, index_col_stride,
    RATIO: tl.constexpr, DRAFT: tl.constexpr, RING_SLOTS: tl.constexpr,
    KEY_WIDTH: tl.constexpr, KEY_BLOCK: tl.constexpr, ACCEPT_WIDTH: tl.constexpr,
):
    row = tl.program_id(0)
    layer = tl.program_id(1)
    req = tl.load(requests + row).to(tl.int64)
    accepted = tl.load(accept_lens + row)
    key_offsets = tl.arange(0, KEY_BLOCK)
    for ordinal in tl.static_range(ACCEPT_WIDTH):
        active = (req > 0) & (ordinal < accepted)
        step = tl.load(
            accept_index + row * index_row_stride + ordinal * index_col_stride,
            mask=active, other=0,
        ).to(tl.int64) - row * DRAFT
        active = active & (step >= 0) & (step < DRAFT)
        candidate = req * RATIO + step
        position = tl.load(staged_positions + candidate, mask=active, other=0)
        live = req * RATIO + position % RATIO
        source = RING_SLOTS + candidate
        layer_base = layer.to(tl.int64) * (2 * RING_SLOTS) * KEY_WIDTH
        key = tl.load(
            key_state + layer_base + source * KEY_WIDTH + key_offsets,
            mask=active & (key_offsets < KEY_WIDTH), other=0,
        )
        tl.store(
            key_state + layer_base + live * KEY_WIDTH + key_offsets,
            key, mask=active & (key_offsets < KEY_WIDTH),
        )
        if layer == 0:
            for axis in tl.static_range(3):
                coord = tl.load(rope_state + source * 3 + axis, mask=active, other=0)
                tl.store(rope_state + live * 3 + axis, coord, mask=active)


class QSATokenToKVPool(HybridLinearKVPool):
    """Hybrid KV pool with the minimal BF16 state required by simple QSA."""

    # DSV4-style compressed addressing: the full-KV allocator is paged with
    # the page a multiple of the compress ratio, so every compression
    # group's raw tokens are contiguous in one page and
    # ``compressed_slot = full_slot // compress_ratio`` is a stable,
    # collision-free mapping (any of the group's slots floor-divides to the
    # same value). The compressed cache therefore has exactly
    # ``full_slots // ratio`` slots, its lifecycle rides the full-KV
    # allocator and radix tree (page-granular sharing shares compressed
    # slots by arithmetic), and no ownership bookkeeping exists. Full slot 0
    # is the pools' reserved padding slot, so compressed slot 0 stays the
    # inert dump target for non-boundary rows.
    index_state_dtype = torch.bfloat16

    @classmethod
    def qsa_bytes_per_token(
        cls, *, kv_heads: int, head_dim: int, compress_ratio: int, num_layers: int
    ) -> int:
        """Per-token cost of the QSA index caches: the compressed keys only.

        Pre-compression state is a per-request ring of ``compress_ratio``
        slots (the pending group's members), not a per-token cache, so it
        does not price per token; its total is bounded by the request-slot
        count and stays outside this budget like the other per-request
        buffers.
        """
        index_k_bytes = _index_k_bytes(
            kv_heads=kv_heads, head_dim=head_dim, dtype=cls.index_state_dtype
        )
        return index_k_bytes // compress_ratio * num_layers


    def __init__(
        self,
        *,
        size: int,
        dtype: torch.dtype,
        page_size: int,
        head_num: int,
        head_dim: int,
        full_attention_layer_ids: List[int],
        device: str,
        mamba_pool: MambaPool,
        qsa_index_kv_heads: int,
        qsa_index_head_dim: int,
        qsa_compress_ratio: int,
        qsa_token_topk: int,
        num_request_slots: int,
        enable_memory_saver: bool = False,
        enable_kv_cache_copy: bool = False,
        start_layer: Optional[int] = None,
        full_kv_pool_class: Optional[type] = None,
        quant_method=None,
        post_capture_active: bool = False,
        full_kv_pool_gpu_size: int = 0,
    ):
        if page_size <= 1 or page_size % qsa_compress_ratio != 0:
            raise ValueError(
                "compressed QSA requires a paged full-KV cache with the page "
                "a multiple of the compress ratio (compressed slots are "
                f"full_slot // ratio): page_size={page_size}, "
                f"ratio={qsa_compress_ratio}. With MambaRadixCache this "
                "needs the mamba extra-buffer strategy or "
                "--disable-radix-cache (see the Qwen4-Exp arg overrides)."
            )
        # The base __init__ computes mem_usage through the overridden
        # get_kv_size_bytes before the QSA buffers exist; give them empty
        # placeholders first and recompute mem_usage at the end.
        self.qsa_key_state_buffer_pool = []
        self.qsa_compressed_k_buffer_pool = []
        self.qsa_key_state_flat = torch.empty(0)
        self.qsa_rope_state = torch.empty(0, dtype=torch.int64)
        self.qsa_verify_row_slots = torch.empty(0, dtype=torch.int64)
        self.qsa_verify_group_locs = torch.empty(0, dtype=torch.int32)
        self.qsa_verify_positions = torch.empty(0, dtype=torch.int64)
        self.qsa_rope_position_buffer = torch.empty(0)
        super().__init__(
            size=size,
            dtype=dtype,
            page_size=page_size,
            head_num=head_num,
            head_dim=head_dim,
            full_attention_layer_ids=full_attention_layer_ids,
            device=device,
            mamba_pool=mamba_pool,
            enable_memory_saver=enable_memory_saver,
            enable_kv_cache_copy=enable_kv_cache_copy,
            use_mla=False,
            start_layer=start_layer,
            full_kv_pool_class=full_kv_pool_class,
            quant_method=quant_method,
            post_capture_active=post_capture_active,
            full_kv_pool_gpu_size=full_kv_pool_gpu_size,
        )
        if (
            min(
                qsa_index_kv_heads,
                qsa_index_head_dim,
                qsa_compress_ratio,
                qsa_token_topk,
            )
            <= 0
        ):
            raise ValueError("QSA cache configuration values must be positive")
        if qsa_token_topk % qsa_compress_ratio != 0:
            raise ValueError("qsa_token_topk must be divisible by qsa_compress_ratio")
        self.qsa_compress_ratio = int(qsa_compress_ratio)
        self.qsa_index_head_dim = int(qsa_index_head_dim)
        self.qsa_index_kv_heads = int(qsa_index_kv_heads)
        self.qsa_token_topk = int(qsa_token_topk)
        self.qsa_block_topk = self.qsa_token_topk // self.qsa_compress_ratio
        state_size = size + page_size
        # Compressed slots mirror the full-KV slot space 1:ratio; the "page"
        # seen by the scoring kernels is one full-KV page's worth of groups.
        self.qsa_compressed_page_size = page_size // self.qsa_compress_ratio
        self.qsa_compressed_capacity = -(state_size // -self.qsa_compress_ratio)
        # Pre-compression index-K state is a per-request RING, not a
        # per-token cache: once a group's compressed key is written, its raw
        # members are never read again, and page-granular prefix sharing
        # keeps every extend chunk group-aligned, so the only state that
        # must survive a forward is the pending group's members -- at most
        # ``ratio`` tokens per request, addressed as
        # ``req_pool_idx * ratio + position % ratio``. Request slot 0 is
        # never allocated, so ring rows [0, ratio) double as the inert dump
        # for tokens whose group already compressed in the same forward.
        if num_request_slots <= 0:
            raise ValueError(
                f"QSA pending ring needs request slots, got {num_request_slots}"
            )
        self.qsa_num_request_slots = int(num_request_slots)
        self._init_qsa_pending_state(len(full_attention_layer_ids), device)
        # One contiguous allocation behind per-layer views: every layer's
        # compressed pages are addressable from a single base pointer.
        self.qsa_compressed_flat = torch.zeros(
            (
                len(full_attention_layer_ids),
                self.qsa_compressed_capacity
                * self.qsa_index_kv_heads
                * self.qsa_index_head_dim,
            ),
            dtype=self.index_state_dtype,
            device=device,
        )
        self.qsa_compressed_k_buffer_pool = [
            self.qsa_compressed_flat[layer_offset].view(
                self.qsa_compressed_capacity,
                self.qsa_index_kv_heads,
                self.qsa_index_head_dim,
            )
            for layer_offset in range(len(full_attention_layer_ids))
        ]
        k_size, v_size = self.get_kv_size_bytes()
        self.mem_usage = (k_size + v_size) / GB

    def _init_qsa_pending_state(self, num_layers: int, device) -> None:
        """Live rings and isolated verify rows, bounded by the request pool.

        The first half never changes during TARGET_VERIFY. The second half
        stages candidates by request and tree step, not position modulo ratio.
        Complete groups can therefore mix the immutable accepted prefix with
        candidate members without rejected tails overwriting that prefix.
        """
        ring_slots = self.qsa_num_request_slots * self.qsa_compress_ratio
        self.qsa_ring_slots = ring_slots
        self.qsa_key_state_flat = torch.zeros(
            (num_layers, 2 * ring_slots, self.qsa_index_kv_heads, self.qsa_index_head_dim),
            dtype=self.index_state_dtype, device=device,
        )
        self.qsa_key_state_buffer_pool = [
            self.qsa_key_state_flat[layer, :ring_slots]
            for layer in range(num_layers)
        ]
        self.qsa_rope_state = torch.zeros(
            (2 * ring_slots, 3), dtype=torch.int64, device=device
        )
        self.qsa_rope_position_buffer = self.qsa_rope_state[:ring_slots]
        self.qsa_verify_positions = torch.zeros(ring_slots, dtype=torch.int64, device=device)
        self.qsa_verify_row_slots = torch.zeros(ring_slots, dtype=torch.int64, device=device)
        self.qsa_verify_group_locs = torch.zeros(
            (ring_slots, self.qsa_compress_ratio), dtype=torch.int32, device=device
        )

    def get_qsa_verify_key_state_buffer(self, layer_id: int) -> torch.Tensor:
        return self.qsa_key_state_flat[self._transfer_full_attention_id(layer_id)]

    def prepare_qsa_verify(
        self, requests, positions, group_ends, group_rows, draft_token_num: int
    ):
        """Graph-stable stage slots and mixed live/candidate compression plan."""
        num_rows, num_groups = positions.numel(), group_ends.numel()
        if not 0 < draft_token_num <= self.qsa_compress_ratio:
            raise ValueError("QSA verify window must fit the request candidate ring")
        if max(num_rows, num_groups) > self.qsa_ring_slots:
            raise ValueError("QSA verify rows exceed the actual request pool capacity")
        slots = self.qsa_verify_row_slots[:num_rows]
        groups = self.qsa_verify_group_locs[:num_groups]
        if max(num_rows, num_groups) == 0:
            return slots, groups
        if positions.is_cuda:
            _qsa_verify_plan_kernel[(max(num_rows, num_groups),)](
                requests, positions, group_ends, group_rows, slots, groups,
                self.qsa_verify_positions, num_rows, num_groups,
                RATIO=self.qsa_compress_ratio, DRAFT=draft_token_num,
                RING_SLOTS=self.qsa_ring_slots,
            )
            return slots, groups
        rows = torch.arange(num_rows, device=positions.device)
        candidate_slots = requests.long() * self.qsa_compress_ratio + rows % draft_token_num
        slots.copy_(candidate_slots + self.qsa_ring_slots)
        self.qsa_verify_positions[candidate_slots] = positions.long()
        group_rows = group_rows.long()
        group_requests = requests[group_rows].long()
        prefix = positions[group_rows] - group_rows % draft_token_num
        members = (
            group_ends[:, None]
            - torch.arange(self.qsa_compress_ratio - 1, -1, -1, device=positions.device)
        ).clamp_min(0)
        groups.copy_(torch.where(
            (group_requests[:, None] > 0) & (members >= prefix[:, None]),
            self.qsa_ring_slots + group_requests[:, None] * self.qsa_compress_ratio
            + members - prefix[:, None],
            group_requests[:, None] * self.qsa_compress_ratio
            + members % self.qsa_compress_ratio,
        ))
        return slots, groups

    def commit_qsa_state_after_verify(
        self, req_pool_indices, accept_lens, accept_index, draft_token_num: int
    ) -> None:
        """Commit exactly the accepted nodes; all other live ring rows survive."""
        bs = accept_lens.numel()
        if bs == 0 or accept_index.numel() == 0:
            return
        if self.qsa_key_state_flat.is_cuda:
            width = self.qsa_index_kv_heads * self.qsa_index_head_dim
            _qsa_verify_commit_kernel[(bs, self.qsa_key_state_flat.shape[0])](
                self.qsa_key_state_flat, self.qsa_rope_state, self.qsa_verify_positions,
                req_pool_indices, accept_lens, accept_index,
                accept_index.stride(0), accept_index.stride(1),
                RATIO=self.qsa_compress_ratio, DRAFT=draft_token_num,
                RING_SLOTS=self.qsa_ring_slots, KEY_WIDTH=width,
                KEY_BLOCK=triton.next_power_of_2(width),
                ACCEPT_WIDTH=accept_index.shape[1],
            )
            return
        ordinals = torch.arange(accept_index.shape[1], device=accept_lens.device)[None, :]
        steps = accept_index.long() - (
            torch.arange(bs, device=accept_lens.device)[:, None] * draft_token_num
        )
        requests = req_pool_indices[:bs].long()[:, None].expand_as(steps)
        active = (requests > 0) & (ordinals < accept_lens[:, None])
        active &= (steps >= 0) & (steps < draft_token_num)
        candidates = requests[active] * self.qsa_compress_ratio + steps[active]
        live = requests[active] * self.qsa_compress_ratio + (
            self.qsa_verify_positions[candidates] % self.qsa_compress_ratio
        )
        source = self.qsa_ring_slots + candidates
        self.qsa_key_state_flat[:, live] = self.qsa_key_state_flat[:, source]
        self.qsa_rope_state[live] = self.qsa_rope_state[source]

    def get_qsa_key_state_buffer(self, layer_id: int) -> torch.Tensor:
        return self.qsa_key_state_buffer_pool[
            self._transfer_full_attention_id(layer_id)
        ]

    def set_qsa_key_state_buffer(
        self, layer_id: int, loc: torch.Tensor, token_k: torch.Tensor
    ) -> None:
        buffer = self.get_qsa_key_state_buffer(layer_id)
        buffer[loc.long()] = token_k.to(buffer.dtype)

    def set_qsa_rope_position_buffer(
        self, loc: torch.Tensor, positions: torch.Tensor
    ) -> None:
        positions = positions.long()
        if positions.ndim == 1:
            positions = positions.unsqueeze(0).expand(3, -1)
        if positions.ndim != 2 or positions.shape[0] != 3:
            raise ValueError(
                f"QSA RoPE positions must be [tokens] or [3, tokens], got {positions.shape}"
            )
        self.qsa_rope_position_buffer[loc.long()] = positions.transpose(0, 1)

    def get_qsa_rope_position_buffer(self, loc: torch.Tensor) -> torch.Tensor:
        return self.qsa_rope_position_buffer[loc.long()]

    def get_qsa_compressed_k_buffer(self, layer_id: int) -> torch.Tensor:
        return self.qsa_compressed_k_buffer_pool[
            self._transfer_full_attention_id(layer_id)
        ]

    def set_qsa_compressed_k_buffer(
        self, layer_id: int, loc: torch.Tensor, compressed_k: torch.Tensor
    ) -> None:
        buffer = self.get_qsa_compressed_k_buffer(layer_id)
        buffer[loc.long()] = compressed_k.to(buffer.dtype)

    def get_kv_size_bytes(self):
        k_size, v_size = super().get_kv_size_bytes()
        qsa_k_size = (
            self.qsa_key_state_flat.numel() * self.qsa_key_state_flat.element_size()
            + sum(
                tensor.numel() * tensor.element_size()
                for tensor in self.qsa_compressed_k_buffer_pool
            )
            + self.qsa_rope_state.numel() * self.qsa_rope_state.element_size()
            + self.qsa_verify_positions.numel() * self.qsa_verify_positions.element_size()
            + self.qsa_verify_row_slots.numel() * self.qsa_verify_row_slots.element_size()
            + self.qsa_verify_group_locs.numel() * self.qsa_verify_group_locs.element_size()
        )
        return k_size + qsa_k_size, v_size


class QwenDSATokenToKVPool(HybridLinearKVPool):
    """Hybrid KV pool carrying the per-token index-K cache of tokenwise QSA.

    Only the BF16 reference layout is ported: one flat
    ``[size + page_size, index_kv_heads, index_head_dim]`` buffer per
    full-attention (DSA) layer, addressed by raw KV slots.  The FP8
    deep_gemm layout is intentionally not ported yet and the caller-side
    fast paths fail loudly instead of silently degrading.
    """

    index_state_dtype = torch.bfloat16

    @classmethod
    def qsa_bytes_per_token(
        cls, *, kv_heads: int, head_dim: int, num_layers: int
    ) -> int:
        return (
            _index_k_bytes(
                kv_heads=kv_heads, head_dim=head_dim, dtype=cls.index_state_dtype
            )
            * num_layers
        )

    def __init__(
        self,
        *,
        size: int,
        dtype: torch.dtype,
        page_size: int,
        head_num: int,
        head_dim: int,
        full_attention_layer_ids: List[int],
        device: str,
        mamba_pool: MambaPool,
        qsa_index_kv_heads: int,
        qsa_index_head_dim: int,
        qsa_token_budget: int,
        enable_memory_saver: bool = False,
        enable_kv_cache_copy: bool = False,
        start_layer: Optional[int] = None,
        full_kv_pool_class: Optional[type] = None,
        quant_method=None,
        post_capture_active: bool = False,
        full_kv_pool_gpu_size: int = 0,
    ):
        if page_size != 64:
            raise ValueError(
                "tokenwise QSA requires KV-cache page_size 64 for its paged "
                f"indexer buffer, got {page_size}"
            )
        self.dsa_index_k_buffer_pool = []
        super().__init__(
            size=size,
            dtype=dtype,
            page_size=page_size,
            head_num=head_num,
            head_dim=head_dim,
            full_attention_layer_ids=full_attention_layer_ids,
            device=device,
            mamba_pool=mamba_pool,
            enable_memory_saver=enable_memory_saver,
            enable_kv_cache_copy=enable_kv_cache_copy,
            use_mla=False,
            start_layer=start_layer,
            full_kv_pool_class=full_kv_pool_class,
            quant_method=quant_method,
            post_capture_active=post_capture_active,
            full_kv_pool_gpu_size=full_kv_pool_gpu_size,
        )
        if qsa_index_kv_heads != 1:
            raise ValueError(
                f"tokenwise QSA requires index_kv_heads = 1 (MQA), got "
                f"{qsa_index_kv_heads}"
            )
        if min(qsa_index_kv_heads, qsa_index_head_dim, qsa_token_budget) <= 0:
            raise ValueError("QSA cache configuration values must be positive")
        self.qsa_compress_ratio = 1
        self.qsa_index_kv_heads = int(qsa_index_kv_heads)
        self.qsa_index_head_dim = int(qsa_index_head_dim)
        self.qsa_token_topk = int(qsa_token_budget)
        self.qsa_block_topk = int(qsa_token_budget)
        state_size = size + page_size
        self.dsa_index_k_buffer_pool = [
            torch.zeros(
                (state_size, self.qsa_index_kv_heads, self.qsa_index_head_dim),
                dtype=self.index_state_dtype,
                device=device,
            )
            for _ in full_attention_layer_ids
        ]
        k_size, v_size = self.get_kv_size_bytes()
        self.mem_usage = (k_size + v_size) / GB

    def set_dsa_index_k_buffer(
        self, layer_id: int, loc: torch.Tensor, index_k: torch.Tensor
    ) -> None:
        buffer = self.get_dsa_index_k_buffer(layer_id)
        buffer[loc.long()] = index_k.to(buffer.dtype)

    def get_dsa_index_k_buffer(self, layer_id: int) -> torch.Tensor:
        return self.dsa_index_k_buffer_pool[self._transfer_full_attention_id(layer_id)]

    def get_kv_size_bytes(self):
        k_size, v_size = super().get_kv_size_bytes()
        dsa_k_size = sum(
            tensor.numel() * tensor.element_size()
            for tensor in self.dsa_index_k_buffer_pool
        )
        return k_size + dsa_k_size, v_size
