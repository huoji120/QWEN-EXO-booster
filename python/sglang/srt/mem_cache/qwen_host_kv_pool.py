"""GPU-first FP8 KV with CUDA-mapped pinned memory for overflow rows.

Logical slots address a page-aligned CUDA prefix and a disjoint host suffix.
QSA index/state pools stay on CUDA. Stored K/V retain global FP8 scaling.
"""

from __future__ import annotations

import logging
import math
import threading
import weakref
from pathlib import Path

import torch
import triton
import triton.language as tl

from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool
from sglang.srt.utils.async_probe import maybe_detect_oob

logger = logging.getLogger(__name__)


_HOST_RESERVATIONS = weakref.WeakKeyDictionary()
_HOST_RESERVATION_LOCK = threading.Lock()
_HOST_BASELINE = 0
_HOST_HEADROOM = 8 * 1024**3


def _reserve_host_bytes(owner, requested: int) -> None:
    """Admit main/draft banks, including reservations not yet faulted into RAM."""
    global _HOST_BASELINE
    with _HOST_RESERVATION_LOCK:
        root = Path("/sys/fs/cgroup")
        limit_path, current_path = root / "memory.max", root / "memory.current"
        if limit_path.exists() and current_path.exists():
            limit_text = limit_path.read_text().strip()
            if limit_text != "max":
                limit = int(limit_text)
                current = int(current_path.read_text())
                stats_path = root / "memory.stat"
                if stats_path.exists():
                    stats = dict(
                        (name, int(value))
                        for name, value in (
                            line.split() for line in stats_path.read_text().splitlines()
                        )
                    )
                    # Clean mmap/file cache is reclaimable; shared, dirty,
                    # writeback and locked pages cannot fund pinned banks.
                    reclaimable = max(
                        0,
                        stats.get("file", 0)
                        - sum(
                            stats.get(name, 0)
                            for name in (
                                "shmem", "file_dirty", "file_writeback", "unevictable"
                            )
                        ),
                    )
                    current -= min(current, reclaimable)
                if not _HOST_RESERVATIONS:
                    _HOST_BASELINE = current
                committed = max(
                    current, _HOST_BASELINE + sum(_HOST_RESERVATIONS.values())
                )
                if committed + requested + _HOST_HEADROOM > limit:
                    raise MemoryError(
                        "Qwen host FP8 KV exceeds cgroup memory.max with 8 GiB "
                        f"headroom: committed={committed}, requested={requested}, "
                        f"limit={limit}."
                    )
        _HOST_RESERVATIONS[owner] = requested


def _finish_host_access(device, buffers) -> None:
    # Raw integer pointers do not let the pinned allocator record CUDA use.
    # Hold every backing tensor until all outstanding mapped accesses finish.
    torch.cuda.synchronize(device)


@triton.jit
def _scatter_host_kv(
    cache_k, cache_v, k_address, v_address, gpu_k, gpu_v, slots,
    GPU_ROWS: tl.constexpr, SLOT_STRIDE: tl.constexpr,
    HEADS: tl.constexpr, K_DIM: tl.constexpr, V_DIM: tl.constexpr,
    K_S0: tl.constexpr, K_S1: tl.constexpr, K_S2: tl.constexpr,
    V_S0: tl.constexpr, V_S1: tl.constexpr, V_S2: tl.constexpr,
    CAPACITY: tl.constexpr, BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offsets = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    slot = tl.load(slots + row * SLOT_STRIDE)
    valid_slot = (slot >= 0) & (slot < CAPACITY)
    k_ptr = k_address.to(tl.int64).to(tl.pointer_type(tl.uint8))
    v_ptr = v_address.to(tl.int64).to(tl.pointer_type(tl.uint8))
    k_mask = (offsets < HEADS * K_DIM) & valid_slot
    v_mask = (offsets < HEADS * V_DIM) & valid_slot
    k = tl.load(
        cache_k + row * K_S0 + (offsets // K_DIM) * K_S1
        + (offsets % K_DIM) * K_S2, mask=k_mask, other=0,
    )
    v = tl.load(
        cache_v + row * V_S0 + (offsets // V_DIM) * V_S1
        + (offsets % V_DIM) * V_S2, mask=v_mask, other=0,
    )
    on_gpu = slot < GPU_ROWS
    tl.store(gpu_k + slot * HEADS * K_DIM + offsets, k, mask=k_mask & on_gpu)
    tl.store(gpu_v + slot * HEADS * V_DIM + offsets, v, mask=v_mask & on_gpu)
    host_slot = tl.maximum(slot - GPU_ROWS, 0)
    tl.store(k_ptr + host_slot * HEADS * K_DIM + offsets, k, mask=k_mask & ~on_gpu)
    tl.store(v_ptr + host_slot * HEADS * V_DIM + offsets, v, mask=v_mask & ~on_gpu)


@triton.jit
def _gather_host_rows(
    address, gpu, slots, output, NUM_SLOTS: tl.constexpr,
    GPU_ROWS: tl.constexpr, SLOT_STRIDE: tl.constexpr,
    ROW_DIM: tl.constexpr, CAPACITY: tl.constexpr,
    STORAGE_TYPE: tl.constexpr, BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offsets = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    slot = tl.load(slots + row * SLOT_STRIDE, mask=row < NUM_SLOTS, other=-1)
    source = address.to(tl.int64).to(tl.pointer_type(STORAGE_TYPE))
    valid = (slot >= 0) & (slot < CAPACITY) & (offsets < ROW_DIM)
    host_values = tl.load(
        source + tl.maximum(slot - GPU_ROWS, 0) * ROW_DIM + offsets,
        mask=valid & (slot >= GPU_ROWS), other=0.0,
    )
    gpu_source = gpu.to(tl.pointer_type(STORAGE_TYPE))
    gpu_values = tl.load(
        gpu_source + tl.maximum(slot, 0) * ROW_DIM + offsets,
        mask=valid & (slot < GPU_ROWS), other=0.0,
    )
    values = tl.where(slot < GPU_ROWS, gpu_values, host_values)
    if STORAGE_TYPE != tl.uint8:
        # Triton does not lower every direct FP8-to-FP8 conversion on SM120.
        # FP32 represents all stored FP8 values exactly before the output cast.
        values = values.to(tl.float32)
    tl.store(output + row * ROW_DIM + offsets, values, mask=offsets < ROW_DIM)


class QwenHostFP8KVPool(MHATokenToKVPool):
    """Disjoint CUDA prefix and allocator-owned pinned overflow bank.

    Device consumers gather by logical slot; moves are snapshot assignments.
    """

    def __init__(self, *args, gpu_size=0, **kwargs):
        self.gpu_size = gpu_size
        super().__init__(*args, **kwargs)

    def _create_buffers(self):
        device = torch.device(self.device)
        if device.type != "cuda":
            raise ValueError("Qwen host FP8 KV requires CUDA mapped host memory.")
        if self.dtype not in (torch.float8_e4m3fn, torch.float8_e5m2):
            raise ValueError("Qwen host KV supports only global-scaled CUDA FP8.")
        if self.is_quantized_kv_cache or self.post_capture_active:
            raise ValueError("Qwen host KV does not support quantized recipes or VA backing.")
        if self.use_hnd or self.kv_cache_layout != "nhd":
            raise ValueError("Qwen host FP8 KV requires NHD layout.")
        self._host_cuda_device = torch.device(
            "cuda", device.index if device.index is not None else torch.cuda.current_device()
        )
        super()._create_buffers()

    def _create_buffers_normal(self):
        if not 0 <= self.gpu_size <= self.size or self.gpu_size % self.page_size:
            raise ValueError("GPU KV capacity must be page-aligned and within logical size.")
        k_shape, v_shape = self._kv_buffer_shapes()
        self.gpu_rows = self.gpu_size + self.page_size if self.gpu_size else 0
        host_rows = self.size + self.page_size - self.gpu_rows
        host_k_shape, host_v_shape = (host_rows, *k_shape[1:]), (host_rows, *v_shape[1:])
        gpu_k_shape, gpu_v_shape = (self.gpu_rows, *k_shape[1:]), (self.gpu_rows, *v_shape[1:])
        self.host_bytes = self.layer_num * (math.prod(host_k_shape) + math.prod(host_v_shape))
        _reserve_host_bytes(self, self.host_bytes)
        self.k_buffer, self.v_buffer = [], []
        self.gpu_k_buffer, self.gpu_v_buffer = [], []
        try:
            for _ in range(self.layer_num):
                self.k_buffer.append(torch.empty(host_k_shape, dtype=torch.uint8, device="cpu", pin_memory=True))
                self.v_buffer.append(torch.empty(host_v_shape, dtype=torch.uint8, device="cpu", pin_memory=True))
                self.gpu_k_buffer.append(torch.empty(gpu_k_shape, dtype=torch.uint8, device=self.device))
                self.gpu_v_buffer.append(torch.empty(gpu_v_shape, dtype=torch.uint8, device=self.device))
            # Only the dummy page is readable before explicit KV writes.
            dummy_buffers = (self.gpu_k_buffer + self.gpu_v_buffer) if self.gpu_rows else (self.k_buffer + self.v_buffer)
            for buffer in dummy_buffers:
                buffer[: self.page_size].zero_()
        except BaseException:
            self.k_buffer.clear()
            self.v_buffer.clear()
            self.gpu_k_buffer.clear()
            self.gpu_v_buffer.clear()
            with _HOST_RESERVATION_LOCK:
                _HOST_RESERVATIONS.pop(self, None)
            raise
        self._host_lifetime = weakref.finalize(
            self, _finish_host_access, self._host_cuda_device,
            tuple(self.k_buffer + self.v_buffer + self.gpu_k_buffer + self.gpu_v_buffer),
        )
        logger.info(
            "QWEN_HOST_KV_ALLOCATED tokens=%d layers=%d dtype=%s gpu_tokens=%d host_tokens=%d pinned_host_bytes=%d GPU_raw_KV_bytes=%d",
            self.size, self.layer_num, self.dtype, self.gpu_size, self.size - self.gpu_size,
            self.host_bytes, sum(self.get_kv_size_bytes()),
        )

    def _init_kv_copy_and_warmup(self):
        # A single-launch generic copy races on overlapping slot assignments.
        self._kv_copy_config = None

    def _clear_buffers(self):
        self._host_lifetime()
        super()._clear_buffers()
        del self.gpu_k_buffer
        del self.gpu_v_buffer
        with _HOST_RESERVATION_LOCK:
            _HOST_RESERVATIONS.pop(self, None)

    def get_kv_size_bytes(self):
        # QSA's wrapper separately adds its device index/state buffers.
        return tuple(sum(buffer.numel() for buffer in buffers) for buffers in (self.gpu_k_buffer, self.gpu_v_buffer))

    def _check_slots(self, slots):
        if slots.device != self._host_cuda_device or slots.ndim != 1:
            raise ValueError("Host KV slots must be one-dimensional on the pool CUDA device.")
        if slots.dtype not in (torch.int32, torch.int64):
            raise ValueError("Host KV slots must have int32/int64 dtype.")

    def set_kv_buffer(
        self, layer, loc_info, cache_k, cache_v, k_scale=None, v_scale=None,
        layer_id_override=None, dcp_kv_mask=None,
    ):
        if dcp_kv_mask is not None:
            raise ValueError("Qwen host KV does not support distributed context parallelism.")
        return super().set_kv_buffer(
            layer, loc_info, cache_k, cache_v, k_scale, v_scale,
            layer_id_override=layer_id_override,
        )

    def _store_kv_layer(self, layer_idx, loc, cache_k, cache_v):
        self._check_slots(loc)
        if cache_k.device != loc.device or cache_v.device != loc.device:
            raise ValueError("Host KV writes require CUDA K/V inputs.")
        if cache_k.dtype != torch.uint8 or cache_v.dtype != torch.uint8:
            raise ValueError("Host KV writer expects inherited FP8 byte views.")
        if (tuple(cache_k.shape) != (loc.numel(), self.head_num, self.head_dim)
                or tuple(cache_v.shape) != (loc.numel(), self.head_num, self.v_head_dim)):
            raise ValueError("Host KV writes must match the pool's NHD geometry.")
        if not loc.numel():
            return
        width = max(self.row_dim, self.head_num * self.v_head_dim)
        _scatter_host_kv[(loc.numel(), triton.cdiv(width, 256))](
            cache_k, cache_v,
            self.k_buffer[layer_idx].data_ptr(), self.v_buffer[layer_idx].data_ptr(),
            self.gpu_k_buffer[layer_idx], self.gpu_v_buffer[layer_idx], loc,
            self.gpu_rows, loc.stride(0),
            self.head_num, self.head_dim, self.v_head_dim,
            *cache_k.stride(), *cache_v.stride(), self.size + self.page_size, 256,
        )

    def _gather_rows(self, local_layer_id, slots, output_k, output_v, *, raw=False):
        self._check_slots(slots)
        if not 0 <= local_layer_id < self.layer_num:
            raise IndexError("Host KV local layer id is outside the pool.")
        rows = output_k.shape[0]
        allowed = (torch.uint8,) if raw else (
            torch.float8_e4m3fn, torch.float8_e5m2, torch.bfloat16,
            torch.float16, torch.float32,
        )
        for output, dim in ((output_k, self.head_dim), (output_v, self.v_head_dim)):
            if (output.device != slots.device or not output.is_contiguous()
                    or output.ndim != 3 or tuple(output.shape[1:]) != (self.head_num, dim)):
                raise ValueError("Host KV gather output must be contiguous CUDA NHD with pool geometry.")
            if output.shape[0] != rows or rows < slots.numel():
                raise ValueError("Host KV outputs must hold all selected slots with equal row counts.")
            if output.dtype not in allowed:
                raise ValueError("Unsupported host KV gather output dtype.")
        if not rows:
            return
        storage_type = tl.uint8 if raw else (
            tl.float8e4nv if self.dtype == torch.float8_e4m3fn else tl.float8e5
        )
        for buffer, gpu, output, dim in (
            (self.k_buffer[local_layer_id], self.gpu_k_buffer[local_layer_id], output_k, self.head_dim),
            (self.v_buffer[local_layer_id], self.gpu_v_buffer[local_layer_id], output_v, self.v_head_dim),
        ):
            _gather_host_rows[(rows, triton.cdiv(self.head_num * dim, 256))](
                buffer.data_ptr(), gpu, slots, output, slots.numel(),
                self.gpu_rows, slots.stride(0),
                self.head_num * dim, self.size + self.page_size, storage_type, 256,
            )

    def gather_into(self, local_layer_id, slots, output_k, output_v):
        """Gather stored values, casting only; -1 and padded tail become zero.

        local_layer_id indexes backing lists (pool layer minus start_layer).
        Callers apply the layer K/V global scales, as for the GPU FP8 pool.
        This performs no allocation or host I/O, including graph replay.
        """
        self._gather_rows(local_layer_id, slots, output_k, output_v)

    def get_kv_tokens(self, layer_id, indices, dtype, *, layer_id_override=None):
        pool_layer_id = layer_id if layer_id_override is None else layer_id_override
        if self.layer_transfer_counter is not None:
            self.layer_transfer_counter.wait_until(pool_layer_id - self.start_layer)
        key = torch.empty((indices.numel(), self.head_num, self.head_dim), dtype=dtype, device=self.device)
        value = torch.empty((indices.numel(), self.head_num, self.v_head_dim), dtype=dtype, device=self.device)
        self.gather_into(pool_layer_id - self.start_layer, indices, key, value)
        return key, value

    def move_kv_cache(self, tgt_loc, src_loc):
        self._check_slots(tgt_loc)
        self._check_slots(src_loc)
        if tgt_loc.numel() != src_loc.numel():
            raise ValueError("Host KV move requires matching source and target lengths.")
        if not self.layer_num or not src_loc.numel():
            return
        capacity = self.size + self.page_size
        maybe_detect_oob(tgt_loc, 0, capacity, "move_host_kv_cache target")
        maybe_detect_oob(src_loc, 0, capacity, "move_host_kv_cache source")
        # Snapshot every selected row before writing each layer. Chunked direct
        # copies would corrupt cycles that cross a chunk boundary.
        key = torch.empty((src_loc.numel(), self.head_num, self.head_dim), dtype=torch.uint8, device=self.device)
        value = torch.empty((src_loc.numel(), self.head_num, self.v_head_dim), dtype=torch.uint8, device=self.device)
        for layer_id in range(self.layer_num):
            self._gather_rows(layer_id, src_loc, key, value, raw=True)
            self._store_kv_layer(layer_id, tgt_loc, key, value)

    def get_cpu_copy(self, indices, mamba_indices=None):
        indices = indices.to(device=self._host_cuda_device, dtype=torch.int64)
        result = []
        for layer_id in range(self.layer_num):
            chunks = []
            for offset in range(0, indices.numel(), self.cpu_offloading_chunk_size):
                selected = indices[offset : offset + self.cpu_offloading_chunk_size]
                key = torch.empty((selected.numel(), self.head_num, self.head_dim), dtype=torch.uint8, device=self.device)
                value = torch.empty((selected.numel(), self.head_num, self.v_head_dim), dtype=torch.uint8, device=self.device)
                self._gather_rows(layer_id, selected, key, value, raw=True)
                chunks.append([key.cpu(), value.cpu()])
            result.append(chunks)
        return result

    def load_cpu_copy(self, kv_cache_cpu, indices, mamba_indices=None):
        indices = indices.to(device=self._host_cuda_device, dtype=torch.int64)
        if len(kv_cache_cpu) != self.layer_num:
            raise ValueError("CPU KV snapshot must include every bank layer.")
        for layer_id in range(self.layer_num):
            for offset in range(0, indices.numel(), self.cpu_offloading_chunk_size):
                selected = indices[offset : offset + self.cpu_offloading_chunk_size]
                key_cpu, value_cpu = kv_cache_cpu[layer_id][offset // self.cpu_offloading_chunk_size]
                for payload, dim in ((key_cpu, self.head_dim), (value_cpu, self.v_head_dim)):
                    if (payload.device.type != "cpu" or payload.dtype != torch.uint8
                            or tuple(payload.shape) != (selected.numel(), self.head_num, dim)):
                        raise ValueError("KV snapshot must contain matching CPU byte rows.")
                self._store_kv_layer(layer_id, selected, key_cpu.to(self.device), value_cpu.to(self.device))

    def set_kv_buffer_prefix_valid(self, *args, **kwargs):
        raise NotImplementedError("Prefix-valid scatter is not supported by Qwen host KV.")

    def get_contiguous_buf_infos(self):
        raise NotImplementedError("Disaggregated/HiCache transfer is not supported by Qwen host KV.")
