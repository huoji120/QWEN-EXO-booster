"""GPU-first FP8/NVFP4 KV with CUDA-mapped pinned memory overflow.

Logical slots address a page-aligned CUDA prefix and a disjoint host suffix.
NVFP4 retains packed data and block scales without a logical dequant workspace.
"""

from __future__ import annotations

import logging
import threading
import weakref
from pathlib import Path

import torch
import triton
import triton.language as tl

from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool, unwrap_write_loc
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
                        "Qwen host KV exceeds cgroup memory.max with 8 GiB "
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


@triton.jit
def _selected_host_values(address, gpu, slot, offsets, GPU_ROWS: tl.constexpr,
                          ROW_DIM: tl.constexpr, CAPACITY: tl.constexpr,
                          STORAGE_TYPE: tl.constexpr):
    valid = (slot >= 0) & (slot < CAPACITY) & (offsets < ROW_DIM)
    host = address.to(tl.int64).to(tl.pointer_type(STORAGE_TYPE))
    device = gpu.to(tl.pointer_type(STORAGE_TYPE))
    h = tl.load(host + tl.maximum(slot - GPU_ROWS, 0) * ROW_DIM + offsets,
                mask=valid & (slot >= GPU_ROWS), other=0.0)
    g = tl.load(device + tl.maximum(slot, 0) * ROW_DIM + offsets,
                mask=valid & (slot < GPU_ROWS), other=0.0)
    # Match _gather_host_rows, including FP8 cross-format conversion.
    return tl.where(slot < GPU_ROWS, g, h).to(tl.float32)


@triton.jit
def _gather_selected_host_kv(
    host_k, host_v, gpu_k, gpu_v, req_to_token, req_indices, indices,
    seq_lens, cu_k, output_k, output_v,
    BATCH: tl.constexpr, TOPK: tl.constexpr,
    REQ_ROWS: tl.constexpr, REQ_WIDTH: tl.constexpr,
    R0: tl.constexpr, R1: tl.constexpr, RI: tl.constexpr,
    I0: tl.constexpr, I1: tl.constexpr, LS: tl.constexpr, CS: tl.constexpr,
    GPU_ROWS: tl.constexpr, K_DIM: tl.constexpr, V_DIM: tl.constexpr,
    CAPACITY: tl.constexpr, STORAGE_TYPE: tl.constexpr, BLOCK: tl.constexpr,
):
    packed_row = tl.program_id(0)
    offsets = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    # Upper-bound search handles empty requests and page-padded cu_k alike.
    # One program owns each output row, including the unused packed tail.
    lo, hi = 0, BATCH
    while lo < hi:
        mid = (lo + hi) // 2
        end = tl.load(cu_k + (mid + 1) * CS)
        before = packed_row < end
        hi = tl.where(before, mid, hi)
        lo = tl.where(before, lo, mid + 1)
    row = lo
    active = row < BATCH
    start = tl.load(cu_k + row * CS, mask=active, other=0)
    col = packed_row - start
    req = tl.load(req_indices + row * RI, mask=active, other=-1)
    length = tl.load(seq_lens + row * LS, mask=active, other=0)
    pos = tl.load(indices + row * I0 + col * I1,
                  mask=active & (col >= 0) & (col < TOPK), other=-1)
    valid = active & (req >= 0) & (req < REQ_ROWS)
    valid = valid & (pos >= 0) & (pos < length) & (pos < REQ_WIDTH)
    slot = tl.load(req_to_token + req * R0 + pos * R1, mask=valid, other=-1)
    k = _selected_host_values(host_k, gpu_k, slot, offsets, GPU_ROWS,
                              K_DIM, CAPACITY, STORAGE_TYPE)
    v = _selected_host_values(host_v, gpu_v, slot, offsets, GPU_ROWS,
                              V_DIM, CAPACITY, STORAGE_TYPE)
    tl.store(output_k + packed_row * K_DIM + offsets, k, mask=offsets < K_DIM)
    tl.store(output_v + packed_row * V_DIM + offsets, v, mask=offsets < V_DIM)


class QwenHostKVPool(MHATokenToKVPool):
    """Disjoint CUDA prefix and allocator-owned pinned overflow bank.

    Device consumers gather by logical slot; moves are snapshot assignments.
    """

    def __init__(self, *args, gpu_size=0, **kwargs):
        self.gpu_size = gpu_size
        super().__init__(*args, **kwargs)

    def _create_buffers(self):
        device = torch.device(self.device)
        if device.type != "cuda":
            raise ValueError("Qwen host KV requires CUDA mapped host memory.")
        if self.is_quantized_kv_cache:
            if self.quant_method.name != "nvfp4":
                raise ValueError("Qwen host KV supports only FP8 or NVFP4.")
            if self.head_dim != self.v_head_dim or self.head_dim % 16:
                raise ValueError("NVFP4 host KV requires equal K/V head dimensions divisible by 16.")
        elif self.dtype not in (torch.float8_e4m3fn, torch.float8_e5m2):
            raise ValueError("Qwen host KV supports only FP8 or NVFP4.")
        if self.post_capture_active:
            raise ValueError("Qwen host KV does not support VA backing.")
        if self.use_hnd or self.kv_cache_layout != "nhd":
            raise ValueError("Qwen host KV requires NHD layout.")
        self._host_cuda_device = torch.device(
            "cuda", device.index if device.index is not None else torch.cuda.current_device()
        )
        self.store_dtype = torch.uint8
        self.dq_k_buffer = self.dq_v_buffer = None
        self._create_buffers_normal()
        # No inherited pointer tables: they would describe only the host suffix
        # as if it were a CUDA-addressable logical bank.
        self._kv_buffer_descs = []

    def _create_buffers_normal(self):
        if not 0 <= self.gpu_size <= self.size or self.gpu_size % self.page_size:
            raise ValueError("GPU KV capacity must be page-aligned and within logical size.")
        self.gpu_rows = self.gpu_size + self.page_size if self.gpu_size else 0
        host_rows = self.size + self.page_size - self.gpu_rows
        divisor = 2 if self.is_quantized_kv_cache else 1
        self._raw_dims = (self.head_dim // divisor, self.v_head_dim // divisor)
        if self.is_quantized_kv_cache:
            self._raw_dims += (self.head_dim // 16, self.v_head_dim // 16)
        self.k_buffer, self.v_buffer = [], []
        self.gpu_k_buffer, self.gpu_v_buffer = [], []
        self.k_scale_buffer = [] if self.is_quantized_kv_cache else None
        self.v_scale_buffer = [] if self.is_quantized_kv_cache else None
        self.gpu_k_scale_buffer = [] if self.is_quantized_kv_cache else None
        self.gpu_v_scale_buffer = [] if self.is_quantized_kv_cache else None
        host_banks, gpu_banks = self._raw_banks()
        self.host_bytes = self.layer_num * host_rows * self.head_num * sum(self._raw_dims)
        _reserve_host_bytes(self, self.host_bytes)
        try:
            for _ in range(self.layer_num):
                for host, gpu, dim in zip(host_banks, gpu_banks, self._raw_dims):
                    host.append(torch.empty((host_rows, self.head_num, dim), dtype=torch.uint8,
                                            device="cpu", pin_memory=True))
                    gpu.append(torch.empty((self.gpu_rows, self.head_num, dim), dtype=torch.uint8,
                                           device=self.device))
            # Only the dummy page is readable before explicit KV writes.
            for bank in gpu_banks if self.gpu_rows else host_banks:
                for buffer in bank:
                    buffer[: self.page_size].zero_()
        except BaseException:
            for bank in host_banks + gpu_banks:
                bank.clear()
            with _HOST_RESERVATION_LOCK:
                _HOST_RESERVATIONS.pop(self, None)
            raise
        self._host_lifetime = weakref.finalize(
            self, _finish_host_access, self._host_cuda_device,
            tuple(buffer for bank in host_banks + gpu_banks for buffer in bank),
        )
        logger.info(
            "QWEN_HOST_KV_ALLOCATED tokens=%d layers=%d dtype=%s recipe=%s gpu_tokens=%d host_tokens=%d pinned_host_bytes=%d GPU_raw_KV_bytes=%d",
            self.size, self.layer_num, self.dtype, self.quant_method.name,
            self.gpu_size, self.size - self.gpu_size,
            self.host_bytes, sum(self.get_kv_size_bytes()),
        )

    def _raw_banks(self):
        host = (self.k_buffer, self.v_buffer)
        gpu = (self.gpu_k_buffer, self.gpu_v_buffer)
        if self.is_quantized_kv_cache:
            host += (self.k_scale_buffer, self.v_scale_buffer)
            gpu += (self.gpu_k_scale_buffer, self.gpu_v_scale_buffer)
        return host, gpu

    def _init_kv_copy_and_warmup(self):
        # A single-launch generic copy races on overlapping slot assignments.
        self._kv_copy_config = None

    def _clear_buffers(self):
        self._host_lifetime()
        super()._clear_buffers()
        del self.gpu_k_buffer
        del self.gpu_v_buffer
        del self.gpu_k_scale_buffer
        del self.gpu_v_scale_buffer
        with _HOST_RESERVATION_LOCK:
            _HOST_RESERVATIONS.pop(self, None)

    def get_kv_size_bytes(self):
        # QSA's wrapper separately adds its device index/state buffers.
        _, gpu = self._raw_banks()
        return tuple(sum(buffer.nbytes for bank in gpu[kind::2] for buffer in bank)
                     for kind in range(2))

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
        if self.is_quantized_kv_cache:
            from sglang.srt.layers.quantization.kvfp4_tensor import NVFP4KVQuantizeUtil

            loc, _, _ = unwrap_write_loc(loc_info)
            self._check_slots(loc)
            maybe_detect_oob(loc, 0, self.size + self.page_size, "set_host_nvfp4_kv")
            pool_layer = layer.layer_id if layer_id_override is None else layer_id_override
            local_layer = pool_layer - self.start_layer
            global_layer = layer.layer_id if layer is not None else pool_layer
            if not 0 <= local_layer < self.layer_num:
                raise IndexError("Host KV local layer id is outside the pool.")
            for value, dim in ((cache_k, self.head_dim), (cache_v, self.v_head_dim)):
                if (value.device != loc.device or value.dtype not in (torch.bfloat16, torch.float16)
                        or tuple(value.shape) != (loc.numel(), self.head_num, dim)):
                    raise ValueError("NVFP4 writes require matching CUDA BF16/FP16 NHD inputs.")
            if not loc.numel():
                return
            k_scale, v_scale = self._quantized_scales(global_layer, k_scale, v_scale)
            key, ks, _ = NVFP4KVQuantizeUtil.quantize(cache_k.contiguous(), k_scale)
            value, vs, _ = NVFP4KVQuantizeUtil.quantize(cache_v.contiguous(), v_scale)
            self._store_raw_rows(local_layer, loc, tuple(
                tensor.view(torch.uint8) for tensor in (key, value, ks, vs)
            ))
            return
        return super().set_kv_buffer(
            layer, loc_info, cache_k, cache_v, k_scale, v_scale,
            layer_id_override=layer_id_override,
        )

    def _store_kv_layer(self, layer_idx, loc, cache_k, cache_v):
        self._store_raw_rows(layer_idx, loc, (cache_k, cache_v))

    def _store_raw_rows(self, layer_idx, loc, payloads):
        self._check_slots(loc)
        if not 0 <= layer_idx < self.layer_num:
            raise IndexError("Host KV local layer id is outside the pool.")
        if len(payloads) != len(self._raw_dims):
            raise ValueError("Host KV writes must include data and all block scales.")
        for payload, dim in zip(payloads, self._raw_dims):
            if (payload.device != loc.device or payload.dtype != torch.uint8
                    or tuple(payload.shape) != (loc.numel(), self.head_num, dim)):
                raise ValueError("Host KV writes require matching CUDA NHD byte rows.")
        if not loc.numel():
            return
        host, gpu = self._raw_banks()
        for pair in range(0, len(payloads), 2):
            key, value = payloads[pair : pair + 2]
            kd, vd = self._raw_dims[pair : pair + 2]
            _scatter_host_kv[(loc.numel(), triton.cdiv(self.head_num * max(kd, vd), 256))](
                key, value, host[pair][layer_idx].data_ptr(), host[pair + 1][layer_idx].data_ptr(),
                gpu[pair][layer_idx], gpu[pair + 1][layer_idx], loc,
                self.gpu_rows, loc.stride(0), self.head_num, kd, vd,
                *key.stride(), *value.stride(), self.size + self.page_size, 256,
            )

    def gather_selected_into(self, local_layer_id, req_to_token, req_indices,
                             indices, seq_lens, cu_k, output_k, output_v, *, layer_id=None):
        """Gather QSA selections; NVFP4 applies global scales, FP8 only casts.

        cu_k is the same monotonic packed/page-padded prefix plan used by QSA.
        No scratch allocations or host synchronization occur during replay.
        """
        if self.is_quantized_kv_cache:
            from sglang.srt.layers.attention.qsa.nvfp4 import gather_nvfp4_selected

            gather_nvfp4_selected(
                *self._nvfp4_device_banks(local_layer_id),
                req_to_token, req_indices, indices, seq_lens, cu_k, output_k, output_v,
                **self._nvfp4_gather_kwargs(local_layer_id, layer_id),
            )
            return
        self._check_slots(req_indices)
        self._check_slots(seq_lens)
        self._check_slots(cu_k)
        batch, topk = indices.shape
        for tensor in (req_to_token, indices):
            if (tensor.device != self._host_cuda_device or tensor.ndim != 2
                    or tensor.dtype not in (torch.int32, torch.int64)):
                raise ValueError("QSA mapping must be two-dimensional CUDA integer tensors.")
        if (req_indices.numel() != batch or seq_lens.numel() != batch
                or cu_k.numel() != batch + 1):
            raise ValueError("QSA mapping batch dimensions must match.")
        if any(s <= 0 for t in (req_to_token, req_indices, indices, seq_lens, cu_k)
               for s in t.stride()):
            raise ValueError("QSA mapping requires positive strides.")
        self._check_gather_outputs(local_layer_id, output_k, output_v, 0)
        rows = output_k.shape[0]
        if not rows:
            return
        _gather_selected_host_kv[(rows, triton.cdiv(max(self.row_dim, self.head_num * self.v_head_dim), 256))](
            self.k_buffer[local_layer_id].data_ptr(), self.v_buffer[local_layer_id].data_ptr(),
            self.gpu_k_buffer[local_layer_id], self.gpu_v_buffer[local_layer_id],
            req_to_token, req_indices, indices, seq_lens, cu_k, output_k, output_v,
            batch, topk, *req_to_token.shape, *req_to_token.stride(),
            req_indices.stride(0), *indices.stride(), seq_lens.stride(0), cu_k.stride(0),
            self.gpu_rows, self.row_dim, self.head_num * self.v_head_dim,
            self.size + self.page_size,
            tl.float8e4nv if self.dtype == torch.float8_e4m3fn else tl.float8e5, 256,
        )

    def _check_gather_outputs(self, local_layer_id, output_k, output_v, num_slots):
        if not 0 <= local_layer_id < self.layer_num:
            raise IndexError("Host KV local layer id is outside the pool.")
        rows = output_k.shape[0]
        allowed = (torch.float8_e4m3fn, torch.float8_e5m2, torch.bfloat16,
                   torch.float16, torch.float32)
        for output, dim in ((output_k, self.head_dim), (output_v, self.v_head_dim)):
            if (output.device != self._host_cuda_device or not output.is_contiguous()
                    or output.ndim != 3 or tuple(output.shape[1:]) != (self.head_num, dim)):
                raise ValueError("Host KV gather output must be contiguous CUDA NHD with pool geometry.")
            if output.shape[0] != rows or rows < num_slots:
                raise ValueError("Host KV outputs must hold all selected slots with equal row counts.")
            if output.dtype not in allowed:
                raise ValueError("Unsupported host KV gather output dtype.")

    def _gather_rows(self, local_layer_id, slots, output_k, output_v):
        self._check_slots(slots)
        self._check_gather_outputs(local_layer_id, output_k, output_v, slots.numel())
        rows = output_k.shape[0]
        if not rows:
            return
        storage_type = tl.float8e4nv if self.dtype == torch.float8_e4m3fn else tl.float8e5
        for buffer, gpu, output, dim in (
            (self.k_buffer[local_layer_id], self.gpu_k_buffer[local_layer_id], output_k, self.head_dim),
            (self.v_buffer[local_layer_id], self.gpu_v_buffer[local_layer_id], output_v, self.v_head_dim),
        ):
            _gather_host_rows[(rows, triton.cdiv(self.head_num * dim, 256))](
                buffer.data_ptr(), gpu, slots, output, slots.numel(),
                self.gpu_rows, slots.stride(0),
                self.head_num * dim, self.size + self.page_size, storage_type, 256,
            )

    def _nvfp4_device_banks(self, local_layer_id):
        if not 0 <= local_layer_id < self.layer_num:
            raise IndexError("Host KV local layer id is outside the pool.")
        return tuple(bank[local_layer_id] for bank in self._raw_banks()[1])

    def _nvfp4_gather_kwargs(self, local_layer_id, layer_id):
        if layer_id is None:
            raise ValueError("NVFP4 gathers require the original global layer_id.")
        k_scale, v_scale = self.quant_method.get_bmm_scales(layer_id)
        return dict(k_scale=k_scale, v_scale=v_scale,
                    host_k=self.k_buffer[local_layer_id],
                    host_v=self.v_buffer[local_layer_id],
                    host_ks=self.k_scale_buffer[local_layer_id],
                    host_vs=self.v_scale_buffer[local_layer_id])

    def gather_into(self, local_layer_id, slots, output_k, output_v, *, layer_id=None):
        """Gather selected rows without device allocations or synchronization.

        NVFP4 outputs include global scales selected by original layer_id.
        FP8 outputs are cast-only; their caller applies global scales as before.
        Invalid slots and the unused output tail are zero.
        """
        if self.is_quantized_kv_cache:
            from sglang.srt.layers.attention.qsa.nvfp4 import gather_nvfp4_rows

            gather_nvfp4_rows(
                *self._nvfp4_device_banks(local_layer_id), slots, output_k, output_v,
                **self._nvfp4_gather_kwargs(local_layer_id, layer_id),
            )
        else:
            self._gather_rows(local_layer_id, slots, output_k, output_v)

    def _new_raw_rows(self, count):
        return tuple(torch.empty((count, self.head_num, dim), dtype=torch.uint8,
                                 device=self.device) for dim in self._raw_dims)

    def _gather_raw_rows(self, local_layer_id, slots, outputs):
        self._check_slots(slots)
        if not slots.numel():
            return
        host, gpu = self._raw_banks()
        for host_bank, gpu_bank, output, dim in zip(host, gpu, outputs, self._raw_dims):
            _gather_host_rows[(slots.numel(), triton.cdiv(self.head_num * dim, 256))](
                host_bank[local_layer_id].data_ptr(), gpu_bank[local_layer_id], slots,
                output, slots.numel(), self.gpu_rows, slots.stride(0),
                self.head_num * dim, self.size + self.page_size, tl.uint8, 256,
            )

    def get_kv_tokens(self, layer_id, indices, dtype, *, layer_id_override=None):
        pool_layer_id = layer_id if layer_id_override is None else layer_id_override
        if self.layer_transfer_counter is not None:
            self.layer_transfer_counter.wait_until(pool_layer_id - self.start_layer)
        key = torch.empty((indices.numel(), self.head_num, self.head_dim), dtype=dtype, device=self.device)
        value = torch.empty((indices.numel(), self.head_num, self.v_head_dim), dtype=dtype, device=self.device)
        self.gather_into(pool_layer_id - self.start_layer, indices, key, value, layer_id=layer_id)
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
        payloads = self._new_raw_rows(src_loc.numel())
        for layer_id in range(self.layer_num):
            self._gather_raw_rows(layer_id, src_loc, payloads)
            self._store_raw_rows(layer_id, tgt_loc, payloads)

    def get_cpu_copy(self, indices, mamba_indices=None):
        indices = indices.to(device=self._host_cuda_device, dtype=torch.int64)
        result = []
        for layer_id in range(self.layer_num):
            chunks = []
            for offset in range(0, indices.numel(), self.cpu_offloading_chunk_size):
                selected = indices[offset : offset + self.cpu_offloading_chunk_size]
                payloads = self._new_raw_rows(selected.numel())
                self._gather_raw_rows(layer_id, selected, payloads)
                chunks.append([payload.cpu() for payload in payloads])
            result.append(chunks)
        return result

    def load_cpu_copy(self, kv_cache_cpu, indices, mamba_indices=None):
        indices = indices.to(device=self._host_cuda_device, dtype=torch.int64)
        if len(kv_cache_cpu) != self.layer_num:
            raise ValueError("CPU KV snapshot must include every bank layer.")
        for layer_id in range(self.layer_num):
            for offset in range(0, indices.numel(), self.cpu_offloading_chunk_size):
                selected = indices[offset : offset + self.cpu_offloading_chunk_size]
                payloads = kv_cache_cpu[layer_id][offset // self.cpu_offloading_chunk_size]
                if len(payloads) != len(self._raw_dims):
                    raise ValueError("KV snapshot must include data and all block scales.")
                for payload, dim in zip(payloads, self._raw_dims):
                    if (payload.device.type != "cpu" or payload.dtype != torch.uint8
                            or tuple(payload.shape) != (selected.numel(), self.head_num, dim)):
                        raise ValueError("KV snapshot must contain matching CPU byte rows.")
                self._store_raw_rows(layer_id, selected, tuple(payload.to(self.device) for payload in payloads))

    def _get_key_buffer(self, layer_id):
        raise NotImplementedError("Split host KV requires selected-row gathers, not a logical key buffer.")

    def _get_value_buffer(self, layer_id):
        raise NotImplementedError("Split host KV requires selected-row gathers, not a logical value buffer.")

    def get_raw_kv_buffer(self, layer_id):
        raise NotImplementedError("Split host KV has no contiguous logical raw buffer.")

    def _pd_registerable_tensors(self):
        raise NotImplementedError("Disaggregated/HiCache transfer is not supported by Qwen host KV.")

    def set_kv_buffer_prefix_valid(self, *args, **kwargs):
        raise NotImplementedError("Prefix-valid scatter is not supported by Qwen host KV.")

    def get_contiguous_buf_infos(self):
        raise NotImplementedError("Disaggregated/HiCache transfer is not supported by Qwen host KV.")
