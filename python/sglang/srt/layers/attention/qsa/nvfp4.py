"""Selected-row NVFP4 decoding for QSA, including mapped pinned-host banks.

Even dimensions occupy the low nibble. Each 16 values share one E4M3FN
scale byte. Decode and both scale multiplications are FP32, followed by one
cast to the consumer dtype. No full-bank decode or device metadata read occurs.
"""

import math

import torch
import triton
import triton.language as tl


@triton.jit
def _load_nvfp4(
    data, scales, host_data, host_scales, slots, heads, dims, valid,
    global_scale, GPU_ROWS: tl.constexpr, HOST_ROWS: tl.constexpr,
    DS: tl.constexpr, SS: tl.constexpr, HDS: tl.constexpr, HSS: tl.constexpr,
):
    valid = valid & (slots >= 0) & (slots < GPU_ROWS + HOST_ROWS)
    gpu_valid = valid & (slots < GPU_ROWS)
    packed = tl.load(
        data + slots * DS[0] + heads * DS[1] + (dims // 2) * DS[2],
        mask=gpu_valid, other=0,
    )
    scale_bits = tl.load(
        scales + slots * SS[0] + heads * SS[1] + (dims // 16) * SS[2],
        mask=gpu_valid, other=0,
    )
    if HOST_ROWS > 0:
        host_valid = valid & (slots >= GPU_ROWS)
        host_slot = slots - GPU_ROWS
        host_bytes = host_data.to(tl.int64).to(tl.pointer_type(tl.uint8))
        host_sf = host_scales.to(tl.int64).to(tl.pointer_type(tl.uint8))
        hp = tl.load(
            host_bytes + host_slot * HDS[0] + heads * HDS[1] + (dims // 2) * HDS[2],
            mask=host_valid, other=0,
        )
        hs = tl.load(
            host_sf + host_slot * HSS[0] + heads * HSS[1] + (dims // 16) * HSS[2],
            mask=host_valid, other=0,
        )
        packed = tl.where(host_valid, hp, packed)
        scale_bits = tl.where(host_valid, hs, scale_bits)
    nibble = (packed >> ((dims % 2) * 4)) & 15
    magnitude = nibble & 7
    value = tl.where(
        magnitude < 4, magnitude.to(tl.float32) * 0.5,
        tl.where(magnitude < 6, magnitude.to(tl.float32) - 2.0,
                 tl.where(magnitude == 6, 4.0, 6.0)),
    )
    value = tl.where((nibble & 8) != 0, -value, value)
    block_scale = scale_bits.to(tl.float8e4nv, bitcast=True).to(tl.float32)
    return value * block_scale * global_scale


@triton.jit(do_not_specialize=["SLOT_COUNT"])
def _gather(
    k, v, ks, vs, hk, hv, hks, hvs, slots, req_to_token, req_indices,
    indices, seq_lens, cu_k, out_k, out_v, k_scale, v_scale,
    GPU_ROWS: tl.constexpr, HOST_ROWS: tl.constexpr,
    KDS: tl.constexpr, VDS: tl.constexpr, KSS: tl.constexpr, VSS: tl.constexpr,
    HKDS: tl.constexpr, HVDS: tl.constexpr, HKSS: tl.constexpr, HVSS: tl.constexpr,
    OKS: tl.constexpr, OVS: tl.constexpr, DIM: tl.constexpr,
    SELECTED: tl.constexpr, SLOT_COUNT, SLOT_STRIDE: tl.constexpr,
    BATCH: tl.constexpr, TOPK: tl.constexpr, REQ_ROWS: tl.constexpr,
    REQ_WIDTH: tl.constexpr, RS: tl.constexpr, RIS: tl.constexpr,
    IS: tl.constexpr, LS: tl.constexpr, CS: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    head = tl.program_id(1)
    d = tl.arange(0, DIM)
    if SELECTED:
        lo, hi = 0, BATCH
        while lo < hi:
            mid = (lo + hi) // 2
            end = tl.load(cu_k + (mid + 1) * CS)
            before = row < end
            hi = tl.where(before, mid, hi)
            lo = tl.where(before, lo, mid + 1)
        batch = lo
        active = batch < BATCH
        start = tl.load(cu_k + batch * CS, mask=active, other=0)
        col = row - start
        req = tl.load(req_indices + batch * RIS, mask=active, other=-1)
        length = tl.load(seq_lens + batch * LS, mask=active, other=0)
        valid = active & (col >= 0) & (col < TOPK) & (req >= 0) & (req < REQ_ROWS)
        position = tl.load(indices + batch * IS[0] + col * IS[1], mask=valid, other=-1)
        valid = valid & (position >= 0) & (position < length) & (position < REQ_WIDTH)
        slot = tl.load(req_to_token + req.to(tl.int64) * RS[0] + position.to(tl.int64) * RS[1],
                       mask=valid, other=-1).to(tl.int64)
    else:
        slot = tl.load(slots + row * SLOT_STRIDE, mask=row < SLOT_COUNT, other=-1).to(tl.int64)
    key = _load_nvfp4(k, ks, hk, hks, slot, head, d, True, k_scale,
                     GPU_ROWS, HOST_ROWS, KDS, KSS, HKDS, HKSS)
    value = _load_nvfp4(v, vs, hv, hvs, slot, head, d, True, v_scale,
                       GPU_ROWS, HOST_ROWS, VDS, VSS, HVDS, HVSS)
    tl.store(out_k + row * OKS[0] + head * OKS[1] + d * OKS[2], key)
    tl.store(out_v + row * OVS[0] + head * OVS[1] + d * OVS[2], value)


def _check_banks(k, v, ks, vs, k_scale, v_scale, host_k, host_v, host_ks, host_vs):
    if k.ndim != 3 or not k.is_cuda or k.dtype != torch.uint8:
        raise ValueError("NVFP4 data must be CUDA uint8 NHD/2 banks.")
    rows, heads, half_dim = k.shape
    dim = half_dim * 2
    if heads == 0 or dim < 16 or dim & (dim - 1):
        raise ValueError("NVFP4 requires nonempty heads and a power-of-two head dimension >= 16.")
    for bank, shape in ((v, k.shape), (ks, (rows, heads, dim // 16)),
                        (vs, (rows, heads, dim // 16))):
        if bank.shape != shape or bank.dtype != torch.uint8 or bank.device != k.device:
            raise ValueError("NVFP4 data and block-scale banks must have matching uint8 layouts.")
    host = (host_k, host_v, host_ks, host_vs)
    if any(t is not None for t in host) and not all(t is not None for t in host):
        raise ValueError("NVFP4 host data and both block-scale banks must be supplied together.")
    host_rows = 0
    if host_k is not None:
        host_rows = host_k.shape[0]
        for bank, width in zip(host, (half_dim, half_dim, dim // 16, dim // 16)):
            if (bank.shape != (host_rows, heads, width) or bank.dtype != torch.uint8
                    or bank.device.type != "cpu" or (host_rows and not bank.is_pinned())):
                raise ValueError("NVFP4 host banks must be pinned uint8 NHD data/scales.")
    for bank in (k, v, ks, vs, *host):
        if bank is not None and any(s <= 0 for s in bank.stride()):
            raise ValueError("NVFP4 banks require positive strides.")
    for value in (k_scale, v_scale):
        if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
            raise ValueError("NVFP4 global scales must be finite positive CPU numbers.")
    return dim, host_rows


def _metadata(tensor, device, ndim):
    if (tensor.device != device or tensor.ndim != ndim
            or tensor.dtype not in (torch.int32, torch.int64)
            or any(s <= 0 for s in tensor.stride())):
        raise ValueError("NVFP4 mapping metadata must be positive-stride CUDA int32/int64 tensors.")


def _outputs(k, dim, out_k, out_v, minimum):
    rows = out_k.shape[0]
    for out in (out_k, out_v):
        if (out.shape != (rows, k.shape[1], dim) or rows < minimum
                or out.device != k.device or out.dtype not in (torch.float16, torch.bfloat16, torch.float32)
                or any(s <= 0 for s in out.stride())):
            raise ValueError("NVFP4 outputs must hold matching CUDA NHD decoded rows.")
    return rows


def _bank_args(k, v, ks, vs, host_k, host_v, host_ks, host_vs):
    host = (host_k, host_v, host_ks, host_vs)
    pointers = tuple(t.data_ptr() if t is not None else 0 for t in host)
    strides = tuple(t.stride() if t is not None else (0, 0, 0) for t in host)
    return pointers, (k.stride(), v.stride(), ks.stride(), vs.stride(), *strides)


@triton.jit
def _compact_indices(indices, lengths, output, WIDTH: tl.constexpr,
                     IS0: tl.constexpr, IS1: tl.constexpr, LS: tl.constexpr,
                     OS0: tl.constexpr, OS1: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    col = tl.arange(0, BLOCK)
    position = tl.load(indices + row * IS0 + col * IS1, mask=col < WIDTH, other=-1)
    length = tl.load(lengths + row * LS)
    valid = (col < WIDTH) & (position >= 0) & (position < length)
    count = tl.sum(valid.to(tl.int32), 0)
    rank = tl.cumsum(valid.to(tl.int32), 0) - 1
    # A permutation of [0, WIDTH): no initialization/scatter race, even when
    # draft-tail selections follow invalid frozen-prefix columns.
    destination = tl.where(valid, rank, count + col - rank - 1)
    tl.store(output + row * OS0 + destination * OS1,
             tl.where(valid, position, -1), mask=col < WIDTH)


def compact_nvfp4_indices(indices, lengths, output):
    """Stable causal compaction into caller-owned graph scratch."""
    _metadata(indices, indices.device, 2)
    _metadata(lengths, indices.device, 1)
    _metadata(output, indices.device, 2)
    if (not indices.is_cuda or output.shape != indices.shape
            or lengths.numel() != indices.shape[0]):
        raise ValueError("NVFP4 compact indices require matching CUDA query rows.")
    if indices.shape[0] and indices.shape[1]:
        _compact_indices[(indices.shape[0],)](
            indices, lengths, output, indices.shape[1], *indices.stride(),
            lengths.stride(0), *output.stride(),
            BLOCK=triton.next_power_of_2(indices.shape[1]), num_warps=4,
        )


def gather_nvfp4_rows(k, v, ks, vs, slots, out_k, out_v, *, k_scale: float,
                      v_scale: float, host_k=None, host_v=None, host_ks=None, host_vs=None):
    """Decode only ``slots``; zero invalid slots and the unused output tail.

    The slot count is a runtime bound so growing staged prefixes reuse a kernel.
    """
    dim, host_rows = _check_banks(k, v, ks, vs, k_scale, v_scale, host_k, host_v, host_ks, host_vs)
    _metadata(slots, k.device, 1)
    rows = _outputs(k, dim, out_k, out_v, slots.numel())
    if not rows:
        return
    pointers, strides = _bank_args(k, v, ks, vs, host_k, host_v, host_ks, host_vs)
    _gather[(rows, k.shape[1])](
        k, v, ks, vs, *pointers, slots, slots, slots, slots, slots, slots,
        out_k, out_v, float(k_scale), float(v_scale), k.shape[0], host_rows,
        *strides, out_k.stride(), out_v.stride(), dim,
        False, slots.numel(), slots.stride(0), 0, 0, 0, 0, (0, 0), 0, (0, 0), 0, 0,
        num_warps=4,
    )


def gather_nvfp4_selected(k, v, ks, vs, req_to_token, req_indices, indices,
                          seq_lens, cu_k, out_k, out_v, *, k_scale: float,
                          v_scale: float, host_k=None, host_v=None, host_ks=None, host_vs=None):
    """Decode the existing packed/page-padded selected-row plan, without scratch.

    ``cu_k`` is a monotonic prefix plan beginning at zero. Padding selections
    must be -1. Rows beyond its final boundary are zeroed as well.
    """
    dim, host_rows = _check_banks(k, v, ks, vs, k_scale, v_scale, host_k, host_v, host_ks, host_vs)
    for tensor, ndim in ((req_to_token, 2), (req_indices, 1), (indices, 2), (seq_lens, 1), (cu_k, 1)):
        _metadata(tensor, k.device, ndim)
    batch, topk = indices.shape
    if req_indices.numel() != batch or seq_lens.numel() != batch or cu_k.numel() != batch + 1:
        raise ValueError("NVFP4 selected mapping batch dimensions must match.")
    rows = _outputs(k, dim, out_k, out_v, 0)
    if not rows:
        return
    pointers, strides = _bank_args(k, v, ks, vs, host_k, host_v, host_ks, host_vs)
    _gather[(rows, k.shape[1])](
        k, v, ks, vs, *pointers, req_indices, req_to_token, req_indices,
        indices, seq_lens, cu_k, out_k, out_v, float(k_scale), float(v_scale),
        k.shape[0], host_rows, *strides, out_k.stride(), out_v.stride(), dim,
        True, 0, 0, batch, topk, *req_to_token.shape, req_to_token.stride(),
        req_indices.stride(0), indices.stride(), seq_lens.stride(0), cu_k.stride(0), num_warps=4,
    )


def sparse_gqa_nvfp4_paged(q, k, v, ks, vs, indices, cu_q, cu_k, kv_lens,
                           scale, kv_locations, *, max_q: int, k_scale: float,
                           v_scale: float, host_k=None, host_v=None, host_ks=None, host_vs=None):
    """Attend directly to selected NVFP4 rows, never materializing a KV prefix.

    ``cu_k`` indexes ``kv_locations``. Indices are logical sequence positions;
    future positions, invalid physical slots and empty rows are masked/zeroed.
    The caller supplies max_q and valid monotonic sequence boundaries on CUDA.
    """
    from sglang.srt.layers.attention.qsa.sparse_attn import (
        _get_best_config,
        _sparse_gqa_chunk_prefill,
    )

    dim, host_rows = _check_banks(k, v, ks, vs, k_scale, v_scale, host_k, host_v, host_ks, host_vs)
    if (q.ndim != 3 or q.device != k.device or q.dtype not in (torch.float16, torch.bfloat16)
            or q.shape[2] != dim or q.shape[1] == 0 or q.shape[1] % k.shape[1]
            or any(s <= 0 for s in q.stride())):
        raise ValueError("NVFP4 QSA requires CUDA FP16/BF16 NHD queries with grouped KV heads.")
    if type(max_q) is not int or max_q < 0 or (q.shape[0] and not max_q):
        raise ValueError("NVFP4 QSA max_q must be a nonnegative CPU integer.")
    for tensor in (cu_q, cu_k, kv_lens, kv_locations):
        _metadata(tensor, q.device, 1)
    _metadata(indices, q.device, indices.ndim)
    batch = cu_q.numel() - 1
    if (batch < 0 or cu_k.numel() != batch + 1 or kv_lens.numel() != batch
            or indices.ndim not in (2, 3) or indices.shape[0] != q.shape[0]
            or (indices.ndim == 3 and indices.shape[1] != k.shape[1])):
        raise ValueError("NVFP4 QSA metadata shapes must match queries and requests.")
    if type(scale) not in (int, float) or not math.isfinite(scale):
        raise ValueError("NVFP4 QSA attention scale must be a finite CPU number.")
    out = torch.empty_like(q)
    if not q.shape[0]:
        return out
    if not batch:
        raise ValueError("Nonempty NVFP4 queries require request boundaries.")
    pointers, strides = _bank_args(k, v, ks, vs, host_k, host_v, host_ks, host_vs)
    hk, hv, hks, hvs = pointers
    _, _, _, _, hkds, hvds, hkss, hvss = strides
    group_size = q.shape[1] // k.shape[1]
    block_n, warps, stages = _get_best_config(q.shape[0])
    _sparse_gqa_chunk_prefill[(max_q, batch * k.shape[1])](
        q, k, v, out, indices, cu_q, cu_k, kv_lens, scale, indices.shape[-1],
        *q.stride(), *k.stride(), *v.stride(), *out.stride(),
        indices.stride(0), indices.stride(1) if indices.ndim == 3 else 0,
        indices.stride(-1), NUM_KV_HEADS=k.shape[1], GROUP_SIZE=group_size,
        BLOCK_M=max(16, triton.next_power_of_2(group_size)), BLOCK_N=block_n,
        HEAD_DIM=dim, PAGED=True, kv_locations=kv_locations, host_k=hk, host_v=hv,
        k_scale=float(k_scale), v_scale=float(v_scale), HAS_HOST=host_rows > 0,
        GPU_ROWS=k.shape[0], HOST_ROWS=host_rows, LOCATION_COUNT=kv_locations.numel(),
        sl_n=kv_locations.stride(0), scq_n=cu_q.stride(0), sck_n=cu_k.stride(0),
        skl_n=kv_lens.stride(0), NVFP4=True, k_scales=ks, v_scales=vs,
        host_k_scales=hks, host_v_scales=hvs, KSS=ks.stride(), VSS=vs.stride(),
        HKDS=hkds, HVDS=hvds, HKSS=hkss, HVSS=hvss,
        num_warps=warps, num_stages=stages,
    )
    return out
