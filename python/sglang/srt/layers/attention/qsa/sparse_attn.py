"""Validated sparse GQA operators migrated from the QSA reference branch."""

from typing import Optional

import torch
import triton
import triton.language as tl

_H20_CONFIGS = [
    (32, (32, 8, 2)),
    (64, (64, 8, 2)),
    (1024, (32, 4, 2)),
    (float("inf"), (16, 1, 2)),
]
_L20_CONFIGS = [
    (32, (32, 8, 2)),
    (64, (64, 8, 2)),
    (128, (64, 4, 2)),
    (512, (32, 4, 2)),
    (float("inf"), (16, 1, 2)),
]


def _get_best_config(total_q: int):
    table = _H20_CONFIGS if "H20" in torch.cuda.get_device_name(0) else _L20_CONFIGS
    return next(cfg for limit, cfg in table if total_q <= limit)


@triton.jit
def _sparse_gqa_prefill(
    q,
    k,
    v,
    out,
    indices,
    cu_seqlens,
    scale,
    topk,
    sq_m: tl.constexpr,
    sq_h: tl.constexpr,
    sq_d: tl.constexpr,
    sk_n: tl.constexpr,
    sk_h: tl.constexpr,
    sk_d: tl.constexpr,
    sv_n: tl.constexpr,
    sv_h: tl.constexpr,
    sv_d: tl.constexpr,
    so_m: tl.constexpr,
    so_h: tl.constexpr,
    so_d: tl.constexpr,
    si_m: tl.constexpr,
    si_g: tl.constexpr,
    si_n: tl.constexpr,
    NUM_KV_HEADS: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    batch_group = tl.program_id(1)
    group = batch_group % NUM_KV_HEADS
    batch = batch_group // NUM_KV_HEADS
    seq_start = tl.load(cu_seqlens + batch).to(tl.int64)
    seq_end = tl.load(cu_seqlens + batch + 1).to(tl.int64)
    query_relative = tl.program_id(0).to(tl.int64)
    query = seq_start + query_relative
    if query >= seq_end:
        return

    row_topk = tl.minimum(topk, query_relative + 1)
    row_limit = tl.minimum(topk, ((row_topk + BLOCK_N - 1) // BLOCK_N) * BLOCK_N)
    offs_h = tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_DIM)
    head_start = group * GROUP_SIZE
    q_values = tl.load(
        q
        + query * sq_m
        + (head_start + offs_h[:, None]) * sq_h
        + offs_d[None, :] * sq_d,
        mask=(offs_h < GROUP_SIZE)[:, None],
        other=0.0,
    )
    q_values = (q_values * scale * 1.4426950408).to(q_values.dtype)
    k_base = k + seq_start * sk_n + group * sk_h
    v_base = v + seq_start * sv_n + group * sv_h
    idx_row = indices + query * si_m + group * si_g
    max_value = tl.full([BLOCK_M], -float("inf"), tl.float32)
    normalizer = tl.zeros([BLOCK_M], tl.float32)
    accumulator = tl.zeros([BLOCK_M, HEAD_DIM], tl.float32)
    offs_n = tl.arange(0, BLOCK_N)
    for start in range(0, row_limit, BLOCK_N):
        current = start + offs_n
        token = tl.load(idx_row + current * si_n, mask=current < topk, other=-1)
        valid = token >= 0
        keys = tl.load(
            k_base + token[None, :] * sk_n + offs_d[:, None] * sk_d,
            mask=valid[None, :],
            other=0.0,
        )
        values = tl.load(
            v_base + token[:, None] * sv_n + offs_d[None, :] * sv_d,
            mask=valid[:, None],
            other=0.0,
        )
        scores = tl.where(valid[None, :], tl.dot(q_values, keys), -float("inf"))
        next_max = tl.maximum(max_value, tl.max(scores, 1))
        alpha = tl.math.exp2(max_value - next_max)
        probabilities = tl.math.exp2(scores - next_max[:, None])
        accumulator = tl.dot(
            probabilities.to(values.dtype), values, accumulator * alpha[:, None]
        )
        normalizer = normalizer * alpha + tl.sum(probabilities, 1)
        max_value = next_max
    output = accumulator / normalizer[:, None]
    tl.store(
        out
        + query * so_m
        + (head_start + offs_h[:, None]) * so_h
        + offs_d[None, :] * so_d,
        output,
        mask=(offs_h < GROUP_SIZE)[:, None],
    )


def sparse_gqa_fwd_interface_triton(q, k, v, max_seqlen_k, indices, cu_seqlens, scale):
    total_q, num_q_heads, head_dim = q.shape
    num_kv_heads = k.shape[1]
    group_size = num_q_heads // num_kv_heads
    block_m = max(16, triton.next_power_of_2(group_size))
    block_n, warps, stages = _get_best_config(total_q)
    out = torch.empty_like(q)
    _sparse_gqa_prefill[(max_seqlen_k, (cu_seqlens.shape[0] - 1) * num_kv_heads)](
        q,
        k,
        v,
        out,
        indices,
        cu_seqlens,
        scale,
        indices.shape[-1],
        q.stride(0),
        q.stride(1),
        q.stride(2),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        indices.stride(0),
        indices.stride(1) if indices.ndim == 3 else 0,
        indices.stride(2) if indices.ndim == 3 else indices.stride(1),
        NUM_KV_HEADS=num_kv_heads,
        GROUP_SIZE=group_size,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        HEAD_DIM=head_dim,
        num_warps=warps,
        num_stages=stages,
    )
    return out


@triton.jit
def _sparse_gqa_chunk_prefill(
    q,
    k,
    v,
    out,
    indices,
    cu_q,
    cu_k,
    kv_lens,
    scale,
    topk,
    sq_m: tl.constexpr,
    sq_h: tl.constexpr,
    sq_d: tl.constexpr,
    sk_n: tl.constexpr,
    sk_h: tl.constexpr,
    sk_d: tl.constexpr,
    sv_n: tl.constexpr,
    sv_h: tl.constexpr,
    sv_d: tl.constexpr,
    so_m: tl.constexpr,
    so_h: tl.constexpr,
    so_d: tl.constexpr,
    si_m: tl.constexpr,
    si_g: tl.constexpr,
    si_n: tl.constexpr,
    NUM_KV_HEADS: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    PAGED: tl.constexpr = False,
    kv_locations=None,
    host_k=0,
    host_v=0,
    k_scale=1.0,
    v_scale=1.0,
    HAS_HOST: tl.constexpr = False,
    GPU_ROWS: tl.constexpr = 0,
    HOST_ROWS: tl.constexpr = 0,
    LOCATION_COUNT: tl.constexpr = 0,
    sl_n: tl.constexpr = 1,
    scq_n: tl.constexpr = 1,
    sck_n: tl.constexpr = 1,
    skl_n: tl.constexpr = 1,
    shk_n: tl.constexpr = 0,
    shk_h: tl.constexpr = 0,
    shk_d: tl.constexpr = 0,
    shv_n: tl.constexpr = 0,
    shv_h: tl.constexpr = 0,
    shv_d: tl.constexpr = 0,
    SCALE_K: tl.constexpr = False,
    SCALE_V: tl.constexpr = False,
):
    query_relative = tl.program_id(0).to(tl.int64)
    batch_group = tl.program_id(1)
    group = batch_group % NUM_KV_HEADS
    batch = batch_group // NUM_KV_HEADS
    q_start = tl.load(cu_q + batch * scq_n)
    q_end = tl.load(cu_q + (batch + 1) * scq_n)
    query = (q_start + query_relative).to(tl.int64)
    if query >= q_end:
        return
    k_start = tl.load(cu_k + batch * sck_n).to(tl.int64)
    kv_len = tl.load(kv_lens + batch * skl_n).to(tl.int64)
    visible = query_relative + kv_len - (q_end - q_start) + 1
    row_topk = tl.minimum(topk, visible)
    row_limit = tl.minimum(topk, ((row_topk + BLOCK_N - 1) // BLOCK_N) * BLOCK_N)
    offs_h = tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_DIM)
    q_values = tl.load(
        q
        + query * sq_m
        + (group * GROUP_SIZE + offs_h[:, None]) * sq_h
        + offs_d[None, :] * sq_d,
        mask=(offs_h < GROUP_SIZE)[:, None],
        other=0.0,
    )
    q_values = (q_values * scale * 1.4426950408).to(q_values.dtype)
    if PAGED:
        k_base = k + group * sk_h
        v_base = v + group * sv_h
        if HAS_HOST:
            host_k_base = host_k.to(tl.int64).to(k.dtype) + group * shk_h
            host_v_base = host_v.to(tl.int64).to(v.dtype) + group * shv_h
    else:
        k_base = k + k_start * sk_n + group * sk_h
        v_base = v + k_start * sv_n + group * sv_h
    idx_row = indices + query * si_m + group * si_g
    max_value = tl.full([BLOCK_M], -float("inf"), tl.float32)
    normalizer = tl.zeros([BLOCK_M], tl.float32)
    accumulator = tl.zeros([BLOCK_M, HEAD_DIM], tl.float32)
    offs_n = tl.arange(0, BLOCK_N)
    for start in range(0, row_limit, BLOCK_N):
        current = start + offs_n
        token = tl.load(idx_row + current * si_n, mask=current < topk, other=-1)
        valid = token >= 0
        if PAGED:
            logical = k_start + token.to(tl.int64)
            valid = valid & (token < kv_len) & (logical >= 0)
            valid = valid & (logical < LOCATION_COUNT)
            slot = tl.load(
                kv_locations + logical * sl_n,
                mask=valid,
                other=-1,
            ).to(tl.int64)
            valid = valid & (slot >= 0) & (slot < GPU_ROWS + HOST_ROWS)
            gpu_valid = valid & (slot < GPU_ROWS)
            # FP32 is exact for stored FP8 and avoids unsupported direct FP8
            # conversions on SM120. Round to q.dtype before applying scales.
            keys = tl.load(
                k_base + slot[None, :] * sk_n + offs_d[:, None] * sk_d,
                mask=gpu_valid[None, :],
                other=0.0,
            ).to(tl.float32)
            values = tl.load(
                v_base + slot[:, None] * sv_n + offs_d[None, :] * sv_d,
                mask=gpu_valid[:, None],
                other=0.0,
            ).to(tl.float32)
            if HAS_HOST:
                host_valid = valid & (slot >= GPU_ROWS)
                host_slot = slot - GPU_ROWS
                host_keys = tl.load(
                    host_k_base
                    + host_slot[None, :] * shk_n
                    + offs_d[:, None] * shk_d,
                    mask=host_valid[None, :],
                    other=0.0,
                ).to(tl.float32)
                host_values = tl.load(
                    host_v_base
                    + host_slot[:, None] * shv_n
                    + offs_d[None, :] * shv_d,
                    mask=host_valid[:, None],
                    other=0.0,
                ).to(tl.float32)
                keys = tl.where(host_valid[None, :], host_keys, keys)
                values = tl.where(host_valid[:, None], host_values, values)
            keys = keys.to(q_values.dtype)
            values = values.to(q_values.dtype)
            if SCALE_K:
                keys = (keys.to(tl.float32) * k_scale).to(q_values.dtype)
            if SCALE_V:
                values = (values.to(tl.float32) * v_scale).to(q_values.dtype)
        else:
            keys = tl.load(
                k_base + token[None, :] * sk_n + offs_d[:, None] * sk_d,
                mask=valid[None, :],
                other=0.0,
            )
            values = tl.load(
                v_base + token[:, None] * sv_n + offs_d[None, :] * sv_d,
                mask=valid[:, None],
                other=0.0,
            )
        scores = tl.where(valid[None, :], tl.dot(q_values, keys), -float("inf"))
        next_max = tl.maximum(max_value, tl.max(scores, 1))
        alpha = tl.math.exp2(max_value - next_max)
        probabilities = tl.math.exp2(scores - next_max[:, None])
        accumulator = tl.dot(
            probabilities.to(values.dtype), values, accumulator * alpha[:, None]
        )
        normalizer = normalizer * alpha + tl.sum(probabilities, 1)
        max_value = next_max
    output = accumulator / normalizer[:, None]
    tl.store(
        out
        + query * so_m
        + (group * GROUP_SIZE + offs_h[:, None]) * so_h
        + offs_d[None, :] * so_d,
        output,
        mask=(offs_h < GROUP_SIZE)[:, None],
    )


def sparse_gqa_fwd_interface_triton_ck(
    q, k, v, indices, cu_q, cu_k, kv_lens, scale, *, max_q: int
):
    k, v = k.contiguous(), v.contiguous()
    total_q, num_q_heads, head_dim = q.shape
    num_kv_heads = k.shape[1]
    group_size = num_q_heads // num_kv_heads
    block_m = max(16, triton.next_power_of_2(group_size))
    block_n, warps, stages = _get_best_config(total_q)
    out = torch.empty_like(q)
    _sparse_gqa_chunk_prefill[(max_q, (cu_q.shape[0] - 1) * num_kv_heads)](
        q,
        k,
        v,
        out,
        indices,
        cu_q,
        cu_k,
        kv_lens,
        scale,
        indices.shape[-1],
        q.stride(0),
        q.stride(1),
        q.stride(2),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        indices.stride(0),
        indices.stride(1) if indices.ndim == 3 else 0,
        indices.stride(2) if indices.ndim == 3 else indices.stride(1),
        NUM_KV_HEADS=num_kv_heads,
        GROUP_SIZE=group_size,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        HEAD_DIM=head_dim,
        num_warps=warps,
        num_stages=stages,
    )
    return out


def sparse_gqa_fwd_interface_triton_paged(
    q,
    k,
    v,
    indices,
    cu_q,
    cu_k,
    kv_lens,
    scale,
    kv_locations,
    *,
    max_q: int,
    k_scale: Optional[float] = None,
    v_scale: Optional[float] = None,
    host_k=None,
    host_v=None,
):
    """Attend to selected logical positions directly in raw NHD KV banks.

    ``cu_k`` indexes the flattened ``kv_locations`` table, not the raw banks.
    Slots below ``k.shape[0]`` are on CUDA; remaining slots index the optional
    pinned host suffix. FP8 byte banks must be viewed as their storage dtype.
    The caller owns host backing lifetime and supplies a CPU ``max_q`` at least
    as large as every query segment. Sequence metadata, integer selections and
    their causal validity retain the dense interface's caller contract. Scales
    are Python numbers, not device tensors. No device-to-host metadata read,
    KV/position copy or full-prefix temporary is created.
    """
    query_dtypes = (torch.float16, torch.bfloat16, torch.float32)
    storage_dtypes = (*query_dtypes, torch.float8_e4m3fn, torch.float8_e5m2)
    if q.ndim != 3 or not q.is_cuda or q.dtype not in query_dtypes:
        raise ValueError("Paged QSA queries must be CUDA NHD FP16/BF16/FP32.")
    total_q, num_q_heads, head_dim = q.shape
    if head_dim < 16 or head_dim & (head_dim - 1):
        raise ValueError("Paged QSA head dimension must be a power of two >= 16.")
    if (
        k.ndim != 3
        or v.shape != k.shape
        or k.shape[1] == 0
        or k.shape[2] != head_dim
        or k.device != q.device
        or v.device != q.device
        or k.dtype not in storage_dtypes
        or v.dtype != k.dtype
        or not k.is_contiguous()
        or not v.is_contiguous()
    ):
        raise ValueError("Paged QSA requires matching contiguous CUDA NHD KV banks.")
    num_kv_heads = k.shape[1]
    if num_q_heads == 0 or num_q_heads % num_kv_heads:
        raise ValueError("Paged QSA query heads must be a multiple of KV heads.")
    if type(max_q) is not int or max_q < 0 or (total_q and max_q == 0):
        raise ValueError("Paged QSA max_q must be a nonnegative CPU integer.")
    batch = cu_q.numel() - 1
    if (host_k is None) != (host_v is None):
        raise ValueError("Paged QSA host K and V must be supplied together.")
    if host_k is not None:
        if host_k.shape != host_v.shape:
            raise ValueError("Paged QSA host KV banks must have matching shapes.")
        for tensor in (host_k, host_v):
            if (
                tensor.ndim != 3
                or tensor.shape[1:] != k.shape[1:]
                or tensor.device.type != "cpu"
                or tensor.dtype != k.dtype
                or not tensor.is_contiguous()
                or (tensor.numel() and not tensor.is_pinned())
            ):
                raise ValueError(
                    "Paged QSA host KV must be pinned contiguous NHD with CUDA-bank dtype."
                )
    if not (k_scale is None or isinstance(k_scale, (int, float))) or not (
        v_scale is None or isinstance(v_scale, (int, float))
    ):
        raise ValueError("Paged QSA KV scales must be Python numbers or None.")

    out = torch.empty_like(q)
    if total_q == 0:
        return out
    group_size = num_q_heads // num_kv_heads
    block_m = max(16, triton.next_power_of_2(group_size))
    block_n, warps, stages = _get_best_config(total_q)
    if (
        head_dim == 256
        and q.dtype in (torch.float16, torch.bfloat16)
        and torch.cuda.get_device_capability(q.device) == (12, 0)
    ):
        # Indirect FP8 loads need more warps than the dense 16-token tile.
        # A 64-token/two-stage tile exceeds SM120 shared-memory capacity.
        block_n, warps, stages = 32, 4, 2
    has_host = host_k is not None and host_k.shape[0] > 0
    _sparse_gqa_chunk_prefill[(max_q, batch * num_kv_heads)](
        q,
        k,
        v,
        out,
        indices,
        cu_q,
        cu_k,
        kv_lens,
        scale,
        indices.shape[-1],
        *q.stride(),
        *k.stride(),
        *v.stride(),
        *out.stride(),
        indices.stride(0),
        indices.stride(1) if indices.ndim == 3 else 0,
        indices.stride(2) if indices.ndim == 3 else indices.stride(1),
        NUM_KV_HEADS=num_kv_heads,
        GROUP_SIZE=group_size,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        HEAD_DIM=head_dim,
        PAGED=True,
        kv_locations=kv_locations,
        host_k=host_k.data_ptr() if has_host else 0,
        host_v=host_v.data_ptr() if has_host else 0,
        k_scale=1.0 if k_scale is None else k_scale,
        v_scale=1.0 if v_scale is None else v_scale,
        HAS_HOST=has_host,
        GPU_ROWS=k.shape[0],
        HOST_ROWS=host_k.shape[0] if has_host else 0,
        LOCATION_COUNT=kv_locations.numel(),
        sl_n=kv_locations.stride(0),
        scq_n=cu_q.stride(0),
        sck_n=cu_k.stride(0),
        skl_n=kv_lens.stride(0),
        shk_n=host_k.stride(0) if has_host else 0,
        shk_h=host_k.stride(1) if has_host else 0,
        shk_d=host_k.stride(2) if has_host else 0,
        shv_n=host_v.stride(0) if has_host else 0,
        shv_h=host_v.stride(1) if has_host else 0,
        shv_d=host_v.stride(2) if has_host else 0,
        SCALE_K=k_scale is not None,
        SCALE_V=v_scale is not None,
        num_warps=warps,
        num_stages=stages,
    )
    return out


@triton.jit
def _fa2_valid_counts(
    seq_lens,
    indices,
    counts,
    topk: tl.constexpr,
    stride_i: tl.constexpr,
    BLOCK_TOPK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_TOPK)
    length = tl.load(seq_lens + row)
    positions = tl.load(
        indices + row * stride_i + cols,
        mask=cols < topk,
        other=-1,
    )
    valid = (positions >= 0) & (positions < length)
    tl.store(counts + row, tl.sum(valid.to(tl.int32), axis=0))


@triton.jit
def _fa2_prefix_sum(counts, cu_k, batch, BLOCK_B: tl.constexpr):
    rows = tl.arange(0, BLOCK_B)
    valid_rows = rows < batch
    row_counts = tl.load(counts + rows, mask=valid_rows, other=0)
    tl.store(cu_k, 0)
    tl.store(cu_k + rows + 1, tl.cumsum(row_counts, 0), mask=valid_rows)


def qwen_sparse_fa2_cu_seqlens_triton(
    seq_lens, indices, counts, cu_k, batch, topk, block_b: Optional[int] = None
):
    block_b = block_b or triton.next_power_of_2(batch)
    # Count one request per program. The previous implementation formed a
    # [next_power_of_2(topk), next_power_of_2(batch)] tensor in one program;
    # topk=2051 and batch=512 therefore exceeded Triton's 1M-element limit.
    _fa2_valid_counts[(batch,)](
        seq_lens,
        indices,
        counts,
        topk,
        indices.stride(0),
        BLOCK_TOPK=triton.next_power_of_2(topk),
        num_warps=8,
    )
    # Prefix sum is only over the batch dimension and remains a small 1-D
    # tensor, including during CUDA graph capture.
    _fa2_prefix_sum[(1,)](
        counts,
        cu_k,
        batch,
        BLOCK_B=block_b,
        num_warps=8,
    )


@triton.jit
def _compact_kv(
    k,
    v,
    req_to_token,
    req_indices,
    indices,
    seq_lens,
    cu_k,
    out_k,
    out_v,
    topk: tl.constexpr,
    heads: tl.constexpr,
    dim: tl.constexpr,
    req_stride: tl.constexpr,
    idx_stride: tl.constexpr,
    BLOCK_TOPK: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    batch, head, block = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    cols = block * BLOCK_TOPK + tl.arange(0, BLOCK_TOPK)
    dims = tl.arange(0, BLOCK_D)
    length = tl.load(seq_lens + batch)
    req = tl.load(req_indices + batch)
    pack_start = tl.load(cu_k + batch)
    valid_count = tl.load(cu_k + batch + 1) - pack_start
    positions = tl.load(indices + batch * idx_stride + cols, mask=cols < topk, other=-1)
    valid = (cols < valid_count) & (positions >= 0) & (positions < length)
    slots = tl.load(
        req_to_token + req * req_stride + tl.where(valid, positions, 0),
        mask=valid,
        other=0,
    )
    src = slots[:, None] * heads * dim + head * dim + dims[None, :]
    dst = (pack_start + cols)[:, None] * heads * dim + head * dim + dims[None, :]
    mask = valid[:, None] & (dims[None, :] < dim)
    tl.store(out_k + dst, tl.load(k + src, mask=mask, other=0.0), mask=mask)
    tl.store(out_v + dst, tl.load(v + src, mask=mask, other=0.0), mask=mask)


def qwen_sparse_valid_counts_triton(seq_lens, indices, counts, batch, topk):
    """Valid-count pass alone, for consumers that need per-row lengths but
    not the packed cu_seqlens prefix sum (trtllm paged decode packs rows at
    a fixed page-aligned stride instead)."""
    _fa2_valid_counts[(batch,)](
        seq_lens,
        indices,
        counts,
        topk,
        indices.stride(0),
        BLOCK_TOPK=triton.next_power_of_2(topk),
        num_warps=8,
    )


def qwen_sparse_kv_extraction_compact_triton(
    k, v, req_to_token, req_indices, indices, seq_lens, cu_k, out_k, out_v, batch, topk
):
    _, heads, dim = k.shape
    block_topk = 16
    _compact_kv[(batch, heads, triton.cdiv(topk, block_topk))](
        k,
        v,
        req_to_token,
        req_indices,
        indices,
        seq_lens,
        cu_k,
        out_k,
        out_v,
        topk,
        heads,
        dim,
        req_to_token.stride(0),
        indices.stride(0),
        BLOCK_TOPK=block_topk,
        BLOCK_D=triton.next_power_of_2(dim),
        num_warps=8,
    )


__all__ = [
    "qwen_sparse_fa2_cu_seqlens_triton",
    "qwen_sparse_valid_counts_triton",
    "qwen_sparse_kv_extraction_compact_triton",
    "sparse_gqa_fwd_interface_triton",
    "sparse_gqa_fwd_interface_triton_ck",
    "sparse_gqa_fwd_interface_triton_paged",
]
