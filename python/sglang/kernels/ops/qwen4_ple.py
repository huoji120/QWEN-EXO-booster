"""Fused kernels for Qwen4 PLE hashing, state movement and row decoding."""

from __future__ import annotations

import torch
import triton
import triton.language as tl

_QWEN4_NGRAM_SIZE = 3
_QWEN4_HEADS_PER_NGRAM = 8
_QWEN4_NGRAM_HEADS = 16
_QWEN4_HC_COUNT = 4
_QWEN4_HIDDEN_SIZE = 2560
_QWEN4_MAX_SHORT_CONV_STATE_LEN = 16


@triton.jit
def _round_bf16_to_fp32(value):
    """RNE-round fp32 to BF16 precision while retaining an fp32 register."""

    bits = value.to(tl.int32, bitcast=True)
    rounding_bias = 0x7FFF + ((bits >> 16) & 1)
    rounded_bits = (bits + rounding_bias) & -65536
    return rounded_bits.to(tl.float32, bitcast=True)


@triton.jit
def _qwen4_ngram_hash_kernel(
    contexts_ptr,
    multipliers_ptr,
    vocab_sizes_ptr,
    offsets_ptr,
    output_ptr,
    num_outputs,
    eos_token_id,
    NGRAM_SIZE: tl.constexpr,
    HEADS_PER_NGRAM: tl.constexpr,
    NGRAM_HEADS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    output_idx = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = output_idx < num_outputs
    token_idx = output_idx // NGRAM_HEADS
    head_idx = output_idx % NGRAM_HEADS
    context_base = token_idx * NGRAM_SIZE

    token_0 = tl.load(contexts_ptr + context_base, mask=mask, other=0)
    token_1 = tl.load(contexts_ptr + context_base + 1, mask=mask, other=0)
    token_2 = tl.load(contexts_ptr + context_base + 2, mask=mask, other=0)
    multiplier_0 = tl.load(multipliers_ptr)
    multiplier_1 = tl.load(multipliers_ptr + 1)
    multiplier_2 = tl.load(multipliers_ptr + 2)

    # Only the final position of each three-token context is materialized. Its
    # shift-1 source is valid unless the immediately preceding token is EOS;
    # its shift-2 source is valid only if neither preceding token is EOS.
    previous_1 = tl.where(token_1 == eos_token_id, eos_token_id, token_1)
    previous_2 = tl.where(
        (token_0 == eos_token_id) | (token_1 == eos_token_id),
        eos_token_id,
        token_0,
    )
    mixed = (token_2 * multiplier_0) ^ (previous_1 * multiplier_1)
    mixed_3 = mixed ^ (previous_2 * multiplier_2)
    mixed = tl.where(head_idx < HEADS_PER_NGRAM, mixed, mixed_3)

    vocab_size = tl.load(vocab_sizes_ptr + head_idx, mask=mask, other=1)
    offset = tl.load(offsets_ptr + head_idx, mask=mask, other=0)
    tl.store(output_ptr + output_idx, mixed % vocab_size + offset, mask=mask)


def can_fuse_qwen4_ngram_hash(
    contexts: torch.Tensor,
    multipliers: torch.Tensor,
    vocab_sizes: torch.Tensor,
    offsets: torch.Tensor,
) -> bool:
    """Return whether inputs match the fixed Qwen4 PLE hash contract."""

    return (
        contexts.is_cuda
        and contexts.dtype == torch.long
        and contexts.dim() == 2
        and contexts.shape[1] == _QWEN4_NGRAM_SIZE
        and contexts.is_contiguous()
        and multipliers.is_cuda
        and multipliers.dtype == torch.long
        and multipliers.numel() == _QWEN4_NGRAM_SIZE
        and vocab_sizes.is_cuda
        and vocab_sizes.dtype == torch.long
        and vocab_sizes.numel() == _QWEN4_NGRAM_HEADS
        and offsets.is_cuda
        and offsets.dtype == torch.long
        and offsets.numel() == _QWEN4_NGRAM_HEADS
    )


def fused_qwen4_ngram_hash(
    contexts: torch.Tensor,
    multipliers: torch.Tensor,
    vocab_sizes: torch.Tensor,
    offsets: torch.Tensor,
    eos_token_id: int,
) -> torch.Tensor:
    """Return the 16 Qwen4 PLE N-gram IDs in one kernel launch."""

    if not can_fuse_qwen4_ngram_hash(contexts, multipliers, vocab_sizes, offsets):
        raise ValueError("unsupported input for fused Qwen4 PLE N-gram hash")
    output = torch.empty(
        (contexts.shape[0], _QWEN4_NGRAM_HEADS),
        dtype=torch.long,
        device=contexts.device,
    )
    num_outputs = output.numel()
    if num_outputs:
        block_size = 256
        _qwen4_ngram_hash_kernel[(triton.cdiv(num_outputs, block_size),)](
            contexts,
            multipliers,
            vocab_sizes,
            offsets,
            output,
            num_outputs,
            eos_token_id,
            NGRAM_SIZE=_QWEN4_NGRAM_SIZE,
            HEADS_PER_NGRAM=_QWEN4_HEADS_PER_NGRAM,
            NGRAM_HEADS=_QWEN4_NGRAM_HEADS,
            BLOCK_SIZE=block_size,
            num_warps=4,
        )
    return output


@triton.jit
def _qwen4_gate_value_kernel(
    gate_ptr,
    value_ptr,
    output_ptr,
    num_tokens,
    HC_COUNT: tl.constexpr,
    HIDDEN_SIZE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    token_group = tl.program_id(0)
    token = token_group // HC_COUNT
    hidden = tl.arange(0, BLOCK_SIZE)
    mask = (token < num_tokens) & (hidden < HIDDEN_SIZE)

    # `gate` is already the BF16 output of the existing multiply, reduction, and
    # division kernels.  Reproduce every remaining eager BF16 rounding boundary
    # while broadcasting the scalar over its 2,560-element value group.
    gate = tl.load(gate_ptr + token_group).to(tl.float32)
    magnitude = tl.maximum(tl.abs(gate), 1.0e-6)
    root = _round_bf16_to_fp32(tl.sqrt(magnitude))
    sign = tl.where(gate > 0.0, 1.0, tl.where(gate < 0.0, -1.0, 0.0))
    transformed = _round_bf16_to_fp32(root * sign)
    activated = _round_bf16_to_fp32(tl.sigmoid(transformed))

    value = tl.load(value_ptr + token * HIDDEN_SIZE + hidden, mask=mask, other=0.0).to(
        tl.float32
    )
    output_offset = token_group * HIDDEN_SIZE + hidden
    tl.store(output_ptr + output_offset, activated * value, mask=mask)


def can_fuse_qwen4_gate_value(gate: torch.Tensor, value: torch.Tensor) -> bool:
    """Return whether inputs match Qwen4's fixed BF16 gate/value contract."""

    return (
        gate.is_cuda
        and gate.dtype == torch.bfloat16
        and gate.dim() == 3
        and gate.shape[1:] == (_QWEN4_HC_COUNT, 1)
        and gate.is_contiguous()
        and value.is_cuda
        and value.dtype == gate.dtype
        and value.shape == (gate.shape[0], _QWEN4_HIDDEN_SIZE)
        and value.is_contiguous()
    )


def fused_qwen4_gate_value(gate: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
    """Apply Qwen4's post-reduction gate and value broadcast in one kernel."""

    if not can_fuse_qwen4_gate_value(gate, value):
        raise ValueError("unsupported input for fused Qwen4 PLE gate/value")
    output = torch.empty(
        (gate.shape[0], _QWEN4_HC_COUNT, _QWEN4_HIDDEN_SIZE),
        dtype=value.dtype,
        device=value.device,
    )
    if gate.shape[0]:
        _qwen4_gate_value_kernel[(gate.shape[0] * _QWEN4_HC_COUNT,)](
            gate,
            value,
            output,
            gate.shape[0],
            HC_COUNT=_QWEN4_HC_COUNT,
            HIDDEN_SIZE=_QWEN4_HIDDEN_SIZE,
            BLOCK_SIZE=4096,
            num_warps=8,
        )
    return output


@triton.jit
def _qwen4_short_conv_state_kernel(
    state_ptr,
    state_indices_ptr,
    x_ptr,
    conv_input_ptr,
    num_tokens,
    CHANNELS: tl.constexpr,
    STATE_LEN: tl.constexpr,
    BLOCK_CHANNELS: tl.constexpr,
    BLOCK_STATE_LEN: tl.constexpr,
):
    token = tl.program_id(0)
    channel = tl.program_id(1) * BLOCK_CHANNELS + tl.arange(0, BLOCK_CHANNELS)[:, None]
    state_col = tl.arange(0, BLOCK_STATE_LEN)[None, :]
    channel_mask = (token < num_tokens) & (channel < CHANNELS)
    state_mask = channel_mask & (state_col < STATE_LEN)
    state_index = tl.load(state_indices_ptr + token, mask=token < num_tokens, other=0)
    state_base = state_index * CHANNELS * STATE_LEN
    state_offset = state_base + channel * STATE_LEN + state_col
    output_base = token * CHANNELS * (STATE_LEN + 1)
    output_offset = output_base + channel * (STATE_LEN + 1) + state_col

    # Each lane owns one complete channel history.  Materialize every old value
    # before advancing the in-place cache so the convolution input is exactly
    # the native index_select + cat result.
    old_state = tl.load(state_ptr + state_offset, mask=state_mask, other=0.0)
    tl.store(conv_input_ptr + output_offset, old_state, mask=state_mask)
    x = tl.load(x_ptr + token * CHANNELS + channel, mask=channel_mask, other=0.0)
    tl.store(
        conv_input_ptr + output_base + channel * (STATE_LEN + 1) + STATE_LEN,
        x,
        mask=channel_mask,
    )
    tl.debug_barrier()

    # Slot zero is the CUDA-graph padding slot and can occur in multiple rows.
    # Its post-step value is unobservable; skipping it also avoids duplicate
    # writers.  Real request slots are unique within a decode batch.
    update_mask = state_mask & (state_col < STATE_LEN - 1) & (state_index != 0)
    next_value = tl.load(
        conv_input_ptr + output_offset + 1, mask=update_mask, other=0.0
    )
    tl.store(state_ptr + state_offset, next_value, mask=update_mask)
    tl.store(
        state_ptr + state_base + channel * STATE_LEN + STATE_LEN - 1,
        x,
        mask=channel_mask & (state_index != 0),
    )


def can_fuse_qwen4_short_conv_state(
    state: torch.Tensor,
    state_indices: torch.Tensor,
    x: torch.Tensor,
) -> bool:
    """Return whether decode state movement can use the exact fused kernel."""

    return (
        state.is_cuda
        and state.dtype in (torch.bfloat16, torch.float16)
        and state.dim() == 3
        and state.is_contiguous()
        and 0 < state.shape[2] <= _QWEN4_MAX_SHORT_CONV_STATE_LEN
        and state_indices.is_cuda
        and state_indices.dtype == torch.long
        and state_indices.dim() == 1
        and state_indices.is_contiguous()
        and x.is_cuda
        and x.dtype == state.dtype
        and x.dim() == 2
        and x.is_contiguous()
        and x.shape == (state_indices.shape[0], state.shape[1])
    )


def fused_qwen4_short_conv_state(
    state: torch.Tensor,
    state_indices: torch.Tensor,
    x: torch.Tensor,
) -> torch.Tensor:
    """Build ``[selected state, x]`` and advance real decode slots in one launch."""

    if not can_fuse_qwen4_short_conv_state(state, state_indices, x):
        raise ValueError("unsupported input for fused Qwen4 short-conv state")
    state_len = state.shape[2]
    conv_input = torch.empty(
        (x.shape[0], x.shape[1], state_len + 1),
        dtype=x.dtype,
        device=x.device,
    )
    if x.shape[0]:
        block_channels = 128
        block_state_len = triton.next_power_of_2(state_len)
        _qwen4_short_conv_state_kernel[
            (x.shape[0], triton.cdiv(x.shape[1], block_channels))
        ](
            state,
            state_indices,
            x,
            conv_input,
            x.shape[0],
            CHANNELS=state.shape[1],
            STATE_LEN=state_len,
            BLOCK_CHANNELS=block_channels,
            BLOCK_STATE_LEN=block_state_len,
            num_warps=8,
        )
    return conv_input


# Packed prefill port: sgl-project/sglang PR #43032,
# head 723037ce4f52515b9c24ae6326381fe9c34f8bfb.
@triton.jit
def _qwen4_varlen_conv_kernel(
    x_ptr,
    weight_ptr,
    state_ptr,
    state_indices_ptr,
    query_start_loc_ptr,
    req_indices_ptr,
    token_offsets_ptr,
    output_ptr,
    num_tokens,
    CHANNELS: tl.constexpr,
    KERNEL_SIZE: tl.constexpr,
    DILATION: tl.constexpr,
    STATE_LEN: tl.constexpr,
    BLOCK_CHANNELS: tl.constexpr,
):
    token = tl.program_id(0).to(tl.int64)
    channel = tl.program_id(1) * BLOCK_CHANNELS + tl.arange(0, BLOCK_CHANNELS)
    mask = (token < num_tokens) & (channel < CHANNELS)
    req = tl.load(req_indices_ptr + token, mask=token < num_tokens, other=0)
    token_offset = tl.load(token_offsets_ptr + token, mask=token < num_tokens, other=0)
    seq_start = tl.load(query_start_loc_ptr + req, mask=token < num_tokens, other=0)
    state_index = tl.load(state_indices_ptr + req, mask=token < num_tokens, other=0)
    acc = tl.zeros((BLOCK_CHANNELS,), dtype=tl.float32)
    for weight_col in tl.static_range(0, KERNEL_SIZE):
        source_col = token_offset - (KERNEL_SIZE - 1 - weight_col) * DILATION
        from_tokens = source_col >= 0
        token_value = tl.load(
            x_ptr + (seq_start + source_col) * CHANNELS + channel,
            mask=mask & from_tokens,
            other=0.0,
        ).to(tl.float32)
        state_value = tl.load(
            state_ptr
            + state_index * CHANNELS * STATE_LEN
            + channel * STATE_LEN
            + STATE_LEN
            + source_col,
            mask=mask & ~from_tokens,
            other=0.0,
        ).to(x_ptr.dtype.element_ty)
        state_value = state_value.to(tl.float32)
        weight = tl.load(
            weight_ptr + channel * KERNEL_SIZE + weight_col,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        acc += tl.where(from_tokens, token_value, state_value) * weight
    tl.store(output_ptr + token * CHANNELS + channel, acc, mask=mask)


@triton.jit
def _qwen4_varlen_state_writeback_kernel(
    x_ptr,
    state_ptr,
    state_indices_ptr,
    query_start_loc_ptr,
    track_indices_ptr,
    track_offsets_ptr,
    CHANNELS: tl.constexpr,
    STATE_LEN: tl.constexpr,
    HAS_TRACK: tl.constexpr,
    BLOCK_CHANNELS: tl.constexpr,
    BLOCK_STATE_LEN: tl.constexpr,
):
    req = tl.program_id(0)
    channel_1d = tl.program_id(1) * BLOCK_CHANNELS + tl.arange(0, BLOCK_CHANNELS)
    channel = channel_1d[:, None]
    state_col = tl.arange(0, BLOCK_STATE_LEN)[None, :]
    channel_mask_1d = channel_1d < CHANNELS
    channel_mask = channel_mask_1d[:, None]
    state_mask = channel_mask & (state_col < STATE_LEN)
    state_index = tl.load(state_indices_ptr + req)
    seq_start = tl.load(query_start_loc_ptr + req)
    length = tl.load(query_start_loc_ptr + req + 1) - seq_start
    state_base = state_index * CHANNELS * STATE_LEN

    # A row can shift by less than STATE_LEN, so every old value must be
    # materialized before this program starts overwriting the in-place slot.
    old_state = tl.load(
        state_ptr + state_base + channel * STATE_LEN + state_col,
        mask=state_mask,
        other=0.0,
    ).to(x_ptr.dtype.element_ty)
    tl.debug_barrier()

    for output_col in tl.static_range(0, STATE_LEN):
        source_col = length - STATE_LEN + output_col
        old_col = STATE_LEN + source_col
        old_value = tl.sum(tl.where(state_col == old_col, old_state, 0.0), axis=1)
        token_value = tl.load(
            x_ptr + (seq_start + source_col) * CHANNELS + channel_1d,
            mask=channel_mask_1d & (source_col >= 0),
            other=0.0,
        )
        value = tl.where(source_col >= 0, token_value, old_value)
        tl.store(
            state_ptr + state_base + channel_1d * STATE_LEN + output_col,
            value,
            mask=channel_mask_1d & (length > 0) & (state_index != 0),
        )

    if HAS_TRACK:
        track_index = tl.load(track_indices_ptr + req)
        track_offset = tl.load(track_offsets_ptr + req)
        track_base = track_index * CHANNELS * STATE_LEN
        for output_col in tl.static_range(0, STATE_LEN):
            source_col = track_offset - STATE_LEN + output_col
            old_col = STATE_LEN + source_col
            old_value = tl.sum(tl.where(state_col == old_col, old_state, 0.0), axis=1)
            token_value = tl.load(
                x_ptr + (seq_start + source_col) * CHANNELS + channel_1d,
                mask=channel_mask_1d & (source_col >= 0),
                other=0.0,
            )
            value = tl.where(source_col >= 0, token_value, old_value)
            tl.store(
                state_ptr + track_base + channel_1d * STATE_LEN + output_col,
                value,
                mask=channel_mask_1d & (length > 0) & (track_index != 0),
            )


def can_fuse_qwen4_varlen_conv(
    x,
    weight,
    state,
    state_indices,
    query_start_loc,
    req_indices,
    token_offsets,
    dilation,
):
    """Return whether ordinary PLE prefill can use the packed CUDA path."""

    requests = state_indices.shape[0] if state_indices.ndim == 1 else -1
    return (
        x.is_cuda
        and torch.version.hip is None
        and x.dtype == torch.bfloat16
        and x.ndim == 2
        and x.shape[1] > 0
        and x.is_contiguous()
        and weight.device == x.device
        and weight.dtype == torch.bfloat16
        and weight.ndim == 3
        and weight.shape[:2] == (x.shape[1], 1)
        and weight.is_contiguous()
        and state.device == x.device
        and state.dtype in (torch.bfloat16, torch.float32)
        and state.ndim == 3
        and state.shape[1] == x.shape[1]
        and 0 < state.shape[2] <= _QWEN4_MAX_SHORT_CONV_STATE_LEN
        and state.is_contiguous()
        and isinstance(dilation, int)
        and dilation > 0
        and state.shape[2] == (weight.shape[2] - 1) * dilation
        and state_indices.device == x.device
        and state_indices.dtype == torch.int64
        and state_indices.ndim == 1
        and state_indices.is_contiguous()
        and query_start_loc.device == x.device
        and query_start_loc.dtype == torch.int64
        and query_start_loc.shape == (requests + 1,)
        and query_start_loc.is_contiguous()
        and req_indices.device == x.device
        and req_indices.dtype == torch.int64
        and req_indices.shape == (x.shape[0],)
        and req_indices.is_contiguous()
        and token_offsets.device == x.device
        and token_offsets.dtype == torch.int64
        and token_offsets.shape == (x.shape[0],)
        and token_offsets.is_contiguous()
    )


def fused_qwen4_varlen_conv(
    x,
    weight,
    state,
    state_indices,
    query_start_loc,
    req_indices,
    token_offsets,
    dilation,
    *,
    track_indices=None,
    track_offsets=None,
):
    """Compute packed PLE convolution and update main/radix state slots.

    This raw API intentionally validates only tensor metadata that is available
    without a device-to-host synchronization. The caller must guarantee that
    ``query_start_loc`` is a nondecreasing prefix sum starting at zero and
    ending at ``x.shape[0]``; every ``req_indices[t]`` is in
    ``[0, state_indices.shape[0])``; and every ``token_offsets[t]`` is in
    ``[0, query_start_loc[req + 1] - query_start_loc[req])`` for
    ``req = req_indices[t]``.

    The caller must also guarantee that nonzero main slots in
    ``state_indices`` are unique, nonzero radix slots in ``track_indices`` are
    unique, and the two sets of nonzero slots are disjoint. Slot zero is
    reserved for dummy or empty rows and is never written. These value-level
    contracts are not checked here because doing so would introduce host
    synchronization.
    """

    if not can_fuse_qwen4_varlen_conv(
        x,
        weight,
        state,
        state_indices,
        query_start_loc,
        req_indices,
        token_offsets,
        dilation,
    ):
        raise ValueError("unsupported input for Qwen4 packed-varlen convolution")
    has_track = track_indices is not None or track_offsets is not None
    if has_track and (
        track_indices is None
        or track_offsets is None
        or track_indices.device != x.device
        or track_offsets.device != x.device
        or track_indices.dtype != torch.int64
        or track_offsets.dtype != torch.int64
        or track_indices.shape != state_indices.shape
        or track_offsets.shape != state_indices.shape
        or not track_indices.is_contiguous()
        or not track_offsets.is_contiguous()
    ):
        raise ValueError("invalid Qwen4 packed-varlen track metadata")

    output = torch.empty_like(x)
    block_channels = 128
    if x.shape[0]:
        _qwen4_varlen_conv_kernel[
            (x.shape[0], triton.cdiv(x.shape[1], block_channels))
        ](
            x,
            weight,
            state,
            state_indices,
            query_start_loc,
            req_indices,
            token_offsets,
            output,
            x.shape[0],
            CHANNELS=x.shape[1],
            KERNEL_SIZE=weight.shape[2],
            DILATION=dilation,
            STATE_LEN=state.shape[2],
            BLOCK_CHANNELS=block_channels,
        )

    requests = state_indices.shape[0]
    if requests:
        writeback_channels = 64
        _qwen4_varlen_state_writeback_kernel[
            (requests, triton.cdiv(x.shape[1], writeback_channels))
        ](
            x,
            state,
            state_indices,
            query_start_loc,
            track_indices if has_track else state_indices,
            track_offsets if has_track else state_indices,
            CHANNELS=x.shape[1],
            STATE_LEN=state.shape[2],
            HAS_TRACK=has_track,
            BLOCK_CHANNELS=writeback_channels,
            BLOCK_STATE_LEN=triton.next_power_of_2(state.shape[2]),
        )
    return output



@triton.jit
def _qwen4_fp8_e4m3_rowscale_kernel(
    rows_ptr,
    scales_ptr,
    output_ptr,
    num_rows,
    hidden_size,
    BLOCK_SIZE: tl.constexpr,
):
    """Decode E4M3FN bytes and apply one FP32 scale per row."""

    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = (row < num_rows) & (cols < hidden_size)
    bits = tl.load(
        rows_ptr + row * hidden_size + cols, mask=mask, other=0
    ).to(tl.int32)
    exponent = (bits >> 3) & 0xF
    mantissa = bits & 0x7
    subnormal = mantissa.to(tl.float32) * (2.0 ** -9)
    # Normal E4M3FN values map exactly to FP32 exponent/mantissa bits.
    normal_bits = ((exponent + 120) << 23) | (mantissa << 20)
    normal = normal_bits.to(tl.float32, bitcast=True)
    value = tl.where(exponent == 0, subnormal, normal)
    # Exponent=15/mantissa=7 is NaN; retain sign bits, including negative zero.
    value_bits = value.to(tl.int32, bitcast=True)
    value_bits = tl.where(
        (exponent == 15) & (mantissa == 7), 0x7FC00000, value_bits
    )
    value_bits = value_bits | ((bits & 0x80) << 24)
    value = value_bits.to(tl.float32, bitcast=True)
    scale = tl.load(scales_ptr + row).to(tl.float32)
    # Triton emits the default IEEE FP32 round-to-nearest multiply (without
    # fast-math/FTZ), matching the CPU decoder's FP32 product.
    scaled = value * scale
    # CUDA multiplication may canonicalize NaN to an all-ones payload; the
    # integer RNE bias would overflow that payload into signed zero.
    rounded = tl.where(scaled != scaled, scaled, _round_bf16_to_fp32(scaled))
    tl.store(
        output_ptr + row * hidden_size + cols,
        rounded,
        mask=mask,
    )


def can_fuse_qwen4_fp8_rowscale(
    rows: torch.Tensor, scales: torch.Tensor, output: torch.Tensor
) -> bool:
    """Return whether the native FP8 rowscale decode contract is supported."""

    return (
        rows.is_cuda
        and rows.dtype == torch.uint8
        and rows.dim() == 2
        and rows.is_contiguous()
        and scales.is_cuda
        and scales.dtype == torch.float32
        and scales.dim() == 1
        and scales.is_contiguous()
        and output.is_cuda
        and output.dtype == torch.bfloat16
        and output.dim() == 2
        and output.is_contiguous()
        and rows.shape == output.shape
        and scales.shape[0] == rows.shape[0]
        and rows.device == scales.device == output.device
    )


def fused_qwen4_fp8_rowscale(
    rows: torch.Tensor, scales: torch.Tensor, output: torch.Tensor
) -> torch.Tensor:
    """Decode native FP8 rows directly into an existing contiguous BF16 buffer."""

    if not can_fuse_qwen4_fp8_rowscale(rows, scales, output):
        raise ValueError("unsupported input for fused Qwen4 FP8 rowscale decode")
    num_rows, hidden_size = rows.shape
    if num_rows:
        block_size = triton.next_power_of_2(hidden_size)
        _qwen4_fp8_e4m3_rowscale_kernel[(num_rows,)](
            rows,
            scales,
            output,
            num_rows,
            hidden_size,
            BLOCK_SIZE=block_size,
            num_warps=4 if block_size <= 256 else 8,
        )
    return output
