"""Attention KV cache adapters for the MLX backend."""

from __future__ import annotations

from typing import TYPE_CHECKING

import mlx.core as mx

if TYPE_CHECKING:
    from sglang.srt.hardware_backend.mlx.kv_cache.attention_kv_pool import (
        MlxAttentionKVPool,
    )


class AttentionOffsetCache:
    """Data-free shim satisfying mlx-lm's cache protocol.

    Provides ``make_mask`` and ``state`` without storing actual K/V.
    """

    def __init__(self, offset: int = 0):
        self.offset = offset

    @property
    def state(self):
        return ()  # Empty — safe for mx.eval unpacking

    def make_mask(self, N, **kwargs):
        return None if N == 1 else "causal"

    def update_and_fetch(self, keys, values):
        raise RuntimeError("AttentionOffsetCache should not store data")


_DEFAULT_MAX_SEQ_LEN = 4096


class ContiguousAttentionKVCache:
    """Pre-allocated attention KV buffer for one request and one layer.

    Shape ``(1, n_kv_heads, max_seq_len, head_dim)``.  Slice assignment
    instead of ``mx.concatenate``.  Lazy-allocated on first write.
    """

    __slots__ = ("keys", "max_seq_len", "offset", "values")

    def __init__(
        self,
        n_kv_heads: int | None = None,
        head_dim: int | None = None,
        max_seq_len: int = _DEFAULT_MAX_SEQ_LEN,
        dtype: mx.Dtype | None = None,
    ):
        if n_kv_heads is not None and head_dim is not None and dtype is not None:
            self.keys = mx.zeros((1, n_kv_heads, max_seq_len, head_dim), dtype=dtype)
            self.values = mx.zeros((1, n_kv_heads, max_seq_len, head_dim), dtype=dtype)
        else:
            self.keys = None
            self.values = None
        self.offset = 0
        self.max_seq_len = max_seq_len

    def _allocate(self, keys: mx.array) -> None:
        """Allocate buffers matching the first key tensor's shape."""
        B, n_kv_heads, _, head_dim = keys.shape
        self.keys = mx.zeros(
            (B, n_kv_heads, self.max_seq_len, head_dim), dtype=keys.dtype
        )
        self.values = mx.zeros(
            (B, n_kv_heads, self.max_seq_len, head_dim), dtype=keys.dtype
        )

    @property
    def state(self):
        """Arrays for ``mx.eval`` unpacking."""
        if self.keys is None:
            return ()
        return (self.keys, self.values)

    def make_mask(self, N, **kwargs):
        return None if N == 1 else "causal"

    def _grow(self, required: int) -> None:
        """Double the buffer until it can hold *required* tokens."""
        new_max = self.max_seq_len
        while new_max < required:
            new_max *= 2
        B, n_kv_heads, _, head_dim = self.keys.shape
        new_k = mx.zeros((B, n_kv_heads, new_max, head_dim), dtype=self.keys.dtype)
        new_v = mx.zeros((B, n_kv_heads, new_max, head_dim), dtype=self.values.dtype)
        if self.offset > 0:
            new_k[:, :, : self.offset, :] = self.keys[:, :, : self.offset, :]
            new_v[:, :, : self.offset, :] = self.values[:, :, : self.offset, :]
        self.keys = new_k
        self.values = new_v
        self.max_seq_len = new_max

    def update_and_fetch(
        self, keys: mx.array, values: mx.array
    ) -> tuple[mx.array, mx.array]:
        """Append K/V and return all valid K/V up to current offset."""
        if self.keys is None:
            self._allocate(keys)
        S = keys.shape[2]
        end = self.offset + S
        if end > self.max_seq_len:
            self._grow(end)
        self.keys[:, :, self.offset : end, :] = keys
        self.values[:, :, self.offset : end, :] = values
        self.offset = end
        return self.keys[:, :, :end, :], self.values[:, :, :end, :]

    def write_token(self, k: mx.array, v: mx.array) -> None:
        """Write one token. k, v shape: (1, n_kv_heads, 1, head_dim)."""
        end = self.offset + 1
        if end > self.max_seq_len:
            self._grow(end)
        self.keys[:, :, self.offset : end, :] = k
        self.values[:, :, self.offset : end, :] = v
        self.offset = end

    def get_kv(self) -> tuple[mx.array, mx.array]:
        """Return valid K/V: (1, n_kv_heads, offset, head_dim)."""
        return self.keys[:, :, : self.offset, :], self.values[:, :, : self.offset, :]

    def get_kv_slice(self, start: int, end: int) -> tuple[mx.array, mx.array]:
        """Return a token slice without changing the cache offset."""
        return self.keys[:, :, start:end, :], self.values[:, :, start:end, :]


class QuantizedContiguousAttentionKVCache:
    """Per-request MXFP8 attention cache.

    MLX stores four E4M3 values in each ``uint32`` plus one E8M0 scale per
    32-value group. Values are dequantized to the model compute dtype only for
    the valid prefix consumed by attention, keeping the persistent cache close
    to one byte per element instead of two.
    """

    __slots__ = (
        "_bits",
        "_group_size",
        "_mode",
        "dtype",
        "key_scales",
        "keys",
        "max_seq_len",
        "offset",
        "value_scales",
        "values",
    )

    def __init__(
        self,
        n_kv_heads: int | None = None,
        head_dim: int | None = None,
        max_seq_len: int = _DEFAULT_MAX_SEQ_LEN,
        dtype: mx.Dtype = mx.float16,
        *,
        group_size: int = 32,
        bits: int = 8,
        mode: str = "mxfp8",
    ):
        if mode != "mxfp8" or group_size != 32 or bits != 8:
            raise ValueError("MLX quantized KV cache currently supports mxfp8 only")
        self.keys = None
        self.key_scales = None
        self.values = None
        self.value_scales = None
        self.offset = 0
        self.max_seq_len = max_seq_len
        self.dtype = dtype
        # mlx-lm treats any cache exposing a public ``bits`` attribute as its
        # own affine QuantizedKVCache and routes to affine quantized_matmul.
        # Keep MXFP8 metadata private because this adapter dequantizes before
        # returning K/V to SDPA.
        self._group_size = group_size
        self._bits = bits
        self._mode = mode
        if n_kv_heads is not None and head_dim is not None:
            self._allocate_shape(1, n_kv_heads, head_dim)

    def _allocate_shape(self, batch: int, n_kv_heads: int, head_dim: int) -> None:
        if head_dim % self._group_size != 0:
            raise ValueError(
                f"MXFP8 KV head_dim={head_dim} must be divisible by "
                f"group_size={self._group_size}"
            )
        packed_dim = head_dim * self._bits // 32
        scale_dim = head_dim // self._group_size
        packed_shape = (batch, n_kv_heads, self.max_seq_len, packed_dim)
        scale_shape = (batch, n_kv_heads, self.max_seq_len, scale_dim)
        self.keys = mx.zeros(packed_shape, dtype=mx.uint32)
        self.key_scales = mx.zeros(scale_shape, dtype=mx.uint8)
        self.values = mx.zeros(packed_shape, dtype=mx.uint32)
        self.value_scales = mx.zeros(scale_shape, dtype=mx.uint8)

    def _allocate(self, keys: mx.array) -> None:
        batch, n_kv_heads, _, head_dim = keys.shape
        self.dtype = keys.dtype
        self._allocate_shape(batch, n_kv_heads, head_dim)

    @property
    def state(self):
        if self.keys is None:
            return ()
        return self.keys, self.key_scales, self.values, self.value_scales

    def make_mask(self, N, **kwargs):
        return None if N == 1 else "causal"

    def _grow(self, required: int) -> None:
        new_max = self.max_seq_len
        while new_max < required:
            new_max *= 2

        def grow(array: mx.array) -> mx.array:
            shape = list(array.shape)
            shape[2] = new_max
            expanded = mx.zeros(tuple(shape), dtype=array.dtype)
            if self.offset > 0:
                expanded[:, :, : self.offset, :] = array[:, :, : self.offset, :]
            return expanded

        self.keys = grow(self.keys)
        self.key_scales = grow(self.key_scales)
        self.values = grow(self.values)
        self.value_scales = grow(self.value_scales)
        self.max_seq_len = new_max

    def _quantize(self, value: mx.array) -> tuple[mx.array, mx.array]:
        packed, scales = mx.quantize(
            value,
            group_size=self._group_size,
            bits=self._bits,
            mode=self._mode,
        )
        return packed, scales

    def _dequantize(self, packed: mx.array, scales: mx.array) -> mx.array:
        return mx.dequantize(
            packed,
            scales,
            group_size=self._group_size,
            bits=self._bits,
            mode=self._mode,
            dtype=self.dtype,
        )

    def update_and_fetch(
        self, keys: mx.array, values: mx.array
    ) -> tuple[mx.array, mx.array]:
        self._append(keys, values)
        return self.get_kv()

    def _append(self, keys: mx.array, values: mx.array) -> None:
        if self.keys is None:
            self._allocate(keys)
        steps = keys.shape[2]
        end = self.offset + steps
        if end > self.max_seq_len:
            self._grow(end)
        packed_k, scales_k = self._quantize(keys)
        packed_v, scales_v = self._quantize(values)
        self.keys[:, :, self.offset : end, :] = packed_k
        self.key_scales[:, :, self.offset : end, :] = scales_k
        self.values[:, :, self.offset : end, :] = packed_v
        self.value_scales[:, :, self.offset : end, :] = scales_v
        self.offset = end

    def write_token(self, k: mx.array, v: mx.array) -> None:
        self._append(k, v)

    def get_kv(self) -> tuple[mx.array, mx.array]:
        return self.get_kv_slice(0, self.offset)

    def get_kv_slice(self, start: int, end: int) -> tuple[mx.array, mx.array]:
        return (
            self._dequantize(
                self.keys[:, :, start:end, :],
                self.key_scales[:, :, start:end, :],
            ),
            self._dequantize(
                self.values[:, :, start:end, :],
                self.value_scales[:, :, start:end, :],
            ),
        )


class PoolBackedAttentionKVCache:
    """Lazily gathers cached attention KV from the shared pool during forward.

    Each ``update_and_fetch`` gathers this layer's prefix from the pool
    on demand, keeping operations in the lazy compute graph.  Convert to
    ``ContiguousAttentionKVCache`` via ``to_contiguous`` after the forward pass.
    """

    __slots__ = (
        "_full_keys",
        "_full_values",
        "_layer_idx",
        "_new_keys",
        "_new_values",
        "_pool",
        "_slots",
        "offset",
    )

    def __init__(
        self,
        pool: MlxAttentionKVPool,
        layer_idx: int,
        slots: mx.array,
        prefix_len: int,
    ):
        self._pool = pool
        self._layer_idx = layer_idx
        self._slots = slots
        self.offset = prefix_len
        self._full_keys: mx.array | None = None
        self._full_values: mx.array | None = None
        self._new_keys: mx.array | None = None
        self._new_values: mx.array | None = None

    @property
    def keys(self) -> mx.array | None:
        return self._full_keys

    @property
    def values(self) -> mx.array | None:
        return self._full_values

    @property
    def state(self):
        if self._full_keys is not None:
            return (self._full_keys, self._full_values)
        return ()

    def make_mask(self, N, **kwargs):
        return None if N == 1 else "causal"

    def update_and_fetch(
        self, keys: mx.array, values: mx.array
    ) -> tuple[mx.array, mx.array]:
        """Gather cached prefix from pool, concatenate with new K/V."""
        S = keys.shape[2]

        if self.offset > 0:
            k_cached, v_cached = self._pool.get_kv(
                self._layer_idx, self._slots[: self.offset]
            )
            # Pool layout (S, n_kv_heads, head_dim) → cache (1, n_kv_heads, S, head_dim)
            k_cached = k_cached.transpose(1, 0, 2)[None]
            v_cached = v_cached.transpose(1, 0, 2)[None]
            k_all = mx.concatenate([k_cached, keys], axis=2)
            v_all = mx.concatenate([v_cached, values], axis=2)
        else:
            k_all = keys
            v_all = values

        self.offset += S
        self._full_keys = k_all
        self._full_values = v_all
        self._new_keys = keys
        self._new_values = values
        return k_all, v_all

    def to_contiguous(
        self,
        max_seq_len: int = 4096,
        quantization_mode: str | None = None,
    ) -> ContiguousAttentionKVCache | QuantizedContiguousAttentionKVCache:
        """Convert to contiguous attention KV reusing forward-pass arrays."""
        cache = (
            QuantizedContiguousAttentionKVCache(
                max_seq_len=max_seq_len,
                dtype=(
                    self._full_keys.dtype
                    if self._full_keys is not None
                    else self._pool.dtype
                ),
            )
            if quantization_mode == "mxfp8"
            else ContiguousAttentionKVCache(max_seq_len=max_seq_len)
        )
        if self._full_keys is not None:
            cache.update_and_fetch(self._full_keys, self._full_values)
        return cache
