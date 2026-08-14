"""Flat attention KV pool for the MLX backend.

Each layer buffer has shape ``(pool_size, n_kv_heads, head_dim)``.
This v1 pool is intentionally uniform: every wrapped softmax-attention
layer must share the same KV shape and full-context KV semantics.
Heterogeneous KV shapes and sliding-window KV need per-layer/window-aware
pools before they can use MLX radix reuse.

Slot 0 is reserved as padding (1-based indexing).
"""

import logging

import mlx.core as mx

logger = logging.getLogger(__name__)


class MlxAttentionKVPool:
    """Pre-allocated attention KV pool indexed by integer slot IDs."""

    def __init__(
        self,
        pool_size: int,
        num_layers: int,
        n_kv_heads: int,
        head_dim: int,
        dtype: mx.Dtype = mx.float16,
        quantization_mode: str | None = None,
    ):
        self.pool_size = pool_size
        self.num_layers = num_layers
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self.dtype = dtype
        self.quantization_mode = quantization_mode

        if quantization_mode not in {None, "mxfp8"}:
            raise ValueError(
                "MLX attention KV pool supports only None or mxfp8 quantization"
            )
        if quantization_mode == "mxfp8" and head_dim % 32 != 0:
            raise ValueError(
                f"MXFP8 KV head_dim={head_dim} must be divisible by group_size=32"
            )

        # Per-attention-layer buffers: (pool_size, n_kv_heads, head_dim)
        packed_dim = head_dim // 4 if quantization_mode == "mxfp8" else head_dim
        buffer_dtype = mx.uint32 if quantization_mode == "mxfp8" else dtype
        self.k_buffer: list[mx.array] = [
            mx.zeros((pool_size, n_kv_heads, packed_dim), dtype=buffer_dtype)
            for _ in range(num_layers)
        ]
        self.v_buffer: list[mx.array] = [
            mx.zeros((pool_size, n_kv_heads, packed_dim), dtype=buffer_dtype)
            for _ in range(num_layers)
        ]
        scale_shape = (pool_size, n_kv_heads, head_dim // 32)
        self.k_scales: list[mx.array] = (
            [mx.zeros(scale_shape, dtype=mx.uint8) for _ in range(num_layers)]
            if quantization_mode == "mxfp8"
            else []
        )
        self.v_scales: list[mx.array] = (
            [mx.zeros(scale_shape, dtype=mx.uint8) for _ in range(num_layers)]
            if quantization_mode == "mxfp8"
            else []
        )

        mem_mb = sum(array.nbytes for array in self.all_buffers()) / (1024 * 1024)
        logger.info(
            f"MlxAttentionKVPool: {pool_size} slots x {num_layers} layers "
            f"x {n_kv_heads} heads x {head_dim} dim, "
            f"dtype={dtype}, quantization={quantization_mode or 'none'}, "
            f"~{mem_mb:.1f} MB"
        )

    @staticmethod
    def bytes_per_slot(
        *,
        num_layers: int,
        n_kv_heads: int,
        head_dim: int,
        dtype: mx.Dtype,
        quantization_mode: str | None,
    ) -> int:
        if quantization_mode == "mxfp8":
            # One byte per E4M3 element plus one E8M0 scale per group of 32,
            # for both K and V.
            return 2 * num_layers * n_kv_heads * (head_dim + head_dim // 32)
        return 2 * num_layers * n_kv_heads * head_dim * dtype.size

    def _quantize(self, value: mx.array) -> tuple[mx.array, mx.array]:
        packed, scales = mx.quantize(
            value,
            group_size=32,
            bits=8,
            mode="mxfp8",
        )
        return packed, scales

    def _dequantize(self, packed: mx.array, scales: mx.array) -> mx.array:
        return mx.dequantize(
            packed,
            scales,
            group_size=32,
            bits=8,
            mode="mxfp8",
            dtype=self.dtype,
        )

    def set_kv(self, layer_id: int, slots: mx.array, k: mx.array, v: mx.array) -> None:
        """Scatter K/V into *slots* for one layer."""
        if self.quantization_mode == "mxfp8":
            packed_k, scales_k = self._quantize(k)
            packed_v, scales_v = self._quantize(v)
            self.k_buffer[layer_id][slots] = packed_k
            self.k_scales[layer_id][slots] = scales_k
            self.v_buffer[layer_id][slots] = packed_v
            self.v_scales[layer_id][slots] = scales_v
        else:
            self.k_buffer[layer_id][slots] = k
            self.v_buffer[layer_id][slots] = v

    def get_kv(self, layer_id: int, slots: mx.array) -> tuple[mx.array, mx.array]:
        """Gather K/V from *slots* for one layer."""
        if self.quantization_mode == "mxfp8":
            return (
                self._dequantize(
                    self.k_buffer[layer_id][slots], self.k_scales[layer_id][slots]
                ),
                self._dequantize(
                    self.v_buffer[layer_id][slots], self.v_scales[layer_id][slots]
                ),
            )
        return self.k_buffer[layer_id][slots], self.v_buffer[layer_id][slots]

    def get_kv_all_layers(self, slots: mx.array) -> tuple[mx.array, mx.array]:
        """Gather K/V from *slots* across all layers."""
        pairs = [self.get_kv(i, slots) for i in range(self.num_layers)]
        k_all = mx.stack([pair[0] for pair in pairs])
        v_all = mx.stack([pair[1] for pair in pairs])
        return k_all, v_all

    def set_kv_all_layers(
        self, slots: mx.array, k_all: mx.array, v_all: mx.array
    ) -> None:
        """Scatter K/V into *slots* across all layers."""
        for i in range(self.num_layers):
            self.set_kv(i, slots, k_all[i], v_all[i])

    def all_buffers(self) -> list[mx.array]:
        """Return all buffer arrays (for ``mx.eval``)."""
        return self.k_buffer + self.v_buffer + self.k_scales + self.v_scales

    def clear(self) -> None:
        """Zero all buffers."""
        packed_dim = (
            self.head_dim // 4 if self.quantization_mode == "mxfp8" else self.head_dim
        )
        shape = (self.pool_size, self.n_kv_heads, packed_dim)
        buffer_dtype = mx.uint32 if self.quantization_mode == "mxfp8" else self.dtype
        for i in range(self.num_layers):
            self.k_buffer[i] = mx.zeros(shape, dtype=buffer_dtype)
            self.v_buffer[i] = mx.zeros(shape, dtype=buffer_dtype)
            if self.quantization_mode == "mxfp8":
                scale_shape = (
                    self.pool_size,
                    self.n_kv_heads,
                    self.head_dim // 32,
                )
                self.k_scales[i] = mx.zeros(scale_shape, dtype=mx.uint8)
                self.v_scales[i] = mx.zeros(scale_shape, dtype=mx.uint8)
