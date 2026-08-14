"""Correctness and memory checks for the native MLX MXFP8 KV cache."""

from __future__ import annotations

import importlib.util
import unittest

from sglang.test.ci.ci_register import register_cpu_ci, register_mlx_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")
register_mlx_ci(est_time=2, suite="stage-a-unit-test-mlx")

_HAS_MLX = importlib.util.find_spec("mlx") is not None

if _HAS_MLX:
    import mlx.core as mx
    from sglang.srt.hardware_backend.mlx.kv_cache import (
        MlxAttentionKVPool,
        QuantizedContiguousAttentionKVCache,
    )


@unittest.skipUnless(_HAS_MLX, "requires mlx")
class TestMxfp8KVCache(unittest.TestCase):
    def test_pool_roundtrip_and_memory_reduction(self):
        pool = MlxAttentionKVPool(
            pool_size=16,
            num_layers=2,
            n_kv_heads=4,
            head_dim=256,
            dtype=mx.float16,
            quantization_mode="mxfp8",
        )
        slots = mx.array([2, 5, 9], dtype=mx.int32)
        keys = mx.random.normal((3, 4, 256)).astype(mx.float16)
        values = mx.random.normal((3, 4, 256)).astype(mx.float16)
        pool.set_kv(0, slots, keys, values)
        restored_k, restored_v = pool.get_kv(0, slots)
        mx.eval(restored_k, restored_v, *pool.all_buffers())

        self.assertEqual(restored_k.shape, keys.shape)
        self.assertEqual(restored_k.dtype, mx.float16)
        self.assertLess(float(mx.mean(mx.abs(restored_k - keys)).item()), 0.05)
        self.assertLess(float(mx.mean(mx.abs(restored_v - values)).item()), 0.05)

        quantized_bytes = sum(array.nbytes for array in pool.all_buffers())
        float_bytes = 16 * 2 * 4 * 256 * 2 * mx.float16.size
        self.assertLess(quantized_bytes, float_bytes * 0.55)

    def test_contiguous_cache_appends_grows_and_slices(self):
        cache = QuantizedContiguousAttentionKVCache(
            max_seq_len=2,
            dtype=mx.float16,
        )
        self.assertFalse(
            hasattr(cache, "bits"),
            "mlx-lm reserves public cache.bits for its affine KV cache",
        )
        keys = mx.random.normal((1, 2, 3, 256)).astype(mx.float16)
        values = mx.random.normal((1, 2, 3, 256)).astype(mx.float16)
        restored_k, restored_v = cache.update_and_fetch(keys, values)
        sliced_k, sliced_v = cache.get_kv_slice(1, 3)
        mx.eval(restored_k, restored_v, sliced_k, sliced_v, *cache.state)

        self.assertEqual(cache.offset, 3)
        self.assertEqual(cache.max_seq_len, 4)
        self.assertEqual(restored_k.shape, keys.shape)
        self.assertEqual(sliced_k.shape, (1, 2, 2, 256))
        self.assertLess(float(mx.mean(mx.abs(restored_k - keys)).item()), 0.05)
        self.assertLess(
            float(mx.mean(mx.abs(sliced_v - values[:, :, 1:3, :])).item()),
            0.05,
        )


if __name__ == "__main__":
    unittest.main()
