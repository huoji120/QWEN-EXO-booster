"""Selected QSA gather: old four-launch sequence versus fused mapping/K/V.

Warm runs replay a fixed read set. Cold runs flush L2 before replaying a
pre-captured workload (flush excluded from timing). No production KV is used.
The pool is capped at 32768 rows, one layer, plus marker's bounded L2 flush.
"""

import os

import torch
import triton
import triton.language as tl

from sglang.jit_kernel.benchmark import marker
from sglang.srt.mem_cache.qwen_host_kv_pool import QwenHostFP8KVPool
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=120, stage="base-b-kernel-benchmark", runner_config="1-gpu-large")


@triton.jit
def old_mapping(req_to_token, reqs, indices, lengths, cu, slots,
                TOPK: tl.constexpr, RS: tl.constexpr, IS: tl.constexpr,
                BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    start = tl.load(cu + row)
    count = tl.load(cu + row + 1) - start
    length = tl.load(lengths + row)
    req = tl.load(reqs + row)
    positions = tl.load(indices + row * IS + cols, mask=cols < TOPK, other=-1)
    valid = (cols < count) & (positions >= 0) & (positions < length)
    selected = tl.load(req_to_token + req * RS + tl.where(valid, positions, 0),
                       mask=valid, other=-1)
    tl.store(slots + start + cols, selected, mask=(cols < TOPK) & (cols < count))


@marker.parametrize("batch", [1, 2, 6, 10], [1, 2])
@marker.parametrize("verify_width", [1, 4], [4])
@marker.parametrize("topk", [128, 512, 2048, 2051], [2048, 2051])
@marker.parametrize("host_fraction", [0.0, 0.25, 0.5, 1.0], [0.0, 0.5])
@marker.parametrize("read_set", ["warm", "cold"], ["warm"])
@marker.benchmark("implementation", ["old", "fused"])
def benchmark(batch, verify_width, topk, host_fraction, read_set, implementation):
    # Flash-Next TP1 full KV geometry: 2 heads x 256; indexer geometry differs.
    # Override only when testing a different checkpoint/TP geometry.
    heads = int(os.environ.get("QSA_BENCH_KV_HEADS", "2"))
    dim = int(os.environ.get("QSA_BENCH_HEAD_DIM", "256"))
    size = 32768
    if heads * dim > 4096:
        raise ValueError("Geometry exceeds the bounded benchmark allocation")
    pool = QwenHostFP8KVPool(
        size=size, gpu_size=size // 2, page_size=64,
        dtype=torch.float8_e4m3fn, head_num=heads, head_dim=dim,
        v_head_dim=dim, layer_num=1, device="cuda", enable_memory_saver=False,
        enable_alt_stream=False, enable_kv_cache_copy=True, kv_cache_layout="nhd",
        start_layer=0, end_layer=0,
    )
    try:
        for value, bank in enumerate((pool.k_buffer[0], pool.v_buffer[0],
                                      pool.gpu_k_buffer[0], pool.gpu_v_buffer[0])):
            bank.fill_(48 + value * 4)
        rows = batch * verify_width
        stride = triton.cdiv(topk, 64) * 64
        positions = torch.arange(topk, device="cuda").expand(rows, -1).contiguous()
        reqs = torch.arange(rows, device="cuda", dtype=torch.int32)
        lengths = torch.full((rows,), topk, device="cuda", dtype=torch.int32)
        cu = torch.arange(rows + 1, device="cuda", dtype=torch.int32) * stride
        ids = torch.arange(rows * topk, device="cuda").reshape(rows, topk)
        # Interleave host rows rather than clustering all host traffic by CTA.
        on_host = ids.remainder(4) < round(host_fraction * 4)
        mapping = torch.where(on_host, pool.gpu_rows + ids * 17 % (size + 64 - pool.gpu_rows),
                              64 + ids * 17 % (pool.gpu_rows - 64)).to(torch.int32)
        outputs = tuple(torch.empty((rows * stride, heads, dim), device="cuda", dtype=torch.bfloat16)
                        for _ in range(2))
        slots = torch.empty(rows * stride, device="cuda", dtype=torch.int32)

        def old():
            slots.fill_(-1)
            old_mapping[(rows, triton.cdiv(topk, 16))](
                mapping, reqs, positions, lengths, cu, slots, topk, mapping.stride(0),
                positions.stride(0), 16)
            pool.gather_into(0, slots, *outputs)
            outputs[0].mul_(0.75)
            outputs[1].mul_(1.5)

        def fused():
            pool.gather_selected_into(0, mapping, reqs, positions, lengths, cu, *outputs)
            outputs[0].mul_(0.75)
            outputs[1].mul_(1.5)

        fn = old if implementation == "old" else fused
        # Also check each benchmark fixture numerically before timing it.
        old()
        reference = tuple(t.clone() for t in outputs)
        fused()
        for actual, expected in zip(outputs, reference):
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        del reference
        if read_set == "cold":
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(3):
                    fn()
            torch.cuda.current_stream().wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                fn()
            fn = graph.replay
        return marker.do_bench(
            fn, use_cuda_graph=read_set == "warm", warmup_iters=5, replay_iters=50,
            graph_clone_args=None, graph_clone_kwargs=None,
            memory_args=None, memory_output=outputs, disable_log_bandwidth=True,
        )
    finally:
        pool._clear_buffers()


if __name__ == "__main__":
    benchmark.run()
