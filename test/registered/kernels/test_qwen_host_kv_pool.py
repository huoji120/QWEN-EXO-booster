"""Numerical CUDA regressions for GPU-first FP8 KV with pinned overflow."""

from types import SimpleNamespace

import pytest
import torch

from sglang.srt.mem_cache.qwen_host_kv_pool import QwenHostFP8KVPool
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=30, stage="base-b", runner_config="1-gpu-small")
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.fixture
def make_pool():
    pools = []

    def create(dtype=torch.float8_e4m3fn, *, size=64, gpu_size=0, page_size=4,
               layers=2, k_dim=256, v_dim=128):
        pool = QwenHostFP8KVPool(
            size=size, gpu_size=gpu_size, page_size=page_size, dtype=dtype, head_num=2,
            head_dim=k_dim, v_head_dim=v_dim, layer_num=layers, device="cuda",
            enable_memory_saver=False, enable_alt_stream=False,
            enable_kv_cache_copy=True, kv_cache_layout="nhd",
            start_layer=3, end_layer=3 + layers - 1,
        )
        pools.append(pool)
        return pool

    yield create
    for pool in pools:
        pool._clear_buffers()


def values(rows, dim, offset=0):
    return (
        torch.arange(rows * 2 * dim, device="cuda").reshape(rows, 2, dim)
        .remainder(127).float().sub(63).div(16).add(offset)
    ).to(torch.bfloat16)


@pytest.mark.parametrize("dtype", [torch.float8_e4m3fn, torch.float8_e5m2])
@pytest.mark.parametrize("output_dtype", [torch.bfloat16, torch.float8_e4m3fn])
@pytest.mark.parametrize("gpu_size", [0, 32, 64])
def test_host_scatter_gather_graph_reorders_and_zeroes_padding(make_pool, dtype, output_dtype, gpu_size):
    pool = make_pool(dtype, gpu_size=gpu_size)
    layer = SimpleNamespace(layer_id=3)
    # Deliberately non-contiguous source rows: the writer must preserve QKV
    # view strides, not assume that every token is packed back-to-back.
    k_storage = torch.empty((4, 2, 512), device="cuda", dtype=dtype)
    v_storage = torch.empty((4, 2, 256), device="cuda", dtype=dtype)
    keys, vals = k_storage[:, :, ::2], v_storage[:, :, ::2]
    keys.copy_(values(4, 256))
    vals.copy_(values(4, 128, 1))
    # Native MTP transposes [request, step] and supplies strided slot views.
    write_slots = torch.empty(12, device="cuda", dtype=torch.int64)[::3]
    selected = torch.empty(14, device="cuda", dtype=torch.int32)[::2]
    write_slots.copy_(torch.tensor([4, 35, 36, 67], device="cuda"))
    selected.copy_(torch.tensor([36, -1, 4, 35, 67, 68, -3], device="cuda", dtype=torch.int32))
    out_k = torch.empty((10, 2, 256), device="cuda", dtype=output_dtype)
    out_v = torch.empty((10, 2, 128), device="cuda", dtype=output_dtype)

    def run():
        pool.set_kv_buffer(layer, write_slots, keys, vals)
        pool.gather_into(0, selected, out_k, out_v)

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            run()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()

    for locations, order, read_slots, offset in (
        ([4, 35, 36, 67], [2, -1, 0, 1, 3, -1, -1], [36, -1, 4, 35, 67, 68, -3], 0),
        ([36, 4, 67, 35], [2, 0, -1, 3, 1, -1, -1], [67, 36, -1, 35, 4, 68, -3], 2),
        ([35, 67, 4, 36], [0, 3, 1, -1, 2, -1, -1], [35, 36, 67, -1, 4, 68, -3], -2),
    ):
        keys.copy_(values(4, 256, offset))
        vals.copy_(values(4, 128, offset + 1))
        write_slots.copy_(torch.tensor(locations, device="cuda"))
        selected.copy_(torch.tensor(read_slots, device="cuda", dtype=torch.int32))
        graph.replay()
        for output, source in ((out_k, keys), (out_v, vals)):
            expected = torch.zeros(output.shape, device="cuda", dtype=torch.float32)
            for row, source_row in enumerate(order):
                if source_row >= 0:
                    expected[row].copy_(source[source_row].float().to(output_dtype).float())
            torch.testing.assert_close(output.float(), expected, rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.float8_e4m3fn, torch.float8_e5m2])
@pytest.mark.parametrize("gpu_size", [0, 32, 64])
def test_global_scales_layer_mapping_and_cpu_snapshot(make_pool, dtype, gpu_size):
    pool = make_pool(dtype, gpu_size=gpu_size)
    pool.cpu_offloading_chunk_size = 2
    source = torch.tensor([4, 35, 36, 67], device="cuda")
    expected = []
    for local_layer in range(2):
        key, val = values(4, 256, local_layer), values(4, 128, -local_layer)
        k_scale, v_scale = 0.5, 2.0
        k_expected = (key.float() / k_scale).to(dtype).float()
        v_expected = (val.float() / v_scale).to(dtype).float()
        pool.set_kv_buffer(SimpleNamespace(layer_id=90 + local_layer), source,
                           key, val, k_scale, v_scale, layer_id_override=3 + local_layer)
        actual = pool.get_kv_tokens(90 + local_layer, source, torch.float32,
                                    layer_id_override=3 + local_layer)
        torch.testing.assert_close(actual[0] * k_scale, k_expected * k_scale, rtol=0, atol=0)
        torch.testing.assert_close(actual[1] * v_scale, v_expected * v_scale, rtol=0, atol=0)
        expected.append((k_expected, v_expected))

    selected = torch.tensor([4, 35, 4, 36, 67], device="cuda")
    snapshot = pool.get_cpu_copy(selected)
    destination = torch.tensor([36, 34, 35, 4, 60], device="cuda")
    # Changing the original rows must not change the snapshot or restored KV.
    for local_layer in range(2):
        pool.set_kv_buffer(SimpleNamespace(layer_id=3 + local_layer), source,
                           torch.zeros((4, 2, 256), device="cuda", dtype=dtype),
                           torch.zeros((4, 2, 128), device="cuda", dtype=dtype))
    pool.load_cpu_copy(snapshot, destination)
    order = torch.tensor([0, 1, 0, 2, 3], device="cuda")
    restored = pool.get_cpu_copy(destination)
    for local_layer in range(2):
        actual = pool.get_kv_tokens(3 + local_layer, destination, torch.float32)
        torch.testing.assert_close(actual[0], expected[local_layer][0][order], rtol=0, atol=0)
        torch.testing.assert_close(actual[1], expected[local_layer][1][order], rtol=0, atol=0)
        for kind in range(2):
            chunks = [chunk[kind] for chunk in snapshot[local_layer]]
            assert all(chunk.device.type == "cpu" and chunk.dtype == torch.uint8 for chunk in chunks)
            snapshot_bytes = torch.cat(chunks)
            expected_bytes = expected[local_layer][kind][order].to(dtype).view(torch.uint8).cpu()
            torch.testing.assert_close(snapshot_bytes, expected_bytes, rtol=0, atol=0)
            restored_bytes = torch.cat([chunk[kind] for chunk in restored[local_layer]])
            torch.testing.assert_close(restored_bytes, snapshot_bytes, rtol=0, atol=0)

    # Duplicate snapshot entries and restored rows must not alias each other.
    duplicate_before = snapshot[0][1][0][0].clone()
    snapshot[0][0][0][0].zero_()
    torch.testing.assert_close(snapshot[0][1][0][0], duplicate_before, rtol=0, atol=0)
    pool.set_kv_buffer(SimpleNamespace(layer_id=3), destination[2:3],
                       values(1, 256, 20), values(1, 128, -20))
    unchanged = pool.get_kv_tokens(3, destination[:1], torch.float32)
    torch.testing.assert_close(unchanged[0], expected[0][0][:1], rtol=0, atol=0)
    torch.testing.assert_close(unchanged[1], expected[0][1][:1], rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.float8_e4m3fn, torch.float8_e5m2])
@pytest.mark.parametrize("gpu_size", [0, 256, 520])
def test_acceptance_moves_are_parallel_assignments_across_large_cycles(make_pool, dtype, gpu_size):
    pool = make_pool(dtype, size=520, gpu_size=gpu_size)
    source = torch.arange(4, 517, device="cuda", dtype=torch.int64)
    target = source.roll(257)
    for layer in range(2):
        pool.set_kv_buffer(SimpleNamespace(layer_id=3 + layer), source,
                           values(513, 256, layer), values(513, 128, -layer))
    before = pool.get_cpu_copy(source)
    pool.move_kv_cache(target, source)
    after = pool.get_cpu_copy(target)
    for layer in range(2):
        for kind in range(2):
            expected = torch.cat([chunk[kind] for chunk in before[layer]])
            actual = torch.cat([chunk[kind] for chunk in after[layer]])
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    # Accepted destination must retain the accepted row when a draft/source
    # slot is reused, rather than keeping an alias to the old logical slot.
    accepted_source = torch.tensor([4], device="cuda")
    accepted_target = torch.tensor([523], device="cuda")
    accepted = pool.get_kv_tokens(3, accepted_source, torch.float32)
    pool.move_kv_cache(accepted_target, accepted_source)
    pool.set_kv_buffer(SimpleNamespace(layer_id=3), accepted_source,
                       values(1, 256, 20), values(1, 128, -20))
    actual = pool.get_kv_tokens(3, accepted_target, torch.float32)
    torch.testing.assert_close(actual[0], accepted[0], rtol=0, atol=0)
    torch.testing.assert_close(actual[1], accepted[1], rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.float8_e4m3fn, torch.float8_e5m2])
@pytest.mark.parametrize("gpu_size", [0, 32768, 65536])
def test_large_virtual_bank_has_bounded_resident_allocations(make_pool, dtype, gpu_size):
    # Initialize CUDA before measuring the pool's allocation, not its context.
    torch.empty(1, device="cuda")
    before = torch.cuda.memory_allocated()
    pool = make_pool(dtype, size=65536, gpu_size=gpu_size, page_size=32, layers=1)
    device_bytes = torch.cuda.memory_allocated() - before
    gpu_rows = gpu_size + 32 if gpu_size else 0
    host_rows = 65536 - gpu_size if gpu_size else 65568
    key_bytes, value_bytes = gpu_rows * 2 * 256, gpu_rows * 2 * 128
    assert pool.host_bytes == host_rows * 2 * (256 + 128)
    assert sum(buffer.numel() * buffer.element_size()
               for buffer in pool.k_buffer + pool.v_buffer) == pool.host_bytes
    assert pool.get_kv_size_bytes() == (key_bytes, value_bytes)
    assert key_bytes + value_bytes <= device_bytes < key_bytes + value_bytes + 1024**2
    # First/final valid rows and both sides of the mixed residency boundary.
    locations = torch.tensor([32, 32799, 32800, 65567], device="cuda")
    key, val = values(4, 256), values(4, 128)
    expected_key = key.to(pool.dtype).float()
    expected_val = val.to(pool.dtype).float()
    pool.set_kv_buffer(SimpleNamespace(layer_id=3), locations, key, val)
    actual = pool.get_kv_tokens(3, locations.flip(0), torch.float32)
    torch.testing.assert_close(actual[0], expected_key.flip(0), rtol=0, atol=0)
    torch.testing.assert_close(actual[1], expected_val.flip(0), rtol=0, atol=0)
    dummy = pool.get_kv_tokens(3, torch.arange(32, device="cuda"), torch.float32)
    assert torch.count_nonzero(dummy[0]) == 0
    assert torch.count_nonzero(dummy[1]) == 0
