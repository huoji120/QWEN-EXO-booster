"""Real CUDA QSA attention over GPU and mapped-host FP8 banks."""

from functools import partial
from types import SimpleNamespace

import pytest
import torch

from sglang.srt.layers.attention import qwen_sparse_attn_backend as backend_module
from sglang.srt.layers.attention.qwen_sparse_attn_backend import QwenSparseAttnBackend
from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
from sglang.srt.mem_cache.qsa_kv_pool import QSATokenToKVPool
from sglang.srt.mem_cache.qwen_host_kv_pool import QwenHostFP8KVPool
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=90, stage="base-b-kernel-unit", runner_config="1-gpu-large")
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def _pair(*, gpu_size=0, dtype=torch.float8_e4m3fn):
    torch.manual_seed(183)
    req_pool = ReqToTokenPool(
        size=3, max_context_len=64, device="cuda", enable_memory_saver=False
    )
    req_pool.req_to_token.zero_()
    # Non-monotonic physical pages make a logical/physical mix-up observable.
    req_pool.req_to_token[1] = torch.cat(
        [torch.arange(96, 128), torch.arange(32, 64)]
    ).to(device="cuda", dtype=torch.int32)
    req_pool.req_to_token[2] = torch.cat(
        [torch.arange(160, 192), torch.arange(64, 96)]
    ).to(device="cuda", dtype=torch.int32)
    config = SimpleNamespace(
        model_type="qwen4_exp", indexer_n_heads=8, indexer_kv_heads=1,
        indexer_head_dim=128, indexer_budget=2048, indexer_compress_ratio=4,
    )
    layer = SimpleNamespace(
        layer_id=11, tp_q_head_num=8, head_dim=128, scaling=128 ** -0.5,
        k_scale_float=0.75, v_scale_float=1.5,
    )
    layers = [SimpleNamespace(**{**vars(layer), "layer_id": 3}), layer]
    banks = []
    inputs = [
        (torch.randn(260, 2, 128, device="cuda", dtype=torch.bfloat16),
         torch.randn(260, 2, 128, device="cuda", dtype=torch.bfloat16))
        for _ in layers
    ]
    for k, v in inputs:
        k[0].zero_()
        v[0].zero_()
    for pool_class in (None, partial(QwenHostFP8KVPool, gpu_size=gpu_size)):
        pool = QSATokenToKVPool(
            size=256, dtype=dtype, page_size=4,
            head_num=2, head_dim=128, full_attention_layer_ids=[3, 11],
            device="cuda", mamba_pool=None, qsa_index_kv_heads=1,
            qsa_index_head_dim=128, qsa_compress_ratio=4,
            qsa_token_topk=2048, num_request_slots=4,
            full_kv_pool_class=pool_class,
        )
        for current_layer, (k, v) in zip(layers, inputs):
            # Base set_kv_buffer divides by non-unit layer global scales.
            pool.set_kv_buffer(
                current_layer, torch.arange(260, device="cuda"), k.clone(), v.clone(),
                k_scale=current_layer.k_scale_float,
                v_scale=current_layer.v_scale_float,
            )
        runner = SimpleNamespace(
            device=torch.device("cuda"), token_to_kv_pool=pool,
            req_to_token_pool=req_pool,
            model_config=SimpleNamespace(context_len=64, hf_text_config=config),
        )
        banks.append(QwenSparseAttnBackend(runner))
    return banks, layer


def _batch(mode, reqs, lengths, width):
    rows = 2 * width
    extends = torch.full((2,), width, dtype=torch.int32, device="cuda")
    spec = SimpleNamespace(
        topk=1, draft_token_num=width, extend_seq_lens_tensor=extends,
        extend_seq_lens_cpu=[width, width],
    )
    return SimpleNamespace(
        forward_mode=mode, batch_size=2, req_pool_indices=reqs,
        seq_lens=lengths, seq_lens_cpu=lengths.cpu(), spec_info=spec,
        input_ids=torch.zeros(rows, dtype=torch.int64, device="cuda"),
        positions=torch.zeros(rows, dtype=torch.int64, device="cuda"),
        out_cache_loc=torch.ones(rows, dtype=torch.int64, device="cuda"),
        extend_seq_lens=extends if mode.is_draft_extend_v2() else None,
        _original_forward_mode=None, mrope_positions=None,
    )


def _reference(backend, layer, q, topk):
    metadata = backend.forward_metadata
    raw_k = backend.token_to_kv_pool.get_key_buffer(layer.layer_id).to(torch.float32)
    raw_v = backend.token_to_kv_pool.get_value_buffer(layer.layer_id).to(torch.float32)
    result = []
    for row in range(q.shape[0]):
        length = int(metadata.sequence_lengths[row])
        req = int(metadata.row_req_pool_indices[row])
        logical = topk[row]
        logical = logical[(logical >= 0) & (logical < length)].long()
        slots = backend.req_to_token_pool.req_to_token[req, logical].long()
        k = (raw_k[slots] * layer.k_scale_float).to(q.dtype).float()
        v = (raw_v[slots] * layer.v_scale_float).to(q.dtype).float()
        k = k.repeat_interleave(4, dim=1)
        v = v.repeat_interleave(4, dim=1)
        scores = torch.einsum("hd,nhd->hn", q[row].float(), k) * layer.scaling
        result.append(torch.einsum("hn,nhd->hd", scores.softmax(-1), v))
    return torch.stack(result).reshape(q.shape[0], -1).to(q.dtype)


@pytest.mark.parametrize("mode,width", [
    (ForwardMode.DECODE, 1),
    (ForwardMode.TARGET_VERIFY, 4),
    (ForwardMode.DRAFT_EXTEND_V2, 4),
])
@pytest.mark.parametrize("dtype", [torch.float8_e4m3fn, torch.float8_e5m2])
@pytest.mark.parametrize("gpu_size", [0, 96, 256])
def test_host_qsa_graph_reordered_requests_causal_rows_and_padding(mode, width, dtype, gpu_size):
    if backend_module._resolve_trtllm_sparse_decode() is None:
        pytest.skip("Mapped-host sparse attention requires native TRTLLM decode")
    backends, layer = _pair(gpu_size=gpu_size, dtype=dtype)
    rows = 2 * width
    reqs = torch.tensor([1, 2], dtype=torch.int32, device="cuda")
    lengths = torch.tensor([12, 17], dtype=torch.int32, device="cuda")
    batch = _batch(mode, reqs, lengths, width)
    q = torch.randn(rows, 8, 128, dtype=torch.bfloat16, device="cuda")
    topk = torch.full((rows, 65), -1, dtype=torch.int32, device="cuda")
    # Valid entries are a prefix, as emitted by the QSA top-k expansion; their
    # order is deliberately not sorted. Slots 99 and 101 straddle the mixed
    # pool's GPU/host boundary at 100. Later entries violate causal length.
    topk[:, :4] = torch.tensor([3, 0, 5, 1], dtype=torch.int32, device="cuda")
    topk[:, 4] = 63
    graphs, outputs = [], []
    for backend in backends:
        backend.init_cuda_graph_state(2, rows)
        backend.init_forward_metadata_out_graph(batch, in_capture=True)
        backend.init_forward_metadata_out_graph(batch)
        for _ in range(2):
            backend._forward_paged_attention(q, layer, batch, topk)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            # Actual metadata kernels are part of capture, including the
            # per-candidate row mapping and replay-dependent sequence lengths.
            backend.init_forward_metadata_out_graph(batch)
            out = backend._forward_paged_attention(q, layer, batch, topk)
        graphs.append(graph)
        outputs.append(out)
    for reorder in (False, True):
        if reorder:
            reqs.copy_(torch.tensor([2, 1], dtype=torch.int32, device="cuda"))
            lengths.copy_(torch.tensor([20, 8], dtype=torch.int32, device="cuda"))
            q.mul_(0.7)
            topk[:, :4] = torch.tensor([6, 2, 0, 4], dtype=torch.int32, device="cuda")
            if mode.is_draft_extend_v2():
                batch.spec_info.extend_seq_lens_tensor.copy_(
                    torch.tensor([3, 2], dtype=torch.int32, device="cuda")
                )
                # The last three captured rows now map to reserved request 0.
                topk[5:].fill_(-1)
                topk[5:, 0] = 0
        for graph in graphs:
            graph.replay()
        torch.cuda.synchronize()
        expected_reqs, expected_lens = [], []
        for req, length, extend in zip(
            reqs.cpu().tolist(), lengths.cpu().tolist(),
            batch.spec_info.extend_seq_lens_tensor.cpu().tolist(),
        ):
            if mode.is_decode():
                expected_reqs.append(req)
                expected_lens.append(length)
            else:
                prefix = length if mode.is_target_verify() else length - extend
                count = width if mode.is_target_verify() else extend
                expected_reqs.extend([req] * count)
                expected_lens.extend(range(prefix + 1, prefix + count + 1))
        expected_reqs.extend([0] * (rows - len(expected_reqs)))
        expected_lens.extend([1] * (rows - len(expected_lens)))
        for backend in backends:
            assert backend.forward_metadata.row_req_pool_indices.cpu().tolist() == expected_reqs
            assert backend.forward_metadata.sequence_lengths.cpu().tolist() == expected_lens
        torch.testing.assert_close(outputs[1], outputs[0], atol=0.01, rtol=0.01)
        torch.testing.assert_close(outputs[1], _reference(backends[0], layer, q, topk), atol=0.02, rtol=0.02)
        host = backends[1]
        gpu = backends[0]
        host_k, host_v = next(iter(host._fa2_scratch.values()))
        gpu_k, gpu_v = next(iter(gpu._fa2_scratch.values()))
        counts = host.forward_metadata.fa2_valid_counts.cpu().tolist()
        for row, count in enumerate(counts):
            start = row * 128
            torch.testing.assert_close(host_k[start:start + count], gpu_k[start:start + count], rtol=0, atol=0)
            torch.testing.assert_close(host_v[start:start + count], gpu_v[start:start + count], rtol=0, atol=0)
            assert torch.count_nonzero(host_k[start + count:start + 128]) == 0
            assert torch.count_nonzero(host_v[start + count:start + 128]) == 0


@pytest.mark.parametrize("dtype", [torch.float8_e4m3fn, torch.float8_e5m2])
@pytest.mark.parametrize("gpu_size", [0, 96, 256])
def test_host_qsa_chunk_prefill_reads_old_prefix_and_commits_current_chunk(dtype, gpu_size):
    backends, layer = _pair(gpu_size=gpu_size, dtype=dtype)
    reqs = torch.tensor([2, 1], dtype=torch.int32, device="cuda")
    lengths = torch.tensor([15, 10], dtype=torch.int32, device="cuda")
    extends = torch.tensor([3, 2], dtype=torch.int32, device="cuda")
    positions = torch.tensor([12, 13, 14, 8, 9], device="cuda")
    locations = torch.cat([
        backends[0].req_to_token_pool.req_to_token[2, 12:15],
        backends[0].req_to_token_pool.req_to_token[1, 8:10],
    ]).long()
    batch = SimpleNamespace(
        forward_mode=ForwardMode.EXTEND, batch_size=2, req_pool_indices=reqs,
        seq_lens=lengths, seq_lens_cpu=lengths.cpu(), spec_info=None,
        positions=positions, out_cache_loc=locations, extend_seq_lens=extends,
        extend_seq_lens_cpu=[3, 2], _original_forward_mode=None,
        mrope_positions=None,
        input_ids=torch.zeros(5, dtype=torch.int64, device="cuda"),
    )
    q = torch.randn(5, 8, 128, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(5, 2, 128, device="cuda", dtype=torch.bfloat16)
    v = torch.randn_like(k)
    topk = torch.stack([
        torch.tensor([0, 3, int(pos), -1], device="cuda", dtype=torch.int32)
        for pos in positions.cpu()
    ])
    outputs = []
    for backend in backends:
        backend.init_forward_metadata(batch)
        outputs.append(backend.forward_extend(
            q, k.clone(), v.clone(), layer, batch, topk_indices=topk
        ))
    torch.testing.assert_close(outputs[1], outputs[0], atol=0.01, rtol=0.01)
    # Independent dense calculation includes old prefix rows and quantized
    # current-chunk rows at each query's own causal boundary.
    gpu = backends[0]
    keys = gpu.token_to_kv_pool.get_key_buffer(layer.layer_id).float() * layer.k_scale_float
    values = gpu.token_to_kv_pool.get_value_buffer(layer.layer_id).float() * layer.v_scale_float
    expected = []
    for row, req in enumerate([2, 2, 2, 1, 1]):
        slots = gpu.req_to_token_pool.req_to_token[req, topk[row, :3].long()].long()
        selected_k = keys[slots].to(q.dtype).float().repeat_interleave(4, dim=1)
        selected_v = values[slots].to(q.dtype).float().repeat_interleave(4, dim=1)
        scores = torch.einsum("hd,nhd->hn", q[row].float(), selected_k) * layer.scaling
        expected.append(torch.einsum("hn,nhd->hd", scores.softmax(-1), selected_v))
    torch.testing.assert_close(outputs[1], torch.stack(expected).reshape(5, -1).to(q.dtype), atol=0.02, rtol=0.02)
    host_k, host_v = backends[1].token_to_kv_pool.get_kv_tokens(layer.layer_id, locations, q.dtype)
    gpu_k, gpu_v = gpu.token_to_kv_pool.get_kv_tokens(layer.layer_id, locations, q.dtype)
    torch.testing.assert_close(host_k, gpu_k, atol=0, rtol=0)
    torch.testing.assert_close(host_v, gpu_v, atol=0, rtol=0)
    # Routing a chunk write must not corrupt unrelated rows or the other layer.
    all_slots = torch.arange(260, device="cuda")
    for layer_id in (3, 11):
        host_rows = backends[1].token_to_kv_pool.get_kv_tokens(layer_id, all_slots, torch.float32)
        gpu_rows = gpu.token_to_kv_pool.get_kv_tokens(layer_id, all_slots, torch.float32)
        for host_rows_kind, gpu_rows_kind in zip(host_rows, gpu_rows):
            torch.testing.assert_close(host_rows_kind, gpu_rows_kind, atol=0, rtol=0)
