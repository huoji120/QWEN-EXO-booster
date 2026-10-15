"""Real CUDA QSA attention over GPU and mapped-host FP8 banks."""

from functools import partial
from types import SimpleNamespace

import pytest
import torch

from sglang.srt.layers.attention import qwen_sparse_attn_backend as backend_module
from sglang.srt.layers.attention.qsa.sparse_attn import (
    sparse_gqa_fwd_interface_triton_ck,
    sparse_gqa_fwd_interface_triton_paged,
)
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
@pytest.mark.parametrize("gpu_size", [0, 128, 192, 256])
def test_host_qsa_chunk_prefill_reads_old_prefix_and_commits_current_chunk(
    dtype, gpu_size, monkeypatch
):
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

    def forbidden_prefill_path(*args, **kwargs):
        pytest.fail("Prefill crossed the GPU-only direct-attention boundary")

    for backend in backends:
        backend.init_forward_metadata(batch)
        full_pool = backend.token_to_kv_pool.full_kv_pool
        with monkeypatch.context() as patch:
            if isinstance(full_pool, QwenHostFP8KVPool) and gpu_size < full_pool.size:
                # gpu_size=192 leaves a host suffix even though this batch's
                # selected slots all fit on GPU. It must still materialize.
                patch.setattr(
                    backend_module,
                    "sparse_gqa_fwd_interface_triton_paged",
                    forbidden_prefill_path,
                )
            else:
                # GPU-only pools must retain the no-prefix-materialization path.
                patch.setattr(
                    backend.token_to_kv_pool, "get_kv_tokens", forbidden_prefill_path
                )
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


def _paged_prefill_inputs(storage_dtype, query_dtype, per_head, head_dim=128, q_heads=8):
    torch.manual_seed(927)
    # A long prefix with unequal current chunks exercises multiple sparse tiles
    # and the second request's nonzero cu_q/cu_k offsets, without model weights.
    lengths = [337, 263]
    query_lengths = [3, 2]
    raw_k = torch.randn(772, 2, head_dim, device="cuda").to(storage_dtype)
    raw_v = torch.randn(772, 2, head_dim, device="cuda").to(storage_dtype)
    q = torch.randn(5, q_heads, head_dim, device="cuda", dtype=query_dtype)
    locations = (
        (torch.arange(sum(lengths), device="cuda", dtype=torch.int64) * 137 + 73)
        % raw_k.shape[0]
    )
    rows = []
    for length, query_length in zip(lengths, query_lengths):
        for relative in range(query_length):
            visible = length - query_length + relative + 1
            heads = []
            for _ in range(2):
                candidates = torch.arange(visible, dtype=torch.int32)
                candidates = candidates[
                    (candidates != 0) & (candidates != 3)
                    & (candidates != visible - 1)
                ]
                selected = candidates[torch.randperm(candidates.numel())[:93]]
                selected = torch.cat([
                    torch.tensor([0, 3, visible - 1], dtype=torch.int32),
                    selected,
                ])
                selected[10::11] = -1
                selected[-1] = -1
                heads.append(selected)
            rows.append(torch.stack(heads))
    indices = torch.stack(rows).to(device="cuda")
    if not per_head:
        indices = indices[:, 0, :].contiguous()
    cu_q = torch.tensor([0, 3, 5], device="cuda", dtype=torch.int32)
    cu_k = torch.tensor([0, 337, 600], device="cuda", dtype=torch.int32)
    kv_lens = torch.tensor(lengths, device="cuda", dtype=torch.int32)
    return q, raw_k, raw_v, indices, cu_q, cu_k, kv_lens, locations


def _assert_paged_matches_materialized(inputs, k, v, *, host_k=None, host_v=None):
    q, raw_k, raw_v, indices, cu_q, cu_k, kv_lens, locations = inputs
    fp8 = raw_k.dtype in (torch.float8_e4m3fn, torch.float8_e5m2)
    k_scale, v_scale = (0.73, 1.07) if fp8 else (None, None)
    # Convert the raw bank before indexing: torch advanced indexing need not
    # implement FP8. This is the old gather/cast, then in-place scale contract.
    dense_k = raw_k.to(q.dtype)[locations]
    dense_v = raw_v.to(q.dtype)[locations]
    if fp8:
        dense_k.mul_(k_scale)
        dense_v.mul_(v_scale)
    scale = q.shape[-1] ** -0.5
    expected = sparse_gqa_fwd_interface_triton_ck(
        q, dense_k, dense_v, indices, cu_q, cu_k, kv_lens, scale, max_q=3
    )
    actual = sparse_gqa_fwd_interface_triton_paged(
        q, k, v, indices, cu_q, cu_k, kv_lens, scale, locations,
        max_q=3, k_scale=k_scale, v_scale=v_scale,
        host_k=host_k, host_v=host_v,
    )
    # Different load/warp schedules can change FP32 accumulation near zero.
    # Allow 1e-6 absolute error there and one output-dtype rounding step
    # elsewhere. The scale-order regression below remains bit-exact.
    precision = torch.finfo(q.dtype)
    torch.testing.assert_close(
        actual, expected, atol=1e-6, rtol=precision.eps
    )


@pytest.mark.parametrize(
    "storage_dtype,query_dtype,per_head,head_dim,q_heads",
    [
        (torch.float8_e4m3fn, torch.bfloat16, False, 128, 8),
        (torch.float8_e5m2, torch.float16, True, 128, 8),
        (torch.bfloat16, torch.bfloat16, True, 128, 8),
        (torch.bfloat16, torch.float16, False, 128, 8),
        (torch.float8_e4m3fn, torch.bfloat16, False, 256, 24),
    ],
)
def test_paged_qsa_prefill_gpu_bank_matches_dense(
    storage_dtype, query_dtype, per_head, head_dim, q_heads
):
    inputs = _paged_prefill_inputs(
        storage_dtype, query_dtype, per_head, head_dim, q_heads
    )
    _assert_paged_matches_materialized(inputs, inputs[1], inputs[2])


@pytest.mark.parametrize(
    "storage_dtype,query_dtype,per_head,gpu_size",
    [
        (torch.float8_e4m3fn, torch.bfloat16, True, 384),
        (torch.float8_e5m2, torch.float16, False, 384),
        (torch.float8_e4m3fn, torch.bfloat16, False, 0),
        (torch.float8_e5m2, torch.float16, True, 768),
    ],
)
def test_paged_qsa_prefill_mapped_bank_matches_dense(
    storage_dtype, query_dtype, per_head, gpu_size
):
    inputs = _paged_prefill_inputs(storage_dtype, query_dtype, per_head)
    pool = QwenHostFP8KVPool(
        size=768, page_size=4, dtype=storage_dtype, head_num=2, head_dim=128,
        layer_num=1, device="cuda", enable_memory_saver=False, gpu_size=gpu_size,
    )
    # Preserve the pool's real mapped-host ownership and uint8 storage layout.
    # These copies initialize raw FP8 values, not a second quantization path.
    for raw, gpu_bank, host_bank in (
        (inputs[1], pool.gpu_k_buffer[0], pool.k_buffer[0]),
        (inputs[2], pool.gpu_v_buffer[0], pool.v_buffer[0]),
    ):
        gpu_bank.copy_(raw[:pool.gpu_rows].view(torch.uint8))
        host_bank.copy_(raw[pool.gpu_rows:].view(torch.uint8).cpu())
    if 0 < gpu_size < pool.size:
        # Logical selections 0 and 3 cross banks in both requests.
        selected = inputs[-1][torch.tensor([0, 3, 337, 340], device="cuda")]
        assert (selected[:2] < pool.gpu_rows).tolist() == [True, False]
        assert (selected[2:] < pool.gpu_rows).tolist() == [False, True]
    try:
        _assert_paged_matches_materialized(
            inputs,
            pool.gpu_k_buffer[0].view(storage_dtype),
            pool.gpu_v_buffer[0].view(storage_dtype),
            host_k=pool.k_buffer[0].view(storage_dtype),
            host_v=pool.v_buffer[0].view(storage_dtype),
        )
    finally:
        # The kernel receives integer host pointers; retain the pool until all
        # mapped accesses finish, including when an assertion raises.
        torch.cuda.synchronize()


def test_paged_qsa_prefill_rounds_scaled_fp8_values_before_attention():
    q = torch.zeros(1, 8, 128, device="cuda", dtype=torch.bfloat16)
    raw_k = torch.zeros(4, 2, 128, device="cuda").to(torch.float8_e4m3fn)
    raw_v = torch.tensor([1.25, 0.0, 0.0, 1.125], device="cuda")
    raw_v = raw_v[:, None, None].expand(4, 2, 128).contiguous()
    raw_v = raw_v.to(torch.float8_e4m3fn)
    locations = torch.tensor([3, 0], device="cuda", dtype=torch.int64)
    indices = torch.tensor([[0, 1, -1, -1]], device="cuda", dtype=torch.int32)
    cu_q = torch.tensor([0, 1], device="cuda", dtype=torch.int32)
    cu_k = torch.tensor([0, 2], device="cuda", dtype=torch.int32)
    kv_lens = torch.tensor([2], device="cuda", dtype=torch.int32)
    dense_k = raw_k.to(q.dtype)[locations].mul_(0.73)
    dense_v = raw_v.to(q.dtype)[locations].mul_(1.07)
    expected = sparse_gqa_fwd_interface_triton_ck(
        q, dense_k, dense_v, indices, cu_q, cu_k, kv_lens, 128 ** -0.5, max_q=1
    )
    # Equal logits make attention an average. Rounding each scaled value gives
    # a different BF16 answer than averaging raw FP8 then applying the scale.
    late_scaled = (raw_v.float()[locations].mean(dim=0) * 1.07).to(q.dtype)
    assert torch.all(expected[0] != late_scaled.repeat_interleave(4, dim=0))
    actual = sparse_gqa_fwd_interface_triton_paged(
        q, raw_k, raw_v, indices, cu_q, cu_k, kv_lens, 128 ** -0.5, locations,
        max_q=1, k_scale=0.73, v_scale=1.07,
    )
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
