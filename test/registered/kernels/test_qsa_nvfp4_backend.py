"""NVFP4 QSA cache writes, sparse attention and native MTP graph integration."""

from functools import partial
from types import SimpleNamespace

import pytest
import torch

from sglang.srt.layers.attention import qwen_sparse_attn_backend as backend_module
from sglang.srt.layers.attention.qwen_sparse_attn_backend import QwenSparseAttnBackend
from sglang.srt.layers.quantization.fp4_kv_cache_quant_method import NVFP4KVCacheMethod
from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
from sglang.srt.mem_cache.qsa_kv_pool import QSATokenToKVPool
from sglang.srt.mem_cache.qwen_host_kv_pool import QwenHostKVPool
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=90, stage="base-b", runner_config="1-gpu-small")
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] not in (10, 12),
    reason="Native NVFP4 writer requires Blackwell",
)


def _make_backend(gpu_size):
    torch.manual_seed(731)
    req_pool = ReqToTokenPool(3, 64, "cuda", enable_memory_saver=False)
    req_pool.req_to_token.zero_()
    req_pool.req_to_token[1] = torch.arange(64, 128, device="cuda")
    req_pool.req_to_token[2] = torch.arange(128, 192, device="cuda")
    config = SimpleNamespace(
        model_type="qwen4_exp", indexer_n_heads=8, indexer_kv_heads=1,
        indexer_head_dim=128, indexer_budget=2048, indexer_compress_ratio=4,
    )
    method = NVFP4KVCacheMethod(12, "cuda", attention_backend="qsa")
    method.k_scales_float[11], method.v_scales_float[11] = .75, 1.5
    method.k_scales_gpu[11], method.v_scales_gpu[11] = .75, 1.5
    pool = QSATokenToKVPool(
        size=256, dtype=torch.float4_e2m1fn_x2, page_size=4,
        head_num=2, head_dim=256, full_attention_layer_ids=[3, 11],
        device="cuda", mamba_pool=None, qsa_index_kv_heads=1,
        qsa_index_head_dim=128, qsa_compress_ratio=4, qsa_token_topk=2048,
        num_request_slots=4, quant_method=method,
        full_kv_pool_class=None if gpu_size is None else partial(QwenHostKVPool, gpu_size=gpu_size),
    )
    layer = SimpleNamespace(
        layer_id=11, tp_q_head_num=8, tp_k_head_num=2, head_dim=256,
        scaling=256 ** -.5, k_scale_float=.75, v_scale_float=1.5,
    )
    slots = torch.arange(260, device="cuda")
    for layer_id in (3, 11):
        key = torch.randn(260, 2, 256, device="cuda", dtype=torch.bfloat16)
        value = torch.randn_like(key)
        key[0].zero_()
        value[0].zero_()
        pool.set_kv_buffer(SimpleNamespace(layer_id=layer_id), slots, key, value)
    backend = QwenSparseAttnBackend(SimpleNamespace(
        device=torch.device("cuda"), token_to_kv_pool=pool, req_to_token_pool=req_pool,
        model_config=SimpleNamespace(context_len=64, hf_text_config=config),
    ))
    return backend, layer


def _reference(backend, layer, q, indices, reqs, lengths):
    output = []
    for row, (req, length) in enumerate(zip(reqs, lengths)):
        positions = indices[row]
        positions = positions[(positions >= 0) & (positions < length)].long()
        if positions.numel() == 0:
            output.append(torch.zeros_like(q[row], dtype=torch.float32))
            continue
        slots = backend.req_to_token_pool.req_to_token[req, positions].long()
        keys, values = backend.token_to_kv_pool.get_kv_tokens(layer.layer_id, slots, q.dtype)
        keys = keys.float().repeat_interleave(4, dim=1)
        values = values.float().repeat_interleave(4, dim=1)
        scores = torch.einsum("hd,nhd->hn", q[row].float(), keys) * layer.scaling
        output.append(torch.einsum("hn,nhd->hd", scores.softmax(-1), values))
    return torch.stack(output).reshape(q.shape[0], -1).to(q.dtype)


@pytest.mark.parametrize("gpu_size", [None, 0, 96, 256])
def test_nvfp4_chunk_prefill_and_prefix_reuse(gpu_size):
    backend, layer = _make_backend(gpu_size)
    try:
        reqs = torch.tensor([2, 1], dtype=torch.int32, device="cuda")
        lengths = torch.tensor([15, 10], dtype=torch.int32, device="cuda")
        positions = torch.tensor([12, 13, 14, 8, 9], device="cuda")
        locations = torch.cat((
            backend.req_to_token_pool.req_to_token[2, 12:15],
            backend.req_to_token_pool.req_to_token[1, 8:10],
        )).long()
        batch = SimpleNamespace(
            forward_mode=ForwardMode.EXTEND, batch_size=2, req_pool_indices=reqs,
            seq_lens=lengths, seq_lens_cpu=lengths.cpu(), spec_info=None,
            positions=positions, out_cache_loc=locations,
            extend_seq_lens=torch.tensor([3, 2], dtype=torch.int32, device="cuda"),
            extend_seq_lens_cpu=[3, 2], _original_forward_mode=None,
            mrope_positions=None, input_ids=torch.zeros(5, dtype=torch.int64, device="cuda"),
        )
        q = torch.randn(5, 8, 256, device="cuda", dtype=torch.bfloat16)
        k = torch.randn(5, 2, 256, device="cuda", dtype=torch.bfloat16)
        v = torch.randn_like(k)
        topk = torch.tensor([[0, 3, int(p), -1] for p in positions.cpu()], device="cuda", dtype=torch.int32)
        backend.init_forward_metadata(batch)
        actual = backend.forward_extend(q, k, v, layer, batch, topk_indices=topk)
        expected = _reference(backend, layer, q, topk, [2, 2, 2, 1, 1], [13, 14, 15, 9, 10])
        torch.testing.assert_close(actual, expected, rtol=.025, atol=.025)
        reused = backend.forward_extend(q, k, v, layer, batch, save_kv_cache=False, topk_indices=topk)
        torch.testing.assert_close(reused, actual, rtol=0, atol=0)
    finally:
        backend.token_to_kv_pool.full_kv_pool._clear_buffers()


@pytest.mark.parametrize("gpu_size", [None, 0, 160, 256])
def test_nvfp4_staged_prefill_bounds_and_reuses_storage_across_layers(gpu_size):
    backend, layer = _make_backend(gpu_size)
    try:
        assert backend._nvfp4_prefill_scratch is None
        req_ids = [2, 3, 1]
        backend.req_to_token_pool.req_to_token[3] = torch.arange(192, 256, device="cuda")
        reqs = torch.tensor(req_ids, dtype=torch.int32, device="cuda")
        scratch_ptrs = None
        # Each request fits context64, but their combined histories do not.
        # The middle request has no queries; cached prefixes remain page-aligned.
        for seq_lengths, query_lengths in (
            ([41, 20, 35], [5, 0, 3]),
            ([58, 20, 47], [2, 0, 7]),
            ([64, 20, 53], [4, 0, 1]),
        ):
            assert sum(seq_lengths) > backend.max_context_len
            assert max(seq_lengths) <= backend.max_context_len
            row_reqs = []
            position_values = []
            location_parts = []
            for req, length, count in zip(req_ids, seq_lengths, query_lengths):
                row_reqs.extend([req] * count)
                position_values.extend(range(length - count, length))
                location_parts.append(
                    backend.req_to_token_pool.req_to_token[req, length - count:length]
                )
            positions = torch.tensor(position_values, device="cuda")
            lengths = torch.tensor(seq_lengths, dtype=torch.int32, device="cuda")
            rows = len(position_values)
            batch = SimpleNamespace(
                forward_mode=ForwardMode.EXTEND, batch_size=3,
                req_pool_indices=reqs, seq_lens=lengths,
                seq_lens_cpu=lengths.cpu(), spec_info=None, positions=positions,
                out_cache_loc=torch.cat(location_parts).long(),
                extend_seq_lens=torch.tensor(query_lengths, dtype=torch.int32, device="cuda"),
                extend_seq_lens_cpu=query_lengths, _original_forward_mode=None,
                mrope_positions=None,
                input_ids=torch.zeros(rows, dtype=torch.int64, device="cuda"),
            )
            q = torch.randn(rows, 8, 256, device="cuda", dtype=torch.bfloat16)
            k = torch.randn(rows, 2, 256, device="cuda", dtype=q.dtype)
            v = torch.randn_like(k)
            # At gpu_size160, request 2 selects both GPU and host slots;
            # request 1 stays on GPU. Holes must not shift the query spans.
            topk = torch.tensor(
                [[0, -1, 7, min(31, p), p, p + 1, p + 65, -1] for p in position_values],
                dtype=torch.int32, device="cuda",
            )
            # Both masked holes and exclusively future selections yield zero,
            # while mixed rows retain only in-range causal positions.
            topk[1].fill_(-1)
            topk[-1].fill_(position_values[-1] + 1)
            backend.init_forward_metadata(batch)
            for layer_id in (11, 3, 11):
                current_layer = SimpleNamespace(**{**vars(layer), "layer_id": layer_id})
                actual = backend.forward_extend(
                    q, k, v, current_layer, batch,
                    save_kv_cache=layer_id == 11, topk_indices=topk,
                )
                expected = _reference(
                    backend, current_layer, q, topk, row_reqs,
                    [p + 1 for p in position_values],
                )
                torch.testing.assert_close(actual, expected, rtol=.025, atol=.025)
                scratch = backend._nvfp4_prefill_scratch
                assert len(scratch) == 2
                full_pool = backend.token_to_kv_pool.full_kv_pool
                expected_shape = (
                    backend.max_context_len, full_pool.head_num, full_pool.head_dim,
                )
                for tensor in scratch:
                    assert tensor.shape == expected_shape
                    assert tensor.dtype == q.dtype
                    assert tensor.device == q.device
                byte_bound = (
                    2 * backend.max_context_len * full_pool.head_num
                    * full_pool.head_dim * q.element_size()
                )
                assert sum(t.untyped_storage().nbytes() for t in scratch) <= byte_bound
                current_ptrs = tuple(t.data_ptr() for t in scratch)
                assert current_ptrs[0] != current_ptrs[1]
                if scratch_ptrs is None:
                    scratch_ptrs = current_ptrs
                else:
                    assert current_ptrs == scratch_ptrs
    finally:
        backend.token_to_kv_pool.full_kv_pool._clear_buffers()


@pytest.mark.parametrize("mode,width", [
    (ForwardMode.DECODE, 1), (ForwardMode.TARGET_VERIFY, 4), (ForwardMode.DRAFT_EXTEND_V2, 4),
])
@pytest.mark.parametrize("gpu_size", [None, 0, 96, 256])
def test_nvfp4_mtp_graph_replay_changes_requests_and_cached_values(mode, width, gpu_size):
    if backend_module._resolve_trtllm_sparse_decode() is None:
        pytest.skip("Native sparse TRTLLM decode required")
    backend, layer = _make_backend(gpu_size)
    try:
        rows = 2 * width
        reqs = torch.tensor([1, 2], dtype=torch.int32, device="cuda")
        lengths = torch.tensor([12, 17], dtype=torch.int32, device="cuda")
        extends = torch.full((2,), width, dtype=torch.int32, device="cuda")
        batch = SimpleNamespace(
            forward_mode=mode, batch_size=2, req_pool_indices=reqs, seq_lens=lengths,
            seq_lens_cpu=lengths.cpu(), spec_info=SimpleNamespace(
                topk=1, draft_token_num=width, extend_seq_lens_tensor=extends,
                extend_seq_lens_cpu=[width, width],
            ),
            input_ids=torch.zeros(rows, dtype=torch.int64, device="cuda"),
            positions=torch.zeros(rows, dtype=torch.int64, device="cuda"),
            out_cache_loc=torch.ones(rows, dtype=torch.int64, device="cuda"),
            extend_seq_lens=extends if mode.is_draft_extend_v2() else None,
            _original_forward_mode=None, mrope_positions=None,
        )
        q = torch.randn(rows, 8, 256, dtype=torch.bfloat16, device="cuda")
        topk = torch.full((rows, 65), -1, dtype=torch.int32, device="cuda")
        topk[:, :4] = torch.tensor([3, 0, 5, 1], device="cuda")
        backend.init_cuda_graph_state(2, rows)
        backend.init_forward_metadata_out_graph(batch, in_capture=True)
        for _ in range(2):
            backend.init_forward_metadata_out_graph(batch)
            backend._forward_paged_attention(q, layer, batch, topk)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            backend.init_forward_metadata_out_graph(batch)
            actual = backend._forward_paged_attention(q, layer, batch, topk)
        for iteration in range(2):
            if iteration:
                reqs.copy_(reqs.flip(0))
                lengths.copy_(torch.tensor([20, 8], device="cuda"))
                q.mul_(.7)
                # Native store updates both packed bytes and scales read by the graph.
                slots = torch.tensor([64, 128], device="cuda")
                key = torch.randn(2, 2, 256, device="cuda", dtype=q.dtype) * 4
                backend.token_to_kv_pool.set_kv_buffer(layer, slots, key, -key)
            graph.replay()
            torch.cuda.synchronize()
            metadata = backend.forward_metadata
            expected = _reference(
                backend, layer, q, topk, metadata.row_req_pool_indices.cpu().tolist(),
                metadata.sequence_lengths.cpu().tolist(),
            )
            torch.testing.assert_close(actual, expected, rtol=.025, atol=.025)
    finally:
        backend.token_to_kv_pool.full_kv_pool._clear_buffers()


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__]))
