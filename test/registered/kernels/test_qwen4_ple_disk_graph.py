from types import SimpleNamespace

import pytest
import torch
from torch import nn
from safetensors.torch import save_file

from sglang.srt.environ import envs
from sglang.srt.runtime_context import get_parallel
from sglang.srt.mem_cache.ple_state_pool import NGramPool, ShortConvPool
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.srt.model_executor.forward_context import ForwardContext, forward_context
from sglang.srt.model_executor.runner_utils.capture_mode import model_capture_mode
from sglang.srt.models.qwen4_exp import (
    Qwen4ExpForConditionalGeneration,
    Qwen4ExpPLELayer,
    _prepare_ple_batch,
    _commit_ple_batch,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=30, stage="base-b", runner_config="1-gpu-small")
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def make_fixture(tmp_path, mode):
    import json

    # Native hash heads have prime sizes 7, 11, 13 and 17: exactly 48 rows.
    key = "model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_0.weight"
    values = (torch.arange(192).reshape(48, 4).float() / 128).to(torch.bfloat16)
    save_file({key: values}, str(tmp_path / "table.safetensors"))
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {key: "table.safetensors"}})
    )
    config = SimpleNamespace(
        hidden_size=8, hc_count=4, ple_embed_dim=16, ple_conv_kernel_size=2,
        rms_norm_eps=1e-6, ngram_size=3, heads_per_ngram=2, vocab_size=32,
        ngram_vocab_size_base=7, make_ngram_vocab_size_divisible_by=1,
        eos_token_id=31, split_ngram_parts=1, seed=1234,
        ple_offload_embedding=False,
    )
    with get_parallel().override(tp_size=1), envs.SGLANG_QWEN4_PLE_NVME_PATH.override(str(tmp_path)):
        ple = Qwen4ExpPLELayer(config, layer_id=1).to(device="cuda", dtype=torch.bfloat16)
    with torch.no_grad(), torch.random.fork_rng(devices=[torch.cuda.current_device()]):
        torch.manual_seed(1725)
        for parameter in ple.parameters():
            parameter.uniform_(-0.1, 0.1)
    owner = Qwen4ExpForConditionalGeneration.__new__(Qwen4ExpForConditionalGeneration)
    nn.Module.__init__(owner)
    owner.model = nn.Module()
    owner.model.ple_ngram_size = 3
    owner.model.ple_ngram_eos_token_id = 31
    owner._disk_ple_graph_layers = (ple,)
    width = 4 if mode.is_target_verify() else 1
    slots = 2 if mode.is_target_verify() else 3
    ngram = NGramPool(size=5, context_len=2, eos_token_id=31, device="cuda",
                     spec_state_size=2, speculative_num_draft_tokens=4)
    conv = ShortConvPool(size=5, state_shape=(32, 3), layer_ids=[1],
                        dtype=torch.bfloat16, device="cuda", spec_state_size=2,
                        speculative_num_draft_tokens=4)
    physical = torch.tensor([0, 2, 1, 3, 4], device="cuda")
    pool = SimpleNamespace(
        ple_window_cache=None,
        get_mamba_indices=lambda indices: physical.index_select(0, indices.long()),
        get_ngram_context=ngram.get_context,
        set_ngram_context=ngram.set_context,
        set_ngram_intermediate_context=ngram.set_intermediate_context,
        short_conv_layer_cache=lambda _: conv.conv_state[0],
        short_conv_layer_intermediate_cache=lambda _: conv.intermediate_conv_state[0],
    )
    batch = ForwardBatch(
        forward_mode=mode, batch_size=slots,
        input_ids=torch.zeros(slots * width, dtype=torch.long, device="cuda"),
        req_pool_indices=torch.tensor(([1, 2] if width > 1 else [1, 2, 0]), device="cuda"),
        seq_lens=torch.full((slots,), 8, dtype=torch.int32, device="cuda"),
        out_cache_loc=torch.arange(1, slots * width + 1, device="cuda"),
        seq_lens_sum=slots * 8, return_logprob=False,
        positions=torch.arange(slots * width, device="cuda"),
        spec_info=SimpleNamespace(topk=1, draft_token_num=width) if width > 1 else None,
        extend_seq_lens=torch.full((slots,), width, device="cuda") if width > 1 else None,
    )
    if width == 1:
        batch.out_cache_loc[-1] = 0
    hidden = (torch.arange(slots * width * 32, device="cuda").reshape(-1, 32) / 100).to(torch.bfloat16)
    buffers = owner.allocate_disk_ple_graph_buffers(batch)
    return SimpleNamespace(ple=ple, owner=owner, ngram=ngram, conv=conv,
                           pool=pool, batch=batch, hidden=hidden, buffers=buffers)


@pytest.mark.parametrize("mode", [ForwardMode.DECODE, ForwardMode.TARGET_VERIFY])
@torch.no_grad()
def test_disk_ple_graph_replays_changed_rows_and_preserves_state(tmp_path, mode):
    f = make_fixture(tmp_path, mode)
    try:
        with forward_context(ForwardContext(attn_backend=SimpleNamespace(req_to_token_pool=f.pool))):
            def run():
                batch = _prepare_ple_batch(f.batch.input_ids, f.batch,
                                          ngram_size=3, ngram_eos_token_id=31)
                result = f.ple(f.hidden, f.batch, batch)
                _commit_ple_batch(batch, f.batch)
                return result

            f.batch.qwen4_ple_graph_embeddings = f.buffers
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(2):
                    run()
            torch.cuda.current_stream().wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            with model_capture_mode(), torch.cuda.graph(graph):
                graph_output = run()
            f.ngram.context.fill_(31)
            f.conv.conv_state.zero_()
            pointers = {key: value.data_ptr() for key, value in f.buffers.items()}

            for turn, requests in enumerate(([1, 2], [2, 1], [1, 2])):
                f.batch.req_pool_indices[:2].copy_(torch.tensor(requests, device="cuda"))
                f.batch.input_ids.copy_(torch.arange(f.batch.input_ids.numel(), device="cuda") + 5 + turn * 3)
                if mode.is_target_verify():
                    f.batch.extend_seq_lens.copy_(torch.tensor([2, 4], device="cuda"))
                before_ngram = f.ngram.context.clone()
                before_conv = f.conv.conv_state.clone()
                f.batch.qwen4_ple_graph_embeddings = None
                expected = run().clone()
                expected_ngram = f.ngram.context.clone()
                expected_conv = f.conv.conv_state.clone()
                expected_intermediate = f.ngram.intermediate_context.clone()
                f.ngram.context.copy_(before_ngram)
                f.conv.conv_state.copy_(before_conv)

                actual_batch = f.batch
                if mode.is_decode():
                    from dataclasses import replace
                    actual_batch = replace(f.batch, batch_size=2, input_ids=f.batch.input_ids[:2],
                                           req_pool_indices=f.batch.req_pool_indices[:2],
                                           out_cache_loc=f.batch.out_cache_loc[:2])
                f.owner.prepare_disk_ple_graph_inputs(actual_batch, f.buffers)
                torch.testing.assert_close(f.ngram.context, before_ngram, rtol=0, atol=0)
                torch.testing.assert_close(f.conv.conv_state, before_conv, rtol=0, atol=0)
                assert {key: value.data_ptr() for key, value in f.buffers.items()} == pointers
                if mode.is_decode():
                    assert torch.count_nonzero(f.buffers[1][-1]).item() == 0
                f.batch.qwen4_ple_graph_embeddings = f.buffers
                graph.replay()
                torch.cuda.synchronize()
                real_rows = 2 if mode.is_decode() else 8
                torch.testing.assert_close(graph_output[:real_rows], expected[:real_rows], rtol=0.01, atol=0.001)
                # Slot 0 is inert padding; every real slot, including untouched slots, must match.
                torch.testing.assert_close(f.ngram.context[1:], expected_ngram[1:], rtol=0, atol=0)
                torch.testing.assert_close(f.conv.conv_state[:, 1:], expected_conv[:, 1:], rtol=0.01, atol=0.001)
                if mode.is_target_verify():
                    torch.testing.assert_close(f.ngram.context, before_ngram, rtol=0, atol=0)
                    torch.testing.assert_close(f.ngram.intermediate_context, expected_intermediate, rtol=0, atol=0)
    finally:
        f.ple.ple_embedding.ngram_embedding.close()
