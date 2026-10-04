import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from sglang.srt.configs.qwen3_5 import Qwen3_5Config

from sglang.srt.mem_cache.ple_state_pool import NGramPool, ShortConvPool
from sglang.srt.mem_cache.qsa_kv_pool import QSATokenToKVPool
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.models.qwen4_exp import Qwen4ExpForConditionalGeneration
from sglang.srt.speculative.spec_utils import commit_mamba_states_after_verify


DRAFT = 4


def _conv_scatter_cpu(dst, src, slots, steps):
    rows = torch.arange(steps.numel())
    active = steps >= 0
    dst[:, slots[active]] = src[:, rows[active], steps[active]]


def _cursor_commit_cpu(*, write_pos, cache_base, is_flush, num_accepted,
                       state_batch_indices, max_cache_len, max_spec_len, null_block_id):
    active = state_batch_indices > null_block_id
    slots, counts = state_batch_indices[active], num_accepted[active]
    old_write = write_pos[slots]
    flush = is_flush[slots] != 0
    cache_base[slots] = torch.where(
        flush & (counts > 0), (cache_base[slots] + old_write) % max_cache_len,
        cache_base[slots],
    )
    next_write = torch.where(flush, counts, old_write + counts)
    write_pos[slots] = next_write
    is_flush[slots] = (next_write + 2 * max_spec_len > max_cache_len).to(torch.int8)


def _fixture(replay):
    conv = ShortConvPool(
        size=5, state_shape=(2, 2), layer_ids=[1], dtype=torch.bfloat16,
        device="cpu", spec_state_size=2, speculative_num_draft_tokens=DRAFT,
    )
    ngram = NGramPool(
        size=5, context_len=2, eos_token_id=248044, device="cpu",
        spec_state_size=2, speculative_num_draft_tokens=DRAFT,
    )
    # Real sliding windows of two distinct request-local token sequences.
    for row in range(2):
        window = torch.full((2, 2), -1, dtype=torch.bfloat16)
        context = torch.full((2,), 248044, dtype=torch.int64)
        for step in range(DRAFT):
            token = 10 + row * 100 + step
            window = torch.cat((window[:, 1:], torch.tensor([[token], [token + 1]], dtype=window.dtype)), dim=1)
            context = torch.cat((context[1:], torch.tensor([token])))
            conv.intermediate_conv_state[0, row, step] = window
            ngram.intermediate_context[row, step] = context
    physical = torch.tensor([0, 2, 1])
    gdn = SimpleNamespace(
        conv=[torch.zeros(1, 5, 2, 2, dtype=torch.bfloat16)],
        intermediate_conv_window=[conv.intermediate_conv_state.clone()],
        replayssm_d=torch.zeros(1, 1, 16, 2),
    )
    mamba = SimpleNamespace(
        replayssm_cache_base=torch.zeros(5, dtype=torch.int32) if replay else None,
        replayssm_write_pos=torch.zeros(5, dtype=torch.int32),
        replayssm_is_flush=torch.zeros(5, dtype=torch.int8),
        replayssm_is_kda=False,
    )
    req_pool = SimpleNamespace(
        short_conv_pool=conv, ngram_pool=ngram, mamba_pool=mamba,
        get_mamba_indices=lambda requests: physical[requests.long()],
        get_speculative_mamba2_params_all_layers=lambda: gdn,
    )
    qsa = QSATokenToKVPool.__new__(QSATokenToKVPool)
    qsa.qsa_num_request_slots = 4
    qsa.qsa_compress_ratio = DRAFT
    qsa.qsa_index_kv_heads = 1
    qsa.qsa_index_head_dim = 4
    qsa.full_attention_layer_id_mapping = {0: 0}
    qsa._init_qsa_pending_state(1, "cpu")
    qsa.get_qsa_key_state_buffer(0).fill_(-5)
    qsa.qsa_rope_position_buffer.fill_(-7)
    positions = torch.tensor([3, 4, 5, 6, 5, 6, 7, 8])
    rows = torch.tensor([1, 2]).repeat_interleave(DRAFT)
    slots, _ = qsa.prepare_qsa_verify(
        rows, positions, torch.tensor([3, 7]), torch.tensor([0, 6]), DRAFT,
    )
    candidates = torch.arange(32, dtype=torch.bfloat16).reshape(8, 1, 4) + 20
    coords = torch.stack((positions, positions + 2, positions + 4), dim=1)
    qsa.get_qsa_verify_key_state_buffer(0)[slots] = candidates
    qsa.qsa_rope_state[slots] = coords
    model = Qwen4ExpForConditionalGeneration.__new__(Qwen4ExpForConditionalGeneration)
    torch.nn.Module.__init__(model)
    runner = SimpleNamespace(
        model=model, model_config=SimpleNamespace(hf_config=Qwen3_5Config(), is_draft_model=False),
        req_to_token_pool=req_pool, token_to_kv_pool=qsa, attn_backend=SimpleNamespace(),
    )
    batch = SimpleNamespace(
        forward_mode=ForwardMode.TARGET_VERIFY, req_pool_indices=torch.tensor([1, 2]),
        seq_lens=torch.tensor([3, 1]), mamba_track_indices=None,
    )
    return SimpleNamespace(
        worker=SimpleNamespace(model_runner=runner), batch=batch, conv=conv,
        ngram=ngram, mamba=mamba, gdn=gdn, qsa=qsa,
        candidates=candidates, coords=coords,
    )


class TestQwen4VerifyAcceptedState(unittest.TestCase):
    def test_replayssm_also_commits_real_ple_and_qsa_state(self):
        f = _fixture(True)
        accepted = torch.tensor([1, 3], dtype=torch.int32)
        indices = torch.tensor([[0, -1, -1, -1], [4, 5, 6, -1]], dtype=torch.int32)
        old_keys = f.qsa.get_qsa_key_state_buffer(0).clone()
        old_rope = f.qsa.qsa_rope_position_buffer.clone()
        # Only CUDA-only GDN kernels use numerical CPU references. The common
        # dispatcher and both Qwen4 accepted-state hooks run unchanged.
        with patch("sglang.kernels.ops.attention.fla.gdn_replayssm_spec_decode.commit_gdn_replayssm_spec", new=_cursor_commit_cpu), patch(
            "sglang.kernels.ops.mamba.mamba_state_scatter_triton.fused_conv_window_scatter_with_mask", new=_conv_scatter_cpu
        ):
            commit_mamba_states_after_verify(f.worker, f.batch, accepted, indices, DRAFT)
        torch.testing.assert_close(f.conv.conv_state[0, 2], f.conv.intermediate_conv_state[0, 0, 0], rtol=0, atol=0)
        torch.testing.assert_close(f.conv.conv_state[0, 1], f.conv.intermediate_conv_state[0, 1, 2], rtol=0, atol=0)
        torch.testing.assert_close(f.ngram.context[2], f.ngram.intermediate_context[0, 0], rtol=0, atol=0)
        torch.testing.assert_close(f.ngram.context[1], f.ngram.intermediate_context[1, 2], rtol=0, atol=0)
        torch.testing.assert_close(f.gdn.conv[0][0, 2], f.conv.intermediate_conv_state[0, 0, 0], rtol=0, atol=0)
        self.assertEqual(f.mamba.replayssm_write_pos.tolist(), [0, 3, 1, 0, 0])
        expected_keys, expected_rope = old_keys, old_rope
        for row, count, prefix in ((0, 1, 3), (1, 3, 5)):
            for step in range(count):
                slot = (row + 1) * DRAFT + (prefix + step) % DRAFT
                expected_keys[slot] = f.candidates[row * DRAFT + step]
                expected_rope[slot] = f.coords[row * DRAFT + step]
        torch.testing.assert_close(f.qsa.get_qsa_key_state_buffer(0), expected_keys, rtol=0, atol=0)
        torch.testing.assert_close(f.qsa.qsa_rope_position_buffer, expected_rope, rtol=0, atol=0)

    def test_ordinary_commit_uses_accepted_tree_nodes_and_track_crossing(self):
        f = _fixture(False)
        f.batch.mamba_track_indices = torch.tensor([3, 4])
        accepted = torch.tensor([2, 3], dtype=torch.int32)
        indices = torch.tensor([[0, 2, -1, -1], [4, 5, 7, -1]], dtype=torch.int32)
        with patch("sglang.srt.speculative.spec_utils.get_server_args",
                   return_value=SimpleNamespace(mamba_track_interval=4)):
            commit_mamba_states_after_verify(f.worker, f.batch, accepted, indices, DRAFT)
        for slot, row, step in ((2, 0, 2), (1, 1, 3), (3, 0, 0), (4, 1, 3)):
            torch.testing.assert_close(f.conv.conv_state[0, slot], f.conv.intermediate_conv_state[0, row, step], rtol=0, atol=0)
            torch.testing.assert_close(f.ngram.context[slot], f.ngram.intermediate_context[row, step], rtol=0, atol=0)
        # Acceptance is a tree-node list, not a contiguous row count.
        torch.testing.assert_close(f.qsa.get_qsa_key_state_buffer(0)[5], f.candidates[2], rtol=0, atol=0)
        torch.testing.assert_close(f.qsa.get_qsa_key_state_buffer(0)[8], f.candidates[7], rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
