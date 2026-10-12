from types import SimpleNamespace

import torch

from sglang.srt.managers import overlap_utils
from sglang.srt.managers.overlap_utils import FutureMap, RelayPayload
from sglang.srt.speculative import spec_utils
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm


def test_eagle_target_only_relay_before_and_after_speculative_batches(monkeypatch):
    monkeypatch.setattr(overlap_utils, "_is_cuda", False)
    monkeypatch.setattr(spec_utils, "spec_need_hidden_states", lambda: True)
    pool = SimpleNamespace(req_to_token=torch.zeros((4, 16), dtype=torch.int64))
    future = FutureMap(torch.device("cpu"), SpeculativeAlgorithm.EAGLE, pool)
    slot = torch.tensor([1])
    future.stash(slot, RelayPayload(bonus_tokens=torch.tensor([17])))
    assert future.output_tokens_buf[1].item() == 17
    assert not future._forward_buf_initialized

    draft = SimpleNamespace(
        bonus_tokens=torch.tensor([23]), topk_p=torch.tensor([[.8, .2]]),
        topk_index=torch.tensor([[7, 9]]), hidden_states=torch.tensor([[1., 2., 3.]]),
        draft_probs=None, dsa_topk_indices=None,
    )
    future.stash(slot, RelayPayload.from_draft_input(draft))
    assert future._forward_buf_initialized
    torch.testing.assert_close(future.topk_p_buf[slot], draft.topk_p)
    torch.testing.assert_close(future.hidden_states_buf[slot], draft.hidden_states)
    # Internal target-only work owns no draft payload and must not touch the
    # speculative buffers, whether it uses a new slot or a recycled one.
    future.stash(torch.tensor([2]), RelayPayload(bonus_tokens=torch.tensor([31])))
    future.stash(slot, RelayPayload(bonus_tokens=torch.tensor([37])))
    assert future.output_tokens_buf[2].item() == 31
    assert future.output_tokens_buf[1].item() == 37
    torch.testing.assert_close(future.topk_p_buf[slot], draft.topk_p)
    torch.testing.assert_close(future.hidden_states_buf[slot], draft.hidden_states)

    draft.bonus_tokens = torch.tensor([41])
    draft.topk_p = torch.tensor([[.6, .4]])
    draft.topk_index = torch.tensor([[11, 13]])
    draft.hidden_states = torch.tensor([[4., 5., 6.]])
    future.stash(slot, RelayPayload.from_draft_input(draft))
    assert future.output_tokens_buf[1].item() == 41
    torch.testing.assert_close(future.topk_p_buf[slot], draft.topk_p)
    torch.testing.assert_close(future.topk_index_buf[slot], draft.topk_index)
    torch.testing.assert_close(future.hidden_states_buf[slot], draft.hidden_states)
