import sys
from types import SimpleNamespace

import pytest
import torch

from sglang.srt.models.dflash import (
    CandidateSelector,
    DFlash2DraftModel,
    _grouped_conv,
)
from sglang.srt.speculative.dflash_utils import parse_dflash_draft_config
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


def test_dflash_unary_logit_transform():
    logits = torch.tensor([[-100.0, 0.0, 100.0]], dtype=torch.bfloat16)
    for fields in ({}, {"output_multiplier": 0.2, "final_logit_softcapping": 20.0}):
        config = parse_dflash_draft_config(
            draft_hf_config={
                "num_hidden_layers": 5,
                "dflash_config": {
                    "selector_rank": 256,
                    "selector_top_k": 16,
                    **fields,
                },
            }
        )
        actual = DFlash2DraftModel._transform_unary_logits(
            SimpleNamespace(draft_config=config), logits
        )
        expected = logits.float() * config.output_multiplier
        if config.final_logit_softcapping is not None:
            expected = torch.tanh(expected / config.final_logit_softcapping)
            expected *= config.final_logit_softcapping
        torch.testing.assert_close(actual, expected)


def test_selector_greedy_row_walk_is_deterministic_in_a_mixed_batch():
    """A greedy row walks the argmax, so the q it hands verify has to be the point
    mass there. Greedy reaches the selector as top_k=1 with the temperature reset
    to 1.0, so a softmax q stays a real distribution and verify would
    rejection-sample a deterministic request against it. The row must also not
    depend on who else is in the batch."""
    selector = CandidateSelector(hidden_size=4, vocab_size=16, state_rank=2, top_k=4)
    torch.manual_seed(1)
    candidate_ids = torch.randint(0, 16, (2, 3, 4))
    scores = torch.randn(2, 3, 4, 4)
    uniforms = torch.tensor([[0.2, 0.7, 0.4], [0.8, 0.1, 0.6]])
    temperatures = torch.tensor([1.0, 0.7])
    greedy_mask = torch.tensor([True, False])

    mixed_tokens, mixed_q = selector.sample_path(
        candidate_ids=candidate_ids,
        scores=scores,
        uniforms=uniforms,
        temperatures=temperatures,
        greedy_mask=greedy_mask,
    )
    assert torch.all((mixed_q[0] == 0) | (mixed_q[0] == 1))
    for row in range(2):
        tokens, q_rows = selector.sample_path(
            candidate_ids=candidate_ids[row : row + 1],
            scores=scores[row : row + 1],
            uniforms=uniforms[row : row + 1],
            temperatures=temperatures[row : row + 1],
            greedy_mask=greedy_mask[row : row + 1],
        )
        torch.testing.assert_close(mixed_tokens[row], tokens[0])
        torch.testing.assert_close(mixed_q[row], q_rows[0])


def test_selector_rejects_a_quantized_target_lm_head():
    """The candidate matmuls read the lm_head weight directly, so a packed or
    absent weight would be read as if it were dense."""
    model = SimpleNamespace(
        lm_head=SimpleNamespace(weight=torch.empty(8, 4, dtype=torch.int8)),
        candidate_selector=SimpleNamespace(top_k=4),
    )
    with pytest.raises(RuntimeError, match="requires a dense"):
        DFlash2DraftModel.compute_candidates(model, torch.randn(2, 4))


def test_dflash_unsupported_features_use_target_only_fallback():
    """Hidden-state capture is the per-request feature DFLASH still routes to
    the target; internal jobs stay target-only unless marked eligible."""
    from sglang.srt.speculative.dflash_utils import (
        is_dflash_target_only_request,
        validate_dflash_request,
    )

    req = SimpleNamespace(
        return_logprob=False,
        return_hidden_states=False,
        sampling_params=SimpleNamespace(
            json_schema=None,
            regex=None,
            ebnf=None,
            structural_tag=None,
            custom_params={"qwen_exo_kind": "internal"},
        ),
    )
    assert is_dflash_target_only_request(req)
    assert validate_dflash_request(req, enable_overlap=True) is None

    req.sampling_params.custom_params = {"qwen_exo_kind": "user"}
    req.return_hidden_states = True
    assert is_dflash_target_only_request(req)
    assert validate_dflash_request(req, enable_overlap=True) is None


def test_grammar_constrained_requests_run_dflash():
    """Strict tool calls carry a structural_tag. Routing grammar requests
    target-only made every agent turn decode without DFLASH (eager, ~16 tok/s
    at bs1) and parked them out of DFLASH batches; verify now applies a
    per-draft-position grammar mask instead."""
    from sglang.srt.speculative.dflash_utils import (
        is_dflash_target_only_request,
        validate_dflash_request,
    )

    req = SimpleNamespace(
        return_logprob=False,
        return_hidden_states=False,
        sampling_params=SimpleNamespace(
            json_schema=None,
            regex=None,
            ebnf=None,
            structural_tag='{"type": "structural_tag"}',
            custom_params={"qwen_exo_kind": "user"},
        ),
    )
    assert not is_dflash_target_only_request(req)
    assert validate_dflash_request(req, enable_overlap=True) is None

    req.sampling_params.structural_tag = None
    req.sampling_params.json_schema = {"type": "object"}
    req.sampling_params.custom_params = {
        "qwen_exo_kind": "internal",
        "qwen_exo_dflash": "eligible",
    }
    assert not is_dflash_target_only_request(req)
    assert validate_dflash_request(req, enable_overlap=True) is None


class _CountingGrammar:
    """Allows exactly token ``10 + accepted`` next; tracks net acceptance."""

    def __init__(self):
        self.accepted = []

    def allocate_vocab_mask(self, vocab_size, batch_size, device):
        return torch.full((batch_size, (vocab_size + 31) // 32), -1, dtype=torch.int32)

    def fill_vocab_mask(self, vocab_mask, idx):
        allowed = 10 + len(self.accepted)
        vocab_mask[idx] = 0
        vocab_mask[idx, allowed // 32] = 1 << (allowed % 32)

    def accept_token(self, token):
        self.accepted.append(token)

    def rollback(self, count):
        del self.accepted[-count:]

    def is_terminated(self):
        return False


def test_grammar_vocab_mask_follows_the_draft_chain_and_rolls_back():
    """Row i must be the grammar mask after the committed bonus token plus
    draft tokens 1..i; rows past the first grammar-invalid draft and rows of
    non-grammar requests stay all-allowed, and the matcher ends unchanged."""
    from sglang.srt.speculative.dflash_info import DFlashVerifyInput
    from sglang.srt.speculative.dflash_utils import build_dflash_grammar_vocab_mask

    grammar = _CountingGrammar()
    reqs = [SimpleNamespace(grammar=grammar), SimpleNamespace(grammar=None)]
    draft_tokens = torch.tensor([[7, 10, 11, 40, 13], [1, 2, 3, 4, 5]])
    verify_input = DFlashVerifyInput(
        draft_token=draft_tokens.flatten(),
        positions=torch.zeros(10, dtype=torch.int64),
        draft_token_num=5,
    )

    mask = build_dflash_grammar_vocab_mask(
        reqs=reqs, draft_tokens=draft_tokens, vocab_size=64, verify_input=verify_input
    )

    def allowed(row):
        bits = mask[row].tolist()
        return [t for t in range(64) if (bits[t // 32] >> (t % 32)) & 1]

    assert [allowed(row) for row in range(3)] == [[10], [11], [12]]
    assert all(len(allowed(row)) == 64 for row in range(3, 10))
    assert grammar.accepted == []
    assert verify_input.grammar is grammar


def test_dflash_target_logprob_need_is_request_scoped():
    from sglang.srt.speculative.dflash_utils import (
        dflash_request_needs_target_logprobs,
    )

    def request(*, kind, job_type=None, return_logprob=False):
        return SimpleNamespace(
            return_logprob=return_logprob,
            sampling_params=SimpleNamespace(
                custom_params={
                    "qwen_exo_kind": kind,
                    **({"qwen_exo_job_type": job_type} if job_type else {}),
                }
            ),
        )

    assert dflash_request_needs_target_logprobs(request(kind="user"))
    assert dflash_request_needs_target_logprobs(
        request(kind="internal", job_type="query_probe")
    )
    assert dflash_request_needs_target_logprobs(
        request(kind="internal", return_logprob=True)
    )
    assert not dflash_request_needs_target_logprobs(
        request(kind="internal", job_type="response_compaction")
    )


def test_dflash_think_acceptance_is_relative_and_phase_limited():
    from sglang.srt.speculative.dflash_utils import dflash_think_acceptance_mask

    # Candidate positions are candidates[:, 1:]. The first target row is a
    # near-top candidate (exp(-0.5) ~= 0.607), while the second is below 0.60.
    candidates = torch.tensor([[11, 2, 3]])
    target_logits = torch.tensor(
        [
            [0.0, 0.0, -0.5, -4.0],
            [0.0, 0.0, -4.0, -0.6],
            [0.0, 0.0, -4.0, -0.6],
        ]
    )
    think_mask = torch.tensor([[True, True]])

    force_mask, relative_probability = dflash_think_acceptance_mask(
        candidates=candidates,
        target_logits=target_logits,
        think_mask=think_mask,
        probability_threshold=0.60,
    )

    assert force_mask.tolist() == [[True, False]]
    assert relative_probability[0, 0].item() == pytest.approx(
        torch.exp(torch.tensor(-0.5)).item(), rel=1e-5
    )

    answer_only_mask, _ = dflash_think_acceptance_mask(
        candidates=candidates,
        target_logits=target_logits,
        think_mask=torch.zeros_like(think_mask),
        probability_threshold=0.0,
    )
    assert not bool(answer_only_mask.any())


def test_dflash_think_acceptance_rejects_invalid_shapes_and_thresholds():
    from sglang.srt.speculative.dflash_utils import dflash_think_acceptance_mask

    candidates = torch.tensor([[0, 1]])
    logits = torch.zeros((2, 4))
    phase = torch.ones((1, 1), dtype=torch.bool)
    with pytest.raises(ValueError, match=r"within \[0, 1\]"):
        dflash_think_acceptance_mask(
            candidates=candidates,
            target_logits=logits,
            think_mask=phase,
            probability_threshold=1.1,
        )
    with pytest.raises(ValueError, match="row count"):
        dflash_think_acceptance_mask(
            candidates=candidates,
            target_logits=logits[:1],
            think_mask=phase,
            probability_threshold=0.6,
        )


def test_grouped_conv_supports_runtime_block_sizes():
    """The conv indexes a position inside the block, so it must follow whatever
    block size the worker resolved -- including one that is not a power of two."""
    torch.manual_seed(0)
    groups, group_size, taps = 3, 2, 2
    hidden_size = groups * group_size
    batch_size = 2

    for block_size in (5, 8, 16):
        hidden = torch.randn(batch_size * block_size, hidden_size)
        delta = torch.randn(batch_size * block_size, taps, groups)
        base = torch.randn(taps, hidden_size)

        actual = _grouped_conv(
            hidden, delta, base, block_size, groups, group_size, taps
        )

        expected = torch.empty_like(hidden)
        hidden_3d = hidden.view(batch_size, block_size, groups, group_size)
        delta_4d = delta.view(batch_size, block_size, taps, groups)
        base_3d = base.view(taps, groups, group_size)
        for batch in range(batch_size):
            for position in range(block_size):
                value = torch.zeros(groups, group_size)
                for tap in range(min(taps, position + 1)):
                    coefficient = base_3d[tap] + delta_4d[batch, position, tap, :, None]
                    value += coefficient * hidden_3d[batch, position - tap]
                expected[batch * block_size + position] = value.flatten()
        torch.testing.assert_close(actual, expected)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
