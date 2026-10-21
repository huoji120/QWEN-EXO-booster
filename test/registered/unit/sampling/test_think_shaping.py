"""Thinking-time logit shaping: wait-token suppression and </think> soft landing."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

from types import SimpleNamespace

import torch

from sglang.srt.sampling.think_shaping import (
    QWEN_EXO_THINK_SHAPING_KEY,
    ThinkShapingConfig,
    build_think_shaping,
    ThinkShapingWorkspace,
)
from sglang.srt.speculative.dflash_utils import apply_dflash_verify_logits_adjustments

END = 9
WAIT = 3
CONFIG = ThinkShapingConfig(
    end_token_id=END,
    suppress_after=4,
    suppress_token_ids=(WAIT,),
    suppress_strength=5.0,
    landing_start=6,
    landing_end=10,
    landing_bias=8.0,
)


def _req(output_ids, config=CONFIG):
    custom = {QWEN_EXO_THINK_SHAPING_KEY: config.to_params()} if config else None
    return SimpleNamespace(
        output_ids=list(output_ids),
        sampling_params=SimpleNamespace(custom_params=custom),
        qwen_exo_think_shaping=None,
    )


class _VerifyInfo(SimpleNamespace):
    def __len__(self):
        return 1


def test_suppression_starts_after_the_threshold_and_landing_ramps_to_the_budget():
    """Derived schedule: nothing before ``suppress_after``; wait tokens at
    -strength after it; </think> bias linear from landing_start to the budget,
    capped at landing_bias beyond it."""
    assert CONFIG.row_bias(3) == {}
    assert CONFIG.row_bias(4) == {WAIT: -5.0}
    assert CONFIG.row_bias(8) == {WAIT: -5.0, END: 4.0}
    assert CONFIG.row_bias(20)[END] == 8.0


def test_bias_stops_once_thinking_ends_and_skips_requests_without_shaping():
    """After </think> the answer must be untouched (no suppressed words, no
    second </think> push), including when </think> arrives between scans."""
    thinking = _req([1] * 8)
    plain = _req([1] * 8, config=None)
    shaping = build_think_shaping([thinking, plain], device="cpu")
    assert shaping.token_ids.tolist() == [WAIT, END]
    assert shaping.bias.tolist() == [[-5.0, 4.0], [0.0, 0.0]]

    thinking.output_ids += [END, 1, 1]
    assert build_think_shaping([thinking, plain], device="cpu") is None
    thinking.output_ids += [1] * 20
    assert build_think_shaping([thinking], device="cpu") is None
    assert build_think_shaping([_req([1] * 2)], device="cpu") is None


def test_bias_is_applied_on_decode_and_on_every_speculative_verify_position():
    """The DFLASH verify broadcast path must add the bias at each draft
    position exactly once (the dense fallback goes through apply_logits_bias)."""
    shaping = build_think_shaping([_req([1] * 8)], device="cpu")
    logits = torch.zeros(1, 12)
    shaping.apply(logits)
    assert logits[0, WAIT] == -5.0 and logits[0, END] == 4.0 and logits.sum() == -1.0

    draft = 3
    verify = torch.zeros(draft, 12)
    info = _VerifyInfo(
        has_custom_logit_processor=False,
        acc_linear_penalties=None,
        penalizer_orchestrator=None,
        vocab_mask=None,
        logit_bias=None,
        think_shaping=shaping,
    )
    apply_dflash_verify_logits_adjustments(
        next_token_logits=verify, sampling_info=info, draft_token_num=draft
    )
    assert verify[:, WAIT].tolist() == [-5.0] * draft
    assert verify[:, END].tolist() == [4.0] * draft
    assert verify.sum() == -1.0 * draft


def test_shaping_params_survive_the_scheduler_ipc_round_trip():
    """custom_params values must be scalars (``CustomParamValue``); a token-id
    list nested in the shaping dict failed msgpack validation on the way to
    the scheduler."""
    import msgspec

    from sglang.srt.sampling.sampling_params import SamplingParams

    params = SamplingParams(
        max_new_tokens=8,
        custom_params={QWEN_EXO_THINK_SHAPING_KEY: CONFIG.to_params()},
    )
    decoded = msgspec.msgpack.decode(msgspec.msgpack.encode(params), type=SamplingParams)
    restored = ThinkShapingConfig.from_params(
        decoded.custom_params[QWEN_EXO_THINK_SHAPING_KEY]
    )
    assert restored == CONFIG


def test_bias_reaches_eagle_verify_rows_grouped_per_request():
    """Prod runs EAGLE: its verify logits are [bs * draft, vocab] and never
    went through apply_logits_bias, so the shaping had no effect there."""
    shaping = build_think_shaping([_req([1] * 8), _req([1] * 8, config=None)], device="cpu")
    draft = 3
    logits = torch.zeros(2 * draft, 12)
    shaping.apply_repeated(logits, draft)
    assert logits[:draft, WAIT].tolist() == [-5.0] * draft
    assert logits[:draft, END].tolist() == [4.0] * draft
    assert logits[draft:].abs().sum() == 0


def test_only_single_token_suppress_words_are_biased():
    """A multi-token word must not be biased through its first piece (which
    other words share)."""
    from qwen_exo_booster.think_shaping_policy import suppress_token_ids

    vocab = {"Wait": [1], " Wait": [2], "Hmm": [3], " Hmm": [4, 5], "等等": [6]}
    tokenizer = SimpleNamespace(
        encode=lambda text, add_special_tokens=False: vocab.get(text, [7, 8])
    )
    assert suppress_token_ids(tokenizer) == (1, 2, 3, 6)


def test_workspace_snapshots_survive_reuse_reordering_and_think_completion():
    workspace = ThinkShapingWorkspace("cpu")
    overlap_config = ThinkShapingConfig(
        end_token_id=WAIT, suppress_after=4, suppress_token_ids=(WAIT, WAIT, 5),
        suppress_strength=5.0, landing_start=6, landing_end=10, landing_bias=8.0,
    )
    first = _req([1] * 4)
    overlapping = _req([1] * 6, overlap_config)
    plain = _req([1] * 8, None)
    snapshots = []
    for requests, configs in (
        ([first, overlapping], [CONFIG, overlap_config]),
        ([overlapping, first, plain], [overlap_config, CONFIG, None]),
        ([first], [CONFIG]),
        ([overlapping, plain, first, plain], [overlap_config, None, CONFIG, None]),
    ):
        for tokens in (6, 8, 12):
            first.output_ids = [1] * tokens
            overlapping.output_ids = [1] * tokens
            shaping = workspace.build(requests)
            expected = torch.zeros(len(requests), 3, 12)
            for row, config in enumerate(configs):
                if config is not None:
                    for token, value in config.row_bias(tokens).items():
                        expected[row, :, token] = value
            snapshots.append((shaping, expected))
    first.output_ids.append(END)
    overlapping.output_ids.append(WAIT)
    assert workspace.build([first, overlapping, plain]) is None
    # Delayed consumers must still see the old forward's values and token layout.
    for shaping, expected in snapshots:
        logits = torch.zeros_like(expected)
        shaping.apply_repeated(logits.view(-1, logits.shape[-1]), 3)
        torch.testing.assert_close(logits, expected, rtol=0, atol=0)


def test_workspace_repeated_bias_matches_each_logits_dtype_without_touching_other_columns():
    for dtype in (torch.float16, torch.bfloat16, torch.float32):
        workspace = ThinkShapingWorkspace("cpu")
        thinking, plain = _req([1] * 8), _req([1] * 8, None)
        shaping = workspace.build([thinking, plain])
        logits = torch.arange(2 * 4 * 12, dtype=dtype).reshape(8, 12)
        expected = logits.clone()
        for token, value in CONFIG.row_bias(8).items():
            expected[:4, token] += torch.tensor(value, dtype=dtype)
        shaping.apply_repeated(logits, 4)
        torch.testing.assert_close(logits, expected, rtol=0, atol=0)
