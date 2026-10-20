"""QWEN-EXO policy for thinking-time shaping (see sglang/srt/sampling/think_shaping.py).

Which tokens count as "let me reconsider" and how the configured knobs become
the per-request ``custom_params`` payload the scheduler reads.
"""

from __future__ import annotations

from typing import Any

from sglang.srt.sampling.think_shaping import ThinkShapingConfig

# Words that open another round of self-verification. Only words the tokenizer
# keeps as a single token are used, so no other word shares the biased token.
# Lowercase "wait" is left alone: in coding work it is an ordinary verb
# ("wait for the process").
SUPPRESS_WORDS = (
    "Wait",
    "Hmm",
    "hmm",
    "Hmmm",
    "Alternatively",
    "alternatively",
    "等等",
    "等一下",
)


def suppress_token_ids(tokenizer: Any) -> tuple[int, ...]:
    """Single-token ids of SUPPRESS_WORDS, with and without a leading space."""
    ids: set[int] = set()
    for word in SUPPRESS_WORDS:
        for variant in (word, " " + word):
            encoded = tokenizer.encode(variant, add_special_tokens=False)
            if len(encoded) == 1:
                ids.add(int(encoded[0]))
    return tuple(sorted(ids))


def think_shaping_params(
    *,
    config: Any,
    end_token_id: int,
    suppress_ids: tuple[int, ...],
    max_reasoning_tokens: int,
) -> dict[str, Any] | None:
    """The scheduler payload for one request, or ``None`` when both knobs are off."""
    suppress_on = config.think_suppress_after_tokens > 0 and config.think_suppress_strength > 0
    landing_on = config.think_landing_bias > 0 and config.think_landing_start_ratio < 1
    if not (suppress_on or landing_on):
        return None
    return ThinkShapingConfig(
        end_token_id=int(end_token_id),
        suppress_after=config.think_suppress_after_tokens if suppress_on else 0,
        suppress_token_ids=suppress_ids if suppress_on else (),
        suppress_strength=float(config.think_suppress_strength),
        landing_start=int(max_reasoning_tokens * config.think_landing_start_ratio),
        landing_end=int(max_reasoning_tokens),
        landing_bias=float(config.think_landing_bias) if landing_on else 0.0,
    ).to_params()


__all__ = ["SUPPRESS_WORDS", "suppress_token_ids", "think_shaping_params"]
