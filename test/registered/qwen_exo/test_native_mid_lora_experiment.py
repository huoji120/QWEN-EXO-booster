"""CPU-only contracts for native mid-LoRA experiment selection/evaluation."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from qwen_exo_booster import native_mid_lora_experiment as experiment


@pytest.mark.parametrize("arms", [("ordinary_sft",), ("ordinary_sft", "self_think_sft")])
def test_run_experiment_rejects_unsupported_arm_selection_before_output(tmp_path, arms):
    output = tmp_path / "experiment"
    with pytest.raises(ValueError, match="arms must be self_think_sft"):
        experiment.run_experiment(
            profile="unused-profile", output=output, mode="experiment",
            device="cpu", arms=arms,
        )
    assert not output.exists()


def test_answer_evaluation_masks_supplied_think_without_mutating_training_labels():
    example = SimpleNamespace(
        answer_suffix_ids=(8, 9, 10),
        answer_suffix_labels=(-100, 9, 10),
        provenance={"answer_separator_tokens": 1},
    )
    assembled = {
        "input_ids": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "labels": [-100, -100, -100, 4, 5, 6, 7, -100, 9, 10],
    }
    original_labels = list(assembled["labels"])

    answer_view, answer_start = experiment._answer_evaluation_view(assembled, example)

    assert answer_start == 8
    assert answer_view["input_ids"] == assembled["input_ids"]
    assert answer_view["labels"] == [-100] * 8 + [9, 10]
    assert assembled["labels"] == original_labels
