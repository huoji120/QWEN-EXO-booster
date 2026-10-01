from types import SimpleNamespace

import torch

from sglang.srt.managers.scheduler_components.batch_result_processor import (
    SchedulerBatchResultProcessor,
)


def test_move_logprobs_to_cpu_accepts_tensors_and_python_lists():
    output = SimpleNamespace(
        next_token_logprobs=[-0.25],
        input_token_logprobs=torch.tensor([-0.5, -0.75]),
        next_token_top_logprobs_val=[[-0.1]],
        next_token_top_logprobs_idx=[torch.tensor([3])],
        next_token_token_ids_logprobs_val=[[-0.2]],
    )

    SchedulerBatchResultProcessor.move_logprobs_to_cpu(
        object.__new__(SchedulerBatchResultProcessor),
        batch=SimpleNamespace(return_logprob=True),
        logits_output=output,
    )

    assert output.next_token_logprobs == [-0.25]
    assert output.input_token_logprobs == (-0.5, -0.75)
    assert output.next_token_top_logprobs_val == [[-0.1]]
    assert output.next_token_top_logprobs_idx == [[3]]
    assert output.next_token_token_ids_logprobs_val == [[-0.2]]


def test_decode_preserves_mixed_tensor_list_and_empty_logprob_rows():
    output = SimpleNamespace(
        next_token_logprobs=[-0.25, -0.5, -0.75],
        next_token_top_logprobs_val=[torch.tensor([-0.25]), [-0.5], []],
        next_token_top_logprobs_idx=[torch.tensor([3]), [4], []],
        next_token_token_ids_logprobs_val=[torch.tensor([-1.25]), [-1.5], []],
    )
    ids, logprobs = SchedulerBatchResultProcessor._normalize_decode_outputs(
        object.__new__(SchedulerBatchResultProcessor),
        batch=SimpleNamespace(
            return_logprob=True,
            spec_algorithm=SimpleNamespace(is_none=lambda: True),
        ),
        result=None,
        logits_output=output,
        next_token_ids=torch.tensor([3, 4, 5]),
    )
    assert ids == [[3], [4], [5]]
    assert logprobs == [-0.25, -0.5, -0.75]
    assert output.next_token_top_logprobs_val == [[-0.25], [-0.5], []]
    assert output.next_token_top_logprobs_idx == [[3], [4], []]
    assert output.next_token_token_ids_logprobs_val == [[-1.25], [-1.5], []]
