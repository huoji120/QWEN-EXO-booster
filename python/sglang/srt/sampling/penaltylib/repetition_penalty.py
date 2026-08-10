import logging

import torch
from torch._dynamo.exc import BackendCompilerFailed

from sglang.srt.sampling.penaltylib.orchestrator import _BatchedPenalizer
from sglang.srt.utils import get_compiler_backend, is_npu

logger = logging.getLogger(__name__)
_is_npu = is_npu()
_use_compiled_scaling_penalties = not _is_npu


def _apply_scaling_penalties_eager(logits, scaling_penalties):
    logits[:] = torch.where(
        logits < 0,
        logits * scaling_penalties,
        logits / scaling_penalties,
    )


_compiled_apply_scaling_penalties = torch.compile(
    _apply_scaling_penalties_eager,
    dynamic=True,
    backend=get_compiler_backend(),
    disable=_is_npu,
)


def apply_scaling_penalties(logits, scaling_penalties):
    global _use_compiled_scaling_penalties

    if _use_compiled_scaling_penalties:
        try:
            _compiled_apply_scaling_penalties(logits, scaling_penalties)
            return
        except BackendCompilerFailed:
            logger.warning(
                "Disabling compiled repetition penalties after backend failure",
                exc_info=True,
            )
            _use_compiled_scaling_penalties = False

    _apply_scaling_penalties_eager(logits, scaling_penalties)


class BatchedRepetitionPenalizer(_BatchedPenalizer):
    """
    Repetition penalizer penalizes tokens based on their presence in the generated output.
    """

    is_multiplicative: bool = True

    def _is_required(self) -> bool:
        return any(
            req.sampling_params.repetition_penalty != 1.0
            for req in self.orchestrator.reqs()
        )

    def _prepare(self):
        self.cumulated_repetition_penalties = torch.ones(
            (len(self.orchestrator.reqs()), self.orchestrator.vocab_size),
            dtype=torch.float32,
            device=self.orchestrator.device,
        )
        self.repetition_penalties = (
            torch.tensor(
                data=[
                    req.sampling_params.repetition_penalty
                    for req in self.orchestrator.reqs()
                ],
                dtype=torch.float32,
                device=self.orchestrator.device,
            )
        ).unsqueeze_(1)

    def _cumulate_output_tokens(self, output_ids: torch.Tensor):
        self.cumulated_repetition_penalties.scatter_(
            dim=1,
            index=output_ids.unsqueeze(1),
            src=self.repetition_penalties,
        )

    def _apply(self, logits: torch.Tensor) -> torch.Tensor:
        apply_scaling_penalties(logits, self.cumulated_repetition_penalties)
        return logits

    def get_scaling_penalties(self) -> torch.Tensor:
        return self.cumulated_repetition_penalties

    def _filter(self, keep_indices: torch.Tensor):
        self.repetition_penalties = self.repetition_penalties[keep_indices]
        self.cumulated_repetition_penalties = self.cumulated_repetition_penalties[
            keep_indices
        ]

    def _merge(self, their: "BatchedRepetitionPenalizer"):
        self.repetition_penalties = torch.cat(
            [self.repetition_penalties, their.repetition_penalties], dim=0
        )
        self.cumulated_repetition_penalties = torch.cat(
            [self.cumulated_repetition_penalties, their.cumulated_repetition_penalties],
            dim=0,
        )

    def _teardown(self) -> None:
        for name in ("repetition_penalties", "cumulated_repetition_penalties"):
            if hasattr(self, name):
                delattr(self, name)
