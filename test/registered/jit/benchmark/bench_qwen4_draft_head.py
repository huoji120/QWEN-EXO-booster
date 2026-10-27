"""Full versus selected BF16 draft Head, including target-coordinate scatter.

Run only on an idle GPU. Production geometry needs ~1.7 GiB of head weights
alone; this is a component benchmark, not acceptance-rate or serving evidence.
"""
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.jit_kernel.benchmark import marker
from sglang.srt.layers import logits_processor as logits_module
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.qwen4_draft_head import DraftVocabLogitsProcessor
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=90, stage="base-b-kernel-benchmark", runner_config="1-gpu-large")


@marker.parametrize("rows", [1, 2, 4, 8], [1, 4])
@marker.parametrize("vocab,hidden,selected", [(248320, 2560, 97800)], [(4096, 256, 1600)])
@marker.benchmark("implementation", ["full", "selected"])
def benchmark(rows, vocab, hidden, selected, implementation):
    args = SimpleNamespace(enable_dp_lm_head=False, enable_fp32_lm_head=False,
                           enable_mis=False, rl_on_policy_target=None)
    parallel = SimpleNamespace(tp_size=1, attn_tp_size=1, attn_dp_size=1)
    with ExitStack() as stack, torch.inference_mode():
        # Supply single-rank process metadata, not mocked mathematical work.
        stack.enter_context(patch.object(logits_module, "get_server_args", return_value=args))
        stack.enter_context(patch.object(logits_module, "get_parallel", return_value=parallel))
        stack.enter_context(patch.object(logits_module.triton_symm_mem_ag, "MultimemAllGatherer"))
        stack.enter_context(patch.object(logits_module.triton_symm_mem_ag, "recommended_max_tokens", return_value=128))
        torch.manual_seed(271027)
        head = torch.nn.Linear(hidden, vocab, bias=False, device="cuda", dtype=torch.bfloat16)
        x = torch.randn(rows, hidden, device="cuda", dtype=torch.bfloat16)
        config = SimpleNamespace(vocab_size=vocab, hidden_size=hidden)
        ids = torch.linspace(0, vocab - 1, selected).long().unique()
        baseline = LogitsProcessor(config)
        candidate = DraftVocabLogitsProcessor(config, head, ids)
        expected = baseline._compute_lm_head(x, head)
        actual = candidate._compute_lm_head(x, head)
        selected_ids = ids.cuda()
        torch.testing.assert_close(actual[:, selected_ids], expected[:, selected_ids], rtol=0.02, atol=0.02)
        fn = baseline._compute_lm_head if implementation == "full" else candidate._compute_lm_head
        return marker.do_bench(fn, input_args=(x, head), use_cuda_graph=True,
                               graph_clone_args=None, graph_clone_kwargs=None,
                               memory_args=(x, head.weight if implementation == "full" else candidate.selected_weight))


if __name__ == "__main__":
    benchmark.run()
