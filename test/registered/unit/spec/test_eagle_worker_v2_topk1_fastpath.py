"""Equivalence tests for the EagleDraftWorker topk=1 chain fast path.

For topk=1 the draft tree degenerates to a chain, so `draft_forward` skips the
cat/topk/sort/gather of the slow path and returns pre-allocated constants. These
tests check that the pre-allocated `parent_list` / `top_scores_index` match the
slow path (`organize_draft_results`) for num_steps in {1, 2, 3, 4}.
"""

import contextlib
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.layers.logits_processor import LogitsMetadata, LogitsProcessor
from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardBatch,
    ForwardMode,
)
from sglang.srt.speculative.adaptive_runtime_state import SpecRuntimeState
from sglang.srt.speculative.eagle_utils import organize_draft_results
from sglang.srt.speculative.eagle_worker_v2 import EagleDraftWorker, EAGLEWorkerV2
from sglang.test.ci.ci_register import (
    register_amd_ci,
    register_cpu_ci,
    register_cuda_ci,
)
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=20, stage="base-b", runner_config="1-gpu-small")
register_amd_ci(est_time=20, stage="stage-b", runner_config="1-gpu-small-amd")
register_cpu_ci(est_time=20, suite="base-a-test-cpu")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _fake_server_args(**fields):
    """server_args stand-in: carries fields and the override() entry point."""
    ns = SimpleNamespace(**fields)

    def _override(source, **updates):
        for key, value in updates.items():
            setattr(ns, key, value)

    ns.override = _override
    return ns


def _make_chain_lists(num_steps: int, bs: int):
    """Build the (score, token, parents) lists a topk=1 chain produces.

    Shapes/values mirror `select_top_k_tokens` for topk=1: each step yields one
    token; the first step's parents are [-1, 0], later steps' parents are [i].
    """
    score_list, token_list, parents_list = [], [], []
    for i in range(num_steps):
        # Strictly decreasing scores, as a real chain produces (cumulative probs).
        score_list.append(torch.full((bs, 1, 1), float(num_steps - i), device=DEVICE))
        token_list.append(
            torch.arange(i * bs, (i + 1) * bs, device=DEVICE).unsqueeze(1)
        )
        if i == 0:
            parents_list.append(
                torch.tensor([-1, 0], dtype=torch.long, device=DEVICE).repeat(bs, 1)
            )
        else:
            parents_list.append(torch.full((bs, 1), i, dtype=torch.long, device=DEVICE))
    return score_list, token_list, parents_list


def _make_worker(num_steps: int, num_draft_tokens: int):
    worker = object.__new__(EagleDraftWorker)
    worker.topk = 1
    worker.device = DEVICE
    worker.speculative_num_steps = num_steps
    worker.speculative_num_draft_tokens = num_draft_tokens
    worker.server_args = _fake_server_args(
        cuda_graph_config=SimpleNamespace(decode=SimpleNamespace(max_bs=8)),
        max_running_requests=8,
    )
    return worker


class TestEagleWorkerV2Topk1FastPath(CustomTestCase):
    def test_fast_path_matches_slow_path(self):
        bs = 3
        for num_steps in (1, 2, 3, 4):
            with self.subTest(num_steps=num_steps):
                num_draft_tokens = num_steps + 1
                worker = _make_worker(num_steps, num_draft_tokens)
                worker._rebuild_topk1_chain_buffers()

                score_list, token_list, parents_list = _make_chain_lists(num_steps, bs)
                ref_parent, ref_index, ref_tokens = organize_draft_results(
                    score_list, token_list, parents_list, num_draft_tokens
                )

                fast_parent = worker._topk1_parents_prealloc[:bs]
                fast_index = worker._topk1_score_indices_prealloc[:bs]
                fast_tokens = torch.cat(token_list, dim=1)

                self.assertEqual(fast_parent.shape, ref_parent.shape)
                self.assertEqual(fast_parent.tolist(), ref_parent.long().tolist())
                self.assertEqual(fast_index.tolist(), ref_index.long().tolist())
                self.assertEqual(fast_tokens.tolist(), ref_tokens.tolist())

                # The kernel reads these via data_ptr() as contiguous int64.
                self.assertEqual(fast_parent.dtype, torch.long)
                self.assertEqual(fast_index.dtype, torch.long)
                self.assertTrue(fast_parent.is_contiguous())
                self.assertTrue(fast_index.is_contiguous())

    def test_assert_on_inconsistent_steps_and_draft_tokens(self):
        # num_draft_tokens must equal num_steps + 1 for topk=1.
        worker = _make_worker(num_steps=3, num_draft_tokens=3)
        with self.assertRaises(AssertionError):
            worker._rebuild_topk1_chain_buffers()


class TestEagleWorkerV2BackendFallback(CustomTestCase):
    def test_missing_seed_cuda_graph_fallback(self):
        graph_result = (
            [],
            torch.zeros((1, 1), dtype=torch.long, device=DEVICE),
            torch.zeros((1, 1), dtype=torch.long, device=DEVICE),
            None,
        )
        tree_result = (
            torch.empty((0,), dtype=torch.bool, device=DEVICE),
            torch.zeros((1,), dtype=torch.long, device=DEVICE),
            torch.zeros((1, 2), dtype=torch.long, device=DEVICE),
            torch.zeros((1, 2), dtype=torch.long, device=DEVICE),
            torch.zeros((1, 2), dtype=torch.long, device=DEVICE),
            torch.zeros((2,), dtype=torch.long, device=DEVICE),
        )

        for seed_enabled, seed_present, expect_graph in (
            (True, False, False),
            (True, True, True),
            (False, False, True),
        ):
            with self.subTest(
                seed_enabled=seed_enabled,
                seed_present=seed_present,
            ):
                worker = object.__new__(EagleDraftWorker)
                worker.req_to_token_pool = None
                worker.cuda_graph_runner = SimpleNamespace(
                    execute=MagicMock(return_value=graph_result)
                )
                worker.draft_runner = SimpleNamespace(canary_manager=None)
                worker.topk = 1
                worker.speculative_num_steps = 1
                worker.speculative_num_draft_tokens = 2
                worker.device = DEVICE
                worker.tree_mask_mode = None
                worker.seed_dsa_topk_from_draft_extend = seed_enabled
                worker.index_share_for_mtp_iteration = True
                forward_batch = SimpleNamespace(forward_mode=ForwardMode.DECODE)
                worker.draft_forward = MagicMock(return_value=graph_result)
                attn_backend = SimpleNamespace(
                    get_verify_buffers_to_fill_after_draft=lambda: (None, None),
                    max_context_len=1,
                )
                worker.target_worker = SimpleNamespace(
                    model_runner=SimpleNamespace(attn_backend=attn_backend)
                )
                draft_input = SimpleNamespace(
                    bonus_tokens=torch.zeros((1,), dtype=torch.long, device=DEVICE),
                    dsa_topk_indices=(
                        torch.ones((1, 1), dtype=torch.int32, device=DEVICE)
                        if seed_present
                        else None
                    ),
                )
                batch = SimpleNamespace(
                    spec_info=draft_input,
                    forward_mode=ForwardMode.DECODE,
                    seq_lens_sum=1,
                    seq_lens=torch.ones((1,), dtype=torch.int32, device=DEVICE),
                )

                with patch(
                    "sglang.srt.speculative.eagle_worker_common.build_tree_kernel_efficient",
                    return_value=tree_result,
                ), patch(
                    "sglang.srt.speculative.eagle_worker_v2.prepare_for_draft",
                    return_value=(forward_batch, True),
                ):
                    worker.draft(batch)

                self.assertEqual(worker.cuda_graph_runner.execute.called, expect_graph)
                self.assertEqual(worker.draft_forward.called, not expect_graph)

    def _make_adaptive_worker(self, runner_attn_backend):
        """An EAGLEWorkerV2 with a draft worker whose state-machine fields are
        filled with sentinels, sufficient to drive _override_worker_state /
        apply_runtime_state without touching the GPU."""
        draft_runner = SimpleNamespace(
            draft_attn_backend=object(),
            attn_backend=runner_attn_backend,
        )
        draft_worker = SimpleNamespace(
            speculative_num_steps=2,
            speculative_num_draft_tokens=3,
            draft_attn_backend=object(),
            draft_extend_attn_backend=object(),
            cuda_graph_runner=object(),
            cuda_graph_runner_for_draft_extend=object(),
            draft_runner=draft_runner,
            # _override_worker_state / apply_runtime_state call this hook; the
            # topk=1 buffers are exercised by the fast-path tests above.
            _rebuild_topk1_chain_buffers=lambda: None,
        )
        worker = object.__new__(EAGLEWorkerV2)
        worker._draft_worker = draft_worker
        worker._target_worker = SimpleNamespace(
            model_runner=SimpleNamespace(
                attn_backend=object(), decode_cuda_graph_runner=object()
            )
        )
        worker.speculative_num_steps = 2
        worker.speculative_num_draft_tokens = 3
        worker.server_args = _fake_server_args(
            speculative_num_steps=2,
            speculative_num_draft_tokens=3,
            cuda_graph_bs_decode=None,
            disable_cuda_graph=False,
        )
        return worker, draft_worker

    def test_override_worker_state_restores_runner_attn_backend(self):
        # build_adaptive_runtime_state runs init_attention_backend inside this
        # context for each candidate step; the runner backend it assigns must
        # not leak into the live worker.
        initial_backend = object()
        candidate_backend = object()
        worker, dw = self._make_adaptive_worker(initial_backend)

        with worker._override_worker_state(3, 4):
            dw.draft_runner.attn_backend = candidate_backend
            self.assertIs(dw.draft_runner.attn_backend, candidate_backend)

        self.assertIs(dw.draft_runner.attn_backend, initial_backend)

    def test_apply_runtime_state_updates_runner_attn_backend(self):
        # Switching to another step config must repoint the runner backend at
        # that config's draft-extend backend (read by the draft-extend forward).
        new_extend_backend = object()
        worker, dw = self._make_adaptive_worker(object())

        state = SpecRuntimeState(
            speculative_num_steps=3,
            speculative_num_draft_tokens=4,
            draft_attn_backend=object(),
            cuda_graph_runner=object(),
            target_attn_backend=object(),
            target_graph_runner=object(),
            draft_extend_attn_backend=new_extend_backend,
            cuda_graph_runner_for_draft_extend=object(),
        )
        worker.apply_runtime_state(state)

        self.assertIs(dw.draft_runner.attn_backend, new_extend_backend)

    def test_spec_v2_attn_backends_include_draft_extend_fallback(self):
        target_backend = object()
        decode_backend = object()
        fallback_backend = object()

        worker = object.__new__(EAGLEWorkerV2)
        worker._target_worker = SimpleNamespace(
            model_runner=SimpleNamespace(attn_backend=target_backend)
        )
        worker._draft_worker = SimpleNamespace(
            draft_attn_backend=decode_backend,
            draft_extend_attn_backend=None,
            draft_runner=SimpleNamespace(attn_backend=fallback_backend),
        )

        self.assertEqual(
            worker.spec_v2_attn_backends,
            (target_backend, decode_backend, fallback_backend),
        )


class TestEagleDraftExtendSelectedLogits(CustomTestCase):
    @staticmethod
    def _make_processor(vocab_size):
        # Exercise the real pruning, hidden capture and LM-head arithmetic,
        # without initializing distributed process groups or a model runner.
        processor = object.__new__(LogitsProcessor)
        torch.nn.Module.__init__(processor)
        processor.vocab_size = vocab_size
        processor.do_tensor_parallel_all_gather = False
        processor.do_tensor_parallel_all_gather_dp_attn = False
        processor.use_fp32_lm_head = False
        processor.rl_on_policy_target = None
        processor.logit_scale = None
        processor.final_logit_softcapping = None
        return processor

    def test_draft_extend_logits_hidden_and_seed_selection(self):
        self._check_draft_extend_selection([3, 1, 0, 2], [1, 2, 3, 4])

    def test_single_request_retains_full_lm_head_rows(self):
        for accept_len in (1, 2, 3, 4):
            with self.subTest(accept_len=accept_len):
                self._check_draft_extend_selection([0], [accept_len])

    def _check_draft_extend_selection(self, request_order, accept_lens):
        bs, width, hidden_size = len(request_order), 4, 8
        request_order = torch.tensor(request_order, device=DEVICE)
        accept_lens = torch.tensor(accept_lens, device=DEVICE)[request_order]
        selected = torch.arange(bs, device=DEVICE) * width + accept_lens - 1
        base = torch.arange(bs * width * hidden_size, device=DEVICE).reshape(
            bs, width, hidden_size
        )
        base = torch.sin(base.float() * 0.37)
        winners = torch.arange(bs * width, device=DEVICE).reshape(bs, width) % hidden_size
        base.scatter_add_(2, winners.unsqueeze(-1), torch.full_like(base[:, :, :1], 3))
        base = base[request_order].reshape(bs * width, hidden_size)

        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            for graph_available, graph_runs, gathered, padding in (
                (False, False, False, 0),
                (True, False, False, 4),
                (False, False, True, 4),
                (True, True, False, 4),
            ):
                with self.subTest(
                    dtype=dtype, graph_runs=graph_runs, gathered=gathered, padding=padding
                ):
                    states = torch.cat(
                        (base, torch.full((padding, hidden_size), 99, device=DEVICE))
                    ).to(dtype)
                    hc_states = torch.cat((states + 7, states - 9), dim=-1)
                    head = torch.nn.Linear(hidden_size, hidden_size, bias=False).to(
                        device=DEVICE, dtype=dtype
                    )
                    with torch.no_grad():
                        head.weight.copy_(
                            torch.eye(hidden_size, device=DEVICE)
                            + torch.arange(hidden_size, device=DEVICE).float()[:, None]
                            * 0.001
                        )
                    weight_before = head.weight.detach().clone()
                    processor = self._make_processor(hidden_size)
                    seed = torch.arange(
                        states.shape[0] * 3, device=DEVICE, dtype=torch.int32
                    ).reshape(-1, 3)
                    observed = {}

                    def prepare(spec_info, batch, token_ids, *args, **kwargs):
                        # Preparation must retain all target hidden/HC rows.
                        self.assertIs(spec_info.hidden_states, hc_states)
                        return ForwardBatch(
                            forward_mode=ForwardMode.DRAFT_EXTEND_V2,
                            batch_size=bs,
                            input_ids=token_ids,
                            req_pool_indices=request_order,
                            seq_lens=batch.seq_lens,
                            out_cache_loc=torch.arange(states.shape[0], device=DEVICE),
                            seq_lens_sum=0,
                            spec_info=spec_info,
                            capture_hidden_mode=CaptureHiddenMode.FULL,
                            next_token_logits_buffer=torch.empty(
                                (states.shape[0], hidden_size),
                                device=DEVICE,
                                dtype=torch.float32,
                            ),
                        )

                    def forward(forward_batch):
                        if forward_batch.spec_info.dsa_seed_topk_capture is not None:
                            forward_batch.spec_info.dsa_seed_topk_capture.copy_(seed)
                        metadata = LogitsMetadata.from_forward_batch(forward_batch)
                        with patch.object(
                            processor, "_compute_lm_head", wraps=processor._compute_lm_head
                        ) as compute:
                            output = processor(
                                forward_batch.input_ids,
                                states,
                                head,
                                metadata,
                                aux_hidden_states=[hc_states],
                            )
                        observed["head_rows"] = compute.call_args.args[0].shape[0]
                        observed["logits"] = output.next_token_logits.clone()
                        torch.testing.assert_close(output.hidden_states, hc_states)
                        return output

                    worker = object.__new__(EagleDraftWorker)
                    worker.device = DEVICE
                    worker.topk = 1
                    worker.speculative_num_draft_tokens = width
                    worker.server_args = _fake_server_args(
                        speculative_use_rejection_sampling=False
                    )
                    worker.plan_stream = None
                    worker.plan_stream_ctx = contextlib.nullcontext()
                    worker.seed_dsa_topk_from_draft_extend = True
                    worker.dsa_index_topk = 3
                    worker.dsa_extend_topk_buf = None
                    worker.draft_runner = SimpleNamespace(
                        canary_manager=None,
                        forward=lambda fb: SimpleNamespace(logits_output=forward(fb)),
                    )
                    worker.cuda_graph_runner_for_draft_extend = (
                        SimpleNamespace(
                            can_run_graph=lambda fb: graph_runs,
                            execute=forward,
                            buffers=SimpleNamespace(dsa_seed_topk_capture=seed),
                        )
                        if graph_available
                        else None
                    )
                    batch = SimpleNamespace(
                        seq_lens=torch.ones(bs, device=DEVICE, dtype=torch.int32),
                        sampling_info=SimpleNamespace(),
                    )
                    result = SimpleNamespace(
                        logits_output=SimpleNamespace(hidden_states=hc_states),
                        accept_lens=accept_lens,
                        next_token_ids=torch.arange(states.shape[0], device=DEVICE),
                        next_draft_input=SimpleNamespace(),
                    )
                    with patch(
                        "sglang.srt.speculative.eagle_worker_v2.prepare_for_draft_extend",
                        side_effect=prepare,
                    ), patch(
                        "sglang.srt.speculative.eagle_worker_v2.require_gathered_buffer",
                        return_value=gathered,
                    ), patch("sglang.srt.speculative.eagle_worker_v2._is_hip", False):
                        worker._draft_extend_for_decode(batch, result)

                    pruned = bs > 1 and not graph_runs and not gathered
                    self.assertEqual(
                        observed["head_rows"], bs if pruned else states.shape[0]
                    )
                    reference = (states @ head.weight.T).float()[selected]
                    actual = observed["logits"] if pruned else observed["logits"][selected]
                    tolerance = {
                        torch.float32: 1e-5,
                        torch.float16: 0.002,
                        torch.bfloat16: 0.02,
                    }[dtype]
                    torch.testing.assert_close(
                        actual, reference, rtol=tolerance, atol=tolerance
                    )
                    next_input = result.next_draft_input
                    self.assertEqual(
                        next_input.topk_index.tolist(),
                        reference.argmax(dim=-1, keepdim=True).tolist(),
                    )
                    torch.testing.assert_close(
                        next_input.topk_p, torch.ones(bs, 1, device=DEVICE)
                    )
                    torch.testing.assert_close(
                        next_input.hidden_states, hc_states[selected]
                    )
                    torch.testing.assert_close(
                        next_input.dsa_topk_indices, seed[selected]
                    )
                    torch.testing.assert_close(
                        head.weight, weight_before, rtol=0, atol=0
                    )


if __name__ == "__main__":
    unittest.main()
