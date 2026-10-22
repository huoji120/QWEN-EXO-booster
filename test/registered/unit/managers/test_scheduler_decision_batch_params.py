import inspect
import unittest
from collections import deque
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from qwen_exo_booster.scheduler_fairness import BackgroundPrefillFairness

from sglang.srt.managers.schedule_batch import NextBatchPlan, Req
from sglang.srt.managers.schedule_policy import AddReqResult, PrefillAdder
from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.managers.io_struct import AbortReq
from sglang.srt.managers.scheduler_components.invariant_checker import (
    SchedulerInvariantChecker,
)
from sglang.srt.managers.scheduler_components.pool_stats_observer import (
    SchedulerPoolStatsObserver,
)
from sglang.srt.mem_cache.chunk_cache import ChunkCache
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.mem_cache.memory_pool import ReqToTokenPool

register_cpu_ci(est_time=3, suite="base-a-test-cpu")



class TestDFlashActiveBatchIsolation(CustomTestCase):
    @staticmethod
    def _request(kind, dflash_mode=None):
        custom_params = {"qwen_exo_kind": kind}
        if dflash_mode is not None:
            custom_params["qwen_exo_dflash"] = dflash_mode
        return SimpleNamespace(
            finished=lambda: False,
            return_hidden_states=False,
            sampling_params=SimpleNamespace(
                custom_params=custom_params,
                json_schema=None,
                regex=None,
                ebnf=None,
                structural_tag=None,
            ),
        )

    def test_inflight_last_batch_blocks_opposite_target_only_prefill(self):
        running_batch = SimpleNamespace(reqs=[self._request("user")])
        last_batch = SimpleNamespace(reqs=[self._request("internal")])

        self.assertEqual(
            Scheduler._dflash_active_request_flags(running_batch, last_batch),
            (False, True),
        )

    def test_plain_internal_eligible_request_stays_in_speculative_lane(self):
        batch = SimpleNamespace(
            reqs=[self._request("internal", dflash_mode="eligible")]
        )

        self.assertEqual(
            Scheduler._dflash_active_request_flags(batch),
            (False, True),
        )

    @staticmethod
    def _batch(reqs, *, target_only):
        return SimpleNamespace(
            reqs=reqs,
            spec_algorithm=SimpleNamespace(is_none=lambda: target_only),
        )

    def test_batch_mode_blocks_opposite_lane_when_request_metadata_is_stale(self):
        running_batch = self._batch(
            [self._request("internal", dflash_mode="eligible")],
            target_only=True,
        )
        last_batch = self._batch(
            [self._request("user")],
            target_only=False,
        )

        self.assertEqual(
            Scheduler._dflash_active_request_flags(running_batch, last_batch),
            (True, True),
        )

    def test_preselected_target_only_chunk_owns_lane(self):
        empty_batch = self._batch([], target_only=False)

        self.assertEqual(
            Scheduler._dflash_active_request_flags(
                empty_batch,
                preselected_requests=[self._request("internal")],
            ),
            (True, False),
        )

    def test_preselected_speculative_chunk_owns_lane(self):
        empty_batch = self._batch([], target_only=False)

        self.assertEqual(
            Scheduler._dflash_active_request_flags(
                empty_batch,
                preselected_requests=[self._request("user")],
            ),
            (False, True),
        )


class TestDFlashPriorityYield(CustomTestCase):
    @staticmethod
    def _request(rid, priority, *, target_only=False):
        req = MagicMock(spec=Req)
        req.rid = rid
        req.priority = priority
        req.finished.return_value = False
        req.return_hidden_states = False
        req.sampling_params = SimpleNamespace(
            custom_params={
                "qwen_exo_kind": "internal",
                "qwen_exo_dflash": "target_only" if target_only else "eligible",
            },
            json_schema=None,
            regex=None,
            ebnf=None,
            structural_tag=None,
            max_new_tokens=100,
            ignore_eos=False,
        )
        req.time_stats = SimpleNamespace(wait_queue_entry_time=0)
        req.full_untruncated_fill_ids = [1, 2, 3]
        req.prefix_indices = []
        req.output_ids = [4, 5]
        req.host_hit_length = 0
        req.last_node = MagicMock()
        req.mamba_pool_idx = None
        req.retracted_stain = False
        req.inflight_middle_chunks = 0
        req.req_pool_idx = None
        req.needs_host_load_back.return_value = False

        def set_extend_range(start, end):
            req.extend_range = SimpleNamespace(start=start, end=end, length=end - start)

        req.set_extend_range.side_effect = set_extend_range
        set_extend_range(0, len(req.full_untruncated_fill_ids))
        req.retraction_count = 0
        req.kv = None
        req.input_embeds = None
        return req

    @staticmethod
    def _batch(reqs, *, target_only=False):
        batch = MagicMock()
        batch.reqs = list(reqs)
        batch.spec_algorithm = (
            SpeculativeAlgorithm.NONE if target_only else SpeculativeAlgorithm.DFLASH
        )
        batch.chunked_req = None
        batch.batch_is_full = False
        batch.forward_mode = ForwardMode.DECODE
        batch.batch_size.side_effect = lambda: len(batch.reqs)
        batch.is_prefill_only = False
        batch.return_logprob = False
        batch.input_embeds = None
        batch.is_empty.side_effect = lambda: not batch.reqs

        def filter_batch(*, keep_indices=None, chunked_req_to_exclude=()):
            batch.reqs = [
                req
                for i, req in enumerate(batch.reqs)
                if req not in chunked_req_to_exclude
                and (
                    (keep_indices is None and not req.finished())
                    or (keep_indices is not None and i in keep_indices)
                )
            ]

        def release_req(index, remaining, server_args):
            Req.reset_for_retract(batch.reqs[index])

        batch.filter_batch.side_effect = filter_batch
        batch.release_req.side_effect = release_req
        batch.merge_batch.side_effect = lambda other: batch.reqs.extend(other.reqs)
        return batch

    def _scheduler(self, waiting, *, low_first=False):
        scheduler = object.__new__(Scheduler)
        scheduler.spec_algorithm = SpeculativeAlgorithm.DFLASH
        scheduler.enable_priority_preemption = True
        scheduler.enable_priority_scheduling = True
        scheduler.priority_scheduling_preemption_threshold = 10
        scheduler.server_args = SimpleNamespace(
            schedule_low_priority_values_first=low_first,
            enable_flexkv=False,
            prefill_max_requests=None,
        )
        scheduler.waiting_queue = list(waiting)
        scheduler.chunked_req = None
        scheduler.enable_overlap = False
        scheduler.result_queue = deque()
        scheduler._qwen_exo_parked_batch = None
        scheduler._qwen_exo_active_target_only = None
        scheduler.grammar_manager = MagicMock()
        scheduler.grammar_manager.has_waiting_grammars.return_value = False
        scheduler.enable_hierarchical_cache = False
        scheduler.enable_hicache_storage = False
        scheduler.enable_lora = False
        scheduler.is_hybrid_swa = False
        scheduler.is_mixed_chunk = False
        scheduler.min_free_slots_delayer = None
        scheduler.chunked_prefill_size = None
        scheduler.enable_dynamic_chunking = False
        scheduler.enable_hisparse = False
        scheduler.ps = SimpleNamespace(pp_size=1)
        scheduler.max_prefill_tokens = 10000
        scheduler.max_prefill_bs = 1
        scheduler.max_running_requests = 8
        scheduler.page_size = 1
        scheduler.truncation_align_size = None
        scheduler.dllm_config = None
        scheduler.disaggregation_mode = DisaggregationMode.NULL
        scheduler.new_token_ratio_tracker = SimpleNamespace(current=1.0)
        scheduler.get_num_allocatable_reqs = (
            lambda running_bs, *, num_held_rows=0: 8 - running_bs
        )
        scheduler.policy = MagicMock()
        scheduler.tree_cache = MagicMock()
        scheduler.tree_cache.evictable_size.return_value = 0
        scheduler.token_to_kv_pool_allocator = MagicMock()
        scheduler.token_to_kv_pool_allocator.available_size.return_value = 10000
        scheduler.req_to_token_pool = SimpleNamespace(mamba_allocator=None)
        scheduler.qwen_exo_native_state_bank = None
        scheduler.qwen_exo_hybrid_policy = None
        scheduler.model_config = MagicMock()
        scheduler.tp_worker = SimpleNamespace(
            model_runner=SimpleNamespace(prefill_aware_swa=False)
        )
        scheduler.load_inquirer = MagicMock()
        for name in (
            "_release_finished_qwen_exo_states",
            "_validate_qwen_exo_cached_reuse",
            "_release_qwen_exo_hybrid_state",
            "_bind_qwen_exo_hybrid_state",
            "_release_qwen_exo_reservation",
        ):
            setattr(scheduler, name, MagicMock())
        scheduler._add_request_to_queue = (
            lambda req, is_retracted=False: scheduler.waiting_queue.append(req)
        )
        return scheduler

    def _prefill(
        self, scheduler, running, last=None, *, reject=False, real_admission=False
    ):
        # Exercise real priority/resource accounting; isolate only GPU allocation
        # and prefix matching at the prefill boundary.
        adder = object.__new__(PrefillAdder)
        adder.running_batch = running
        adder.preempt_list = []
        adder.parked_requests = tuple(scheduler._qwen_exo_parked_reqs())
        adder.can_run_list = []
        adder.new_chunked_req = None
        adder.new_token_ratio = 1.0
        adder.priority_scheduling_preemption_threshold = 10
        adder.is_all_swa = adder.is_hybrid_swa = adder.is_hybrid_ssm_cache = False
        adder.tree_cache = scheduler.tree_cache
        adder.token_to_kv_pool_allocator = scheduler.token_to_kv_pool_allocator
        adder.rem_total_token_offset = sum(
            adder._get_running_request_total_token_offset(req) for req in running.reqs
        )
        if adder.parked_requests:
            adder.rem_total_token_offset += scheduler.page_size + sum(
                max(0, req.sampling_params.max_new_tokens - len(req.output_ids))
                for req in adder.parked_requests
            )
        adder.page_size = scheduler.page_size
        adder.rem_chunk_tokens = scheduler.chunked_prefill_size
        adder.rem_input_tokens = scheduler.max_prefill_tokens
        adder.cur_rem_token_offset = 0
        adder.rem_swa_token_offset = 0
        adder.rem_mamba_slots = None
        adder._mamba_slot_cost = 0
        adder.dllm_config = None
        adder.log_hit_tokens = adder.log_input_tokens = 0
        adder.reprocessed_log_hit_tokens = adder.reprocessed_log_input_tokens = 0
        adder.prefill_delayer_single_pass = None
        adder.prefill_max_requests = None
        adder.dsa_prefill_cp_in_seq_split = False

        def add_one(req, **kwargs):
            if reject:
                return AddReqResult.NO_TOKEN
            adder.can_run_list.append(req)
            return AddReqResult.CONTINUE

        if not real_admission:
            adder.add_one_req = add_one

        def init_batch(reqs, *args, **kwargs):
            batch = self._batch(reqs, target_only=args[-1].is_none())
            batch.forward_mode = ForwardMode.EXTEND
            batch.spec_algorithm = args[-1]
            batch.chunked_req = kwargs.get("chunked_req")
            return batch

        module = "sglang.srt.managers.scheduler."
        with (
            patch(module + "PrefillAdder", return_value=adder),
            patch(module + "ScheduleBatch.init_new", side_effect=init_batch),
            patch(module + "PrefillStats.from_adder"),
            patch(module + "set_time_batch"),
            patch(module + "TEST_RETRACT", False),
        ):
            return scheduler._get_new_batch_prefill_raw(None, running, last)

    def test_final_chunk_reuses_its_row_without_blocking_the_free_slot(self):
        for mode in (DisaggregationMode.NULL, DisaggregationMode.PREFILL):
            for parked_count in (0, 1):
                with self.subTest(mode=mode, parked=parked_count):
                    owners = [self._request(f"owner-{i}", 0) for i in range(2)]
                    chunk = self._request("final-chunk", 0)
                    chunk.prefix_indices = [1]
                    chunk.kv_committed_len = 1
                    waiting = [self._request(f"waiting-{i}", 0) for i in range(2)]
                    scheduler = self._scheduler(waiting)
                    scheduler.spec_algorithm = SpeculativeAlgorithm.NONE
                    scheduler.enable_priority_preemption = False
                    scheduler.disaggregation_mode = mode
                    scheduler.max_running_requests = scheduler.max_prefill_bs = 4
                    scheduler.chunked_prefill_size = 8
                    scheduler.chunked_req = chunk
                    scheduler.req_to_token_pool = ReqToTokenPool(4, 16, "cpu", False)
                    scheduler.req_to_token_pool.alloc([*owners, chunk])
                    held_row = chunk.req_pool_idx
                    scheduler.get_num_allocatable_reqs = (
                        Scheduler.get_num_allocatable_reqs.__get__(scheduler)
                    )
                    if parked_count:
                        scheduler._qwen_exo_parked_batch = self._batch(owners[:1])
                        scheduler._qwen_exo_parked_batch.new_tokens_required_next_decode.return_value = 1
                    running = self._batch(owners[parked_count:])
                    with patch(
                        "sglang.srt.managers.scheduler.get_server_args",
                        return_value=SimpleNamespace(pp_max_micro_batch_size=4),
                    ):
                        scheduled, _ = self._prefill(
                            scheduler, running, real_admission=True
                        )
                    self.assertEqual(scheduled.reqs, [chunk, waiting[0]])
                    self.assertEqual(scheduler.waiting_queue, [waiting[1]])
                    rows = scheduler.req_to_token_pool.alloc(scheduled.reqs)
                    self.assertEqual(rows[0], held_row)
                    self.assertEqual(len({r.req_pool_idx for r in [*owners, *scheduled.reqs]}), 4)
                    self.assertEqual(scheduler.req_to_token_pool.available_size(), 0)

    def test_non_speculative_waiter_runs_before_background_decode_finishes(self):
        for exo in (False, True):
            for low_first in (False, True):
                with self.subTest(exo=exo, low_first=low_first):
                    background = self._request("reflection", -100)
                    background.sampling_params.custom_params["qwen_exo_work_class"] = "reflection"
                    foreground = self._request("probe", 0)
                    scheduler = self._scheduler([foreground], low_first=low_first)
                    scheduler.spec_algorithm = SpeculativeAlgorithm.NONE
                    scheduler.server_args.enable_qwen_exo = exo
                    if exo:
                        scheduler._qwen_exo_prefill_fairness = BackgroundPrefillFairness()
                    running = self._batch([background], target_only=True)
                    scheduled, retained = self._prefill(scheduler, running, real_admission=True)
                    self.assertEqual(scheduled.reqs, [foreground])
                    self.assertEqual(retained.reqs, [background])
                    self.assertEqual(scheduler.waiting_queue, [])
                    self.assertEqual(background.retraction_count, 0)

    def test_full_physical_pool_foreground_reaches_background_preemption(self):
        for low_first in (False, True):
            with self.subTest(low_first=low_first):
                foreground = [
                    self._request(f"user-{index}", -1000 if low_first else 1000)
                    for index in range(2)
                ]
                background = [
                    self._request(f"background-{index}", -100)
                    for index in range(4)
                ]
                for index, req in enumerate(background):
                    req.sampling_params.custom_params["qwen_exo_work_class"] = (
                        "reflection" if index == 0 else "maintenance"
                    )
                incoming = self._request("probe", 0)
                scheduler = self._scheduler([incoming], low_first=low_first)
                scheduler.spec_algorithm = SpeculativeAlgorithm.NONE
                scheduler.server_args.enable_qwen_exo = True
                scheduler.max_running_requests = 6
                scheduler.get_num_allocatable_reqs = (
                    lambda running_bs, *, num_held_rows=0: 6 - running_bs
                )
                scheduler._qwen_exo_prefill_fairness = BackgroundPrefillFairness()
                running = self._batch(foreground + background, target_only=True)
                scheduled, retained = self._prefill(
                    scheduler, running, real_admission=True
                )
                self.assertEqual(scheduled.reqs, [incoming])
                self.assertTrue(all(req in retained.reqs for req in foreground))
                self.assertEqual(len(retained.reqs) + len(scheduled.reqs), 6)
                retracted = [req for req in background if req.retraction_count == 1]
                self.assertEqual(len(retracted), 1)
                self.assertEqual(scheduler.waiting_queue, retracted)
                self.assertEqual(retracted[0].output_ids, [4, 5])

    def test_foreground_priority_does_not_bypass_kv_admission(self):
        background = self._request("reflection", -100)
        foreground = self._request("probe", 0)
        scheduler = self._scheduler([foreground])
        scheduler.spec_algorithm = SpeculativeAlgorithm.NONE
        scheduler._qwen_exo_prefill_fairness = BackgroundPrefillFairness()
        # Running reserve=98; incoming input=3, remaining output=98, page slack=1.
        scheduler.token_to_kv_pool_allocator.available_size.return_value = 200
        running = self._batch([background], target_only=True)
        scheduled, retained = self._prefill(scheduler, running, real_admission=True)
        self.assertIsNone(scheduled)
        self.assertEqual(retained.reqs, [background])
        self.assertEqual(scheduler.waiting_queue, [foreground])

    def test_yielded_chunk_preserves_owner_and_resumes_after_scoring_prefill(self):
        background = self._request("reflection", -100)
        background.sampling_params.custom_params["qwen_exo_work_class"] = "reflection"
        background.full_untruncated_fill_ids = list(range(8192))
        background.prefix_indices = list(range(2048))
        background.set_extend_range(0, 2048)
        background.req_pool_idx = 7
        foreground = self._request("scoring", 0)
        foreground.sampling_params.max_new_tokens = 0
        scheduler = self._scheduler([foreground])
        scheduler.spec_algorithm = SpeculativeAlgorithm.NONE
        scheduler.chunked_prefill_size = 2048
        scheduler.chunked_req = background
        scheduler._qwen_exo_prefill_fairness = BackgroundPrefillFairness()
        scheduler._qwen_exo_prefill_fairness.record_prefill([background])
        empty = self._batch([], target_only=True)
        scheduled, _ = self._prefill(scheduler, empty, real_admission=True)
        self.assertEqual(scheduled.reqs, [foreground])
        self.assertIsNone(scheduled.chunked_req)
        self.assertTrue(scheduled.contains_last_prefill_chunk)
        self.assertIs(scheduler.chunked_req, background)
        self.assertEqual(background.inflight_middle_chunks, 0)
        self.assertEqual(background.req_pool_idx, 7)
        self.assertEqual(background.extend_range.end, 2048)
        resumed, _ = self._prefill(scheduler, empty, real_admission=True)
        self.assertEqual(resumed.reqs, [background])
        self.assertIs(resumed.chunked_req, background)
        self.assertEqual(background.inflight_middle_chunks, 1)
        self.assertEqual(background.extend_range.end, 4096)
        self.assertEqual(background.req_pool_idx, 7)

    def test_long_foreground_keeps_single_chunk_owner_and_background_progress(self):
        background = self._request("reflection", -100)
        background.sampling_params.custom_params["qwen_exo_work_class"] = "reflection"
        background.full_untruncated_fill_ids = list(range(8192))
        background.prefix_indices = list(range(2048))
        background.set_extend_range(0, 2048)
        foreground = self._request("long-foreground", 0)
        foreground.full_untruncated_fill_ids = list(range(4096))
        scheduler = self._scheduler([foreground])
        scheduler.spec_algorithm = SpeculativeAlgorithm.NONE
        scheduler.chunked_prefill_size = 2048
        scheduler.chunked_req = background
        scheduler._qwen_exo_prefill_fairness = BackgroundPrefillFairness()
        scheduler._qwen_exo_prefill_fairness.record_prefill([background])
        scheduled, _ = self._prefill(scheduler, self._batch([], target_only=True))
        self.assertEqual(scheduled.reqs, [background])
        self.assertIs(scheduler.chunked_req, background)
        self.assertEqual(background.extend_range.end, 4096)
        self.assertEqual(scheduler.waiting_queue, [foreground])

    def test_background_chunk_yields_to_foreground_decode_without_new_prefill(self):
        background = self._request("reflection", -100)
        background.sampling_params.custom_params["qwen_exo_work_class"] = "reflection"
        background.full_untruncated_fill_ids = list(range(8192))
        background.prefix_indices = list(range(2048))
        background.set_extend_range(0, 2048)
        foreground = self._request("user", 0)
        scheduler = self._scheduler([])
        scheduler.spec_algorithm = SpeculativeAlgorithm.NONE
        scheduler.chunked_prefill_size = 2048
        scheduler.chunked_req = background
        scheduler._qwen_exo_prefill_fairness = BackgroundPrefillFairness()
        scheduler._qwen_exo_prefill_fairness.record_prefill([background])
        running = self._batch([foreground], target_only=True)
        scheduled, retained = self._prefill(scheduler, running)
        self.assertIsNone(scheduled)
        self.assertEqual(retained.reqs, [foreground])
        self.assertIs(scheduler.chunked_req, background)
        self.assertEqual(background.extend_range.end, 2048)
        self.assertEqual(background.inflight_middle_chunks, 0)

    def _qwen4_chunk_decode_scheduler(self, *, decoding=True, decode_turns=1):
        cold = self._request("cold-prefill", 0)
        cold.full_untruncated_fill_ids = list(range(8192))
        cold.prefix_indices = torch.arange(7 * 8192, 7 * 8192 + 2048)
        cold.set_extend_range(0, 2048)
        cold.req_pool_idx = 7
        cold.mamba_pool_idx = torch.tensor(5)
        decoder = self._request("streaming", 0)
        scheduler = self._scheduler([])
        scheduler.spec_algorithm = SpeculativeAlgorithm.EAGLE
        scheduler.server_args.enable_qwen_exo = True
        scheduler.model_config.hf_text_config = SimpleNamespace(model_type="qwen4_exp_text")
        scheduler.chunked_prefill_size = 2048
        scheduler.chunked_req = cold
        scheduler._qwen_exo_prefill_fairness = BackgroundPrefillFairness(
            decode_turns=decode_turns
        )
        scheduler._qwen_exo_prefill_fairness.record_prefill([cold])
        scheduler.running_batch = self._batch([decoder] if decoding else [])
        scheduler.running_batch.spec_algorithm = SpeculativeAlgorithm.EAGLE
        scheduler.last_batch = None
        scheduler.require_mlp_sync = False
        scheduler.enable_fpm = False
        scheduler._pending_chunked_abort_req = None
        scheduler._abort_on_waiting_timeout = MagicMock()
        scheduler._abort_on_running_timeout = MagicMock()
        scheduler.dp_attn_adapter = SimpleNamespace(
            maybe_prepare_mlp_sync_batch=lambda batch, **kwargs: batch
        )
        scheduler.ngram_embedding_manager = SimpleNamespace(
            prepare_for_forward=lambda batch, **kwargs: batch
        )
        # Real cache stashing must advance only the owned chunk's KV prefix.
        pool = SimpleNamespace(req_to_token=torch.arange(8 * 8192).reshape(8, 8192))
        cache = ChunkCache(SimpleNamespace(
            req_to_token_pool=pool, token_to_kv_pool_allocator=None, page_size=1
        ))
        scheduler.stash_chunked_request = lambda req: cache.cache_unfinished_req(req, chunked=True)

        def prefill(running, *, last_batch=None):
            batch, retained = self._prefill(scheduler, running, last_batch, real_admission=True)
            return NextBatchPlan(batch_to_run=batch, running_batch=retained)

        scheduler.get_new_batch_prefill = MagicMock(side_effect=prefill)

        def decode(batch):
            batch.filter_batch()
            batch.forward_mode = ForwardMode.DECODE
            return batch

        scheduler.update_running_batch = decode
        return scheduler, cold, decoder, pool

    @staticmethod
    def _select_chunk_turn(scheduler):
        with patch("sglang.srt.managers.scheduler.set_schedule_time_batch"):
            plan = scheduler.get_next_batch_to_run(scheduler.running_batch, scheduler.last_batch)
        scheduler.running_batch = plan.running_batch
        scheduler.last_batch = plan.batch_to_run
        return plan.batch_to_run

    def test_mtp_long_prefill_and_decode_alternate_without_releasing_chunk_owner(self):
        scheduler, cold, decoder, pool = self._qwen4_chunk_decode_scheduler()
        sequence = []
        for _ in range(6):
            batch = self._select_chunk_turn(scheduler)
            if batch.forward_mode.is_extend():
                sequence.append("prefill")
                cold.inflight_middle_chunks -= int(batch.chunked_req is cold)
            else:
                sequence.append("decode")
                self.assertNotIn(cold, batch.reqs)
                self.assertEqual(batch.reqs, [decoder])
                self.assertEqual(cold.inflight_middle_chunks, 0)
            self.assertEqual(cold.req_pool_idx, 7)
            self.assertEqual(cold.mamba_pool_idx.item(), 5)
            self.assertEqual(cold.output_ids, [4, 5])
            self.assertEqual(cold.retraction_count, 0)
            self.assertTrue(torch.equal(
                cold.prefix_indices, pool.req_to_token[7, :len(cold.prefix_indices)]
            ))
        self.assertEqual(sequence, ["decode", "prefill"] * 3)
        self.assertEqual(cold.extend_range.end, 8192)
        self.assertIsNone(scheduler.chunked_req)
        # Final prefill joins decode once; it is not left parked or lost.
        merged = self._select_chunk_turn(scheduler)
        self.assertEqual(merged.reqs, [decoder, cold])
        self.assertEqual(scheduler.get_new_batch_prefill.call_count, 3)

    def test_mtp_decode_quantum_retains_chunk_and_resumes_prefill_at_bound(self):
        scheduler, cold, decoder, _ = self._qwen4_chunk_decode_scheduler(decode_turns=3)
        decode = scheduler.update_running_batch
        scheduler.update_running_batch = MagicMock(side_effect=decode)
        sequence = []
        for _ in range(8):
            batch = self._select_chunk_turn(scheduler)
            if batch.forward_mode.is_extend():
                sequence.append("prefill")
                self.assertEqual(batch.reqs, [cold])
                cold.inflight_middle_chunks -= int(batch.chunked_req is cold)
            else:
                sequence.append("decode")
                self.assertEqual(batch.reqs, [decoder])
                self.assertNotIn(cold, batch.reqs)
            self.assertEqual(cold.req_pool_idx, 7)
            self.assertEqual(cold.mamba_pool_idx.item(), 5)
            self.assertEqual(cold.retraction_count, 0)
        self.assertEqual(sequence, ["decode", "decode", "decode", "prefill"] * 2)
        self.assertEqual(scheduler.update_running_batch.call_count, 6)
        self.assertEqual(cold.extend_range.end, 6144)

    def test_decoder_finishing_mid_quantum_immediately_resumes_cold_chunk(self):
        scheduler, cold, decoder, _ = self._qwen4_chunk_decode_scheduler(decode_turns=3)
        self.assertEqual(self._select_chunk_turn(scheduler).reqs, [decoder])
        decoder.finished.return_value = True
        batch = self._select_chunk_turn(scheduler)
        self.assertTrue(batch.forward_mode.is_extend())
        self.assertEqual(batch.reqs, [cold])
        self.assertEqual(cold.extend_range.end, 4096)

    def test_no_decoder_runs_cold_chunks_without_artificial_idle_turns(self):
        scheduler, cold, _, _ = self._qwen4_chunk_decode_scheduler(decoding=False)
        for expected_end in (4096, 6144, 8192):
            batch = self._select_chunk_turn(scheduler)
            self.assertTrue(batch.forward_mode.is_extend())
            self.assertEqual(batch.reqs, [cold])
            self.assertEqual(cold.extend_range.end, expected_end)
            cold.inflight_middle_chunks -= int(batch.chunked_req is cold)

    def test_finished_decoder_does_not_delay_owned_prefill(self):
        scheduler, cold, decoder, _ = self._qwen4_chunk_decode_scheduler()
        decoder.finished.return_value = True
        batch = self._select_chunk_turn(scheduler)
        self.assertEqual(batch.reqs, [cold])
        self.assertEqual(cold.extend_range.end, 4096)

    def test_no_prefill_backlog_decodes_every_turn(self):
        scheduler, _, decoder, _ = self._qwen4_chunk_decode_scheduler()
        scheduler.chunked_req = None
        for _ in range(4):
            batch = self._select_chunk_turn(scheduler)
            self.assertEqual(batch.reqs, [decoder])
            self.assertTrue(batch.forward_mode.is_decode())

    def test_unsupported_scheduler_paths_keep_prefill_priority(self):
        overrides = (
            ("spec_algorithm", SpeculativeAlgorithm.DFLASH),
            ("server_args.enable_qwen_exo", False),
            ("model_config.hf_text_config.model_type", "qwen3_5_text"),
            ("ps.pp_size", 2),
            ("disaggregation_mode", DisaggregationMode.PREFILL),
            ("require_mlp_sync", True),
            ("enable_hisparse", True),
            ("is_hybrid_swa", True),
            ("dllm_config", object()),
            ("chunked_prefill_size", None),
            ("is_mixed_chunk", True),
        )
        for path, value in overrides:
            with self.subTest(path=path):
                scheduler, _, _, _ = self._qwen4_chunk_decode_scheduler()
                owner = scheduler
                *parents, field = path.split(".")
                for parent in parents:
                    owner = getattr(owner, parent)
                setattr(owner, field, value)
                self.assertFalse(scheduler._should_yield_qwen4_prefill_to_decode(scheduler.running_batch))

    def test_opposite_lane_waiter_does_not_enter_parked_chunk_decode_turn(self):
        scheduler, cold, decoder, _ = self._qwen4_chunk_decode_scheduler()
        export = self._request("export", 0, target_only=True)
        scheduler.waiting_queue = [export]
        decoded = self._select_chunk_turn(scheduler)
        self.assertEqual(decoded.reqs, [decoder])
        scheduler.get_new_batch_prefill.assert_not_called()
        resumed = self._select_chunk_turn(scheduler)
        self.assertEqual(resumed.reqs, [cold])
        self.assertEqual(resumed.spec_algorithm, SpeculativeAlgorithm.EAGLE)
        self.assertEqual(scheduler.waiting_queue, [export])
        self.assertEqual(cold.retraction_count, 0)

    def test_cancel_parked_chunk_releases_once_without_launching_another_chunk(self):
        scheduler, cold, decoder, _ = self._qwen4_chunk_decode_scheduler()
        self._select_chunk_turn(scheduler)
        scheduler._pending_chunked_abort_req = cold
        cold.time_stats.trace_ctx = MagicMock()
        scheduler.ipc_channels = MagicMock()

        def abort(req, reason):
            req.finished.return_value = True

        def release(req, cache, *, is_insert):
            self.assertFalse(is_insert)
            self.assertEqual(req.inflight_middle_chunks, 0)
            req.req_pool_idx = None
            req.mamba_pool_idx = None

        with (
            patch("sglang.srt.managers.scheduler.prepare_abort", side_effect=abort),
            patch("sglang.srt.managers.scheduler.release_kv_cache", side_effect=release) as free,
        ):
            for _ in range(2):
                batch = self._select_chunk_turn(scheduler)
                self.assertEqual(batch.reqs, [decoder])
            free.assert_called_once()
        self.assertIsNone(scheduler.chunked_req)
        self.assertIsNone(scheduler._pending_chunked_abort_req)
        self.assertIsNone(cold.req_pool_idx)
        scheduler._release_qwen_exo_hybrid_state.assert_called_once_with(cold)

    def test_overlap_decode_turn_commits_pending_chunk_result_once(self):
        scheduler, cold, decoder, _ = self._qwen4_chunk_decode_scheduler()
        scheduler.enable_overlap = True
        scheduler.gracefully_exit = False
        scheduler._engine_paused = False
        scheduler.enable_unified_memory = False
        scheduler.is_generation = False
        scheduler.is_disable_overlap_for_batch = lambda *args, **kwargs: False
        pending = self._batch([cold])
        pending.spec_algorithm = SpeculativeAlgorithm.EAGLE
        pending.forward_mode = ForwardMode.EXTEND
        pending.chunked_req = cold
        scheduler.last_batch = pending
        cold.inflight_middle_chunks = 1

        def receive():
            scheduler.result_queue.append((pending, object()))
            return []

        scheduler.request_receiver = SimpleNamespace(recv_requests=receive)
        scheduler.process_input_requests = MagicMock()

        def commit(batch, result):
            self.assertIs(batch, pending)
            cold.inflight_middle_chunks -= 1

        scheduler.process_batch_result = MagicMock(side_effect=commit)

        def run(batch):
            self.assertEqual(batch.reqs, [decoder])
            self.assertIs(scheduler.chunked_req, cold)
            self.assertEqual(cold.req_pool_idx, 7)
            self.assertEqual(cold.inflight_middle_chunks, 1)
            scheduler.gracefully_exit = True
            return object()

        scheduler.run_batch = run
        scheduler._apply_war_barrier = MagicMock()
        with (
            patch("sglang.srt.managers.scheduler.envs") as envs,
            patch("sglang.srt.managers.scheduler.set_schedule_time_batch"),
        ):
            envs.SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_BUSY.get.return_value = False
            inspect.unwrap(Scheduler.event_loop_overlap)(scheduler)
        scheduler.process_batch_result.assert_called_once()
        self.assertEqual(cold.inflight_middle_chunks, 0)
        self.assertEqual(cold.req_pool_idx, 7)
        self.assertEqual(len(scheduler.result_queue), 1)
        scheduler.get_new_batch_prefill.assert_not_called()

    def _resident_lane_scheduler(self, score_tokens=3):
        scheduler, _, generator, pool = self._qwen4_chunk_decode_scheduler()
        scheduler.ps = SimpleNamespace(
            tp_size=1, pp_size=1, dp_size=1, attn_cp_size=1, attn_dp_size=1,
            dcp_size=1,
        )
        scheduler.enable_pdmux = False
        scheduler.decode_offload_manager = None
        scheduler.chunked_req = None
        scheduler._qwen_exo_active_target_only = False
        scheduler._qwen_exo_prefill_fairness.record_decode()
        generator.to_finish = None
        generator.req_pool_idx = 7
        generator.mamba_pool_idx = torch.tensor(5)
        generator.kv = SimpleNamespace(kv_allocated_len=2048)
        scheduler.running_batch.new_tokens_required_next_decode.return_value = 8
        score = self._request("score", 0, target_only=True)
        score.to_finish = None
        score.req_pool_idx = 6
        score.sampling_params.max_new_tokens = 0
        score.output_ids = []
        score.full_untruncated_fill_ids = list(range(score_tokens))
        score.set_extend_range(0, score_tokens)
        scheduler.waiting_queue = [score]
        return scheduler, generator, score, pool

    def test_equal_priority_score_completes_before_resident_generator_finishes(self):
        scheduler, generator, score, pool = self._resident_lane_scheduler()
        original = scheduler.running_batch
        spec_inputs = original.spec_info
        kv_owner = generator.kv
        gdn_owner = generator.mamba_pool_idx
        ple_row = pool.req_to_token[7].clone()
        scoring = self._select_chunk_turn(scheduler)
        self.assertEqual(scoring.reqs, [score])
        self.assertEqual(scoring.spec_algorithm, SpeculativeAlgorithm.NONE)
        self.assertIs(scheduler._qwen_exo_parked_batch, original)
        self.assertFalse(generator.finished())
        score.finished.return_value = True
        resumed = self._select_chunk_turn(scheduler)
        self.assertIs(resumed, original)
        self.assertIs(resumed.spec_info, spec_inputs)
        self.assertIs(generator.kv, kv_owner)
        self.assertIs(generator.mamba_pool_idx, gdn_owner)
        self.assertEqual(generator.req_pool_idx, 7)
        self.assertTrue(torch.equal(pool.req_to_token[7], ple_row))
        self.assertEqual(generator.retraction_count, 0)
        self.assertEqual(scheduler.waiting_queue, [])

    def test_finished_outgoing_generator_is_removed_before_decode_reservation(self):
        scheduler, generator, score, _ = self._resident_lane_scheduler()
        original = scheduler.running_batch
        completed = self._request("completed", 0)
        completed.finished.return_value = True
        completed.kv = None
        original.reqs.append(completed)

        def reserve_next_decode():
            return sum(req.kv.kv_allocated_len for req in original.reqs) // 256

        original.new_tokens_required_next_decode.side_effect = reserve_next_decode
        selected = self._select_chunk_turn(scheduler)
        self.assertEqual(selected.reqs, [score])
        self.assertEqual(original.reqs, [generator])
        self.assertIs(scheduler._qwen_exo_parked_batch, original)
        self.assertEqual(generator.retraction_count, 0)

    def test_disabled_priority_scheduling_rotates_requests_with_absent_priorities(self):
        for score_priority in (None, 0):
            with self.subTest(score_priority=score_priority):
                scheduler, generator, score, _ = self._resident_lane_scheduler()
                scheduler.enable_priority_scheduling = False
                generator.priority = None
                score.priority = score_priority
                selected = self._select_chunk_turn(scheduler)
                self.assertEqual(selected.reqs, [score])
                score.finished.return_value = True
                self.assertEqual(self._select_chunk_turn(scheduler).reqs, [generator])
                self.assertEqual(generator.retraction_count, 0)

    def test_long_score_chunks_rotate_with_generation_at_every_safe_boundary(self):
        scheduler, generator, score, _ = self._resident_lane_scheduler(8192)
        turns = []
        for _ in range(8):
            batch = self._select_chunk_turn(scheduler)
            if batch.spec_algorithm.is_none():
                turns.append("score")
                self.assertEqual(batch.reqs, [score])
                score.inflight_middle_chunks -= int(batch.chunked_req is score)
                if scheduler.chunked_req is None:
                    score.finished.return_value = True
            else:
                turns.append("generate")
                self.assertEqual(batch.reqs, [generator])
                generator.output_ids.append(6)
            self.assertEqual(generator.req_pool_idx, 7)
            self.assertEqual(generator.mamba_pool_idx.item(), 5)
            self.assertEqual(generator.retraction_count, 0)
        self.assertEqual(turns, ["score", "generate"] * 4)
        self.assertEqual(score.extend_range.end, 8192)
        self.assertEqual(generator.output_ids, [4, 5, 6, 6, 6, 6])
        self.assertTrue(score.finished())
        self.assertFalse(generator.finished())

    def test_failed_score_admission_resumes_generation_without_retraction_or_idle(self):
        for resource in ("tokens", "slots"):
            with self.subTest(resource=resource):
                scheduler, generator, score, _ = self._resident_lane_scheduler()
                original = scheduler.running_batch
                if resource == "tokens":
                    scheduler.token_to_kv_pool_allocator.available_size.return_value = 1
                else:
                    scheduler.get_num_allocatable_reqs = (
                        lambda running_bs, *, num_held_rows=0: 0
                    )
                selected = self._select_chunk_turn(scheduler)
                self.assertIs(selected, original)
                self.assertEqual(selected.reqs, [generator])
                self.assertEqual(scheduler.waiting_queue, [score])
                self.assertEqual(generator.retraction_count, 0)

    def test_resident_handoff_waits_for_overlap_commit_and_stream_fence(self):
        scheduler, generator, score, _ = self._resident_lane_scheduler()
        scheduler.enable_overlap = True
        outgoing = scheduler.running_batch
        scheduler.last_batch = outgoing
        scheduler.result_queue.append((outgoing, object()))
        retained, switched = scheduler._rotate_qwen4_resident_lane(outgoing, outgoing)
        self.assertIs(retained, outgoing)
        self.assertFalse(switched)
        self.assertIsNone(scheduler._qwen_exo_parked_batch)
        scheduler.result_queue.clear()
        scheduler.gracefully_exit = False
        scheduler._engine_paused = False
        scheduler.enable_unified_memory = False
        scheduler.is_generation = False
        scheduler.is_disable_overlap_for_batch = lambda *args, **kwargs: False
        order = []
        scheduler.forward_stream = SimpleNamespace(
            synchronize=lambda: order.append("fence")
        )

        def receive():
            scheduler.result_queue.append((outgoing, object()))
            return []

        def commit(batch, result):
            self.assertIs(batch, outgoing)
            self.assertIsNone(scheduler._qwen_exo_parked_batch)
            generator.output_ids.append(6)
            order.append("commit")

        def run(batch):
            self.assertEqual(order, ["commit", "fence"])
            self.assertEqual(batch.reqs, [score])
            self.assertTrue(batch.spec_algorithm.is_none())
            self.assertEqual(generator.output_ids, [4, 5, 6])
            self.assertEqual(generator.retraction_count, 0)
            scheduler.gracefully_exit = True
            return object()

        scheduler.request_receiver = SimpleNamespace(recv_requests=receive)
        scheduler.process_input_requests = MagicMock()
        scheduler.process_batch_result = MagicMock(side_effect=commit)
        scheduler.run_batch = run
        scheduler._apply_war_barrier = MagicMock()
        with (
            patch("sglang.srt.managers.scheduler.envs") as envs,
            patch("sglang.srt.managers.scheduler.set_schedule_time_batch"),
        ):
            envs.SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_BUSY.get.return_value = False
            inspect.unwrap(Scheduler.event_loop_overlap)(scheduler)
        scheduler.process_batch_result.assert_called_once()
        self.assertEqual(len(scheduler.result_queue), 1)
        self.assertIs(scheduler._qwen_exo_parked_batch, outgoing)

    def test_cancel_parked_scoring_chunk_releases_once_then_generation_continues(self):
        scheduler, generator, score, _ = self._resident_lane_scheduler(8192)
        scored = self._select_chunk_turn(scheduler)
        score.inflight_middle_chunks -= 1
        self._select_chunk_turn(scheduler)
        scheduler._pending_chunked_abort_req = score
        score.time_stats.trace_ctx = MagicMock()
        scheduler.ipc_channels = MagicMock()

        def finish(req, reason):
            req.finished.return_value = True

        def release(req, cache, *, is_insert):
            self.assertIs(req, score)
            self.assertFalse(is_insert)
            self.assertEqual(req.inflight_middle_chunks, 0)
            req.req_pool_idx = None

        with (
            patch("sglang.srt.managers.scheduler.prepare_abort", side_effect=finish),
            patch("sglang.srt.managers.scheduler.release_kv_cache", side_effect=release) as free,
        ):
            for _ in range(2):
                self.assertEqual(self._select_chunk_turn(scheduler).reqs, [generator])
            free.assert_called_once()
        self.assertIsNone(scheduler.chunked_req)
        self.assertIsNone(score.req_pool_idx)
        self.assertEqual(generator.retraction_count, 0)

    def test_resident_generator_abort_and_timeout_are_not_hidden_by_active_score(self):
        for action in ("abort", "timeout"):
            with self.subTest(action=action):
                scheduler, generator, score, _ = self._resident_lane_scheduler()
                self._select_chunk_turn(scheduler)
                generator.time_stats.forward_entry_time = 1
                if action == "abort":
                    scheduler.abort_request(AbortReq(rid=generator.rid))
                else:
                    with (
                        patch("sglang.srt.managers.scheduler.envs") as envs,
                        patch("sglang.srt.managers.scheduler.time.perf_counter", return_value=100),
                    ):
                        envs.SGLANG_REQ_RUNNING_TIMEOUT.get.return_value = 10
                        Scheduler._abort_on_running_timeout(scheduler, scheduler.running_batch)
                self.assertIsNotNone(generator.to_finish)
                self.assertTrue(scheduler._qwen_exo_lane_rotation_due(
                    scheduler.running_batch, scheduler.last_batch
                ))
                self.assertEqual(generator.retraction_count, 0)
                self.assertEqual(generator.req_pool_idx, 7)

    def test_parked_owners_remain_visible_to_slots_load_and_pool_accounting(self):
        scheduler, generator, score, _ = self._resident_lane_scheduler(8192)
        self._select_chunk_turn(scheduler)
        score.kv = SimpleNamespace(kv_allocated_len=2048)
        generator.cache_protected_len = score.cache_protected_len = 0
        # Outgoing overlap metadata and the resident batch reference the same
        # generator. Neither physical owners nor their KV may be double-counted.
        scheduler.last_batch = scheduler._qwen_exo_parked_batch
        scheduler.running_batch.reqs = [score]
        self.assertCountEqual(scheduler._qwen_exo_load_reqs(), [generator, score])
        scheduler.req_to_token_pool.available_size = lambda: 1
        with patch("sglang.srt.managers.scheduler.get_server_args", return_value=SimpleNamespace(
            pp_max_micro_batch_size=2
        )):
            self.assertEqual(Scheduler.get_num_allocatable_reqs(scheduler, 1), 0)
        observer = SimpleNamespace(
            get_last_batch=lambda: scheduler.last_batch,
            get_running_batch=lambda: scheduler.running_batch,
            get_parked_requests=scheduler._qwen_exo_parked_reqs,
        )
        self.assertEqual(SchedulerPoolStatsObserver.active_pool_idxs(observer), {6, 7})
        checker = SimpleNamespace(
            get_last_batch=observer.get_last_batch,
            get_running_batch=observer.get_running_batch,
            get_parked_requests=scheduler._qwen_exo_parked_reqs,
            page_size=1,
            is_hybrid_swa=False,
        )
        self.assertEqual(SchedulerInvariantChecker._get_total_uncached_sizes(checker), (4096, 0))

    def test_resident_chunk_cannot_borrow_parked_decode_headroom(self):
        scheduler, generator, score, _ = self._resident_lane_scheduler(8192)
        scheduler.token_to_kv_pool_allocator.available_size.return_value = 106
        scheduler.tree_cache.supports_mamba.return_value = False
        policy = "sglang.srt.managers.schedule_policy."
        with (
            patch(policy + "is_dsa_prefill_cp_in_seq_split", return_value=False),
            patch(policy + "is_prefill_context_parallel_enabled", return_value=False),
        ):
            adder = PrefillAdder(
                1, scheduler.tree_cache, scheduler.token_to_kv_pool_allocator,
                self._batch([], target_only=True), 0.1, 2048, 2048,
                parked_requests=[generator], parked_decode_tokens=8,
            )
        score.prefix_indices = list(range(2048))
        score.set_extend_range(0, 2048)
        self.assertLess(adder.rem_total_tokens, 0)
        self.assertIs(adder.add_chunked_req(score), score)
        self.assertEqual(adder.can_run_list, [])
        self.assertEqual(score.extend_range.end, 2048)
        self.assertEqual(generator.retraction_count, 0)

    def test_unsupported_native_modes_keep_original_lane_admission(self):
        for field, value in (
            ("ps.tp_size", 2), ("ps.pp_size", 2), ("ps.dp_size", 2),
            ("ps.attn_cp_size", 2), ("ps.dcp_size", 2),
            ("enable_hisparse", True), ("is_hybrid_swa", True),
            ("require_mlp_sync", True), ("is_mixed_chunk", True),
            ("enable_lora", True), ("disaggregation_mode", DisaggregationMode.PREFILL),
        ):
            with self.subTest(field=field):
                scheduler, generator, score, _ = self._resident_lane_scheduler()
                owner = scheduler
                *parents, name = field.split(".")
                for part in parents:
                    owner = getattr(owner, part)
                setattr(owner, name, value)
                outgoing = scheduler.running_batch
                selected, switched = scheduler._rotate_qwen4_resident_lane(outgoing, None)
                self.assertFalse(switched)
                self.assertIs(selected, outgoing)
                self.assertIsNone(scheduler._qwen_exo_parked_batch)

    def test_higher_priority_switch_retracts_entire_lane_and_resumes(self):
        for low_first in (False, True):
            with self.subTest(low_first=low_first):
                sign = -1 if low_first else 1
                victims = [self._request(str(i), -25 * sign) for i in range(2)]
                foreground = self._request("probe", -10 * sign, target_only=True)
                scheduler = self._scheduler([foreground], low_first=low_first)
                running = self._batch(victims)
                last = self._batch(victims)
                new_batch, running = self._prefill(scheduler, running, last)
                self.assertEqual(new_batch.reqs, [foreground])
                self.assertTrue(new_batch.spec_algorithm.is_none())
                self.assertEqual(running.reqs, [])
                self.assertEqual(last.reqs, [])
                self.assertCountEqual(scheduler.waiting_queue, victims)
                for victim in victims:
                    self.assertEqual(victim.retraction_count, 1)
                    self.assertEqual(victim.output_ids, [4, 5])
                foreground.finished.return_value = True
                new_batch.filter_batch()
                resumed, _ = self._prefill(scheduler, new_batch)
                self.assertCountEqual(resumed.reqs, victims)
                self.assertTrue(resumed.spec_algorithm.is_dflash())
                self.assertEqual(scheduler.waiting_queue, [])

    def test_threshold_equal_lower_and_protected_peer_do_not_yield(self):
        for low_first in (False, True):
            for priorities in ((-25,), (-20,), (-15,), (-10,), (-30, -15)):
                with self.subTest(low_first=low_first, priorities=priorities):
                    sign = -1 if low_first else 1
                    victims = [
                        self._request(str(i), p * sign)
                        for i, p in enumerate(priorities)
                    ]
                    foreground = self._request("probe", -15 * sign, target_only=True)
                    scheduler = self._scheduler([foreground], low_first=low_first)
                    running = self._batch(victims)
                    new_batch, _ = self._prefill(scheduler, running)
                    self.assertIsNone(new_batch)
                    self.assertEqual(running.reqs, victims)
                    self.assertEqual(scheduler.waiting_queue, [foreground])
                    self.assertTrue(all(req.retraction_count == 0 for req in victims))

    def test_rejected_foreground_does_not_orphan_retracted_background(self):
        victim = self._request("reflection", -25)
        foreground = self._request("probe", -10, target_only=True)
        foreground.mamba_pool_idx = None
        scheduler = self._scheduler([foreground])
        new_batch, running = self._prefill(
            scheduler, self._batch([victim]), reject=True
        )
        self.assertIsNone(new_batch)
        self.assertEqual(running.reqs, [])
        self.assertEqual(scheduler.waiting_queue, [foreground, victim])
        self.assertEqual(victim.output_ids, [4, 5])

    def test_uncommitted_overlap_result_blocks_release(self):
        victim = self._request("reflection", -25)
        foreground = self._request("probe", -10, target_only=True)
        scheduler = self._scheduler([foreground])
        scheduler.enable_overlap = True
        running = self._batch([victim])
        scheduler.result_queue.append((running, object()))
        new_batch, _ = self._prefill(scheduler, running, running)
        self.assertIsNone(new_batch)
        self.assertEqual(victim.retraction_count, 0)
        self.assertEqual(running.reqs, [victim])

    def test_chunked_or_preselected_work_keeps_ownership(self):
        victim = self._request("reflection", -25)
        foreground = self._request("probe", -10, target_only=True)
        scheduler = self._scheduler([foreground])
        running = self._batch([victim])
        for owner in ("chunked", "last_chunk", "preselected"):
            with self.subTest(owner=owner):
                scheduler.chunked_req = victim if owner == "chunked" else None
                running.chunked_req = victim if owner == "last_chunk" else None
                self.assertFalse(
                    scheduler._can_yield_dflash_mode(
                        foreground,
                        running,
                        running,
                        [victim] if owner == "preselected" else (),
                    )
                )

    def test_qwen4_mtp_native_exports_isolate_then_resume_speculation(self):
        for export_key in ("qwen_exo_native_bank_export", "qwen_exo_session_initial_gdn_export"):
            with self.subTest(export_key=export_key):
                export = self._request("export", 0, target_only=True)
                export.sampling_params.custom_params[export_key] = {"source_digest": "a" * 64}
                ordinary = self._request("generation", 0)
                scheduler = self._scheduler([ordinary, export])
                scheduler.spec_algorithm = SpeculativeAlgorithm.EAGLE
                scheduler.server_args.enable_qwen_exo = True
                scheduler.model_config.hf_text_config = SimpleNamespace(model_type="qwen4_exp_text")
                empty = self._batch([])
                empty.spec_algorithm = SpeculativeAlgorithm.EAGLE
                batch, retained = self._prefill(scheduler, empty)
                self.assertEqual(batch.reqs, [export])
                self.assertEqual(batch.spec_algorithm, SpeculativeAlgorithm.NONE)
                self.assertEqual(retained.reqs, [])
                self.assertEqual(scheduler.waiting_queue, [ordinary])
                export.finished.return_value = True
                batch.filter_batch()
                resumed, _ = self._prefill(scheduler, batch)
                self.assertEqual(resumed.reqs, [ordinary])
                self.assertEqual(resumed.spec_algorithm, SpeculativeAlgorithm.EAGLE)
                self.assertEqual(scheduler.waiting_queue, [])

    def test_target_only_lane_does_not_change_other_eagle_models(self):
        for model_type, exo in (("qwen3_5_text", True), ("qwen4_exp_text", False)):
            with self.subTest(model_type=model_type, exo=exo):
                export = self._request("export", 0, target_only=True)
                scheduler = self._scheduler([export])
                scheduler.spec_algorithm = SpeculativeAlgorithm.EAGLE
                scheduler.server_args.enable_qwen_exo = exo
                scheduler.model_config.hf_text_config = SimpleNamespace(model_type=model_type)
                batch, _ = self._prefill(scheduler, self._batch([]))
                self.assertEqual(batch.reqs, [export])
                self.assertEqual(batch.spec_algorithm, SpeculativeAlgorithm.EAGLE)

    def test_overlap_commits_output_before_retraction_and_processes_once(self):
        for algorithm in (SpeculativeAlgorithm.DFLASH, SpeculativeAlgorithm.EAGLE):
            with self.subTest(algorithm=algorithm):
                self._check_overlap_commits_before_retraction(algorithm)

    def _check_overlap_commits_before_retraction(self, algorithm):
        victim = self._request("reflection", -25)
        foreground = self._request("probe", -10, target_only=True)
        scheduler = self._scheduler([foreground])
        scheduler.spec_algorithm = algorithm
        if algorithm == SpeculativeAlgorithm.EAGLE:
            scheduler.server_args.enable_qwen_exo = True
            scheduler.model_config.hf_text_config = SimpleNamespace(model_type="qwen4_exp_text")
        scheduler.enable_overlap = True
        scheduler.running_batch = self._batch([victim])
        scheduler.running_batch.spec_algorithm = algorithm
        scheduler.last_batch = scheduler.running_batch
        scheduler.gracefully_exit = False
        scheduler._engine_paused = False
        scheduler.enable_unified_memory = False
        scheduler.is_generation = False
        scheduler.request_receiver = MagicMock()

        def receive():
            scheduler.result_queue.append((scheduler.last_batch, object()))
            return []

        scheduler.request_receiver.recv_requests.side_effect = receive
        scheduler.process_input_requests = MagicMock()

        def commit(batch, result):
            self.assertEqual(victim.retraction_count, 0)
            victim.output_ids.append(6)

        scheduler.process_batch_result = MagicMock(side_effect=commit)

        def select(*, running_batch, last_batch):
            new_batch, running = self._prefill(scheduler, running_batch, last_batch)
            self.assertEqual(victim.output_ids, [4, 5, 6])
            self.assertEqual(victim.retraction_count, 1)
            return SimpleNamespace(batch_to_run=new_batch, running_batch=running)

        scheduler.get_next_batch_to_run = select
        scheduler.is_disable_overlap_for_batch = lambda *args, **kwargs: False

        def run(batch):
            scheduler.gracefully_exit = True
            return object()

        scheduler.run_batch = run
        scheduler._apply_war_barrier = MagicMock()
        with patch("sglang.srt.managers.scheduler.envs") as envs:
            envs.SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_BUSY.get.return_value = False
            inspect.unwrap(Scheduler.event_loop_overlap)(scheduler)
        scheduler.process_batch_result.assert_called_once()
        self.assertEqual(len(scheduler.result_queue), 1)
        self.assertEqual(scheduler.waiting_queue, [victim])


if __name__ == "__main__":
    unittest.main()
