import inspect
import unittest
from collections import deque
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from qwen_exo_booster.scheduler_fairness import BackgroundPrefillFairness

from sglang.srt.managers.schedule_batch import Req
from sglang.srt.managers.schedule_policy import AddReqResult, PrefillAdder
from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.srt.managers.scheduler import Scheduler

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
        batch.is_prefill_only = False
        batch.return_logprob = False
        batch.input_embeds = None
        batch.is_empty.side_effect = lambda: not batch.reqs

        def filter_batch(*, keep_indices=None):
            batch.reqs = [
                req
                for i, req in enumerate(batch.reqs)
                if (keep_indices is None and not req.finished())
                or (keep_indices is not None and i in keep_indices)
            ]

        def release_req(index, remaining, server_args):
            Req.reset_for_retract(batch.reqs[index])

        batch.filter_batch.side_effect = filter_batch
        batch.release_req.side_effect = release_req
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
        scheduler.get_num_allocatable_reqs = lambda running_bs: 8 - running_bs
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
                scheduler.get_num_allocatable_reqs = lambda running_bs: 6 - running_bs
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

    def test_overlap_commits_output_before_retraction_and_processes_once(self):
        victim = self._request("reflection", -25)
        foreground = self._request("probe", -10, target_only=True)
        scheduler = self._scheduler([foreground])
        scheduler.enable_overlap = True
        scheduler.running_batch = self._batch([victim])
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
