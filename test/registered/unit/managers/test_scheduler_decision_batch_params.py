import inspect
import unittest
from collections import deque
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.disaggregation.decode import SchedulerDisaggregationDecodeMixin
from sglang.srt.disaggregation.prefill import SchedulerDisaggregationPrefillMixin
from sglang.srt.managers.schedule_batch import Req
from sglang.srt.managers.schedule_policy import AddReqResult, PrefillAdder
from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.srt.managers.scheduler import Scheduler

register_cpu_ci(est_time=3, suite="base-a-test-cpu")

FORBIDDEN_TOKENS = ("self.running_batch", "self.last_batch", "self.cur_batch")

DECISION_METHODS = (
    Scheduler.get_next_batch_to_run,
    Scheduler.get_new_batch_prefill,
    Scheduler._get_new_batch_prefill_raw,
    Scheduler._abort_on_running_timeout,
    Scheduler.is_disable_overlap_for_batch,
    SchedulerDisaggregationPrefillMixin.get_next_disagg_prefill_batch_to_run,
    SchedulerDisaggregationPrefillMixin.process_prefill_chunk,
    SchedulerDisaggregationDecodeMixin.get_new_prebuilt_batch,
    SchedulerDisaggregationDecodeMixin.get_next_disagg_decode_batch_to_run,
)


class TestDecisionMethodsHaveNoHiddenBatchChannel(unittest.TestCase):
    def test_decision_methods_take_batches_as_params_not_self(self):
        for method in DECISION_METHODS:
            source = inspect.getsource(inspect.unwrap(method))
            self.assertIn(f"def {method.__name__}", source)
            for token in FORBIDDEN_TOKENS:
                self.assertNotIn(token, source)


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
        )
        req.time_stats = SimpleNamespace(wait_queue_entry_time=0)
        req.full_untruncated_fill_ids = [1, 2, 3]
        req.prefix_indices = []
        req.output_ids = [4, 5]
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

    def _prefill(self, scheduler, running, last=None, *, reject=False):
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

        def add_one(req, **kwargs):
            if reject:
                return AddReqResult.NO_TOKEN
            adder.can_run_list.append(req)
            return AddReqResult.CONTINUE

        adder.add_one_req = add_one

        def init_batch(reqs, *args, **kwargs):
            return self._batch(reqs, target_only=args[-1].is_none())

        module = "sglang.srt.managers.scheduler."
        with (
            patch(module + "PrefillAdder", return_value=adder),
            patch(module + "ScheduleBatch.init_new", side_effect=init_batch),
            patch(module + "PrefillStats.from_adder"),
            patch(module + "set_time_batch"),
            patch(module + "TEST_RETRACT", False),
        ):
            return scheduler._get_new_batch_prefill_raw(None, running, last)

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
