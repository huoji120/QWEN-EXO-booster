import sys
from array import array
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from qwen_exo_booster.contracts import (
    ContractViolation,
    HybridLifecycleState,
    HybridStateNamespace,
)
from qwen_exo_booster.hybrid_state import HybridRequestPhase, HybridRuntimePolicy
from qwen_exo_booster.scheduler_admission import SchedulerAdmission

if sys.platform != "win32":
    import torch
    from sglang.srt.disaggregation.utils import DisaggregationMode
    from sglang.srt.managers.io_struct import AbortReq
    from sglang.srt.managers.scheduler import Scheduler
else:
    torch = None
    DisaggregationMode = None
    AbortReq = None
    Scheduler = None

requires_scheduler = pytest.mark.skipif(
    Scheduler is None, reason="SGLang scheduler tests require its Linux runtime"
)


def test_admission_reserves_waiting_capacity_and_releases_atomically():
    admission = SchedulerAdmission(page_size=64)
    estimate = admission.estimate(
        prompt_tokens=100, max_new_tokens=32, needs_mamba=True
    )

    first = admission.reserve(
        "request-1",
        estimate,
        available_kv_tokens=512,
        available_request_slots=2,
        available_mamba_slots=2,
    )
    second = admission.reserve(
        "request-2",
        estimate,
        available_kv_tokens=512,
        available_request_slots=2,
        available_mamba_slots=2,
    )

    assert first.admitted
    assert second.admitted
    assert estimate.kv_tokens == 224
    assert admission.reservation_count == 2
    assert admission.release("request-1") == estimate
    assert not admission.is_reserved("request-1")


def test_native_bank_admission_reserves_cached_and_active_mamba_slots():
    estimate = SchedulerAdmission(page_size=64).estimate(
        prompt_tokens=256,
        max_new_tokens=32,
        needs_mamba=True,
        additional_mamba_slots=1,
    )

    assert estimate.mamba_slots == 2


def test_admission_rejects_without_partial_commit():
    admission = SchedulerAdmission(page_size=64)
    estimate = admission.estimate(
        prompt_tokens=200, max_new_tokens=64, needs_mamba=True
    )

    decision = admission.reserve(
        "request-1",
        estimate,
        available_kv_tokens=128,
        available_request_slots=1,
        available_mamba_slots=1,
    )

    assert not decision.admitted
    assert decision.reason == "kv_capacity"
    assert admission.reservation_count == 0


def test_internal_queue_estimate_does_not_consume_user_request_slots():
    admission = SchedulerAdmission(page_size=64)
    estimate = admission.estimate(
        prompt_tokens=100,
        max_new_tokens=16,
        needs_mamba=True,
        request_slots=0,
    )

    decision = admission.reserve(
        "internal:parent-1",
        estimate,
        available_kv_tokens=512,
        available_request_slots=0,
        available_mamba_slots=1,
    )

    assert decision.admitted
    assert estimate.request_slots == 0


def test_peer_rank_rejection_prevents_local_commit():
    admission = SchedulerAdmission(page_size=64, consensus=lambda _local: False)
    estimate = admission.estimate(prompt_tokens=32, max_new_tokens=16, needs_mamba=True)

    decision = admission.reserve(
        "request-1",
        estimate,
        available_kv_tokens=512,
        available_request_slots=4,
        available_mamba_slots=4,
    )

    assert not decision.admitted
    assert decision.reason == "peer_rank_capacity"
    assert admission.reservation_count == 0


def test_workspace_rejection_is_atomic_and_fail_closed():
    admission = SchedulerAdmission(page_size=64)
    estimate = admission.estimate(
        prompt_tokens=32,
        max_new_tokens=16,
        needs_mamba=True,
        workspace_bytes=101,
    )

    decision = admission.reserve(
        "request-1",
        estimate,
        available_kv_tokens=512,
        available_request_slots=1,
        available_mamba_slots=1,
        available_workspace_bytes=100,
    )

    assert not decision.admitted
    assert decision.reason == "workspace_capacity"
    assert decision.available_workspace_bytes == 100
    assert admission.reservation_count == 0


def test_concurrent_reservations_share_fixed_chunk_workspace_peak():
    admission = SchedulerAdmission(page_size=64)
    first = admission.estimate(
        prompt_tokens=32,
        max_new_tokens=16,
        needs_mamba=False,
        workspace_bytes=100,
    )
    second = admission.estimate(
        prompt_tokens=32,
        max_new_tokens=16,
        needs_mamba=False,
        workspace_bytes=80,
    )

    for request_id, estimate in (("request-1", first), ("request-2", second)):
        decision = admission.reserve(
            request_id,
            estimate,
            available_kv_tokens=1024,
            available_request_slots=2,
            available_mamba_slots=None,
            available_workspace_bytes=100,
        )
        assert decision.admitted

    assert admission.reservation_count == 2
    assert admission.release("request-1") == first
    assert admission.release("request-2") == second
    assert admission.reservation_count == 0


class _Req:
    def __init__(self, rid="request-1"):
        self.rid = rid
        self.origin_input_ids = array("q", [1, 2, 3])
        self.full_untruncated_fill_ids = array("q", [1, 2, 3])
        self.output_ids = array("q")
        self.extra_key = None
        self.sampling_params = SimpleNamespace(
            max_new_tokens=8, custom_params={"qwen_exo_kind": "user"}
        )
        self.finished_reason = None
        self.req_pool_idx = None
        self.mamba_pool_idx = None
        self.mamba_ping_pong_track_buffer = None
        self.prefix_indices = torch.empty(0, dtype=torch.int64)
        self.extend_range = SimpleNamespace(start=0, end=3)
        self.time_stats = SimpleNamespace(trace_ctx=MagicMock())

    def finished(self):
        return self.finished_reason is not None


def _scheduler(*, kv_tokens=512, mode=None):
    if mode is None:
        mode = DisaggregationMode.NULL
    scheduler = Scheduler.__new__(Scheduler)
    scheduler.qwen_exo_hybrid_policy = HybridRuntimePolicy(
        tp_size=2,
        dtype="bfloat16",
        page_size=64,
        mamba_strategy="extra_buffer",
        mamba_state_dtype="bfloat16",
    )
    scheduler.qwen_exo_admission = SchedulerAdmission(page_size=64)
    scheduler.page_size = 64
    scheduler.device = "cpu"
    scheduler.model_config = SimpleNamespace(vocab_size=256)
    scheduler.max_running_requests = 4
    scheduler.running_batch = SimpleNamespace(reqs=[])
    scheduler.last_batch = None
    scheduler.chunked_req = None
    scheduler.waiting_queue = []
    scheduler.disaggregation_mode = mode
    scheduler.server_args = SimpleNamespace(
        model_path="model",
        revision=None,
        dtype="bfloat16",
        tokenizer_path="tokenizer",
        tokenizer_revision=None,
        tokenizer_mode="auto",
    )
    scheduler.ps = SimpleNamespace(tp_rank=0, pp_size=1)
    req_to_token = torch.zeros((4, 256), dtype=torch.int64)
    req_to_token[2, :3] = torch.tensor([128, 129, 130])
    scheduler.req_to_token_pool = SimpleNamespace(
        available_size=lambda: 4,
        mamba_allocator=SimpleNamespace(
            available_size=lambda: 4,
            free=MagicMock(),
        ),
        req_to_token=req_to_token,
    )
    scheduler.token_to_kv_pool_allocator = SimpleNamespace(
        available_size=lambda: kv_tokens
    )
    scheduler.ipc_channels = SimpleNamespace(
        send_to_tokenizer=SimpleNamespace(send_output=MagicMock())
    )
    return scheduler


@requires_scheduler
def test_non_cuda_scheduler_skips_cuda_workspace_query(monkeypatch):
    scheduler = _scheduler()
    mem_get_info = MagicMock(side_effect=AssertionError("must not query CUDA"))
    monkeypatch.setattr(torch.cuda, "mem_get_info", mem_get_info)

    assert scheduler._qwen_exo_workspace_estimate_bytes(_Req()) == 6144
    assert scheduler._qwen_exo_available_workspace_bytes() is None
    mem_get_info.assert_not_called()


@requires_scheduler
def test_cuda_workspace_counts_reusable_allocator_cache(monkeypatch):
    scheduler = _scheduler()
    scheduler.device = "cuda"
    reserve = scheduler.qwen_exo_hybrid_policy.workspace_safety_reserve_bytes
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        torch.cuda,
        "mem_get_info",
        lambda: (reserve + 1024, reserve + 16384),
    )
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda: 8192)
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda: 2048)

    assert scheduler._qwen_exo_available_workspace_bytes() == 7168


@requires_scheduler
def test_internal_workspace_shortage_rejects_without_reservation(monkeypatch):
    scheduler = _scheduler()
    scheduler.device = "cuda"
    reserve = scheduler.qwen_exo_hybrid_policy.workspace_safety_reserve_bytes
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        torch.cuda,
        "mem_get_info",
        lambda: (reserve + 6143, reserve + 16384),
    )
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda: 0)
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda: 0)
    req = _Req("internal-workspace")
    req.sampling_params.custom_params = {
        "qwen_exo_kind": "internal",
        "qwen_exo_parent_request_id": "parent-1",
    }

    assert not scheduler._try_qwen_exo_admission(req, is_retracted=False)
    assert scheduler.qwen_exo_admission.reservation_count == 0
    sent = scheduler.ipc_channels.send_to_tokenizer.send_output.call_args.args[0]
    assert sent.finished_reason["code"] == "qwen_exo_capacity_exhausted"
    assert sent.finished_reason["message"] == (
        "QWEN-EXO capacity admission failed: workspace_capacity; "
        "required_workspace_bytes=6144; available_workspace_bytes=6143"
    )


@requires_scheduler
def test_scheduler_admission_counts_evictable_hybrid_cache_capacity():
    scheduler = _scheduler(kv_tokens=0)
    scheduler.tree_cache = SimpleNamespace(
        full_evictable_size=lambda: 256,
        mamba_evictable_size=lambda: 1,
    )
    req = _Req("request-evictable")
    req.origin_input_ids = array("q", range(100))
    req.full_untruncated_fill_ids = req.origin_input_ids

    assert scheduler._try_qwen_exo_admission(req, is_retracted=False)


@requires_scheduler
def test_scheduler_hooks_bind_transition_and_release_native_state():
    scheduler = _scheduler()
    req = _Req()

    assert scheduler._try_qwen_exo_admission(req, is_retracted=False)
    state = req.qwen_exo_hybrid_state
    assert state.phase is HybridRequestPhase.ADMITTED
    assert state.handle.lifecycle is HybridLifecycleState.NEW
    assert scheduler.qwen_exo_admission.is_reserved(req.rid)

    req.req_pool_idx = 2
    req.mamba_pool_idx = torch.tensor([7], dtype=torch.int64)
    req.mamba_ping_pong_track_buffer = torch.tensor([8, 9], dtype=torch.int64)
    scheduler._bind_qwen_exo_hybrid_state(req)
    scheduler._release_qwen_exo_reservation(req)

    state = req.qwen_exo_hybrid_state
    assert state.phase is HybridRequestPhase.PREFILL
    assert state.native_req_pool_idx == 2
    assert state.handle.full_kv_blocks == (2,)
    assert state.handle.recurrent_state_slots == (7, 8, 9)
    assert not scheduler.qwen_exo_admission.is_reserved(req.rid)

    scheduler._advance_qwen_exo_decode_state(req)
    assert req.qwen_exo_hybrid_state.phase is HybridRequestPhase.DECODE

    scheduler._release_qwen_exo_hybrid_state(req)
    assert req.qwen_exo_hybrid_state.phase is HybridRequestPhase.RELEASED
    assert not req.qwen_exo_hybrid_state.handle.has_any_component


@requires_scheduler
def test_internal_children_do_not_consume_logical_user_slots():
    scheduler = _scheduler()
    scheduler.req_to_token_pool.available_size = lambda: 0
    internal = _Req("internal-child")
    internal.sampling_params.custom_params = {
        "qwen_exo_kind": "internal",
        "qwen_exo_parent_request_id": "parent-1",
    }
    scheduler.running_batch = SimpleNamespace(reqs=[internal])
    user = _Req("user-parent")

    assert scheduler._qwen_exo_available_user_slots() == 4
    assert scheduler._try_qwen_exo_admission(user, is_retracted=False)
    assert scheduler.qwen_exo_admission.is_reserved(user.rid)


@requires_scheduler
def test_user_slot_census_deduplicates_overlap_and_pending_reservations():
    scheduler = _scheduler(kv_tokens=2048)
    first = _Req("user-1")
    second = _Req("user-2")
    internal = _Req("internal-child")
    internal.sampling_params.custom_params = {
        "qwen_exo_kind": "internal",
        "qwen_exo_parent_request_id": "parent-1",
    }
    scheduler.running_batch = SimpleNamespace(reqs=[first, internal])
    scheduler.last_batch = SimpleNamespace(reqs=[first, second])
    scheduler.chunked_req = second
    scheduler.running_mbs = [SimpleNamespace(reqs=[second])]
    scheduler.mbs = [SimpleNamespace(reqs=[first, second])]

    assert scheduler._qwen_exo_available_user_slots() == 2
    unreserved_waiting = _Req("user-waiting")
    scheduler.waiting_queue = [unreserved_waiting]
    assert scheduler._qwen_exo_available_user_slots() == 1
    scheduler.waiting_queue = []

    pending = _Req("user-3")
    pending.qwen_exo_reservation_pending = True
    pending_estimate = scheduler.qwen_exo_admission.estimate(
        prompt_tokens=3, max_new_tokens=8, needs_mamba=True
    )
    decision = scheduler.qwen_exo_admission.reserve(
        pending.rid,
        pending_estimate,
        available_kv_tokens=2048,
        available_request_slots=2,
        available_mamba_slots=4,
    )
    assert decision.admitted
    scheduler.waiting_queue = [pending]
    assert scheduler._qwen_exo_available_user_slots() == 2

    fourth = _Req("user-4")
    fifth = _Req("user-5")
    assert scheduler._try_qwen_exo_admission(fourth, is_retracted=False)
    assert not scheduler._try_qwen_exo_admission(fifth, is_retracted=False)
    assert scheduler.qwen_exo_admission.reservation_count == 2
    sent = scheduler.ipc_channels.send_to_tokenizer.send_output.call_args.args[0]
    assert sent.finished_reason["code"] == "qwen_exo_capacity_exhausted"
    assert sent.finished_reason["message"].endswith("request_slots")


@requires_scheduler
def test_internal_batch_reserves_each_child_and_queues_past_user_slots():
    scheduler = _scheduler()
    scheduler.req_to_token_pool.available_size = lambda: 0
    first = _Req("internal-1")
    second = _Req("internal-2")
    for req in (first, second):
        req.sampling_params.custom_params = {
            "qwen_exo_kind": "internal",
            "qwen_exo_parent_request_id": "parent-1",
        }

    assert scheduler._try_qwen_exo_admission(first, is_retracted=False)
    assert scheduler._try_qwen_exo_admission(second, is_retracted=False)
    assert scheduler.qwen_exo_admission.reservation_count == 2
    assert scheduler.qwen_exo_admission.is_reserved("internal:parent-1:internal-1")
    assert scheduler.qwen_exo_admission.is_reserved("internal:parent-1:internal-2")

    scheduler._release_qwen_exo_reservation(first)
    assert scheduler.qwen_exo_admission.reservation_count == 1
    scheduler._release_qwen_exo_reservation(second)
    assert scheduler.qwen_exo_admission.reservation_count == 0


@requires_scheduler
def test_internal_fanout_cannot_overcommit_mamba_capacity():
    scheduler = _scheduler()
    scheduler.req_to_token_pool.available_size = lambda: 0
    scheduler.req_to_token_pool.mamba_allocator.available_size = lambda: 1
    first = _Req("internal-1")
    second = _Req("internal-2")
    for req in (first, second):
        req.sampling_params.custom_params = {
            "qwen_exo_kind": "internal",
            "qwen_exo_parent_request_id": "parent-1",
        }

    assert scheduler._try_qwen_exo_admission(first, is_retracted=False)
    assert not scheduler._try_qwen_exo_admission(second, is_retracted=False)
    assert scheduler.qwen_exo_admission.reservation_count == 1
    assert scheduler.qwen_exo_admission.is_reserved("internal:parent-1:internal-1")


@requires_scheduler
def test_internal_job_without_parent_identity_fails_closed():
    scheduler = _scheduler()
    req = _Req("internal-missing-parent")
    req.sampling_params.custom_params = {"qwen_exo_kind": "internal"}

    assert not scheduler._try_qwen_exo_admission(req, is_retracted=False)
    sent = scheduler.ipc_channels.send_to_tokenizer.send_output.call_args.args[0]
    assert sent.finished_reason["code"] == "qwen_exo_internal_parent_invalid"


@requires_scheduler
def test_scheduler_namespace_mismatch_never_reuses_handle():
    scheduler = _scheduler()
    req = _Req()
    req.qwen_exo_hybrid_state = scheduler.qwen_exo_hybrid_policy.new_request_state(
        request_id=req.rid,
        token_ids=req.origin_input_ids,
        model_fingerprint="model",
        tokenizer_fingerprint="tokenizer",
        tp_rank=0,
        namespace=HybridStateNamespace.EXTERNAL_MEMORY,
    )

    assert not scheduler._try_qwen_exo_admission(req, is_retracted=False)
    assert req.qwen_exo_hybrid_state.phase is HybridRequestPhase.RELEASED
    assert scheduler.qwen_exo_admission.reservation_count == 0
    sent = scheduler.ipc_channels.send_to_tokenizer.send_output.call_args.args[0]
    assert sent.finished_reason["code"] == "qwen_exo_hybrid_identity_invalid"


@requires_scheduler
def test_failed_admission_releases_new_handle_without_reservation():
    scheduler = _scheduler(kv_tokens=0)
    req = _Req()

    assert not scheduler._try_qwen_exo_admission(req, is_retracted=False)
    assert req.qwen_exo_hybrid_state.phase is HybridRequestPhase.RELEASED
    assert scheduler.qwen_exo_admission.reservation_count == 0


@requires_scheduler
def test_retraction_releases_old_handle_and_rebinds_same_request_identity():
    scheduler = _scheduler()
    req = _Req()
    assert scheduler._try_qwen_exo_admission(req, is_retracted=False)
    old_state = req.qwen_exo_hybrid_state

    scheduler._release_qwen_exo_hybrid_state(req)
    assert req.qwen_exo_hybrid_state.phase is HybridRequestPhase.RELEASED
    assert scheduler._try_qwen_exo_admission(req, is_retracted=True)

    new_state = req.qwen_exo_hybrid_state
    assert new_state.phase is HybridRequestPhase.ADMITTED
    assert new_state.handle.request_id == req.rid
    assert new_state.handle.handle_id != old_state.handle.handle_id


@requires_scheduler
def test_owned_chunk_continuation_does_not_revalidate_stale_boundary():
    req = SimpleNamespace(init_next_round_input=MagicMock())

    Scheduler._resume_qwen_exo_chunked_prefill(req)

    req.init_next_round_input.assert_called_once_with()


@requires_scheduler
def test_fresh_reuse_mismatch_rejects_only_request_and_cleans_mamba():
    scheduler = _scheduler()
    req = _Req("fresh-mismatch")
    assert scheduler._try_qwen_exo_admission(req, is_retracted=False)
    req.mamba_pool_idx = torch.tensor(7, dtype=torch.int64)
    req.mamba_cow_src_index = torch.tensor([6], dtype=torch.int64)
    req.mamba_needs_clear = True

    scheduler._reject_qwen_exo_cached_reuse(req, ContractViolation("prefix mismatch"))

    scheduler.req_to_token_pool.mamba_allocator.free.assert_called_once()
    freed = scheduler.req_to_token_pool.mamba_allocator.free.call_args.args[0]
    assert torch.equal(freed, torch.tensor([7], dtype=torch.int64))
    assert req.mamba_pool_idx is None
    assert req.mamba_cow_src_index is None
    assert not req.mamba_needs_clear
    assert req.qwen_exo_hybrid_state.phase is HybridRequestPhase.RELEASED
    assert scheduler.qwen_exo_admission.reservation_count == 0
    sent = scheduler.ipc_channels.send_to_tokenizer.send_output.call_args.args[0]
    assert sent.finished_reason["code"] == "qwen_exo_cache_reuse_invalid"


@requires_scheduler
def test_abort_waiting_request_releases_handle_and_reservation():
    scheduler = _scheduler()
    req = _Req()
    assert scheduler._try_qwen_exo_admission(req, is_retracted=False)
    scheduler.chunked_req = None
    scheduler.waiting_queue = [req]
    scheduler.enable_hicache_storage = False
    scheduler.dllm_config = None
    scheduler.grammar_manager = MagicMock()
    scheduler.running_batch = SimpleNamespace(reqs=[])
    scheduler.last_batch = None

    scheduler.abort_request(AbortReq(rid=req.rid))

    assert req.qwen_exo_hybrid_state.phase is HybridRequestPhase.RELEASED
    assert scheduler.qwen_exo_admission.reservation_count == 0
    assert scheduler.waiting_queue == []


@requires_scheduler
def test_non_null_mode_is_ineligible_without_reservation_or_handle():
    scheduler = _scheduler(mode=DisaggregationMode.PREFILL)
    req = _Req()

    assert not scheduler._try_qwen_exo_admission(req, is_retracted=False)
    assert not hasattr(req, "qwen_exo_hybrid_state")
    assert scheduler.qwen_exo_admission.reservation_count == 0
    sent = scheduler.ipc_channels.send_to_tokenizer.send_output.call_args.args[0]
    assert sent.finished_reason["code"] == "qwen_exo_disaggregation_unsupported"


@requires_scheduler
def test_mixed_middle_chunk_is_not_final_but_true_final_chunk_is():
    middle = _Req("middle")
    middle.full_untruncated_fill_ids = array("q", range(8))
    middle.extend_range = SimpleNamespace(start=0, end=4)
    other = _Req("other")
    other.extend_range = SimpleNamespace(start=0, end=3)

    mask = [Scheduler._qwen_exo_is_final_prefill(req) for req in (middle, other)]
    assert mask == [False, True]

    middle.extend_range = SimpleNamespace(start=4, end=8)
    mask = [Scheduler._qwen_exo_is_final_prefill(req) for req in (middle, other)]
    assert mask == [True, True]


@requires_scheduler
def test_scheduler_eviction_hook_clears_cached_node_evidence():
    scheduler = _scheduler()
    cached = scheduler.qwen_exo_hybrid_policy.new_cached_prefix_state(
        request_id="request-1",
        token_ids=[1, 2, 3],
        model_fingerprint="model",
        tokenizer_fingerprint="tokenizer",
        tp_rank=0,
        generation=0,
        full_kv_blocks=(2,),
        recurrent_state_slots=(7,),
        conv_state_slots=(7,),
    )
    components = [
        SimpleNamespace(value=torch.tensor([128, 129, 130])),
        SimpleNamespace(value=None),
        SimpleNamespace(value=torch.tensor([7])),
    ]
    node = SimpleNamespace(
        id=1, component_data=components, qwen_exo_hybrid_state=cached
    )
    scheduler._qwen_exo_cache_nodes = {node.id: node}

    components[2].value = None
    scheduler._reconcile_qwen_exo_cache_nodes()

    assert node.qwen_exo_hybrid_state.phase is HybridRequestPhase.EVICTED
    assert not node.qwen_exo_hybrid_state.handle.has_any_component


@requires_scheduler
def test_scheduler_reuses_only_matching_persisted_radix_state():
    scheduler = _scheduler()
    first = _Req("request-1")
    second = _Req("request-2")
    assert scheduler._try_qwen_exo_admission(first, is_retracted=False)
    assert scheduler._try_qwen_exo_admission(second, is_retracted=False)

    kv_indices = torch.tensor([128, 129, 130], dtype=torch.int64)
    components = [
        SimpleNamespace(value=kv_indices),
        SimpleNamespace(value=None),
        SimpleNamespace(value=torch.tensor([7], dtype=torch.int64)),
    ]
    node = SimpleNamespace(id=1, component_data=components)
    node.qwen_exo_hybrid_state = scheduler._qwen_exo_cached_prefix_candidate(
        first,
        node=node,
        token_ids=first.origin_input_ids,
        kv_indices=kv_indices,
    )
    second.prefix_indices = kv_indices
    second.last_node = node

    scheduler._validate_qwen_exo_cached_reuse(second)

    second.full_untruncated_fill_ids = array("q", [9, 2, 3])
    with pytest.raises(Exception, match="fingerprint mismatch"):
        scheduler._validate_qwen_exo_cached_reuse(second)


@requires_scheduler
def test_cached_reuse_repairs_stale_metadata_only_for_exact_native_path():
    scheduler = _scheduler()
    scheduler._qwen_exo_cache_nodes = {}
    req = _Req("request-repair")
    kv_indices = torch.tensor([128, 129, 130], dtype=torch.int64)
    root = SimpleNamespace(parent=None)
    node = SimpleNamespace(
        id=17,
        parent=root,
        key=SimpleNamespace(raw_token_ids=lambda: array("q", [1, 2, 3])),
        component_data=[
            SimpleNamespace(value=kv_indices),
            SimpleNamespace(value=None),
            SimpleNamespace(value=torch.tensor([7], dtype=torch.int64)),
        ],
    )
    model_fingerprint, tokenizer_fingerprint = (
        scheduler._qwen_exo_identity_fingerprints()
    )
    node.qwen_exo_hybrid_state = (
        scheduler.qwen_exo_hybrid_policy.new_cached_prefix_state(
            request_id="stale-request",
            token_ids=[9, 9],
            model_fingerprint=model_fingerprint,
            tokenizer_fingerprint=tokenizer_fingerprint,
            tp_rank=0,
            generation=0,
            full_kv_blocks=(2,),
            recurrent_state_slots=(7,),
            conv_state_slots=(7,),
        )
    )
    stale_identity = node.qwen_exo_hybrid_state.handle.prefix_identity
    req.prefix_indices = kv_indices
    req.last_node = node

    scheduler._validate_qwen_exo_cached_reuse(req)

    repaired = node.qwen_exo_hybrid_state
    assert repaired.handle.sequence_length == 3
    assert repaired.handle.prefix_identity != stale_identity
    assert scheduler._qwen_exo_cache_nodes[node.id] is node


@requires_scheduler
def test_scheduler_recovers_complete_native_prefix_metadata():
    scheduler = _scheduler()
    scheduler._qwen_exo_cache_nodes = {}
    req = _Req("request-reuse")
    kv_indices = torch.tensor([128, 129, 130], dtype=torch.int64)
    components = [
        SimpleNamespace(value=kv_indices),
        SimpleNamespace(value=None),
        SimpleNamespace(value=torch.tensor([7], dtype=torch.int64)),
    ]
    root = SimpleNamespace(parent=None)
    node = SimpleNamespace(
        id=9,
        parent=root,
        key=SimpleNamespace(raw_token_ids=lambda: array("q", [1, 2, 3])),
        component_data=components,
    )
    req.prefix_indices = kv_indices
    req.last_node = node

    scheduler._validate_qwen_exo_cached_reuse(req)

    assert node.qwen_exo_hybrid_state.phase is HybridRequestPhase.CACHED
    assert scheduler._qwen_exo_cache_nodes[node.id] is node
    node.qwen_exo_hybrid_state = scheduler.qwen_exo_hybrid_policy.evict_cached_state(
        node.qwen_exo_hybrid_state
    )
    assert node.qwen_exo_hybrid_state.phase is HybridRequestPhase.EVICTED
    req.full_untruncated_fill_ids = array("q", [9, 2, 3])
    with pytest.raises(Exception, match="fingerprint mismatch"):
        scheduler._validate_qwen_exo_cached_reuse(req)
    req.full_untruncated_fill_ids = req.origin_input_ids

    scheduler._validate_qwen_exo_cached_reuse(req)

    assert node.qwen_exo_hybrid_state.phase is HybridRequestPhase.CACHED

    components[2].value = None
    with pytest.raises(RuntimeError, match="state is incomplete"):
        scheduler._validate_qwen_exo_cached_reuse(req)


@requires_scheduler
def test_tracked_node_cannot_recover_when_native_paths_mismatch():
    scheduler = _scheduler()
    root = SimpleNamespace(parent=None)
    actual_kv = torch.tensor([128, 129, 130], dtype=torch.int64)
    node = SimpleNamespace(
        id=10,
        parent=root,
        key=SimpleNamespace(raw_token_ids=lambda: array("q", [1, 2, 3])),
        component_data=[
            SimpleNamespace(value=actual_kv),
            SimpleNamespace(value=None),
            SimpleNamespace(value=torch.tensor([7], dtype=torch.int64)),
        ],
    )
    model_fingerprint, tokenizer_fingerprint = (
        scheduler._qwen_exo_identity_fingerprints()
    )
    existing = scheduler.qwen_exo_hybrid_policy.new_cached_prefix_state(
        request_id="request-original",
        token_ids=[1, 2, 3],
        model_fingerprint=model_fingerprint,
        tokenizer_fingerprint=tokenizer_fingerprint,
        tp_rank=0,
        generation=0,
        full_kv_blocks=(2,),
        recurrent_state_slots=(7,),
        conv_state_slots=(7,),
    )
    candidate = scheduler.qwen_exo_hybrid_policy.new_cached_prefix_state(
        request_id="request-corrupt",
        token_ids=[9, 9, 9],
        model_fingerprint=model_fingerprint,
        tokenizer_fingerprint=tokenizer_fingerprint,
        tp_rank=0,
        generation=0,
        full_kv_blocks=(15,),
        recurrent_state_slots=(7,),
        conv_state_slots=(7,),
    )
    node.qwen_exo_hybrid_state = existing
    scheduler._qwen_exo_cache_nodes = {node.id: node}

    recovered = scheduler._recover_qwen_exo_cached_metadata(
        node=node,
        token_ids=[9, 9, 9],
        kv_indices=torch.tensor([960, 961, 962], dtype=torch.int64),
        candidate=candidate,
    )

    assert not recovered
    assert node.qwen_exo_hybrid_state is existing


@requires_scheduler
def test_split_derives_short_prefix_and_preserves_full_prefix_state():
    scheduler = _scheduler()
    scheduler._qwen_exo_cache_nodes = {}
    policy = scheduler.qwen_exo_hybrid_policy
    full_tokens = list(range(128))
    existing = policy.new_cached_prefix_state(
        request_id="request-a",
        token_ids=full_tokens,
        model_fingerprint="model",
        tokenizer_fingerprint="tokenizer",
        tp_rank=0,
        generation=0,
        full_kv_blocks=(2, 3),
        recurrent_state_slots=(7,),
        conv_state_slots=(7,),
    )

    def key(values):
        return SimpleNamespace(raw_token_ids=lambda: array("q", values))

    root = SimpleNamespace(parent=None)
    parent_components = [
        SimpleNamespace(value=torch.arange(128, 192, dtype=torch.int64)),
        SimpleNamespace(value=None),
        SimpleNamespace(value=None),
    ]
    parent = SimpleNamespace(
        id=1,
        parent=root,
        key=key(full_tokens[:64]),
        component_data=parent_components,
    )
    child = SimpleNamespace(
        id=2,
        parent=parent,
        key=key(full_tokens[64:]),
        component_data=[
            SimpleNamespace(value=torch.arange(192, 256, dtype=torch.int64)),
            SimpleNamespace(value=None),
            SimpleNamespace(value=torch.tensor([7], dtype=torch.int64)),
        ],
    )

    scheduler._reconcile_qwen_exo_split_state(
        new_parent=parent,
        child=child,
        existing=existing,
    )

    assert parent.qwen_exo_hybrid_state.phase is HybridRequestPhase.EVICTED
    assert child.qwen_exo_hybrid_state.phase is HybridRequestPhase.CACHED
    assert child.qwen_exo_hybrid_state.handle.full_kv_blocks == (2, 3)
    assert (
        child.qwen_exo_hybrid_state.handle.prefix_identity
        == existing.handle.prefix_identity
    )

    parent_components[2].value = torch.tensor([8], dtype=torch.int64)
    parent.qwen_exo_hybrid_state = scheduler._qwen_exo_state_for_cache_node(
        parent, existing
    )
    shorter_reuse = policy.new_cached_prefix_state(
        request_id="request-b",
        token_ids=full_tokens[:64],
        model_fingerprint="model",
        tokenizer_fingerprint="tokenizer",
        tp_rank=0,
        generation=0,
        full_kv_blocks=(2,),
        recurrent_state_slots=(8,),
        conv_state_slots=(8,),
    )
    full_reuse = policy.new_cached_prefix_state(
        request_id="request-c",
        token_ids=full_tokens,
        model_fingerprint="model",
        tokenizer_fingerprint="tokenizer",
        tp_rank=0,
        generation=0,
        full_kv_blocks=(2, 3),
        recurrent_state_slots=(7,),
        conv_state_slots=(7,),
    )
    policy.assert_cached_reusable(parent.qwen_exo_hybrid_state, shorter_reuse)
    policy.assert_cached_reusable(child.qwen_exo_hybrid_state, full_reuse)
