import importlib.util
import sys
from pathlib import Path

import pytest

try:
    import torch
except ImportError:
    torch = None

try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None
    tl = None


if triton is not None:
    score_mod_path = (
        Path(__file__).resolve().parents[3]
        / "python/sglang/kernels/ops/attention/score_mod.py"
    )
    spec = importlib.util.spec_from_file_location(
        "qwen_exo_score_mod_test", score_mod_path
    )
    assert spec is not None and spec.loader is not None
    score_mod_module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = score_mod_module
    spec.loader.exec_module(score_mod_module)
    qwen_exo_block_bias_score_mod = score_mod_module.qwen_exo_block_bias_score_mod

    @triton.jit
    def _score_bias_repro_kernel(out, aux):
        rows = tl.arange(0, 8)[:, None]
        cols = tl.arange(0, 16)[None, :]
        mask = (rows < 8) & (cols < 16)
        logits = tl.zeros((8, 16), dtype=tl.float32)
        logits = qwen_exo_block_bias_score_mod(
            logits, rows, cols, rows, 0, mask, aux, 96, 96, 96
        )
        tl.store(out + rows * 16 + cols, logits, mask=mask)

    @triton.jit
    def _score_bias_masked_row_repro_kernel(out, aux, valid_count_ptr):
        rows = tl.arange(0, 8)[:, None]
        cols = tl.arange(0, 16)[None, :]
        valid_count = tl.load(valid_count_ptr)
        valid_rows = rows < valid_count
        attention_mask = valid_rows & (cols < 16)
        q_idx = tl.where(valid_rows, rows, 1 << 28)
        logits = tl.zeros((8, 16), dtype=tl.float32)
        logits = qwen_exo_block_bias_score_mod(
            logits,
            rows,
            cols,
            q_idx,
            0,
            attention_mask,
            aux,
            96,
            96,
            96,
        )
        tl.store(out + rows * 16 + cols, logits, mask=(rows < 8) & (cols < 16))

else:
    qwen_exo_block_bias_score_mod = None


from qwen_exo_booster.attention_signals import (
    AttentionBatchMetadata,
    AttentionSignalTracker,
)
from qwen_exo_booster.score_bias import (
    SCORE_BIAS_SKETCH_DIMENSIONS,
    ScoreBiasRecord,
    block_surprise_records,
    build_score_bias_payload,
    decay_score,
    find_first_token_span,
    find_last_token_span,
)

KEY_SKETCH = (1.0,) + (0.0,) * (SCORE_BIAS_SKETCH_DIMENSIONS - 1)


def test_score_bias_maps_only_model_indexed_middle_blocks():
    record = ScoreBiasRecord(
        token_ids=(4, 5, 6),
        mean_surprisal=2.0,
        step=0,
        source="tool_output",
        key_sketch=KEY_SKETCH,
    )
    prompt = (1, 4, 5, 6, *range(100, 5100))

    payload = build_score_bias_payload(
        prompt,
        (record,),
        current_step=4,
        half_life_steps=2.0,
        min_surprisal=0.5,
        max_bias=0.75,
        max_blocks=4,
    )

    assert len(payload) == 1
    assert payload[0]["start"] == 1
    assert payload[0]["end"] == 4
    assert payload[0]["score"] == 0.75
    assert payload[0]["key_sketch"] == KEY_SKETCH
    assert payload[0]["age_steps"] == 4


def test_score_bias_excludes_tail_and_enforces_candidate_limit():
    prompt = (1, 2, 3, 4, 5, 6, 7, 8)
    records = tuple(
        ScoreBiasRecord(
            (2 * index + 1, 2 * index + 2),
            float(index + 1),
            index,
            "generated",
            KEY_SKETCH,
        )
        for index in range(4)
    )

    payload = build_score_bias_payload(
        prompt,
        records,
        current_step=4,
        half_life_steps=100.0,
        min_surprisal=0.0,
        max_bias=10.0,
        max_blocks=2,
        min_age_steps=1,
        tail_exclusion_tokens=0,
        tail_exclusion_ratio=0.0,
    )

    assert len(payload) == 2
    assert [(item["start"], item["end"]) for item in payload] == [(4, 6), (6, 8)]

    assert (
        build_score_bias_payload(
            prompt,
            records,
            current_step=4,
            half_life_steps=100.0,
            min_surprisal=0.0,
            max_bias=10.0,
            max_blocks=4,
        )
        == ()
    )


def test_block_surprise_records_preserve_aligned_token_buckets():
    records = block_surprise_records(
        (10, 11, 12, 13, 14),
        (1.0, 3.0, 5.0, 7.0, 9.0),
        block_size=2,
        step=3,
        source="generated",
    )

    assert [record.token_ids for record in records] == [(10, 11), (12, 13), (14,)]
    assert [record.mean_surprisal for record in records] == [2.0, 6.0, 9.0]
    assert all(record.step == 3 for record in records)


def test_score_bias_rejects_invalid_inputs():
    with pytest.raises(ValueError, match="half-life"):
        decay_score(1.0, age_steps=1, half_life_steps=0.0, max_bias=1.0)
    with pytest.raises(ValueError, match="equal lengths"):
        block_surprise_records((1,), (), block_size=1, step=0, source="generated")
    assert find_last_token_span((1, 2, 1, 2), (1, 2)) == (2, 4)
    assert find_last_token_span((1, 2), (3,)) is None
    assert find_first_token_span((1, 2, 1, 2), (1, 2)) == (0, 2)


def test_attention_tracker_captures_trajectory_keys_and_ranks_with_live_q():
    tracker = AttentionSignalTracker(num_heads=2, num_kv_heads=1, head_dim=32)
    q_prefill = torch.zeros((8, 2, 32), dtype=torch.float32)
    k_prefill = torch.zeros((8, 1, 32), dtype=torch.float32)
    k_prefill[2:4, 0, 0] = 1.0
    k_prefill[6:8, 0, 1] = 1.0
    captured = tracker.observe(
        q_prefill,
        k_prefill,
        AttentionBatchMetadata(
            is_decode=False,
            is_extend=True,
            contains_last_prefill_chunk=True,
            rids=("request-1",),
            observe_mask=(True,),
            memory_spans=(None,),
            trajectory_spans=(((2, 4), (6, 8)),),
            final_prefill_mask=(True,),
            extend_lens=(8,),
            prefix_lens=(0,),
        ),
    )
    assert captured is not None
    sketches = captured["qwen_exo_trajectory_k_sketch"][0]

    q_decode = torch.zeros((1, 2, 32), dtype=torch.float32)
    q_decode[0, :, 0] = 1.0
    tracker.observe(
        q_decode,
        torch.zeros((1, 1, 32), dtype=torch.float32),
        AttentionBatchMetadata(
            is_decode=True,
            is_extend=False,
            contains_last_prefill_chunk=True,
            rids=("request-1",),
            observe_mask=(True,),
            memory_spans=(None,),
        ),
    )
    ranked = tracker.rank_trajectory_keys(
        "request-1",
        tuple(tuple(float(value) for value in row) for row in sketches),
        limit=1,
        query_window=2,
        min_score=0.0,
        margin=0.005,
    )

    assert ranked[0][0] == 0
    assert ranked[0][1] > 0


def test_attention_tracker_shortlists_with_user_q_before_decode_q_ranking():
    tracker = AttentionSignalTracker(num_heads=1, num_kv_heads=1, head_dim=32)
    q_prefill = torch.zeros((6, 1, 32), dtype=torch.float32)
    q_prefill[0:2, 0, 0] = 1.0
    k_prefill = torch.zeros((6, 1, 32), dtype=torch.float32)
    k_prefill[2:4, 0, 0] = 1.0
    k_prefill[4:6, 0, 1] = 1.0
    captured = tracker.observe(
        q_prefill,
        k_prefill,
        AttentionBatchMetadata(
            is_decode=False,
            is_extend=True,
            contains_last_prefill_chunk=True,
            rids=("request-user-q",),
            observe_mask=(True,),
            memory_spans=(None,),
            trajectory_spans=(((2, 4), (4, 6)),),
            user_query_spans=(((0, 2),),),
            final_prefill_mask=(True,),
            extend_lens=(6,),
            prefix_lens=(0,),
        ),
    )
    assert captured is not None
    assert "qwen_exo_user_query_sketch" in captured
    key_sketches = tuple(
        tuple(float(value) for value in row)
        for row in captured["qwen_exo_trajectory_k_sketch"][0]
    )
    shortlist = tracker.shortlist_trajectory_keys(
        "request-user-q",
        key_sketches,
        limit=1,
        min_score=0.0,
        margin=0.005,
    )
    assert shortlist[0][0] == 0

    q_decode = torch.zeros((1, 1, 32), dtype=torch.float32)
    q_decode[0, 0, 0] = 1.0
    tracker.observe(
        q_decode,
        torch.zeros((1, 1, 32), dtype=torch.float32),
        AttentionBatchMetadata(
            is_decode=True,
            is_extend=False,
            contains_last_prefill_chunk=True,
            rids=("request-user-q",),
            observe_mask=(True,),
            memory_spans=(None,),
        ),
    )
    ranked = tracker.rank_trajectory_keys(
        "request-user-q",
        key_sketches,
        limit=1,
        query_window=1,
        min_score=0.0,
        margin=0.005,
        allowed_indices=tuple(item[0] for item in shortlist),
    )
    assert ranked[0][0] == 0


def test_decode_slot_observer_and_score_bias_survive_row_reordering():
    tracker = AttentionSignalTracker(
        num_heads=1,
        num_kv_heads=1,
        head_dim=32,
        max_requests=4,
        score_bias_max_blocks=2,
        score_bias_selected_blocks=1,
        score_bias_query_window=2,
    )
    q_prefill = torch.zeros((2, 1, 32), dtype=torch.float32)
    q_prefill[0, 0, 0] = 1.0
    q_prefill[1, 0, 1] = 1.0
    metadata = AttentionBatchMetadata(
        is_decode=False,
        is_extend=True,
        contains_last_prefill_chunk=True,
        rids=("request-a", "request-b"),
        observe_mask=(True, True),
        memory_spans=(None, None),
        user_query_spans=(((0, 1),), ((0, 1),)),
        final_prefill_mask=(True, True),
        extend_lens=(1, 1),
        prefix_lens=(0, 0),
    )
    tracker.observe(
        q_prefill,
        torch.zeros((2, 1, 32), dtype=torch.float32),
        metadata,
    )
    key_a = (1.0,) + (0.0,) * 31
    key_b = (0.0, 1.0) + (0.0,) * 30
    tracker.prepare_decode_slots(
        metadata,
        torch.tensor([2, 0], dtype=torch.long),
        (
            ({"start": 3, "end": 5, "score": 0.2, "key_sketch": key_a},),
            ({"start": 7, "end": 9, "score": 0.3, "key_sketch": key_b},),
        ),
        (2, 2),
    )

    q_decode = torch.zeros((3, 1, 32), dtype=torch.float32)
    q_decode[0, 0, 1] = 1.0
    q_decode[1, 0, 0] = 1.0
    q_decode[2, 0, 7] = 1.0
    request_slots = torch.tensor([0, 2, 0], dtype=torch.long)
    row_mask = torch.tensor([True, True, False])
    observed = tracker.observe_decode_slots(q_decode, request_slots, row_mask)
    score_info, aux = tracker.score_bias_decode_slots(request_slots, row_mask)

    torch.testing.assert_close(observed["qwen_exo_q_drift"][:2], torch.zeros(2))
    assert torch.isnan(observed["qwen_exo_q_norm"][2])
    torch.testing.assert_close(
        aux[:, 0, :3],
        torch.tensor([[7.0, 9.0, 0.3], [3.0, 5.0, 0.2], [0.0, 0.0, 0.0]]),
    )
    assert score_info["qwen_exo_score_bias_selected_count"].tolist() == [
        1.0,
        1.0,
        0.0,
    ]

    reordered_slots = torch.tensor([2, 0], dtype=torch.long)
    reordered_mask = torch.tensor([True, True])
    reordered_q = q_decode[[1, 0]]
    second = tracker.observe_decode_slots(reordered_q, reordered_slots, reordered_mask)
    torch.testing.assert_close(second["qwen_exo_q_drift"], torch.zeros(2))



def test_attention_tracker_applies_bounded_system_anchor_without_trajectory_bias():
    tracker = AttentionSignalTracker(
        num_heads=1,
        num_kv_heads=1,
        head_dim=32,
        max_requests=4,
        score_bias_max_blocks=4,
        score_bias_selected_blocks=1,
        score_bias_anchor_bias=0.01,
        score_bias_anchor_max_blocks=2,
    )
    metadata = AttentionBatchMetadata(
        is_decode=False,
        is_extend=True,
        contains_last_prefill_chunk=True,
        rids=("request-anchor",),
        observe_mask=(True,),
        memory_spans=(None,),
        anchor_spans=(((1, 3),),),
        final_prefill_mask=(True,),
        extend_lens=(4,),
        prefix_lens=(0,),
    )
    tracker.observe(
        torch.zeros((4, 1, 32), dtype=torch.float32),
        torch.zeros((4, 1, 32), dtype=torch.float32),
        metadata,
    )
    tracker.prepare_decode_slots(
        metadata,
        torch.tensor([1], dtype=torch.long),
        score_bias_blocks=((),),
        score_bias_phases=(2,),
    )
    tracker.observe_decode_slots(
        torch.zeros((1, 1, 32), dtype=torch.float32),
        torch.tensor([1], dtype=torch.long),
        torch.tensor([True]),
    )
    info, aux = tracker.score_bias_decode_slots(
        torch.tensor([1], dtype=torch.long),
        torch.tensor([True]),
    )

    torch.testing.assert_close(aux[0, 0, :3], torch.tensor([1.0, 3.0, 0.01]))
    assert info["qwen_exo_score_bias_anchor_count"].tolist() == [1.0]
    assert info["qwen_exo_score_bias_selected_count"].tolist() == [0.0]

def test_runtime_builds_original_and_latest_user_query_spans():
    from collections import OrderedDict
    from types import SimpleNamespace

    from qwen_exo_booster.runtime import QwenExoRuntime

    class Tokenizer:
        @staticmethod
        def encode(text, add_special_tokens=False):
            del add_special_tokens
            return {"first task": [10, 11], "latest instruction": [20, 21]}[text]

    class Telemetry:
        def __init__(self):
            self.events = []

        def emit(self, *event):
            self.events.append(event)

    runtime = object.__new__(QwenExoRuntime)
    runtime.config = SimpleNamespace(
        feature_flags=SimpleNamespace(score_bias=True),
        score_bias_mode="trajectory_shadow",
    )
    runtime.tokenizer_manager = SimpleNamespace(tokenizer=Tokenizer())
    runtime.telemetry = Telemetry()
    runtime._request_conversation_keys = {"request-1": "conversation-1"}
    runtime._score_bias_user_queries = OrderedDict([("conversation-1", (KEY_SKETCH,))])
    runtime._request_score_bias_user_query_prepared = set()
    request = SimpleNamespace(
        request_id="request-1",
        input=[
            {"role": "user", "content": "first task"},
            {"type": "function_call_output", "output": "latest instruction"},
            {"role": "user", "content": "latest instruction"},
        ],
    )

    payload = runtime.score_bias_user_query_payload(
        request, [10, 11, 99, 10, 11, 20, 21]
    )

    assert payload["spans"] == [
        {"start": 0, "end": 2, "source": "original"},
        {"start": 5, "end": 7, "source": "latest"},
    ]
    assert payload["persisted_sketches"] == [list(KEY_SKETCH)]
    assert runtime.telemetry.events[-1][1] == "score_bias.user_query_prepared"


def test_runtime_builds_system_instruction_anchor_spans():
    from collections import OrderedDict
    from types import SimpleNamespace

    from qwen_exo_booster.runtime import QwenExoRuntime

    class Tokenizer:
        @staticmethod
        def encode(text, add_special_tokens=False):
            del add_special_tokens
            return {
                "system rules": [30, 31, 32],
                "first task": [10, 11],
            }[text]

    class Telemetry:
        def __init__(self):
            self.events = []

        def emit(self, *event):
            self.events.append(event)

    runtime = object.__new__(QwenExoRuntime)
    runtime.config = SimpleNamespace(
        feature_flags=SimpleNamespace(score_bias=True),
        score_bias_mode="trajectory_shadow",
        score_bias_anchor_bias=0.01,
        score_bias_anchor_max_blocks=2,
    )
    runtime.tokenizer_manager = SimpleNamespace(tokenizer=Tokenizer())
    runtime.telemetry = Telemetry()
    runtime._request_conversation_keys = {"request-anchor": "conversation-anchor"}
    runtime._score_bias_user_queries = OrderedDict()
    runtime._request_score_bias_user_query_prepared = set()
    request = SimpleNamespace(
        request_id="request-anchor",
        instructions="system rules",
        input=[{"role": "user", "content": "first task"}],
    )

    payload = runtime.score_bias_user_query_payload(
        request, [30, 31, 32, 99, 10, 11]
    )

    assert payload["anchor_spans"] == [
        {"start": 0, "end": 3, "source": "system"}
    ]
    assert runtime.telemetry.events[-1][1] == "score_bias.user_query_prepared"


def test_attention_tracker_preserves_persisted_user_q_when_latest_q_is_captured():
    tracker = AttentionSignalTracker(num_heads=1, num_kv_heads=1, head_dim=32)
    persisted = (0.0, 1.0) + (0.0,) * 30
    q_prefill = torch.zeros((2, 1, 32), dtype=torch.float32)
    q_prefill[:, 0, 0] = 1.0

    captured = tracker.observe(
        q_prefill,
        torch.zeros((2, 1, 32), dtype=torch.float32),
        AttentionBatchMetadata(
            is_decode=False,
            is_extend=True,
            contains_last_prefill_chunk=True,
            rids=("request-merged-q",),
            observe_mask=(True,),
            memory_spans=(None,),
            user_query_spans=(((0, 2),),),
            persisted_user_queries=((persisted,),),
            final_prefill_mask=(True,),
            extend_lens=(2,),
            prefix_lens=(0,),
        ),
    )

    assert captured is not None
    assert tracker.user_query_count("request-merged-q") == 2
    sketches = captured["qwen_exo_user_query_sketch"][0]
    assert torch.isfinite(sketches).all()


def test_runtime_does_not_persist_without_prompt_logprobs():
    from collections import OrderedDict
    from types import SimpleNamespace

    from qwen_exo_booster.runtime import QwenExoRuntime

    runtime = object.__new__(QwenExoRuntime)
    runtime.config = SimpleNamespace(feature_flags=SimpleNamespace(score_bias=True))
    runtime._request_conversation_keys = {"request-1": "conversation-1"}
    runtime._request_score_bias_exact_records = {}
    runtime._score_bias_records = OrderedDict()
    runtime.capsule_store = SimpleNamespace(max_records=4)

    runtime._persist_score_bias_records("request-1")

    assert runtime._score_bias_records == {}


def test_runtime_captures_exact_tool_prompt_surprisal():
    from collections import OrderedDict
    from types import SimpleNamespace

    from qwen_exo_booster.runtime import QwenExoRuntime

    class Tokenizer:
        @staticmethod
        def encode(text, add_special_tokens=False):
            del add_special_tokens
            return [10 + index for index, _ in enumerate(text.split())]

    class Telemetry:
        def __init__(self):
            self.events = []

        def emit(self, *event):
            self.events.append(event)

    runtime = object.__new__(QwenExoRuntime)
    runtime.config = SimpleNamespace(
        feature_flags=SimpleNamespace(score_bias=True), score_bias_max_blocks=4
    )
    runtime.tokenizer_manager = SimpleNamespace(tokenizer=Tokenizer())
    runtime.telemetry = Telemetry()
    from qwen_exo_booster.runtime import TrajectoryCaptureBlock

    runtime._request_prompt_ids = {("request-1", 1): (5, 10, 11, 12, 9)}
    runtime._request_trajectory_capture_blocks = {
        "request-1": (TrajectoryCaptureBlock(1, 3, "read", "retrieval"),)
    }
    runtime._request_score_bias_steps = {"request-1": 1}
    runtime._request_conversation_keys = {"request-1": "conversation-1"}
    runtime._request_score_bias_exact_records = {}
    runtime._request_score_bias_scored_marks = {}
    runtime._score_bias_records = OrderedDict()
    runtime.capsule_store = SimpleNamespace(max_records=4)

    runtime._capture_exact_score_bias_records(
        "request-1",
        {
            "meta_info": {
                "input_token_logprobs": [
                    (-0.1, 5),
                    (-1.5, 10),
                    (-2.0, 11),
                    (-0.5, 12),
                    (-0.1, 9),
                ],
                "qwen_exo_trajectory_k_sketch": [list(KEY_SKETCH)],
            }
        },
        generation_index=1,
    )

    exact = runtime._request_score_bias_exact_records["request-1"]
    assert len(exact) == 1
    assert exact[0].mean_surprisal == pytest.approx(1.75)
    assert exact[0].source == "trajectory_exact"
    assert exact[0].key_sketch == KEY_SKETCH
    assert runtime._score_bias_records["conversation-1"] == exact
    assert runtime.telemetry.events[-1][1] == "score_bias.prompt_scored"


def test_runtime_captures_exact_tool_surprisal_from_bounded_prompt_suffix():
    from collections import OrderedDict
    from types import SimpleNamespace

    from qwen_exo_booster.runtime import QwenExoRuntime, TrajectoryCaptureBlock

    class Telemetry:
        def __init__(self):
            self.events = []

        def emit(self, *event):
            self.events.append(event)

    runtime = object.__new__(QwenExoRuntime)
    runtime.config = SimpleNamespace(
        feature_flags=SimpleNamespace(score_bias=True), score_bias_max_blocks=4
    )
    runtime.telemetry = Telemetry()
    runtime._request_prompt_ids = {("request-suffix", 0): (5, 10, 11, 12, 9)}
    runtime._request_trajectory_capture_blocks = {
        "request-suffix": (TrajectoryCaptureBlock(2, 4, "read", "retrieval"),)
    }
    runtime._request_score_bias_steps = {"request-suffix": 1}
    runtime._request_conversation_keys = {"request-suffix": "conversation-suffix"}
    runtime._request_score_bias_exact_records = {}
    runtime._request_score_bias_scored_marks = {}
    runtime._score_bias_records = OrderedDict()
    runtime.capsule_store = SimpleNamespace(max_records=4)

    runtime._capture_exact_score_bias_records(
        "request-suffix",
        {
            "meta_info": {
                "input_token_logprobs": [
                    (None, 10),
                    (-2.0, 11),
                    (-0.5, 12),
                    (-0.1, 9),
                ],
                "qwen_exo_trajectory_k_sketch": [list(KEY_SKETCH)],
            }
        },
        generation_index=0,
    )

    exact = runtime._request_score_bias_exact_records["request-suffix"]
    assert len(exact) == 1
    assert exact[0].token_ids == (11, 12)
    assert exact[0].mean_surprisal == pytest.approx(1.25)
    assert runtime._score_bias_records["conversation-suffix"] == exact


def test_runtime_fails_closed_on_unaligned_prompt_logprobs():
    from types import SimpleNamespace

    from qwen_exo_booster.runtime import QwenExoRuntime

    runtime = object.__new__(QwenExoRuntime)
    runtime.config = SimpleNamespace(feature_flags=SimpleNamespace(score_bias=True))
    runtime.tokenizer_manager = SimpleNamespace(
        tokenizer=SimpleNamespace(encode=lambda *_args, **_kwargs: [10, 11])
    )
    from qwen_exo_booster.runtime import TrajectoryCaptureBlock

    runtime._request_prompt_ids = {("request-1", 1): (5, 10, 11)}
    runtime._request_trajectory_capture_blocks = {
        "request-1": (TrajectoryCaptureBlock(1, 3, "read", "retrieval"),)
    }
    runtime._request_score_bias_steps = {"request-1": 0}
    runtime._request_score_bias_exact_records = {}
    runtime._request_score_bias_scored_marks = {}

    runtime._capture_exact_score_bias_records(
        "request-1",
        {
            "meta_info": {
                "input_token_logprobs": [(None, 99), (-1.0, 11)],
                "qwen_exo_trajectory_k_sketch": [[KEY_SKETCH]],
            }
        },
        generation_index=1,
    )

    assert runtime._request_score_bias_exact_records == {}


def test_runtime_reports_user_shortlist_and_decode_selection():
    from types import SimpleNamespace

    from qwen_exo_booster.runtime import QwenExoRuntime

    class Telemetry:
        def __init__(self):
            self.events = []

        def emit(self, request_id, event_type, payload):
            self.events.append((request_id, event_type, payload))

    runtime = object.__new__(QwenExoRuntime)
    runtime.config = SimpleNamespace(
        feature_flags=SimpleNamespace(score_bias=True),
        score_bias_mode="trajectory_shadow",
    )
    runtime.telemetry = Telemetry()
    runtime._request_score_bias_selection_emitted = set()

    runtime._emit_score_bias_selection_telemetry(
        "request-1",
        {
            "meta_info": {
                "qwen_exo_score_bias_phase": [1, 1],
                "qwen_exo_score_bias_is_decode": [0, 1],
                "qwen_exo_score_bias_candidate_count": [4, 4],
                "qwen_exo_score_bias_user_query_count": [2, 2],
                "qwen_exo_score_bias_shortlist_count": [3, 3],
                "qwen_exo_score_bias_shortlist_max_relevance": [0.4, 0.4],
                "qwen_exo_score_bias_selected_count": [0, 1],
                "qwen_exo_score_bias_max_relevance": [None, 0.2],
                "qwen_exo_score_bias_query_consensus": [0, 2],
                "qwen_exo_score_bias_would_apply_max": [0, 0.01],
            }
        },
        generation_index=0,
    )

    assert [event[1] for event in runtime.telemetry.events] == [
        "score_bias.user_query_shortlist",
        "score_bias.decode_selected",
    ]
    assert runtime.telemetry.events[-1][2]["selected_count"] == 1
    assert runtime.telemetry.events[-1][2]["query_consensus"] == 2


@pytest.mark.skipif(
    triton is None or torch is None or not torch.cuda.is_available(),
    reason="requires Triton and CUDA",
)
def test_qwen_exo_block_bias_score_mod_compiles_and_applies_spans():
    aux = torch.zeros((8, 1, 96), dtype=torch.float32, device="cuda")
    aux[:, 0, :3] = torch.tensor([0.0, 8.0, 0.25], device="cuda")
    out = torch.empty((8, 16), dtype=torch.float32, device="cuda")

    _score_bias_repro_kernel[(1,)](out, aux)
    expected = torch.zeros_like(out)
    expected[:, :8] = 0.25
    torch.testing.assert_close(out, expected, rtol=0.0, atol=0.0)


@pytest.mark.skipif(
    triton is None or torch is None or not torch.cuda.is_available(),
    reason="requires Triton and CUDA",
)
def test_qwen_exo_block_bias_score_mod_does_not_read_masked_query_rows():
    aux = torch.zeros((7, 1, 96), dtype=torch.float32, device="cuda")
    out = torch.empty((8, 16), dtype=torch.float32, device="cuda")
    valid_count = torch.tensor(7, dtype=torch.int32, device="cuda")

    _score_bias_masked_row_repro_kernel[(1,)](out, aux, valid_count)
    torch.cuda.synchronize()

    torch.testing.assert_close(out, torch.zeros_like(out), rtol=0.0, atol=0.0)
