from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
from qwen_exo_booster.attention_signals import (
    AttentionBatchMetadata,
    AttentionSignalTracker,
    inverse_qwen35_rope,
)
from qwen_exo_booster.internal_jobs import InternalScoreResult
from qwen_exo_booster.knowledge import KnowledgeRepository
from qwen_exo_booster.cognition import CognitionRepository
from qwen_exo_booster.native_state_bank import (
    _atomic_torch_save,
    _page_path,
    _quantize_fp8,
)
from qwen_exo_booster.policy_data import PolicyDataRepository
from qwen_exo_booster.tensor_bank import TensorBank


def test_inverse_qwen35_mrope_uses_token_axis_and_interleaved_sections():
    torch.manual_seed(29)
    head_dim = 256
    rotary_dim = 64
    sections = (11, 11, 10)
    frequencies = torch.linspace(0.01, 0.2, rotary_dim // 2)
    angles = torch.arange(16, dtype=torch.float32).unsqueeze(1) * frequencies
    rotary = SimpleNamespace(
        rotary_dim=rotary_dim,
        mrope_section=sections,
        mrope_interleaved=True,
        is_neox_style=True,
        cos_sin_cache=torch.cat((angles.cos(), angles.sin()), dim=-1),
    )
    positions = torch.stack(
        (
            torch.arange(5),
            torch.arange(5) + 2,
            torch.arange(5) + 4,
        )
    )
    raw = torch.randn(5, 2, head_dim)
    cos_sin = rotary.cos_sin_cache[positions]
    cos, sin = cos_sin.chunk(2, dim=-1)
    mixed_cos = cos[0].clone()
    mixed_sin = sin[0].clone()
    mixed_cos[..., 1 : sections[1] * 3 : 3] = cos[1, ..., 1 : sections[1] * 3 : 3]
    mixed_sin[..., 1 : sections[1] * 3 : 3] = sin[1, ..., 1 : sections[1] * 3 : 3]
    mixed_cos[..., 2 : sections[2] * 3 : 3] = cos[2, ..., 2 : sections[2] * 3 : 3]
    mixed_sin[..., 2 : sections[2] * 3 : 3] = sin[2, ..., 2 : sections[2] * 3 : 3]
    first, second = torch.chunk(raw[..., :rotary_dim], 2, dim=-1)
    mixed_cos = mixed_cos.unsqueeze(1)
    mixed_sin = mixed_sin.unsqueeze(1)
    rotated = torch.cat(
        (
            first * mixed_cos - second * mixed_sin,
            second * mixed_cos + first * mixed_sin,
            raw[..., rotary_dim:],
        ),
        dim=-1,
    )

    restored = inverse_qwen35_rope(rotated, positions, rotary=rotary, head_dim=head_dim)

    assert restored.shape == raw.shape
    assert torch.allclose(restored, raw, atol=1e-5, rtol=1e-5)


def test_q_drift_and_memory_key_energy_follow_request_identity():
    tracker = AttentionSignalTracker(num_heads=1, num_kv_heads=1, head_dim=2)
    prefill = AttentionBatchMetadata(
        is_decode=False,
        is_extend=True,
        contains_last_prefill_chunk=True,
        rids=("request-1",),
        observe_mask=(True,),
        memory_spans=((0, 2, "memory-key"),),
        extend_lens=(2,),
        prefix_lens=(0,),
    )

    first = tracker.observe(
        torch.tensor([[0.0, 1.0], [1.0, 0.0]]),
        torch.tensor([[1.0, 0.0], [1.0, 0.0]]),
        prefill,
    )

    assert torch.isnan(first["qwen_exo_q_drift"][0])
    assert first["qwen_exo_memory_energy"][0] == 1.0

    decode = AttentionBatchMetadata(
        is_decode=True,
        is_extend=False,
        contains_last_prefill_chunk=True,
        rids=("request-1",),
        observe_mask=(True,),
        memory_spans=((0, 2, "memory-key"),),
    )
    second = tracker.observe(
        torch.tensor([[0.0, 1.0]]),
        torch.tensor([[0.0, 1.0]]),
        decode,
    )

    assert second["qwen_exo_q_drift"][0] == 1.0
    assert second["qwen_exo_memory_energy"][0] == 0.5


def test_native_bank_keys_register_memory_energy_without_prefix_reprefill():
    tracker = AttentionSignalTracker(num_heads=1, num_kv_heads=1, head_dim=2)
    tracker.register_memory_keys(
        "qwen-exo-native:prefix", torch.tensor([[1.0, 0.0], [1.0, 0.0]])
    )
    decode = AttentionBatchMetadata(
        is_decode=True,
        is_extend=False,
        contains_last_prefill_chunk=True,
        rids=("request-native",),
        observe_mask=(True,),
        memory_spans=((0, 2, "qwen-exo-native:prefix"),),
    )

    result = tracker.observe(
        torch.tensor([[1.0, 0.0]]), torch.tensor([[0.0, 1.0]]), decode
    )

    assert result["qwen_exo_memory_energy"][0] == 1.0


def test_tp_reduction_synchronizes_rank_local_q_and_k_sketches():
    reductions = []

    def reduce_across_tp(value):
        reductions.append(tuple(value.shape))
        return value + value.flip(-1)

    tracker = AttentionSignalTracker(
        num_heads=1,
        num_kv_heads=1,
        head_dim=2,
        reduce_across_tp=reduce_across_tp,
    )
    metadata = AttentionBatchMetadata(
        is_decode=False,
        is_extend=True,
        contains_last_prefill_chunk=True,
        rids=("tp-request",),
        observe_mask=(True,),
        memory_spans=(None,),
        final_prefill_mask=(True,),
        extend_lens=(1,),
        prefix_lens=(0,),
    )

    result = tracker.observe(
        torch.tensor([[2.0, 0.0]]),
        torch.tensor([[0.0, 2.0]]),
        metadata,
    )

    expected = torch.tensor([2**-0.5, 2**-0.5])
    assert torch.allclose(result["qwen_exo_q_sketch"][0], expected)
    assert torch.allclose(result["qwen_exo_k_sketch"][0], expected)
    assert reductions == [(2,), (1, 2)]


def test_full_q_head_experiment_accumulates_chunks_and_preserves_tp_heads():
    gathers = []

    def gather_heads_across_tp(value):
        gathers.append(tuple(value.shape))
        return torch.cat((value, value + 10), dim=0)

    tracker = AttentionSignalTracker(
        num_heads=2,
        num_kv_heads=1,
        head_dim=2,
        total_num_heads=4,
        gather_heads_across_tp=gather_heads_across_tp,
    )
    first = AttentionBatchMetadata(
        is_decode=False,
        is_extend=True,
        contains_last_prefill_chunk=False,
        rids=("full-q",),
        observe_mask=(True,),
        memory_spans=(None,),
        user_query_spans=(((0, 4),),),
        full_query_capture=(True,),
        final_prefill_mask=(False,),
        extend_lens=(2,),
        prefix_lens=(0,),
    )
    second = replace(
        first,
        contains_last_prefill_chunk=True,
        final_prefill_mask=(True,),
        extend_lens=(2,),
        prefix_lens=(2,),
    )
    first_q = torch.tensor([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])
    second_q = torch.tensor([[[9.0, 10.0], [11.0, 12.0]], [[13.0, 14.0], [15.0, 16.0]]])
    keys = torch.zeros((2, 1, 2))

    assert tracker.observe(first_q, keys, first) is None
    result = tracker.observe(second_q, keys, second)

    expected_local = torch.tensor([[7.0, 8.0], [9.0, 10.0]])
    expected = torch.cat((expected_local, expected_local + 10), dim=0)
    assert result["qwen_exo_user_query_full_heads"].shape == (1, 1, 4, 2)
    assert torch.equal(result["qwen_exo_user_query_full_heads"][0, 0], expected)
    assert gathers == [(2, 2)]
    assert not tracker._full_user_query_sums


def test_non_final_prefill_chunk_accumulates_memory_without_emitting_signal():
    tracker = AttentionSignalTracker(num_heads=1, num_kv_heads=1, head_dim=2)
    chunk = AttentionBatchMetadata(
        is_decode=False,
        is_extend=True,
        contains_last_prefill_chunk=False,
        rids=("request-1",),
        observe_mask=(True,),
        memory_spans=((0, 2, "memory-key"),),
        extend_lens=(1,),
        prefix_lens=(0,),
    )

    result = tracker.observe(
        torch.tensor([[1.0, 0.0]]),
        torch.tensor([[1.0, 0.0]]),
        chunk,
    )

    assert result is None
    assert "memory-key" in tracker._memory_anchors


def test_mixed_middle_and_final_prefill_emits_only_final_request():
    tracker = AttentionSignalTracker(num_heads=1, num_kv_heads=1, head_dim=2)
    mixed = AttentionBatchMetadata(
        is_decode=False,
        is_extend=True,
        contains_last_prefill_chunk=False,
        final_prefill_mask=(False, True),
        rids=("middle", "final"),
        observe_mask=(True, True),
        memory_spans=(None, None),
        extend_lens=(1, 1),
        prefix_lens=(0, 0),
    )

    result = tracker.observe(
        torch.tensor([[1.0, 0.0], [0.0, 2.0]]),
        torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
        mixed,
    )

    assert torch.isnan(result["qwen_exo_q_norm"][0])
    assert torch.isfinite(result["qwen_exo_q_norm"][1])
    assert "middle" not in tracker._q_sketches
    assert "final" in tracker._q_sketches


class _BankTokenizer:
    def encode(self, text, add_special_tokens=False):
        return [ord(character) for character in str(text)]

    def decode(self, token_ids, **_kwargs):
        return "".join(chr(int(token_id)) for token_id in token_ids)


class _BankRunner:
    max_fanout = 8

    def __init__(self, root, mode="native"):
        self.root = root
        self.calls = 0
        self.mode = mode
        self.shared_prefix_keys = []
        self.tokenizer_manager = SimpleNamespace(server_args=SimpleNamespace(tp_size=1))

    async def run_score_batch(
        self,
        jobs,
        prompts,
        label_starts,
        _sampling_params,
        *,
        custom_params_per_job,
        extra_keys,
    ):
        del extra_keys
        self.calls += 1
        self.shared_prefix_keys.extend(job.shared_prefix_key for job in jobs)
        results = []
        for job, prompt, custom in zip(jobs, prompts, custom_params_per_job):
            export = custom["qwen_exo_native_bank_export"]
            assert int(label_starts[len(results)]) == int(export["capture_start"]) - 1
            if self.mode != "missing":
                decoded = "".join(chr(int(token)) for token in prompt)
                vector = (
                    [1.0, 0.0]
                    if "WFP" in decoded or "PAGE_A" in decoded
                    else [0.0, 1.0]
                )
                count = int(export["capture_count"])
                rank_count = 2 if self.mode == "mismatch" else 1
                for rank in range(rank_count):
                    rank_vector = vector if rank == 0 else [*vector, 0.0]
                    key = (
                        torch.tensor(rank_vector)
                        .reshape(1, 1, len(rank_vector))
                        .repeat(count, 1, 1)
                    )
                    payload = {
                        "schema": "qwen-exo-native-state-bank-v1",
                        "source_digest": export["source_digest"],
                        "page_id": export["page_id"],
                        "rank": rank,
                        "world_size": rank_count,
                        "model_fingerprint": custom["qwen_exo_bank_model_fingerprint"],
                        "prefix_identity": export["prefix_identity"],
                        "token_start": export["token_start"],
                        "token_end": export["token_start"] + count,
                        "capture_count": count,
                        "token_ids": tuple(prompt[int(export["capture_start"]) :]),
                        "full_layer_ids": (0,),
                        "full_attention": {
                            "0": {
                                "key": _quantize_fp8(key, reduce_dims=(0, 2)),
                                "value": _quantize_fp8(
                                    torch.zeros_like(key), reduce_dims=(0, 2)
                                ),
                            }
                        },
                        "section_delta": {
                            "conv": (
                                _quantize_fp8(
                                    torch.zeros(1, 1, 2, 2), reduce_dims=(3,)
                                ),
                            ),
                            "temporal": _quantize_fp8(
                                torch.zeros(1, 1, 1, 2, 2), reduce_dims=(3, 4)
                            ),
                        },
                    }
                    _atomic_torch_save(
                        payload,
                        _page_path(
                            self.root,
                            export["source_digest"],
                            export["page_id"],
                            rank,
                        ),
                    )
            results.append(
                InternalScoreResult(
                    job=job,
                    token_logprobs=(-1.0,) * int(export["capture_count"]),
                    mean_nll=1.0,
                    prompt_tokens=len(prompt),
                    finish_reason={"type": "stop"},
                    latency_seconds=0.01,
                    metadata=(
                        {"qwen_exo_bank_export_status": ["exported"]}
                        if self.mode != "missing"
                        else {}
                    ),
                )
            )
        return tuple(results)

    async def finish_parent(self, _parent_request_id):
        return None


@pytest.mark.asyncio
async def test_fp8_tensor_bank_ranks_pages_from_raw_attention_heads(tmp_path):
    root = tmp_path / "knowledge"
    root.mkdir()
    (root / "wfp.md").write_text(
        "# WFP\n" + "WFP outbound authorization canonical identifiers. " * 3,
        encoding="utf-8",
    )
    (root / "ctf.md").write_text(
        "# CTF\n" + "Heap exploitation and return oriented programming. " * 3,
        encoding="utf-8",
    )
    repository = KnowledgeRepository(root)
    repository.refresh()
    runner = _BankRunner(tmp_path / "native-bank")
    bank = TensorBank(
        tmp_path / "tensor-bank.pt",
        runner,
        _BankTokenizer(),
        {"knowledge": repository},
        model_fingerprint="model-fingerprint-1",
        max_document_tokens=512,
        salient_token_budget=64,
    )

    snapshot = await bank.ensure_ready()
    rank_audit = {}
    candidates = bank.rank(
        tuple(((1.0, 0.0),) for _ in range(12)),
        query_identity="event-1",
        limit=2,
        audit=rank_audit,
    )

    assert snapshot.storage_dtype == "float8_e4m3fn"
    assert all(keys.dtype is torch.float32 for keys in snapshot.raw_key_heads)
    assert snapshot.model_native_pages == len(snapshot.pages)
    assert all(page.radix_namespace for page in snapshot.pages)
    assert all(page.prefix_identity for page in snapshot.pages)
    page, prefix_ids = bank.page_prefix_token_ids(snapshot.pages[0].page_id)
    assert prefix_ids
    assert page.radix_namespace in runner.shared_prefix_keys
    assert candidates[0].relative_path == "wfp.md"
    assert candidates[0].candidate_origin == "attention_q_native_tensor_bank"
    assert candidates[0].page_ids
    assert candidates[0].source_positions
    assert candidates[0].virtual_positions == tuple(
        range(len(candidates[0].source_positions))
    )
    assert len(candidates[0].token_attributions) == 12 * len(candidates[0].page_ids)
    assert rank_audit["status"] == "ready"
    assert rank_audit["reason"] == "candidates_ready"
    assert rank_audit["considered_documents"] == 2
    assert {row["relative_path"] for row in rank_audit["scored_documents"]} == {
        "wfp.md",
        "ctf.md",
    }

    score_rejection = {}
    assert (
        bank.rank(
            tuple(((1.0, 0.0),) for _ in range(12)),
            query_identity="event-score-rejected",
            limit=2,
            min_tensor_score=2.0,
            min_document_margin=0.0,
            audit=score_rejection,
        )
        == ()
    )
    assert score_rejection["status"] == "rejected"
    assert score_rejection["reason"] == "top_score_below_threshold"
    assert score_rejection["top_score"] < score_rejection["min_tensor_score"]

    margin_rejection = {}
    assert (
        bank.rank(
            tuple(((1.0, 0.0),) for _ in range(12)),
            query_identity="event-margin-rejected",
            limit=2,
            min_tensor_score=-1.0,
            min_document_margin=2.0,
            audit=margin_rejection,
        )
        == ()
    )
    assert margin_rejection["status"] == "rejected"
    assert margin_rejection["reason"] == "document_margin_too_small"
    assert margin_rejection["observed_margin"] < 2.0
    robust_token_keys = []
    for bank_page, keys in zip(snapshot.pages, snapshot.raw_key_heads):
        values = torch.zeros_like(keys)
        if bank_page.relative_path == "wfp.md":
            values[:, :, 1] = 1.0
            values[0, :, 0] = 1.0
            values[0, :, 1] = 0.0
        else:
            values[:, :, 0] = 0.8
            values[:, :, 1] = 0.6
        robust_token_keys.append(values)
    bank._snapshot = replace(snapshot, raw_key_heads=tuple(robust_token_keys))
    bank._token_search_masks.clear()
    bank._rank_key_cache.clear()
    robust_candidates = bank.rank(
        tuple(((1.0, 0.0),) for _ in range(3)),
        query_identity="event-robust",
        limit=2,
        min_document_margin=0.0,
    )
    assert robust_candidates[0].relative_path == "ctf.md"
    bank._snapshot = snapshot
    bank._rank_key_cache.clear()
    bank._token_search_masks.clear()
    before_rank = (tmp_path / "tensor-bank.pt").read_bytes()
    bank.rank((((0.0, 1.0),),), query_identity="event-2", limit=2)
    assert (tmp_path / "tensor-bank.pt").read_bytes() == before_rank

    cached_runner = _BankRunner(tmp_path / "native-bank")
    cached = TensorBank(
        tmp_path / "tensor-bank.pt",
        cached_runner,
        _BankTokenizer(),
        {"knowledge": repository},
        model_fingerprint="model-fingerprint-1",
        max_document_tokens=512,
        salient_token_budget=64,
    )
    cached_snapshot = await cached.ensure_ready()
    assert cached_runner.calls == 0
    assert cached_snapshot.source_digest == snapshot.source_digest
    await cached.ensure_resident((cached_snapshot.pages[0].page_id,))
    assert cached_runner.calls == 0
    await cached.ensure_resident((cached_snapshot.pages[0].page_id,))
    assert cached_runner.calls == 0


@pytest.mark.asyncio
async def test_raw_qk_ranking_uses_strongest_four_head_pairs(tmp_path):
    root = tmp_path / "raw-head-ranking"
    root.mkdir()
    (root / "wfp.md").write_text("WFP target " * 8, encoding="utf-8")
    (root / "ctf.md").write_text("CTF distractor " * 8, encoding="utf-8")
    repository = KnowledgeRepository(root)
    repository.refresh()
    bank = TensorBank(
        tmp_path / "raw-head-ranking.pt",
        _BankRunner(tmp_path / "native-bank"),
        _BankTokenizer(),
        {"knowledge": repository},
        model_fingerprint="model-fingerprint-raw-heads",
        max_document_tokens=128,
        salient_token_budget=64,
    )
    snapshot = await bank.ensure_ready()
    raw_key_heads = []
    for page, keys in zip(snapshot.pages, snapshot.raw_key_heads):
        values = torch.zeros((keys.shape[0], 2, 2), dtype=torch.float32)
        if page.relative_path == "wfp.md":
            values[:, :, 0] = 1.0
        else:
            values[:, :, 0] = 0.6
            values[:, :, 1] = 0.8
        raw_key_heads.append(values)
    bank._snapshot = replace(snapshot, raw_key_heads=tuple(raw_key_heads))
    bank._rank_key_cache.clear()
    audit = {}
    query = (
        tuple((1.0, 0.0) for _ in range(4)) + tuple((-1.0, 0.0) for _ in range(4)),
    )

    candidates = bank.rank(
        query,
        query_identity="raw-head-top4",
        limit=2,
        min_document_margin=0.0,
        audit=audit,
    )

    assert candidates[0].relative_path == "wfp.md"
    assert audit["scoring_method"] == (
        "raw_attention_top4_heads_top4_tokens_top4_queries"
    )
    assert audit["query_head_count"] == 8
    assert audit["key_head_count"] == 2
    assert audit["head_dim"] == 2


@pytest.mark.asyncio
async def test_cognition_page_pads_to_the_next_radix_alignment(tmp_path):
    cognition_root = tmp_path / "cognition-alignment"
    cognition_root.mkdir()
    (cognition_root / "identity.md").write_text(
        "---\nsource_kind: gpt_cognition_identity_card\n---\n" + "C" * 143,
        encoding="utf-8",
    )
    cognition = CognitionRepository(cognition_root)
    cognition.refresh()
    bank = TensorBank(
        tmp_path / "cognition-alignment.pt",
        _BankRunner(tmp_path / "native-bank"),
        _BankTokenizer(),
        {"cognition": cognition},
        model_fingerprint="model-fingerprint-cognition-alignment",
        max_document_tokens=256,
        salient_token_budget=192,
    )

    snapshot = await bank.ensure_ready()
    page = snapshot.pages[0]
    selection = bank.cognition_selection()

    assert page.token_end == 143
    assert page.state_token_count == 192
    assert len(selection.token_ids) == 192
    assert selection.local_positions == tuple(range(192))
    assert selection.source_positions == tuple(range(143))


@pytest.mark.asyncio
async def test_policy_page_preserves_opening_native_prefix(tmp_path):
    cognition_root = tmp_path / "policy-prefix-cognition"
    policy_root = tmp_path / "policy-prefix"
    cognition_root.mkdir()
    policy_root.mkdir()
    (cognition_root / "identity.md").write_text(
        "---\nsource_kind: gpt_cognition_identity_card\n---\nGPT identity",
        encoding="utf-8",
    )
    (policy_root / "execution.md").write_text("P" * 200, encoding="utf-8")
    cognition = CognitionRepository(cognition_root)
    policy = PolicyDataRepository(policy_root)
    cognition.refresh()
    policy.refresh()
    bank = TensorBank(
        tmp_path / "policy-prefix.pt",
        _BankRunner(tmp_path / "native-bank"),
        _BankTokenizer(),
        {"cognition": cognition, "policydata": policy},
        model_fingerprint="model-fingerprint-policy-prefix",
        max_document_tokens=256,
        salient_token_budget=192,
        span_tokens=16,
    )

    snapshot = await bank.ensure_ready()
    policy_page = next(page for page in snapshot.pages if page.lane == "policydata")
    selection = bank.selection_for_page(policy_page.page_id)
    assert policy_page.cognition_token_count == 0
    assert policy_page.token_end == 200
    assert set(range(128)).issubset(selection.source_positions)
    assert policy_page.state_token_count > int(192 * 0.75)
    assert len(selection.local_positions) == 128
    assert len(selection.local_positions) < policy_page.state_token_count
    assert bank.cognition_selection().page_id == policy_page.page_id
    payload = torch.load(
        _page_path(
            tmp_path / "native-bank",
            snapshot.source_digest,
            policy_page.page_id,
            0,
        ),
        map_location="cpu",
        weights_only=True,
    )
    assert len(payload["token_ids"]) == policy_page.state_token_count
    assert payload["section_delta"]["conv"]
    assert payload["section_delta"]["temporal"] is not None


@pytest.mark.asyncio
async def test_small_policy_page_uses_full_native_state(tmp_path):
    cognition_root = tmp_path / "full-policy-cognition"
    policy_root = tmp_path / "full-policy"
    cognition_root.mkdir()
    policy_root.mkdir()
    (cognition_root / "identity.md").write_text(
        "---\nsource_kind: gpt_cognition_identity_card\n---\nGPT identity",
        encoding="utf-8",
    )
    (policy_root / "execution.md").write_text("Policy rule. " * 8, encoding="utf-8")
    cognition = CognitionRepository(cognition_root)
    policy = PolicyDataRepository(policy_root)
    cognition.refresh()
    policy.refresh()
    bank = TensorBank(
        tmp_path / "full-policy.pt",
        _BankRunner(tmp_path / "native-bank"),
        _BankTokenizer(),
        {"cognition": cognition, "policydata": policy},
        model_fingerprint="model-fingerprint-full-policy",
        max_document_tokens=256,
        salient_token_budget=192,
        span_tokens=16,
    )

    snapshot = await bank.ensure_ready()
    policy_page = next(page for page in snapshot.pages if page.lane == "policydata")
    selection = bank.selection_for_page(policy_page.page_id)

    assert policy_page.state_token_count <= int(192 * 0.75)
    assert selection.source_positions == tuple(range(policy_page.token_end))
    assert selection.local_positions == tuple(range(policy_page.state_token_count))


@pytest.mark.asyncio
async def test_tensor_bank_conditions_specialized_state_on_cognition_without_ranking_it(
    tmp_path,
):
    cognition_root = tmp_path / "cognition"
    knowledge_root = tmp_path / "conditioned-knowledge"
    cognition_root.mkdir()
    knowledge_root.mkdir()
    (cognition_root / "identity.md").write_text(
        "---\nsource_kind: gpt_cognition_identity_card\n---\nGPT identity",
        encoding="utf-8",
    )
    (knowledge_root / "wfp.md").write_text(
        "WFP outbound authorization evidence", encoding="utf-8"
    )
    cognition = CognitionRepository(cognition_root)
    knowledge = KnowledgeRepository(knowledge_root)
    cognition.refresh()
    knowledge.refresh()
    bank = TensorBank(
        tmp_path / "conditioned.pt",
        _BankRunner(tmp_path / "native-bank"),
        _BankTokenizer(),
        {"cognition": cognition, "knowledge": knowledge},
        model_fingerprint="model-fingerprint-cognition",
        max_document_tokens=128,
        salient_token_budget=64,
    )

    snapshot = await bank.ensure_ready()
    cognition_ids = bank.cognition_token_ids()
    cognition_page = next(page for page in snapshot.pages if page.lane == "cognition")
    knowledge_page = next(page for page in snapshot.pages if page.lane == "knowledge")
    _page, knowledge_prefix = bank.page_prefix_token_ids(knowledge_page.page_id)
    candidates = bank.rank(
        (((1.0, 0.0),),),
        query_identity="conditioned-query",
        limit=2,
        min_document_margin=0.0,
    )

    assert snapshot.public_dict()["cognition_document_states"] == 1
    assert snapshot.public_dict()["cognition_conditioned_states"] == 1
    assert cognition_page.cognition_token_count == len(cognition_ids)
    assert knowledge_page.cognition_token_count == len(cognition_ids)
    assert knowledge_prefix[: len(cognition_ids)] == cognition_ids
    assert bank.cognition_selection().page_id == cognition_page.page_id
    assert [candidate.lane for candidate in candidates] == ["knowledge"]
    assert candidates[0].native_prefix is not None
    assert candidates[0].native_prefix.token_ids[: len(cognition_ids)] == cognition_ids


@pytest.mark.asyncio
async def test_policydata_opening_conditions_knowledge_and_is_the_default_personality(
    tmp_path,
):
    policy_root = tmp_path / "personality-policy"
    knowledge_root = tmp_path / "personality-knowledge"
    policy_root.mkdir()
    knowledge_root.mkdir()
    (policy_root / "personality.md").write_text(
        "GPT personality and execution policy", encoding="utf-8"
    )
    (knowledge_root / "guide.md").write_text(
        "Specialized SDK evidence", encoding="utf-8"
    )
    policy = PolicyDataRepository(policy_root)
    knowledge = KnowledgeRepository(knowledge_root)
    policy.refresh()
    knowledge.refresh()
    bank = TensorBank(
        tmp_path / "personality-bank.pt",
        _BankRunner(tmp_path / "native-bank"),
        _BankTokenizer(),
        {"policydata": policy, "knowledge": knowledge},
        model_fingerprint="model-fingerprint-personality-policy",
        max_document_tokens=128,
        salient_token_budget=64,
    )

    snapshot = await bank.ensure_ready()
    personality_ids = bank.cognition_token_ids()
    policy_page = next(page for page in snapshot.pages if page.lane == "policydata")
    knowledge_page = next(page for page in snapshot.pages if page.lane == "knowledge")
    _page, knowledge_prefix = bank.page_prefix_token_ids(knowledge_page.page_id)

    assert personality_ids
    assert policy_page.cognition_token_count == 0
    assert knowledge_page.cognition_token_count == len(personality_ids)
    assert knowledge_prefix[: len(personality_ids)] == personality_ids
    assert bank.cognition_selection().page_id == policy_page.page_id


@pytest.mark.asyncio
async def test_tensor_bank_masks_punctuation_only_sink_page(tmp_path):
    root = tmp_path / "knowledge-sinks"
    root.mkdir()
    (root / "sink.md").write_text(". , ; : ! ? 。 ，", encoding="utf-8")
    (root / "useful.md").write_text("UsefulEvidence", encoding="utf-8")
    repository = KnowledgeRepository(root)
    repository.refresh()
    bank = TensorBank(
        tmp_path / "sinks.pt",
        _BankRunner(tmp_path / "native-bank"),
        _BankTokenizer(),
        {"knowledge": repository},
        model_fingerprint="model-fingerprint-sinks",
        max_document_tokens=64,
        salient_token_budget=64,
    )

    snapshot = await bank.ensure_ready()
    token_keys = []
    for page, keys in zip(snapshot.pages, snapshot.raw_key_heads):
        values = torch.zeros_like(keys)
        if page.relative_path == "sink.md":
            values[:, :, 0] = 1.0
        else:
            values[:, :, 0] = 0.8
            values[:, :, 1] = 0.6
        token_keys.append(values)
    bank._snapshot = replace(snapshot, raw_key_heads=tuple(token_keys))
    bank._rank_key_cache.clear()

    candidates = bank.rank(
        tuple(((1.0, 0.0),) for _ in range(3)),
        query_identity="sink-mask",
        limit=2,
    )

    assert [candidate.relative_path for candidate in candidates] == ["useful.md"]


@pytest.mark.asyncio
async def test_tensor_bank_fails_closed_without_native_k_sketch(tmp_path):
    root = tmp_path / "knowledge"
    root.mkdir()
    (root / "wfp.md").write_text("WFP " * 20, encoding="utf-8")
    repository = KnowledgeRepository(root)
    repository.refresh()
    bank = TensorBank(
        tmp_path / "missing.pt",
        _BankRunner(tmp_path / "native-bank", "missing"),
        _BankTokenizer(),
        {"knowledge": repository},
        model_fingerprint="model-fingerprint",
        max_document_tokens=128,
        salient_token_budget=64,
    )

    with pytest.raises(RuntimeError, match="rank artifact is missing"):
        await bank.ensure_ready()
    assert not (tmp_path / "missing.pt").exists()


@pytest.mark.asyncio
async def test_tensor_bank_rejects_rank_local_shape_mismatch(tmp_path):
    root = tmp_path / "knowledge"
    root.mkdir()
    (root / "wfp.md").write_text("WFP " * 20, encoding="utf-8")
    (root / "ctf.md").write_text("CTF " * 20, encoding="utf-8")
    repository = KnowledgeRepository(root)
    repository.refresh()
    bank = TensorBank(
        tmp_path / "mismatch.pt",
        _BankRunner(tmp_path / "native-bank", "mismatch"),
        _BankTokenizer(),
        {"knowledge": repository},
        model_fingerprint="model-fingerprint",
        max_document_tokens=128,
        salient_token_budget=64,
        tp_size=2,
    )

    with pytest.raises(RuntimeError, match="raw-K shape"):
        await bank.ensure_ready()


@pytest.mark.asyncio
async def test_tensor_bank_keeps_policy_and_knowledge_pages_in_separate_lanes(tmp_path):
    knowledge_root = tmp_path / "knowledge"
    policy_root = tmp_path / "policy"
    knowledge_root.mkdir()
    policy_root.mkdir()
    (knowledge_root / "wfp.md").write_text("WFP knowledge " * 8, encoding="utf-8")
    (policy_root / "wfp-policy.md").write_text("WFP policy " * 8, encoding="utf-8")
    knowledge = KnowledgeRepository(knowledge_root)
    knowledge.refresh()
    policy = PolicyDataRepository(policy_root)
    policy.refresh()
    bank = TensorBank(
        tmp_path / "lanes.pt",
        _BankRunner(tmp_path / "native-bank"),
        _BankTokenizer(),
        {"knowledge": knowledge, "policydata": policy},
        model_fingerprint="model-fingerprint",
        max_document_tokens=256,
        salient_token_budget=128,
    )

    snapshot = await bank.ensure_ready()
    assert bank.rank((((1.0, 0.0),),), query_identity="ambiguous", limit=8) == ()
    candidates = bank.rank(
        (((1.0, 0.0),),),
        query_identity="event",
        limit=8,
        min_document_margin=0.0,
    )

    assert {page.lane for page in snapshot.pages} == {"knowledge", "policydata"}
    assert {candidate.lane for candidate in candidates} == {
        "knowledge",
        "policydata",
    }
    assert len({candidate.candidate_id for candidate in candidates}) == 2


@pytest.mark.asyncio
async def test_tensor_bank_document_group_allows_close_shards_to_rank(tmp_path):
    root = tmp_path / "knowledge"
    root.mkdir()
    frontmatter = (
        "---\ncanonical: false\nquality: 0.8\n"
        "document_group: shared-trajectory\n---\n"
    )
    (root / "trajectory-a.md").write_text(
        frontmatter + "WFP trajectory first shard " * 4,
        encoding="utf-8",
    )
    (root / "trajectory-b.md").write_text(
        frontmatter + "WFP trajectory second shard " * 4,
        encoding="utf-8",
    )
    (root / "unrelated.md").write_text("unrelated reference " * 4, encoding="utf-8")
    repository = KnowledgeRepository(root)
    snapshot = repository.refresh()
    runner = _BankRunner(tmp_path / "native-bank")
    bank = TensorBank(
        tmp_path / "grouped.pt",
        runner,
        _BankTokenizer(),
        {"knowledge": repository},
        model_fingerprint="model-fingerprint",
        max_document_tokens=128,
        salient_token_budget=64,
    )

    await bank.ensure_ready()
    candidates = bank.rank((((1.0, 0.0),),), query_identity="grouped", limit=8)

    grouped = {
        document.relative_path: document.document_group
        for document in snapshot.documents
        if document.relative_path.startswith("trajectory-")
    }
    assert grouped == {
        "trajectory-a.md": "shared-trajectory",
        "trajectory-b.md": "shared-trajectory",
    }
    assert {candidate.relative_path for candidate in candidates[:2]} == {
        "trajectory-a.md",
        "trajectory-b.md",
    }


@pytest.mark.asyncio
async def test_tensor_bank_group_score_rejects_single_page_outlier(tmp_path):
    root = tmp_path / "knowledge-group-outlier"
    root.mkdir()
    frontmatter = "---\ndocument_group: shared\n---\n"
    (root / "shared-a.md").write_text(frontmatter + "A " * 20, encoding="utf-8")
    (root / "shared-b.md").write_text(frontmatter + "B " * 20, encoding="utf-8")
    (root / "unrelated.md").write_text("C " * 20, encoding="utf-8")
    repository = KnowledgeRepository(root)
    repository.refresh()
    runner = _BankRunner(tmp_path / "native-bank")
    bank = TensorBank(
        tmp_path / "group-outlier.pt",
        runner,
        _BankTokenizer(),
        {"knowledge": repository},
        model_fingerprint="model-fingerprint-outlier",
        max_document_tokens=128,
        salient_token_budget=64,
    )

    snapshot = await bank.ensure_ready()
    token_keys = []
    for page, keys in zip(snapshot.pages, snapshot.raw_key_heads):
        values = torch.zeros_like(keys)
        if page.relative_path == "shared-a.md":
            values[:, :, 0] = 1.0
        elif page.relative_path == "shared-b.md":
            values[:, :, 0] = 0.8
            values[:, :, 1] = 0.6
        else:
            values[:, :, 0] = 0.95
            values[:, :, 1] = (1.0 - 0.95**2) ** 0.5
        token_keys.append(values)
    bank._snapshot = replace(snapshot, raw_key_heads=tuple(token_keys))
    bank._token_search_masks.clear()
    bank._rank_key_cache.clear()

    candidates = bank.rank(
        (((1.0, 0.0),),),
        query_identity="group-outlier",
        limit=8,
        min_document_margin=0.0,
    )

    assert candidates[0].relative_path == "unrelated.md"
    assert {candidate.relative_path for candidate in candidates[1:]} == {
        "shared-a.md",
        "shared-b.md",
    }


@pytest.mark.asyncio
async def test_short_policy_page_is_padded_and_bindable_as_native_state(tmp_path):
    policy_root = tmp_path / "policy"
    policy_root.mkdir()
    policy_text = "Require verification code NATIVE_41C9."
    (policy_root / "delivery.md").write_text(policy_text, encoding="utf-8")
    policy = PolicyDataRepository(policy_root)
    policy.refresh()
    tokenizer = _BankTokenizer()
    bank = TensorBank(
        tmp_path / "policy.pt",
        _BankRunner(tmp_path / "native-bank"),
        tokenizer,
        {"policydata": policy},
        model_fingerprint="model-fingerprint",
        max_document_tokens=64,
        salient_token_budget=64,
    )

    snapshot = await bank.ensure_ready()
    candidate = policy.rank("Which verification code is required?", limit=1)[0]
    bound = bank.bind_native_prefix(
        candidate, query="Which verification code is required?"
    )

    assert len(snapshot.pages) == 1
    assert snapshot.pages[0].token_end - snapshot.pages[0].token_start < 64
    assert snapshot.pages[0].state_token_count == 64
    assert bound.native_prefix is not None
    assert len(bound.native_prefix.token_ids) == 64
    assert bound.native_prefix.local_positions == tuple(range(64))
    assert bound.native_prefix.source_positions == tuple(range(len(policy_text)))
    assert bound.candidate_origin == "restored_native_tensor_bank"


@pytest.mark.asyncio
async def test_document_selection_preserves_query_token_attribution(tmp_path):
    root = tmp_path / "knowledge"
    root.mkdir()
    (root / "page-a.md").write_text("PAGE_A" + "a" * 58, encoding="utf-8")
    (root / "page-b.md").write_text("PAGE_B" + "b" * 58, encoding="utf-8")
    repository = KnowledgeRepository(root)
    repository.refresh()
    bank = TensorBank(
        tmp_path / "documents.pt",
        _BankRunner(tmp_path / "native-bank"),
        _BankTokenizer(),
        {"knowledge": repository},
        model_fingerprint="model-fingerprint",
        max_document_tokens=64,
        salient_token_budget=64,
    )
    await bank.ensure_ready()

    candidates = bank.rank(
        (((1.0, 0.0),), ((0.0, 1.0),)),
        query_identity="event-documents",
        limit=2,
        min_document_margin=0.0,
    )

    assert {candidate.relative_path for candidate in candidates} == {
        "page-a.md",
        "page-b.md",
    }
    assert all(len(candidate.page_ids) == 1 for candidate in candidates)
    assert all(
        candidate.source_positions == tuple(range(64)) for candidate in candidates
    )
    assert all(
        candidate.virtual_positions == tuple(range(64)) for candidate in candidates
    )
    assert all(len(candidate.token_attributions) == 2 for candidate in candidates)
