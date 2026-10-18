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
from qwen_exo_booster.native_state_bank import _atomic_torch_save
from qwen_exo_booster.policy_data import PolicyDataRepository


def _reference_page_analysis(queries, keys, page, mask, width, selected_heads):
    """CPU reference for original full-scan score and evidence mathematics."""
    count, heads, dim = queries.shape
    key_heads = keys.shape[1]
    empty = ((), (), 0, 0, float("-inf"), ())
    if not mask.any():
        return torch.full((count,), float("-inf")), (empty,) * count
    logits = torch.einsum(
        "qkrd,tkd->qtkr", queries.reshape(count, key_heads, heads // key_heads, dim), keys
    ).flatten(start_dim=2) / dim**0.5
    selected_heads = tuple(head for head in selected_heads if 0 <= head < heads)
    if selected_heads:
        keep = torch.zeros(heads, dtype=torch.bool)
        keep[list(selected_heads)] = True
        logits = logits.masked_fill(~keep, float("-inf"))
    head_top = logits.topk(min(4, len(selected_heads) or heads), dim=2, sorted=False)
    start, end = page.cognition_token_count, page.token_end
    token_scores = head_top.values.mean(dim=2)[:, start:end]
    width = min(max(1, width), token_scores.shape[1])
    if width < 1 or not mask[start:end].any():
        return torch.full((count,), float("-inf")), (empty,) * count
    windows = token_scores.unfold(1, width, 1)
    search = mask[start:end].unfold(0, width, 1)
    finite = search.unsqueeze(0) & torch.isfinite(windows)
    top = windows.masked_fill(~finite, float("-inf")).topk(
        min(4, width), dim=2, sorted=True
    )
    support = top.values.mean(dim=2).masked_fill(
        finite.sum(dim=2) < min(4, width), float("-inf")
    )
    scores, best = support.max(dim=1)
    groups = head_top.indices // (heads // key_heads)
    evidence = []
    for row, window in enumerate(best.tolist()):
        offsets = [
            offset for offset in top.indices[row, window].tolist() if search[window, offset]
        ]
        raw = tuple(sorted(start + window + offset for offset in offsets))
        positions = tuple(sorted(page.source_positions[index] for index in raw))
        evidence.append(
            (
                raw,
                positions,
                page.source_positions[start + window],
                page.source_positions[start + window + width - 1] + 1,
                scores[row].item(),
                tuple(sorted({group for index in raw for group in groups[row, index].tolist()})),
            )
        )
    return scores, tuple(evidence)


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


def test_speculative_q_commits_only_accept_tokens_in_order():
    tracker = AttentionSignalTracker(
        num_heads=1,
        num_kv_heads=1,
        head_dim=2,
        max_requests=4,
    )
    metadata = AttentionBatchMetadata(
        is_decode=False,
        is_extend=True,
        contains_last_prefill_chunk=True,
        rids=("request-1", "request-2"),
        observe_mask=(True, True),
        memory_spans=(None, None),
        final_prefill_mask=(True, True),
    )
    slots = torch.tensor([1, 2], dtype=torch.long)
    tracker.prepare_decode_slots(metadata, slots)
    q_rows = torch.tensor(
        [
            [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]],
            [[1.0, 1.0], [0.0, 1.0], [1.0, 0.0]],
        ]
    )

    result = tracker.observe_accept_q(
        q_rows,
        slots,
        torch.tensor([True, True]),
        torch.tensor([2, 0]),
    )

    assert result["qwen_exo_q_norm"].shape == (2, 3)
    assert torch.isfinite(result["qwen_exo_q_norm"][0, :2]).all()
    assert torch.isnan(result["qwen_exo_q_norm"][0, 2])
    assert torch.isnan(result["qwen_exo_q_norm"][1]).all()
    assert torch.isnan(result["qwen_exo_q_drift"][0, 0])
    assert result["qwen_exo_q_drift"][0, 1] == 1.0
    state = tracker._decode_slots
    assert state is not None
    assert state.history_count[1] == 2
    assert state.history_count[2] == 0
    assert torch.equal(state.previous_q[1], torch.tensor([0.0, 1.0]))


def test_decode_observer_treats_empty_graph_mask_as_disabled_rows():
    tracker = AttentionSignalTracker(num_heads=1, num_kv_heads=1, head_dim=2)

    result = tracker.observe_decode_slots(
        torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
        torch.tensor([1, 2], dtype=torch.long),
        torch.empty(0, dtype=torch.bool),
    )

    assert result["qwen_exo_q_norm"].shape == (2,)
    assert torch.isnan(result["qwen_exo_q_norm"]).all()
    assert torch.isnan(result["qwen_exo_q_sketch"]).all()

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

    async def finish_parent(self, _parent_request_id):
        return None

