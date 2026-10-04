"""Correctness tests for the fused QSA indexer-prep kernels.

Compares the fused kernels (and the QSAIndexer wiring around them) against the
eager indexer path they replace. Outputs must be bit-identical except for rare
last-ulp RMSNorm reduction flips (see assert_bit_comparable):
  - q prep: split -> GemmaRMSNorm -> MRoPE (+ raw-K / RoPE-position stores)
  - compress: gather -> fp32 mean -> GemmaRMSNorm -> MRoPE -> compressed store
"""

from types import SimpleNamespace

import pytest
import torch
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=120, stage="base-b-kernel-unit", runner_config="1-gpu-large")

from sglang.srt.layers.attention.qsa.kernel import (
    average_pool_qsa_keys,
    expand_qsa_block_indices,
    torch_expand_qsa_block_indices,
)
from sglang.srt.layers.attention.qsa.qsa_indexer import QSAIndexer
from sglang.srt.layers.attention.qsa.metadata import QSAIndexerMetadata
from sglang.srt.layers.rotary_embedding.mrope import MRotaryEmbedding

from sglang.srt.runtime_context import get_context


@pytest.fixture(autouse=True)
def _indexer_runtime_context():
    with get_context().override_server_args():
        yield

HEAD_DIM = 128
NUM_Q_HEADS = 4
RATIO = 4
HIDDEN = 2560
EPS = 1e-6


def _make_config():
    return SimpleNamespace(
        indexer_n_heads=NUM_Q_HEADS,
        indexer_kv_heads=1,
        indexer_head_dim=HEAD_DIM,
        indexer_budget=2048,
        indexer_compress_ratio=RATIO,
        hidden_size=HIDDEN,
        rms_norm_eps=EPS,
    )


def _make_rotary(mrope_section, mrope_interleaved, device, dtype=torch.bfloat16):
    return MRotaryEmbedding(
        head_size=HEAD_DIM,
        rotary_dim=HEAD_DIM,
        max_position_embeddings=32768,
        base=1000000,
        is_neox_style=True,
        dtype=dtype,
        mrope_section=mrope_section,
        mrope_interleaved=mrope_interleaved,
    )


def _make_indexer(rotary, device, dtype=torch.bfloat16):
    # Build under the model dtype like ModelRunner does; device-only .to()
    # afterwards so the fp32 cos_sin_cache buffer keeps its dtype.
    prev_dtype = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        indexer = QSAIndexer(
            _make_config(), layer_id=0, quant_config=None, rotary_emb=rotary
        )
        indexer.to(device=device)
    finally:
        torch.set_default_dtype(prev_dtype)
    with torch.no_grad():
        out_features = (NUM_Q_HEADS + 1) * HEAD_DIM
        indexer.index_qk_proj.weight.data.copy_(
            torch.randn(out_features, HIDDEN, device=device, dtype=dtype) * 0.02
        )
        for norm in (indexer.q_layernorm, indexer.k_layernorm):
            w = torch.randn(HEAD_DIM, device=device, dtype=dtype) * 0.1
            norm._weight_loader(norm.weight, w)
    return indexer


class FakePool:
    """Minimal stand-in for the QSA KV pool buffers used by the indexer."""

    def __init__(self, num_slots, num_compressed, device, dtype=torch.bfloat16):
        self.index_state_dtype = dtype
        self.key_state = torch.zeros(
            num_slots, 1, HEAD_DIM, dtype=dtype, device=device
        )
        self.qsa_rope_position_buffer = torch.zeros(
            num_slots, 3, dtype=torch.int64, device=device
        )
        self.compressed = torch.zeros(
            num_compressed, 1, HEAD_DIM, dtype=dtype, device=device
        )

    def get_qsa_key_state_buffer(self, layer_id):
        return self.key_state

    def set_qsa_key_state_buffer(self, layer_id, loc, token_k):
        self.key_state[loc.long()] = token_k.to(self.key_state.dtype)

    def set_qsa_rope_position_buffer(self, loc, positions):
        positions = positions.long()
        if positions.ndim == 1:
            positions = positions.unsqueeze(0).expand(3, -1)
        self.qsa_rope_position_buffer[loc.long()] = positions.transpose(0, 1)

    def get_qsa_rope_position_buffer(self, loc):
        return self.qsa_rope_position_buffer[loc.long()]

    def get_qsa_compressed_k_buffer(self, layer_id):
        return self.compressed

    def set_qsa_compressed_k_buffer(self, layer_id, loc, compressed_k):
        self.compressed[loc.long()] = compressed_k.to(self.compressed.dtype)


def _make_metadata(pool, cache_loc, token_slot_table, write_locs, logical_positions):
    group_ends = logical_positions[(logical_positions + 1) % RATIO == 0]
    members = group_ends[:, None] - torch.arange(
        RATIO - 1, -1, -1, device=cache_loc.device
    )
    return QSAIndexerMetadata(
        sequence_lengths=torch.tensor(
            [int(logical_positions.max()) + 1], dtype=torch.int32, device=cache_loc.device
        ),
        token_to_batch_idx=torch.zeros(cache_loc.numel(), dtype=torch.int32, device=cache_loc.device),
        token_to_kv_pool=pool,
        out_cache_loc=cache_loc,
        token_slot_table=token_slot_table,
        compress_ratio=RATIO,
        block_topk=512,
        write_locs=write_locs,
        compress_group_positions=group_ends,
        compress_sequence_ids=torch.zeros(group_ends.numel(), dtype=torch.int32, device=cache_loc.device),
        compress_group_ring_locs=token_slot_table[0, members.long()],
    )


def _force_eager(indexer):
    """Make an indexer instance take the pre-fusion eager paths."""
    indexer._use_fused_prep = lambda tensor: False
    indexer._use_fused_compress = lambda pool: False
    return indexer


def assert_bit_comparable(actual, expected, max_frac=1e-5, max_abs=0.02):
    """Bit-comparable to the eager path: identical except rare last-ulp flips.

    The eager RMSNorm (flashinfer's CuTe DSL kernel) reduces sums of squares
    in an order that cannot be reproduced exactly, so ~1 row in 30k lands on
    a bf16 rounding boundary and flips by 1-2 ulp (capped at 0.015625 here).
    """
    diff = (actual.float() - expected.float()).abs()
    mismatches = int((diff > 0).sum())
    allowed = max(16, int(max_frac * actual.numel()))
    assert mismatches <= allowed, f"{mismatches} mismatched elements"
    if mismatches:
        peak = diff.max().item()
        assert peak <= max_abs, f"largest deviation {peak} exceeds {max_abs}"


def _run_case(
    num_tokens,
    position_offset,
    mrope_section,
    mrope_interleaved,
    pad_q_heads,
    seed,
):
    device = torch.device("cuda")
    dtype = torch.bfloat16
    torch.manual_seed(seed)
    rotary = _make_rotary(mrope_section, mrope_interleaved, device, dtype)
    indexer = _make_indexer(rotary, device, dtype)

    hidden = torch.randn(num_tokens, HIDDEN, device=device, dtype=dtype) * 0.5
    logical_positions = torch.arange(
        position_offset, position_offset + num_tokens, device=device
    )
    if mrope_section is not None:
        positions = logical_positions.unsqueeze(0).expand(3, -1).contiguous()
    else:
        positions = logical_positions

    # Distinct state slots per token; slot 0 is deliberately unused.
    cache_loc = torch.randperm(max(num_tokens + 8, 4096), device=device)[
        :num_tokens
    ].long() + 1
    token_slot_table = torch.zeros(1, 65536, dtype=torch.int32, device=device)
    token_slot_table[0, logical_positions.long()] = cache_loc.to(torch.int32)

    boundaries = ((logical_positions + 1) % RATIO) == 0
    num_boundaries = int(boundaries.sum())
    write_locs = (
        torch.randperm(2048, device=device)[:num_boundaries].to(torch.int32) + 1
    )

    pool_ref = FakePool(8192, 4096, device, dtype)
    pool_new = FakePool(8192, 4096, device, dtype)

    # Reference: pre-fusion eager path (toggle the real switches off).
    q_ref, token_k_ref, stored_ref = _force_eager(indexer).project_qk(
        hidden, positions
    )
    assert not stored_ref
    indexer.update_key_state_and_compress(
        token_k_ref,
        logical_positions,
        positions,
        _make_metadata(pool_ref, cache_loc, token_slot_table, write_locs, logical_positions),
        state_slots=cache_loc,
    )
    # Restore the fused switches (instance attributes shadow the methods).
    del indexer._use_fused_prep, indexer._use_fused_compress

    # Fused path.
    q_new, token_k_new, stored_new = indexer.project_qk(
        hidden,
        positions,
        pool=pool_new,
        cache_loc=cache_loc,
        q_heads_padded=pad_q_heads,
    )
    assert stored_new
    indexer.update_key_state_and_compress(
        token_k_new,
        logical_positions,
        positions,
        _make_metadata(pool_new, cache_loc, token_slot_table, write_locs, logical_positions),
        state_stored=True,
    )

    expected_heads = pad_q_heads or NUM_Q_HEADS
    assert q_new.shape == (num_tokens, expected_heads, HEAD_DIM)
    assert_bit_comparable(q_new[:, :NUM_Q_HEADS], q_ref)
    if expected_heads > NUM_Q_HEADS:
        assert torch.all(q_new[:, NUM_Q_HEADS:] == 0).item()
    # Raw state stores are plain copies and must match exactly.
    torch.testing.assert_close(pool_new.key_state, pool_ref.key_state, rtol=0, atol=0)
    assert torch.equal(
        pool_new.qsa_rope_position_buffer, pool_ref.qsa_rope_position_buffer
    )
    assert_bit_comparable(pool_new.compressed, pool_ref.compressed)


@pytest.mark.parametrize("pad_q_heads", [None, 8])
@pytest.mark.parametrize(
    "num_tokens, position_offset",
    [
        (1, 3),  # decode landing exactly on a boundary
        (1, 4),  # decode mid-group: no boundary
        (7, 0),  # partial tail group (7 % 4 != 0)
        (8, 1),  # offset tail, boundaries at 3 and 7
        (128, 0),
        (8000, 0),  # prefill shape
    ],
)
@pytest.mark.parametrize(
    "mrope_section, mrope_interleaved",
    [
        ([24, 20, 20], True),  # Qwen4-Exp style interleaved MRoPE
        ([24, 20, 20], False),  # sectioned MRoPE
        (None, False),  # plain 1D RoPE
    ],
)
def test_fused_prep_matches_eager(
    num_tokens, position_offset, mrope_section, mrope_interleaved, pad_q_heads
):
    _run_case(
        num_tokens,
        position_offset,
        mrope_section,
        mrope_interleaved,
        pad_q_heads,
        seed=num_tokens * 31 + position_offset,
    )


def _eager_compress_reference(indexer, pool, group_locs, write_locs):
    """Independent eager compression over selected raw keys and group-start coordinates."""
    key_groups = pool.get_qsa_key_state_buffer(0)[group_locs.long()]
    pooled = average_pool_qsa_keys(key_groups)
    rope_positions = pool.qsa_rope_position_buffer[group_locs[:, 0].long()].transpose(0, 1)
    if not indexer.rotary_emb.mrope_section:
        rope_positions = rope_positions[0]
    normalized = indexer.normalize_compressed_keys(pooled, rope_positions)
    pool.set_qsa_compressed_k_buffer(0, write_locs, normalized)


@pytest.mark.parametrize("num_groups", [1, 5, 2000])
@pytest.mark.parametrize(
    "mrope_section, mrope_interleaved",
    [([24, 20, 20], True), ([24, 20, 20], False), (None, False)],
)
def test_fused_compress_matches_eager(num_groups, mrope_section, mrope_interleaved):
    device = torch.device("cuda")
    dtype = torch.bfloat16
    torch.manual_seed(num_groups)
    rotary = _make_rotary(mrope_section, mrope_interleaved, device, dtype)
    indexer = _make_indexer(rotary, device, dtype)

    pool_ref = FakePool(8192, 4096, device, dtype)
    pool_new = FakePool(8192, 4096, device, dtype)
    pool_new.key_state.copy_(
        pool_ref.key_state.copy_(
            torch.randn(8192, 1, HEAD_DIM, device=device, dtype=dtype)
        )
    )
    positions = torch.randint(0, 30000, (8192, 3), device=device)
    pool_new.qsa_rope_position_buffer.copy_(positions)
    pool_ref.qsa_rope_position_buffer.copy_(positions)

    # Random groups; slot 0 doubles as the CUDA-graph dummy write target, so
    # allow repeats there too.
    group_locs = torch.randint(0, 8192, (num_groups, RATIO), device=device).to(
        torch.int32
    )
    write_locs = torch.randperm(4096, device=device)[:num_groups].to(torch.int32)

    _eager_compress_reference(indexer, pool_ref, group_locs, write_locs)
    indexer._fused_compress_store(pool_new, group_locs, write_locs)

    assert_bit_comparable(pool_new.compressed, pool_ref.compressed)


@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
def test_expand_block_indices_int_inputs(dtype):
    device = torch.device("cuda")
    torch.manual_seed(0)
    rows, block_topk, token_topk, ratio = 37, 512, 2048, 4
    query_positions = torch.randint(0, 8000, (rows,), dtype=dtype, device=device)
    sequence_lengths = (
        query_positions + torch.randint(1, 9, (rows,), dtype=dtype, device=device)
    ).to(dtype)
    # Production contract: top-k only selects blocks inside [0, seq_len//4),
    # so no selected block ever masks out against sequence_lengths.
    counts = torch.randint(0, block_topk + 1, (rows,))
    block_indices = torch.full((rows, block_topk), -1, dtype=torch.int32)
    seq_lens_host = sequence_lengths.cpu()
    for r in range(rows):
        limit = max(int(seq_lens_host[r]) // ratio, 1)
        count = min(int(counts[r]), limit)
        if count:
            block_indices[r, :count] = torch.randperm(limit)[:count].to(torch.int32)
    block_indices = block_indices.to(device)
    out = expand_qsa_block_indices(
        block_indices, query_positions, sequence_lengths, ratio, token_topk
    )
    ref = torch_expand_qsa_block_indices(
        block_indices.cpu(),
        query_positions.cpu(),
        sequence_lengths.cpu(),
        ratio,
        token_topk,
    )
    assert torch.equal(out.cpu(), ref)


def test_decode_selection_equivalent():
    """End-to-end decode selection: eager-prepared vs fused-prepared inputs.

    The last-ulp norm flips must not change the selected top-k blocks: index
    scores are fp32 sums of 128-dim dot products, so a 1-ulp change in one
    component only matters on exact ties.
    """
    from sglang.srt.layers.attention.qsa.mqa import torch_qsa_mqa_decode
    from sglang.srt.layers.attention.qsa.kernel import qsa_fast_topk

    device = torch.device("cuda")
    dtype = torch.bfloat16
    torch.manual_seed(7)
    rotary = _make_rotary([24, 20, 20], True, device, dtype)
    indexer = _make_indexer(rotary, device, dtype)

    batch, max_pages, page_size = 4, 32, 64
    max_model_len = max_pages * page_size
    hidden = torch.randn(batch, HIDDEN, device=device, dtype=dtype)
    positions = (
        torch.arange(8000, 8000 + batch, device=device)
        .unsqueeze(0)
        .expand(3, -1)
        .contiguous()
    )
    qk, _ = indexer.index_qk_proj(hidden)

    # Eager index q.
    q_ref = indexer.q_layernorm(qk[:, : NUM_Q_HEADS * HEAD_DIM].reshape(-1, HEAD_DIM))
    q_ref = q_ref.reshape(batch, NUM_Q_HEADS, HEAD_DIM)
    q_ref = indexer.apply_rope(positions, q_ref)

    # Fused index q.
    pool = FakePool(64, 4096, device, dtype)
    cache_loc = torch.arange(1, batch + 1, device=device)
    q_new, _, stored = indexer.project_qk(
        hidden, positions, pool=pool, cache_loc=cache_loc
    )
    assert stored

    compressed_cache = torch.randn(
        64, page_size, 1, HEAD_DIM, device=device, dtype=dtype
    )
    page_table = torch.arange(max_pages, dtype=torch.int32, device=device).repeat(
        batch, 1
    )
    context_lens = torch.full((batch,), 1500, dtype=torch.int32, device=device)

    def select(q):
        logits = torch_qsa_mqa_decode(
            q, compressed_cache, page_table, context_lens, max_model_len
        )
        row_starts = torch.zeros_like(context_lens)
        return qsa_fast_topk(logits, row_starts, context_lens, topk=512)

    idx_ref = select(q_ref)
    idx_new = select(q_new[:, :NUM_Q_HEADS].contiguous())
    for row in range(batch):
        ref_set = set(idx_ref[row][idx_ref[row] >= 0].tolist())
        new_set = set(idx_new[row][idx_new[row] >= 0].tolist())
        assert ref_set == new_set, f"row {row}: selection mismatch"


def test_target_verify_fused_store_preserves_accepted_prefix():
    from sglang.srt.mem_cache.qsa_kv_pool import QSATokenToKVPool

    device = torch.device("cuda")
    torch.manual_seed(173)
    indexer = _make_indexer(_make_rotary([24, 20, 20], True, device), device)
    pool = QSATokenToKVPool.__new__(QSATokenToKVPool)
    pool.qsa_num_request_slots = 3
    pool.qsa_compress_ratio = RATIO
    pool.qsa_index_kv_heads = 1
    pool.qsa_index_head_dim = HEAD_DIM
    pool.full_attention_layer_id_mapping = {0: 0}
    pool._init_qsa_pending_state(1, device)
    pool.qsa_compressed_k_buffer_pool = [
        torch.zeros(16, 1, HEAD_DIM, dtype=torch.bfloat16, device=device)
    ]
    live = pool.get_qsa_key_state_buffer(0)
    live[4:7] = torch.randn(3, 1, HEAD_DIM, dtype=live.dtype, device=device)
    pool.qsa_rope_position_buffer[4:7] = torch.tensor(
        [[0, 2, 4], [1, 3, 5], [2, 4, 6]], device=device
    )
    old_keys = live.clone()
    old_rope = pool.qsa_rope_position_buffer.clone()
    logical = torch.arange(3, 7, device=device)
    positions = torch.stack((logical, logical + 2, logical + 4))
    slots, groups = pool.prepare_qsa_verify(
        torch.ones(4, dtype=torch.int64, device=device), logical,
        logical[:1], torch.zeros(1, dtype=torch.int64, device=device), RATIO,
    )
    hidden = torch.randn(4, HIDDEN, dtype=torch.bfloat16, device=device)
    _, token_k, stored = indexer.project_qk(
        hidden, positions, pool=pool, cache_loc=slots, is_target_verify=True,
    )
    assert stored
    metadata = SimpleNamespace(
        token_to_kv_pool=pool, compress_member_rows=None, is_cuda_graph=False,
        write_locs=torch.tensor([1], dtype=torch.int32, device=device),
        compress_group_positions=logical[:1], compress_sequence_ids=torch.zeros(
            1, dtype=torch.int64, device=device
        ), compress_group_ring_locs=None,
    )
    indexer.update_key_state_and_compress(
        token_k, logical, positions, metadata, state_slots=slots,
        state_stored=True, verify_group_locs=groups,
    )
    torch.testing.assert_close(live, old_keys, rtol=0, atol=0)
    torch.testing.assert_close(pool.qsa_rope_position_buffer, old_rope, rtol=0, atol=0)
    serial_members = torch.cat((old_keys[4:7], token_k[:1]), dim=0)[None]
    expected = indexer.normalize_compressed_keys(
        average_pool_qsa_keys(serial_members), old_rope[4:5].T
    )
    assert_bit_comparable(pool.get_qsa_compressed_k_buffer(0)[1:2], expected)
    pool.commit_qsa_state_after_verify(
        torch.tensor([1], device=device), torch.tensor([1], device=device),
        torch.arange(4, device=device).reshape(1, 4), RATIO,
    )
    torch.testing.assert_close(live[4:7], old_keys[4:7], rtol=0, atol=0)
    torch.testing.assert_close(live[7:8], token_k[:1], rtol=0, atol=0)
    torch.testing.assert_close(pool.qsa_rope_position_buffer[7], positions[:, 0], rtol=0, atol=0)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v", "-s"]))
