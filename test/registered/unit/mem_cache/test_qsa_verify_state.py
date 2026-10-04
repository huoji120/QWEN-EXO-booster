import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.layers.attention.qsa.qsa_indexer import QSAIndexer
from sglang.srt.layers.layernorm import GemmaRMSNorm
from sglang.srt.layers.rotary_embedding.base import RotaryEmbedding
from sglang.srt.mem_cache.qsa_kv_pool import QSATokenToKVPool


RATIO = 4


def _pool(device="cpu"):
    pool = QSATokenToKVPool.__new__(QSATokenToKVPool)
    pool.qsa_num_request_slots = 6
    pool.qsa_compress_ratio = RATIO
    pool.qsa_index_kv_heads = 1
    pool.qsa_index_head_dim = 4
    pool.full_attention_layer_id_mapping = {0: 0, 2: 1}
    pool._init_qsa_pending_state(2, device)
    pool.qsa_compressed_k_buffer_pool = [
        torch.zeros(64, 1, 4, dtype=torch.bfloat16, device=device) for _ in range(2)
    ]
    return pool


def _indexer():
    indexer = QSAIndexer.__new__(QSAIndexer)
    torch.nn.Module.__init__(indexer)
    indexer.layer_id = 0
    indexer.compress_ratio = RATIO
    indexer.index_head_dim = 4
    indexer.index_kv_heads = 1
    indexer.k_layernorm = GemmaRMSNorm(4)
    # Keep the CPU reference deterministic even on a CUDA-capable test host.
    indexer.k_layernorm._forward_method = indexer.k_layernorm.forward_native
    with patch("sglang.srt.layers.rotary_embedding.base.get_server_args",
               return_value=SimpleNamespace(rl_on_policy_target=None)):
        indexer.rotary_emb = RotaryEmbedding(4, 4, 256, 10000, True, torch.float32)
    return indexer


def _key(request, position):
    return torch.tensor(
        [[[request + position + 1, 2 * position - 3, request * 3 - position, position + 7]]],
        dtype=torch.bfloat16,
    )


def _rope(request, position):
    return torch.tensor([[position + request], [position * 2 + request], [position * 3 + 1]])


def _compressed_slot(request, group_end):
    return request * 8 + group_end // RATIO


def _decode(indexer, pool, request, position):
    boundary = (position + 1) % RATIO == 0
    group_end = position if boundary else RATIO - 1
    group = torch.arange(RATIO)[None, :] + request * RATIO
    metadata = SimpleNamespace(
        token_to_kv_pool=pool, compress_member_rows=None, is_cuda_graph=False,
        write_locs=torch.tensor([_compressed_slot(request, group_end) if boundary else 0]),
        compress_group_positions=torch.tensor([group_end]),
        compress_sequence_ids=torch.tensor([0]), compress_group_ring_locs=group,
        req_pool_indices=torch.tensor([request]),
    )
    indexer.update_key_state_and_compress(
        _key(request, position), torch.tensor([position]), _rope(request, position),
        metadata, state_slots=torch.tensor([request * RATIO + position % RATIO]),
    )


def _seed(indexer, pool, requests, prefixes):
    for request, prefix in zip(requests, prefixes):
        for position in range(prefix):
            _decode(indexer, pool, request, position)


def _verify(indexer, pool, requests, prefixes, *, graph=False, stored=False):
    row_requests = torch.tensor(requests).repeat_interleave(RATIO)
    positions = torch.cat([torch.arange(prefix, prefix + RATIO) for prefix in prefixes])
    # Padding carries no request even if its synthetic length is one.
    positions[row_requests == 0] = 0
    boundary = ((positions + 1) % RATIO == 0) & (row_requests > 0)
    write_locs = torch.tensor([
        _compressed_slot(int(req), int(pos)) if valid else 0
        for req, pos, valid in zip(row_requests, positions, boundary)
    ])
    group_rows = torch.arange(positions.numel()) if graph else boundary.nonzero().flatten()
    group_ends = positions[group_rows]
    writes = write_locs if graph else write_locs[group_rows]
    slots, groups = pool.prepare_qsa_verify(row_requests, positions, group_ends, group_rows, RATIO)
    keys = torch.cat([_key(int(req), int(pos)) for req, pos in zip(row_requests, positions)])
    rope = torch.cat([_rope(int(req), int(pos)) for req, pos in zip(row_requests, positions)], dim=1)
    metadata = SimpleNamespace(
        token_to_kv_pool=pool, compress_member_rows=None, is_cuda_graph=graph,
        write_locs=writes, compress_group_positions=group_ends,
        compress_sequence_ids=group_rows, compress_group_ring_locs=None,
        graph_write_locs=writes, graph_ring_group_locs=groups,
    )
    if stored:
        # The fused Q-norm/RoPE-store writes these same candidate rows before
        # update_key_state_and_compress sees state_stored=True.
        pool.get_qsa_verify_key_state_buffer(0)[slots] = keys
        pool.qsa_rope_state[slots] = rope.T
    indexer.update_key_state_and_compress(
        keys, positions, rope, metadata, state_slots=slots,
        state_stored=stored, verify_group_locs=groups,
    )


class TestQSAVerifyState(unittest.TestCase):
    def test_unaligned_partial_accept_matches_serial_and_next_complete_group(self):
        for accepted in (1, 2, 3, 4):
            for graph in (False, True):
                for stored in (False, True):
                    with self.subTest(accepted=accepted, graph=graph, stored=stored):
                        indexer = _indexer()
                        actual, serial = _pool(), _pool()
                        _seed(indexer, actual, [1], [3])
                        _seed(indexer, serial, [1], [3])
                        old_keys = actual.get_qsa_key_state_buffer(0).clone()
                        old_rope = actual.qsa_rope_position_buffer.clone()
                        _verify(indexer, actual, [1], [3], graph=graph, stored=stored)
                        torch.testing.assert_close(actual.get_qsa_key_state_buffer(0), old_keys, rtol=0, atol=0)
                        torch.testing.assert_close(actual.qsa_rope_position_buffer, old_rope, rtol=0, atol=0)
                        actual.commit_qsa_state_after_verify(
                            torch.tensor([1]), torch.tensor([accepted]),
                            torch.arange(RATIO).reshape(1, RATIO), RATIO,
                        )
                        for position in range(3, 3 + accepted):
                            _decode(indexer, serial, 1, position)
                        torch.testing.assert_close(actual.get_qsa_key_state_buffer(0), serial.get_qsa_key_state_buffer(0), rtol=0, atol=0)
                        torch.testing.assert_close(actual.qsa_rope_position_buffer, serial.qsa_rope_position_buffer, rtol=0, atol=0)
                        # Rejecting positions 4..6 must not poison the next group.
                        next_boundary = ((3 + accepted) // RATIO + 1) * RATIO
                        for position in range(3 + accepted, next_boundary):
                            _decode(indexer, actual, 1, position)
                            _decode(indexer, serial, 1, position)
                        for slot in (8, 9):
                            torch.testing.assert_close(actual.get_qsa_compressed_k_buffer(0)[slot], serial.get_qsa_compressed_k_buffer(0)[slot], rtol=0, atol=0)

    def test_requests_rejected_future_and_padding_are_isolated(self):
        indexer = _indexer()
        actual, serial = _pool(), _pool()
        requests, prefixes, accepts = [2, 1, 4, 0], [3, 6, 8, 0], [1, 0, 4, 4]
        _seed(indexer, actual, requests[:3], prefixes[:3])
        _seed(indexer, serial, requests[:3], prefixes[:3])
        actual.get_qsa_key_state_buffer(0)[12:16].fill_(77)
        serial.get_qsa_key_state_buffer(0)[12:16].fill_(77)
        history = actual.get_qsa_compressed_k_buffer(0).clone()
        _verify(indexer, actual, requests, prefixes, graph=True)
        # Both full-attention layers share the accepted commit boundary.
        actual.qsa_key_state_flat[1].copy_(actual.qsa_key_state_flat[0])
        actual.commit_qsa_state_after_verify(
            torch.tensor(requests), torch.tensor(accepts),
            torch.arange(16).reshape(4, 4), RATIO,
        )
        for request, prefix, accepted in zip(requests[:3], prefixes[:3], accepts[:3]):
            for position in range(prefix, prefix + accepted):
                _decode(indexer, serial, request, position)
        torch.testing.assert_close(actual.get_qsa_key_state_buffer(0), serial.get_qsa_key_state_buffer(0), rtol=0, atol=0)
        torch.testing.assert_close(actual.get_qsa_key_state_buffer(2), serial.get_qsa_key_state_buffer(0), rtol=0, atol=0)
        torch.testing.assert_close(actual.qsa_rope_position_buffer, serial.qsa_rope_position_buffer, rtol=0, atol=0)
        # Already-complete accepted groups are not cleared on rejection.
        torch.testing.assert_close(actual.get_qsa_compressed_k_buffer(0)[8], history[8], rtol=0, atol=0)
        torch.testing.assert_close(actual.get_qsa_compressed_k_buffer(0)[32:34], history[32:34], rtol=0, atol=0)

    def test_empty_verify_preserves_numerical_state(self):
        pool = _pool()
        pool.qsa_key_state_flat.fill_(23)
        pool.qsa_rope_state.fill_(41)
        empty = torch.empty(0, dtype=torch.int64)
        pool.prepare_qsa_verify(empty, empty, empty, empty, RATIO)
        pool.commit_qsa_state_after_verify(empty, empty, empty, RATIO)
        torch.testing.assert_close(pool.qsa_key_state_flat, torch.full_like(pool.qsa_key_state_flat, 23), rtol=0, atol=0)
        torch.testing.assert_close(pool.qsa_rope_state, torch.full_like(pool.qsa_rope_state, 41), rtol=0, atol=0)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_gpu_plan_and_accepted_commit_match_cpu(self):
        cpu, gpu = _pool(), _pool("cuda")
        requests = torch.tensor([2, 1, 0]).repeat_interleave(RATIO)
        positions = torch.tensor([3, 4, 5, 6, 6, 7, 8, 9, 0, 0, 0, 0])
        group_rows = torch.tensor([0, 5])
        group_ends = positions[group_rows]
        for pool in (cpu, gpu):
            device = pool.qsa_key_state_flat.device
            pool.qsa_key_state_flat.copy_(torch.arange(pool.qsa_key_state_flat.numel()).reshape_as(pool.qsa_key_state_flat).to(device))
            pool.qsa_rope_state.copy_(torch.arange(pool.qsa_rope_state.numel()).reshape_as(pool.qsa_rope_state).to(device))
            pool.prepare_qsa_verify(requests.to(device), positions.to(device), group_ends.to(device), group_rows.to(device), RATIO)
            pool.commit_qsa_state_after_verify(
                torch.tensor([2, 1, 0], device=device), torch.tensor([1, 3, 4], device=device),
                torch.arange(12, device=device).reshape(3, 4), RATIO,
            )
        torch.testing.assert_close(gpu.qsa_verify_row_slots.cpu(), cpu.qsa_verify_row_slots, rtol=0, atol=0)
        torch.testing.assert_close(gpu.qsa_verify_group_locs.cpu(), cpu.qsa_verify_group_locs, rtol=0, atol=0)
        torch.testing.assert_close(gpu.qsa_key_state_flat.cpu(), cpu.qsa_key_state_flat, rtol=0, atol=0)
        torch.testing.assert_close(gpu.qsa_rope_state.cpu(), cpu.qsa_rope_state, rtol=0, atol=0)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_verify_plan_and_commit_replay_use_current_device_inputs(self):
        cpu, gpu = _pool(), _pool("cuda")
        initial_keys = torch.arange(cpu.qsa_key_state_flat.numel()).reshape_as(cpu.qsa_key_state_flat).to(torch.bfloat16)
        initial_rope = torch.arange(cpu.qsa_rope_state.numel()).reshape_as(cpu.qsa_rope_state)
        requests = torch.tensor([2, 1, 0], device="cuda")
        row_requests = requests.repeat_interleave(RATIO)
        positions = torch.tensor([3, 4, 5, 6, 6, 7, 8, 9, 0, 0, 0, 0], device="cuda")
        group_rows = torch.tensor([0, 5], device="cuda")
        group_ends = positions[group_rows].clone()
        accepted = torch.tensor([1, 3, 4], device="cuda")
        indices = torch.arange(12, device="cuda").reshape(3, 4)
        gpu.qsa_key_state_flat.copy_(initial_keys)
        gpu.qsa_rope_state.copy_(initial_rope)

        def step():
            gpu.prepare_qsa_verify(row_requests, positions, group_ends, group_rows, RATIO)
            gpu.commit_qsa_state_after_verify(requests, accepted, indices, RATIO)

        step()  # Compile kernels before capture.
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            step()
        for new_positions, new_groups, new_accepts in (
            ([3, 4, 5, 6, 6, 7, 8, 9, 0, 0, 0, 0], [0, 5], [1, 3, 4]),
            ([7, 8, 9, 10, 4, 5, 6, 7, 0, 0, 0, 0], [0, 7], [3, 1, 0]),
        ):
            positions.copy_(torch.tensor(new_positions, device="cuda"))
            group_rows.copy_(torch.tensor(new_groups, device="cuda"))
            group_ends.copy_(positions[group_rows])
            accepted.copy_(torch.tensor(new_accepts, device="cuda"))
            gpu.qsa_key_state_flat.copy_(initial_keys)
            gpu.qsa_rope_state.copy_(initial_rope)
            cpu.qsa_key_state_flat.copy_(initial_keys)
            cpu.qsa_rope_state.copy_(initial_rope)
            cpu_positions, cpu_groups = torch.tensor(new_positions), torch.tensor(new_groups)
            cpu.prepare_qsa_verify(
                row_requests.cpu(), cpu_positions, cpu_positions[cpu_groups], cpu_groups, RATIO
            )
            cpu.commit_qsa_state_after_verify(
                requests.cpu(), torch.tensor(new_accepts), indices.cpu(), RATIO
            )
            graph.replay()
            torch.testing.assert_close(gpu.qsa_verify_group_locs.cpu(), cpu.qsa_verify_group_locs, rtol=0, atol=0)
            torch.testing.assert_close(gpu.qsa_key_state_flat.cpu(), cpu.qsa_key_state_flat, rtol=0, atol=0)
            torch.testing.assert_close(gpu.qsa_rope_state.cpu(), cpu.qsa_rope_state, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
