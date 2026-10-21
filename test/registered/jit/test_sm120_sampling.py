"""Offline GPU checks; no model, checkpoint, server, or inference launch."""

import unittest
from types import SimpleNamespace

import torch

from sglang.jit_kernel.sm120_sampling import (
    exact_chain_sampling,
    prepare_target_distribution,
    sample_truncated_proposal,
    sorted_topk,
)
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=80, stage="base-b-kernel-unit", runner_config="1-gpu-large")


def reference_target(logits, temperatures, ks, ps, depth):
    """Independent dense sorting oracle with complete cutoff and nucleus ties."""
    rows = []
    for r, row in enumerate(logits.double().cpu()):
        b = r // depth
        p = (row / float(temperatures[b])).softmax(-1)
        k = int(ks[b])
        if 0 < k < row.numel():
            pivot = p.sort(descending=True).values[k - 1]
            p = torch.where(p >= pivot, p, 0)
            p /= p.sum()
        if float(ps[b]) < 1:
            ordered = p.sort(descending=True).values
            cutoff = torch.searchsorted(ordered.cumsum(0), float(ps[b])).clamp_max(row.numel() - 1)
            pivot = ordered[cutoff]
            p = torch.where(p >= pivot, p, 0)
            p /= p.sum()
        rows.append(p)
    return torch.stack(rows).float().reshape(-1, depth, logits.shape[-1])


def materialize(distribution, depth):
    batch = distribution.eligible.numel()
    vocab = distribution.dense.shape[-1]
    sparse = torch.zeros((batch, depth, vocab), device="cuda")
    sparse.scatter_(-1, distribution.ids, distribution.sparse)
    return torch.where(
        distribution.eligible[:, None, None].bool(), sparse,
        distribution.dense.reshape(batch, depth, vocab),
    )


def sample_chain(distribution, q, candidates, coins, final_coins):
    batch, depth = candidates.shape
    retrieve = torch.arange(batch * depth, device="cuda", dtype=torch.int64).view(batch, depth)
    predicts = torch.full((batch * depth,), -1, dtype=torch.int32, device="cuda")
    accepted = torch.full((batch, depth), -1, dtype=torch.int32, device="cuda")
    counts = torch.empty((batch,), dtype=torch.int32, device="cuda")
    exact_chain_sampling(
        distribution, predicts=predicts, accept_index=accepted, accept_token_num=counts,
        candidates=candidates, retrieve_index=retrieve, retrieve_next_token=retrieve,
        retrieve_next_sibling=retrieve, uniform_samples=coins,
        uniform_samples_for_final_sampling=final_coins, draft_probs=q,
    )
    return predicts.reshape(batch, depth), accepted, counts


def reference_chain(p, q, candidates, coins, final_coins):
    batch, depth, vocab = p.shape
    predicts = torch.full((batch, depth), -1, dtype=torch.int32)
    indices = torch.full((batch, depth), -1, dtype=torch.int32)
    counts = torch.zeros(batch, dtype=torch.int32)
    for b in range(batch):
        row = 0
        indices[b, 0] = b * depth
        for s in range(1, depth):
            tok = int(candidates[b, s])
            if coins[b, s - 1] * q[b, row, tok] >= p[b, row, tok]:
                break
            predicts[b, row] = tok
            row = s
            counts[b] += 1
            indices[b, counts[b]] = b * depth + s
        weights = p[b, row].clone()
        if row != depth - 1:
            weights = (weights - q[b, row]).clamp_min(0)
        threshold = final_coins[b] * weights.sum()
        matches = (weights.cumsum(0) > threshold).nonzero()
        predicts[b, row] = int(matches[0]) if len(matches) else vocab - 1
    return predicts, indices, counts


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class TestSm120ExactSampling(CustomTestCase):
    def test_sorted_topk(self):
        for vocab in (64, 257, 4096, 248320):
            for k in (1, 8, 56, 64):
                logits = torch.randn((3, vocab), device="cuda")
                # Adversarial ties: selection membership may vary, values may not.
                logits[0, :100] = 8
                values, ids = sorted_topk(logits, k)
                expected = logits.topk(k, dim=-1).values
                torch.testing.assert_close(values, expected)
                torch.testing.assert_close(values, logits.gather(-1, ids))
                for row in ids.cpu():
                    self.assertEqual(row.unique().numel(), k)

    def test_proposal_uses_actual_q_and_no_static_map(self):
        for batch in (1, 3, 10):
            vocab = 248320
            logits = torch.randn((batch, vocab), device="cuda")
            logits[:, 200001] = logits.amax(-1) + 0.1
            temperatures = torch.linspace(0.7, 1.5, batch, device="cuda")[:, None]
            ks = torch.tensor(([1, 9, -1, 64, 100] * 2)[:batch], device="cuda", dtype=torch.int32)
            ps = torch.linspace(0.6, 1.0, batch, device="cuda")
            expo = torch.rand((batch, 64), device="cuda").clamp_min(1e-6)
            q, px, tokens = sample_truncated_proposal(logits, temperatures, ks, ps, expo)
            values, ids = logits.topk(64, dim=-1)
            expected = torch.zeros_like(q)
            for b in range(batch):
                k = min(int(ks[b]), 64) if ks[b] > 0 else 64
                weights = (values[b, :k].double() / temperatures[b]).softmax(-1)
                ordered = weights.sort(descending=True).values
                if ps[b] < 1:
                    cutoff = torch.searchsorted(ordered.cumsum(0), ps[b].double()).clamp_max(k - 1)
                    weights = torch.where(weights >= ordered[cutoff], weights, 0)
                    weights /= weights.sum()
                expected[b, ids[b, :k]] = weights.float()
            torch.testing.assert_close(q, expected, atol=1e-6, rtol=2e-5)
            torch.testing.assert_close(px, q.gather(1, tokens))
            torch.testing.assert_close(q.sum(-1), torch.ones(batch, device="cuda"))
            self.assertTrue(bool((q[:, 200001] > 0).all()))
            candidate_q = q.gather(-1, ids)
            winners = (candidate_q / expo).argmax(-1, keepdim=True)
            torch.testing.assert_close(tokens, ids.gather(1, winners))

    def test_target_dense_fallback_and_complete_ties(self):
        batch, depth, vocab = 10, 4, 4096
        logits = torch.randn((batch * depth, vocab), device="cuda")
        # More than 64 tied cutoff tokens, at both top-k and top-p boundaries.
        logits[0:depth, :130] = 10
        ks = torch.tensor([4, 1, 56, 57, 0, -1, 4096, 8, 13, 3], dtype=torch.int32, device="cuda")
        ps = torch.tensor([0.02, 1, 0.8, 0.9, 1, 0.8, 0.6, 1, 0.5, 0.9], device="cuda")
        temps = torch.linspace(0.6, 1.6, batch, device="cuda")[:, None]
        info = SimpleNamespace(temperatures=temps, top_ks=ks, top_ps=ps)
        distribution = prepare_target_distribution(logits, info, depth)
        actual = materialize(distribution, depth)
        expected = reference_target(logits, temps.cpu(), ks.cpu(), ps.cpu(), depth).cuda()
        torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-4)
        self.assertFalse(bool(distribution.eligible[0]))
        self.assertTrue(bool(distribution.eligible[1]))
        self.assertFalse(bool(distribution.eligible[3:7].any()))
        self.assertEqual(int((actual[0, 0] > 0).sum()), 130)

    def test_accept_zero_through_width_duplicate_ids_and_residual(self):
        for depth in (2, 3, 4, 5):
            batch, vocab = depth, 257
            logits = torch.full((batch * depth, vocab), -float("inf"), device="cuda")
            logits[:, 7] = torch.log(torch.tensor(0.25)).item()
            logits[:, 200] = torch.log(torch.tensor(0.75)).item()
            info = SimpleNamespace(
                temperatures=torch.ones((batch, 1), device="cuda"),
                top_ks=torch.full((batch,), 2, dtype=torch.int32, device="cuda"),
                top_ps=torch.ones(batch, device="cuda"),
            )
            distribution = prepare_target_distribution(logits, info, depth)
            q = torch.zeros((batch, depth - 1, vocab), device="cuda")
            q[:, :, 7] = 0.5
            q[:, :, 12] = 0.5  # target token 200 is outside the proposal support.
            candidates = torch.full((batch, depth), 7, dtype=torch.int64, device="cuda")
            coins = torch.ones((batch, depth), device="cuda")
            for b in range(batch):
                coins[b, :b] = 0
            final = torch.full((batch,), 0.75, device="cuda")
            actual = sample_chain(distribution, q, candidates, coins, final)
            p = reference_target(logits, info.temperatures.cpu(), info.top_ks.cpu(), info.top_ps.cpu(), depth)
            expected = reference_chain(p, q.cpu(), candidates.cpu(), coins.cpu(), final.cpu())
            for got, want in zip(actual, expected):
                torch.testing.assert_close(got.cpu(), want)
            self.assertEqual(actual[2].cpu().tolist(), list(range(depth)))
            self.assertTrue(bool((actual[0][:, -1][actual[2] == depth - 1] == 200).all()))

    def test_identical_coins_sparse_and_dense_against_independent_oracle(self):
        batch, depth, vocab = 10, 4, 257
        logits = torch.randn((batch * depth, vocab), device="cuda")
        logits[:depth, :100] = 8  # tie-overflow request next to sparse requests
        info = SimpleNamespace(
            temperatures=torch.linspace(0.5, 1.7, batch, device="cuda")[:, None],
            top_ks=torch.tensor([3, 4, 5, 56, 57, -1, 1, 2, 12, 0], dtype=torch.int32, device="cuda"),
            top_ps=torch.linspace(0.5, 1, batch, device="cuda"),
        )
        distribution = prepare_target_distribution(logits, info, depth)
        draft_logits = torch.randn((batch * (depth - 1), vocab), device="cuda")
        q, _, tokens = sample_truncated_proposal(
            draft_logits, info.temperatures.repeat_interleave(depth - 1, 0),
            info.top_ks.repeat_interleave(depth - 1), info.top_ps.repeat_interleave(depth - 1),
        )
        q = q.view(batch, depth - 1, vocab)
        candidates = torch.cat((torch.zeros((batch, 1), dtype=torch.int64, device="cuda"), tokens.view(batch, depth - 1)), -1)
        coins = torch.rand((batch, depth), device="cuda")
        final = torch.rand(batch, device="cuda")
        actual = sample_chain(distribution, q, candidates, coins, final)
        p = reference_target(logits, info.temperatures.cpu(), info.top_ks.cpu(), info.top_ps.cpu(), depth)
        expected = reference_chain(p, q.cpu(), candidates.cpu(), coins.cpu(), final.cpu())
        for got, want in zip(actual, expected):
            torch.testing.assert_close(got.cpu(), want)

    def test_first_sample_marginal_is_full_target_p(self):
        batch, depth, vocab = 8192, 2, 64
        logits = torch.full((batch * depth, vocab), -float("inf"), device="cuda")
        logits[:, :3] = torch.tensor([0.2, 0.3, 0.5], device="cuda").log()
        info = SimpleNamespace(
            temperatures=torch.ones((batch, 1), device="cuda"),
            top_ks=torch.full((batch,), 3, dtype=torch.int32, device="cuda"),
            top_ps=torch.ones(batch, device="cuda"),
        )
        distribution = prepare_target_distribution(logits, info, depth)
        q = torch.zeros((batch, 1, vocab), device="cuda")
        q[:, :, 0] = 0.7
        q[:, :, 1] = 0.3
        proposal = (torch.rand(batch, device="cuda") >= 0.7).long()
        candidates = torch.stack((torch.zeros_like(proposal), proposal), -1)
        predicts, _, _ = sample_chain(distribution, q, candidates, torch.rand((batch, depth), device="cuda"), torch.rand(batch, device="cuda"))
        histogram = torch.bincount(predicts[:, 0].long(), minlength=vocab).float() / batch
        torch.testing.assert_close(histogram[:3], torch.tensor([0.2, 0.3, 0.5], device="cuda"), atol=0.025, rtol=0)
        self.assertEqual(float(histogram[3:].sum()), 0)

    def test_graph_private_pool_q_survives_target_reuse_and_new_parameters(self):
        batch, vocab = 10, 4096
        logits = torch.randn((batch, vocab), device="cuda")
        temps = torch.ones((batch, 1), device="cuda")
        ks = torch.full((batch,), 8, dtype=torch.int32, device="cuda")
        ps = torch.full((batch,), 0.9, device="cuda")
        sample_truncated_proposal(logits, temps, ks, ps)  # JIT/warmup before capture
        torch.cuda.synchronize()
        pool = torch.cuda.graph_pool_handle()
        proposal_graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(proposal_graph, pool=pool):
            q, _, token = sample_truncated_proposal(logits, temps, ks, ps)
        target_graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(target_graph, pool=pool):
            # Adversarial writer models a later graph reusing private storage.
            q.fill_(123)
        ks.fill_(3)
        ps.fill_(0.5)
        temps.fill_(0.8)
        proposal_graph.replay()
        owned_q = q.clone()  # Same operation as EAGLEDraftCudaGraphRunner.execute.
        expected = owned_q.clone()
        self.assertTrue(bool((owned_q.gather(1, token) > 0).all()))
        target_graph.replay()
        torch.testing.assert_close(owned_q, expected)
        ks.fill_(1)
        proposal_graph.replay()
        new_q = q.clone()
        self.assertTrue(bool(((new_q > 0).sum(-1) == 1).all()))
        torch.testing.assert_close(owned_q, expected)
        self.assertFalse(torch.equal(new_q, expected))


if __name__ == "__main__":
    unittest.main()
