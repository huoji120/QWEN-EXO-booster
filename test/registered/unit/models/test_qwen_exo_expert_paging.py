"""CPU regressions for route paging and checkpoint-global activation scales."""

import unittest
import tempfile
import weakref
from pathlib import Path
from unittest.mock import patch

import torch
from torch import nn

import qwen_exo_booster.expert_paging as paging
from qwen_exo_booster.expert_paging import (
    NVFP4ExpertBank,
    plan_route_groups,
    remap_route_ids,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestExpertPaging(unittest.TestCase):
    def test_prefill_preserves_weighted_mlp_and_original_token_order(self):
        # More unique experts in a batch than residency. Two nonadjacent rows
        # share a route set, but expert order and probabilities differ.
        ids = torch.tensor([[4, 1], [0, 3], [1, 4], [2, 2]], dtype=torch.int32)
        weights = torch.tensor([[0.8, 0.2], [0.15, 0.6], [0.3, 0.7], [0.4, 0.1]])
        hidden = torch.arange(12, dtype=torch.float32).reshape(4, 3) / 8
        up = torch.arange(30, dtype=torch.float32).reshape(5, 2, 3) / 19
        down = torch.arange(30, dtype=torch.float32).reshape(5, 3, 2) / 23
        bias = torch.arange(15, dtype=torch.float32).reshape(5, 3) / 31

        def compute(x, route_ids, probabilities, w1, w2, expert_bias):
            result = torch.zeros_like(x)
            for row in range(x.shape[0]):
                for route in range(route_ids.shape[1]):
                    expert = int(route_ids[row, route])
                    value = w2[expert] @ torch.nn.functional.silu(w1[expert] @ x[row])
                    result[row] += probabilities[row, route] * (value + expert_bias[expert])
            return result

        expected = compute(hidden, ids, weights, up, down, bias)
        actual = torch.empty_like(hidden)
        groups = plan_route_groups(ids.tolist(), capacity=2, num_experts=5)
        self.assertEqual(groups, [([0, 2], (1, 4)), ([1], (0, 3)), ([3], (2,))])
        for rows, experts in groups:
            indices = torch.tensor(rows)
            local_ids = remap_route_ids(ids[indices], experts)
            selected = list(experts)
            actual[indices] = compute(
                hidden[indices], local_ids, weights[indices],
                up[selected], down[selected], bias[selected],
            )
        torch.testing.assert_close(actual, expected)
        # The unnormalized row and duplicate route row must not be renormalized
        # or deduplicated merely because their experts share a resident slot.
        self.assertEqual(local_ids.tolist(), [[0, 0]])

    def test_invalid_native_routes_fail_instead_of_dropping_contributions(self):
        for routes in ([[0, 1, 2]], [[-1, 1]], [[0, 5]]):
            with self.assertRaises(ValueError):
                plan_route_groups(routes, capacity=2, num_experts=5)
        self.assertEqual(plan_route_groups([], capacity=2, num_experts=5), [])

    def test_unselected_experts_set_global_activation_quantization_scale(self):
        layer = nn.Module()
        layer.top_k = 2
        layer.num_experts = 3
        for name in NVFP4ExpertBank._WEIGHTS:
            layer.register_parameter(name, nn.Parameter(torch.ones(3, 2), requires_grad=False))
        layer.register_parameter("w13_input_scale", nn.Parameter(
            torch.tensor([[1.0, 2.0], [3.0, 4.0], [8.0, 7.0]]), requires_grad=False
        ))
        layer.register_parameter("w2_input_scale", nn.Parameter(
            torch.tensor([1.0, 2.0, 16.0]), requires_grad=False
        ))
        bank = NVFP4ExpertBank(layer)
        bank.finalize()
        self.assertEqual(bank.activation_scales, (8.0, 16.0))
        with torch.no_grad():
            layer.w2_input_scale[2] = float("nan")
        with self.assertRaisesRegex(ValueError, "activation scale"):
            bank.finalize()


class TestHostStorageAdmission(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        for name, value in (("memory.max", 120), ("memory.current", 118)):
            (self.root / name).write_text(str(value * 1024**3))
        for replacement in (
            patch.object(paging, "Path", lambda name: self.root / Path(name).name),
            patch.object(paging, "_HOST_BASELINE", None),
            patch.object(paging, "_HOST_RESERVATIONS", weakref.WeakKeyDictionary()),
        ):
            replacement.start()
            self.addCleanup(replacement.stop)

    def write_stats(self, **protected):
        stats = {"file": 116 * 1024**3, **protected}
        (self.root / "memory.stat").write_text(
            "".join(f"{name} {value}\n" for name, value in stats.items())
        )

    def test_clean_cache_is_reclaimable_but_untouched_banks_still_count(self):
        self.write_stats()
        first, second = nn.Module(), nn.Module()
        paging._reserve_host_storage(first, 64 * 1024**3)
        # Both banks are still empty: memory.current has not risen yet.
        with self.assertRaises(MemoryError):
            paging._reserve_host_storage(second, 48 * 1024**3)

    def test_shared_dirty_writeback_and_locked_pages_are_not_capacity(self):
        for name in ("shmem", "file_dirty", "file_writeback", "unevictable"):
            with self.subTest(name=name):
                self.write_stats(**{name: 8 * 1024**3})
                with self.assertRaises(MemoryError):
                    paging._reserve_host_storage(nn.Module(), 108 * 1024**3)

    def test_missing_cache_statistics_keeps_conservative_admission(self):
        with self.assertRaises(MemoryError):
            paging._reserve_host_storage(nn.Module(), 1024**3)


if __name__ == "__main__":
    unittest.main()
