"""CPU causal-objective and ingestion guards; no model/GPU/optimizer execution."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch.nn import functional as F

from qwen_exo_booster.native_ple_training_backend import chunked_causal_cross_entropy


spec = importlib.util.spec_from_file_location("native_ple_runner", Path(__file__).resolve().parents[3]
    / "scripts" / "qwen_exo" / "train_native_ple_delta.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


@pytest.mark.parametrize("dtype", [torch.float64, torch.float32, torch.bfloat16])
@pytest.mark.parametrize("chunk", [1, 3, 128])
def test_chunked_causal_ce_matches_loss_and_nonunit_gradient(dtype, chunk):
    generator = torch.Generator().manual_seed(732)
    hidden = torch.randn(2, 7, 5, generator=generator, dtype=torch.float64).to(dtype).requires_grad_()
    head = torch.randn(11, 5, generator=generator, dtype=torch.float64).to(dtype)
    # Includes masked interior positions, distinct batch targets, first-label
    # values which must never be supervised, and the final causal target.
    labels = torch.tensor([[10, 1, -100, 4, 2, -100, 8], [9, -100, 7, 6, -100, 0, 3]])
    reference_hidden = hidden.detach().clone().requires_grad_()
    logits = F.linear(reference_hidden[:, :-1], head)
    compute = logits if dtype == torch.float64 else logits.float()
    expected = F.cross_entropy(compute.reshape(-1, 11), labels[:, 1:].reshape(-1), ignore_index=-100)
    actual = chunked_causal_cross_entropy(hidden, head, labels, chunk)
    (actual * 2.75).backward()
    (expected * 2.75).backward()
    tolerance = 2e-2 if dtype == torch.bfloat16 else 2e-6
    torch.testing.assert_close(actual, expected, rtol=tolerance, atol=tolerance)
    torch.testing.assert_close(hidden.grad, reference_hidden.grad, rtol=tolerance, atol=tolerance)
    assert torch.equal(hidden.grad[:, -1], torch.zeros_like(hidden.grad[:, -1]))
    assert head.grad is None


def test_causal_ce_rejects_no_next_token_targets_and_outside_vocabulary():
    hidden = torch.zeros(1, 3, 2, requires_grad=True)
    head = torch.zeros(4, 2)
    with pytest.raises(ValueError, match="no_targets"):
        chunked_causal_cross_entropy(hidden, head, torch.tensor([[2, -100, -100]]))
    with pytest.raises(ValueError, match="label_invalid"):
        chunked_causal_cross_entropy(hidden, head, torch.tensor([[-100, 4, 1]]))
    with pytest.raises(ValueError, match="contract_invalid"):
        chunked_causal_cross_entropy(hidden, head.requires_grad_(), torch.tensor([[-100, 1, 2]]))


def test_source_row_selection_uses_predictor_not_target_position():
    payload = {"row_keys": torch.tensor([[10, 11], [20, 21], [30, 31], [40, 41], [50, 51]]),
               "labels": torch.tensor([-100, 1, -100, 3, 4])}
    keys, counts = runner.source_row_counts(payload)
    assert keys.tolist() == [10, 11, 30, 31, 40, 41]
    assert counts.tolist() == [1, 1, 1, 1, 1, 1]


def test_explicit_start_guard_precedes_all_data_and_cuda_access():
    with pytest.raises(runner.TrainingError, match="explicit_start_training_required"):
        runner.execute(SimpleNamespace(start_training=False))


def test_approved_training_settings_guard_precedes_data_access():
    with pytest.raises(runner.TrainingError, match="approved_one_epoch"):
        runner.execute(SimpleNamespace(start_training=True, epochs=2, batch_size=1, lr=1e-4))


def test_sparse_clipping_coalesces_duplicate_indices_and_rejects_nonfinite():
    delta = torch.nn.Module()
    delta.weight = torch.nn.Parameter(torch.zeros(3, 2))
    model = torch.nn.Module()
    model.delta = delta
    model.frozen = torch.nn.Parameter(torch.ones(2), requires_grad=False)
    delta.weight.grad = torch.sparse_coo_tensor(torch.tensor([[1, 1, 2]]),
        torch.tensor([[3., 0.], [0., 4.], [0., 12.]]), (3, 2))
    receipt = runner.sparse_gradient_gate(model, delta, clip_norm=6.5)
    assert receipt["gradient_norm"] == 13.0
    torch.testing.assert_close(delta.weight.grad.to_dense(), torch.tensor([[0., 0.], [1.5, 2.], [0., 6.]]))
    delta.weight.grad = torch.sparse_coo_tensor(torch.tensor([[0]]), torch.tensor([[float("nan"), 0.]]), (3, 2))
    with pytest.raises(runner.TrainingError, match="nonfinite_sparse_gradient"):
        runner.sparse_gradient_gate(model, delta)
    delta.weight.grad = torch.sparse_coo_tensor(torch.tensor([[0]]), torch.tensor([[1., 0.]]), (3, 2))
    model.frozen.grad = torch.ones_like(model.frozen)
    with pytest.raises(runner.TrainingError, match="frozen_parameter_gradient"):
        runner.sparse_gradient_gate(model, delta)


def test_payload_contract_rejects_changed_history_rows_and_label_mask():
    import hashlib
    from qwen_exo_booster.engram import EngramHashSpec
    from qwen_exo_booster.native_ple_knowledge import native_ple_row_keys
    identity = SimpleNamespace(hash=EngramHashSpec(vocab_size=64, eos_id=63, ngram_size=3,
        heads_per_order=2, multipliers=(3, 5, 7), head_sizes=(17, 19, 23, 29),
        head_offsets=(0, 17, 36, 59)))
    ids, history = torch.tensor([1, 2, 3, 4]), torch.tensor([11, 12])
    rows = native_ple_row_keys(ids, identity, history)
    payload = {"input_ids": ids, "labels": torch.tensor([-100, 2, -100, 4]),
        "loss_mask": torch.tensor([False, True, False, True]), "row_keys": rows,
        "ngram_history": history}
    window = {"assistant_targets": 2, "tokens": 4, "start": 0, "end": 4,
        "row_keys_sha256": hashlib.sha256(rows.numpy().tobytes()).hexdigest()}
    runner.payload_contract(payload, window, identity, 32768)
    changed = {**payload, "ngram_history": torch.tensor([12, 11])}
    with pytest.raises(runner.TrainingError, match="causal_rows_invalid"):
        runner.payload_contract(changed, window, identity, 32768)
    changed = {**payload, "labels": torch.tensor([-100, 2, 3, 4])}
    with pytest.raises(runner.TrainingError, match="label_mask_invalid"):
        runner.payload_contract(changed, window, identity, 32768)


def test_disk_counts_require_repetition_and_retain_case_provenance(tmp_path, monkeypatch):
    import sqlite3
    windows = [{"case_group": "case_1", "number": 1}, {"case_group": "case_2", "number": 2}]
    first = {"row_keys": torch.tensor([[10, 11], [20, 21], [30, 31]]),
             "labels": torch.tensor([-100, 1, -100])}
    second = {"row_keys": torch.tensor([[10, 22], [40, 41]]), "labels": torch.tensor([-100, 2])}
    monkeypatch.setattr(runner, "read_window", lambda root, document, window, identity:
        first if window["number"] == 1 else second)
    database = tmp_path / "counts.sqlite"
    rows = runner.scan_rows(tmp_path, {"windows": windows}, None, database, 2, 10)
    assert rows.tolist() == [10]
    with sqlite3.connect(database) as connection:
        assert connection.execute("SELECT case_group,occurrences FROM counts WHERE row_id=10 "
                                  "ORDER BY case_group").fetchall() == [("case_1", 1), ("case_2", 1)]
    with pytest.raises(runner.TrainingError, match="explicit_budget_or_empty"):
        runner.scan_rows(tmp_path, {"windows": windows}, None, tmp_path / "empty.sqlite", 3, 10)
