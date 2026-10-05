"""Explicit, private two-case native PLE memorization runner; never auto-starts.

Outputs only structural/numerical JSON. Corpus commands and HTML are never executed.
The W4A16 frozen reference is not bit-exact serving W4A4. Evaluation is overlapping
teacher-forced NLL, not held-out generalization or generated task success.
"""
from __future__ import annotations

import argparse
import collections
import contextlib
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import sqlite3
import sys
import tempfile

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "python"))


class TrainingError(ValueError):
    """Stable structural codes only; the CLI never exposes exception text."""


def require(condition, code):
    if not condition:
        raise TrainingError(code)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    fd, temporary = tempfile.mkstemp(prefix=".atomic-", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, sort_keys=True, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def private_new_run(path):
    path = Path(path).resolve()
    require(not path.is_relative_to(REPO_ROOT), "private_run_outside_repository_required")
    require(not path.exists(), "run_already_exists")
    path.mkdir(parents=True, exist_ok=False)
    return path


def payload_contract(payload, window, identity, max_sequence):
    import torch
    from qwen_exo_booster.native_ple_knowledge import native_ple_row_keys
    require(isinstance(payload, dict) and set(payload) == {
        "input_ids", "labels", "loss_mask", "row_keys", "ngram_history"}, "tensor_fields_invalid")
    require(all(isinstance(t, torch.Tensor) and t.device.type == "cpu" for t in payload.values()),
            "cpu_tensor_payload_required")
    ids, labels, mask, rows, history = (payload[n] for n in (
        "input_ids", "labels", "loss_mask", "row_keys", "ngram_history"))
    require(ids.dtype == labels.dtype == rows.dtype == history.dtype == torch.int64
            and mask.dtype == torch.bool and ids.ndim == 1
            and ids.shape == labels.shape == mask.shape, "tensor_dtype_shape_invalid")
    require(2 <= ids.numel() <= max_sequence <= 32768 and not bool(mask[0]), "window_boundary_invalid")
    require(torch.equal(labels, torch.where(mask, ids, -100)), "label_mask_invalid")
    require(history.shape == (2,) and rows.shape == (ids.numel(), len(identity.hash.head_sizes)),
            "row_history_shape_invalid")
    require(torch.equal(rows, native_ple_row_keys(ids, identity, history)), "causal_rows_invalid")
    require(int(mask.sum()) == window["assistant_targets"] > 0 and ids.numel() == window["tokens"]
            and window["end"] - window["start"] == ids.numel(), "target_count_invalid")
    require(hashlib.sha256(rows.numpy().tobytes()).hexdigest() == window["row_keys_sha256"],
            "row_fingerprint_invalid")
    return payload


def read_window(root, manifest, window, identity):
    import torch
    name = window["file"]
    relative = Path(name)
    require(not relative.is_absolute() and ".." not in relative.parts, "window_path_invalid")
    path = (root / relative).resolve()
    require(path.is_relative_to(root), "window_path_escape")
    descriptor = manifest["files"][name]
    require(path.stat().st_size == descriptor["bytes"] and sha(path) == descriptor["sha256"],
            "window_bytes_changed")
    return payload_contract(torch.load(path, map_location="cpu", weights_only=True, mmap=True),
                            window, identity, manifest["max_sequence"])


def validate_manifest(path, profile, expected_groups):
    from transformers import AutoTokenizer
    from qwen_exo_booster.native_ple_knowledge import NativePLEIdentity
    path, profile = Path(path).resolve(), Path(profile).resolve()
    require(not path.is_relative_to(REPO_ROOT), "private_manifest_outside_repository_required")
    document = json.loads(path.read_text(encoding="utf-8"))
    require(document.get("schema_version") == 1 and document.get("status") == "prepared_not_started"
            and document.get("training_started") is False and document.get("generation_started") is False,
            "manifest_execution_state_invalid")
    require(document.get("memorization_cases") is True, "explicit_memorization_manifest_required")
    groups = document.get("selected_case_groups")
    require(isinstance(groups, list) and len(groups) == len(set(groups)) == 2
            and set(groups) == set(expected_groups), "exact_two_case_groups_required")
    require(all(document.get(field) == 0 for field in (
        "truncated_records", "truncated_targets", "dropped_messages")), "source_loss_not_allowed")
    require(sha(document["source_path"]) == document["source_sha256"], "source_bytes_changed")
    require(sha(path.parent / "split-map.json") == document["split_map_sha256"], "split_map_changed")
    require(sha(Path(__file__).with_name("prepare_native_ple_data.py")) == document["code_sha256"],
            "prepared_implementation_changed")
    identity = NativePLEIdentity.from_profile(profile)
    require(identity.to_dict() == document["native_identity"]
            and identity.fingerprint() == document["native_identity_sha256"], "native_identity_changed")
    files = document["tokenizer"]["files"]
    require(all(sha(profile / name) == value for name, value in files.items()), "tokenizer_bytes_changed")
    require(hashlib.sha256(canonical(files)).hexdigest() == document["tokenizer"]["files_sha256"],
            "tokenizer_fingerprint_invalid")
    tokenizer = AutoTokenizer.from_pretrained(str(profile), local_files_only=True, trust_remote_code=False)
    require(hashlib.sha256(tokenizer.chat_template.encode()).hexdigest()
            == document["tokenizer"]["chat_template_sha256"], "tokenizer_template_changed")
    require(tokenizer.convert_tokens_to_ids("<|endoftext|>") == identity.hash.eos_id,
            "tokenizer_eos_mismatch")
    windows = document["windows"]
    require(windows and len({w["file"] for w in windows}) == len(windows), "window_inventory_invalid")
    require(set(w["case_group"] for w in windows) == set(groups)
            and all(w["split"] == "train" for w in windows), "memorization_window_groups_invalid")
    require(document["splits"].get("heldout", {}).get("assistant_targets", 0) == 0,
            "memorization_must_not_claim_heldout")
    per_record = collections.Counter()
    counts = collections.Counter()
    record_windows = collections.Counter()
    for window in windows:
        read_window(path.parent, document, window, identity)
        per_record[window["source_line"]] += window["assistant_targets"]
        counts[window["case_group"]] += window["assistant_targets"]
        record_windows[window["source_line"]] += 1
    require(len(document["records"]) == document["source_records"], "records_count_invalid")
    require(len({r["source_line"] for r in document["records"]}) == len(document["records"]),
            "duplicate_source_records")
    require(set(r["case_group"] for r in document["records"]) == set(groups), "record_groups_invalid")
    require(all(per_record[r["source_line"]] == r["assistant_targets"]
                and record_windows[r["source_line"]] == r["windows"] for r in document["records"]),
            "record_target_coverage_invalid")
    require(set(per_record) <= {r["source_line"] for r in document["records"]}, "unknown_window_record")
    record_groups = {r["source_line"]: r["case_group"] for r in document["records"]}
    require(all(w["case_group"] == record_groups[w["source_line"]] for w in windows),
            "record_window_group_mismatch")
    total = sum(counts.values())
    require(total == document["splits"]["train"]["assistant_targets"], "epoch_target_coverage_invalid")
    return document, identity, counts


def source_row_counts(payload):
    """Count keys at source t predicting supervised labels[t+1], never target t+1."""
    import torch
    selected = payload["row_keys"][:-1][payload["labels"][1:] != -100]
    return torch.unique(selected.reshape(-1), sorted=True, return_counts=True)


def scan_rows(root, document, identity, database, min_count, max_rows):
    import torch
    require(min_count >= 2 and max_rows > 0, "row_budget_invalid")
    connection = sqlite3.connect(database)
    try:
        connection.execute("CREATE TABLE counts(row_id INTEGER, case_group TEXT, occurrences INTEGER, "
                           "PRIMARY KEY(row_id,case_group)) WITHOUT ROWID")
        for window in document["windows"]:
            keys, counts = source_row_counts(read_window(root, document, window, identity))
            connection.executemany("INSERT INTO counts VALUES(?,?,?) ON CONFLICT(row_id,case_group) "
                                   "DO UPDATE SET occurrences=occurrences+excluded.occurrences",
                                   ((int(k), window["case_group"], int(c)) for k, c in zip(keys, counts)))
            connection.commit()
        selected_count = connection.execute("SELECT COUNT(*) FROM (SELECT row_id FROM counts "
            "GROUP BY row_id HAVING SUM(occurrences)>=?)", (min_count,)).fetchone()[0]
        require(0 < selected_count <= max_rows, "selected_rows_exceed_explicit_budget_or_empty")
        # Only the explicitly budgeted, selected row list enters RAM; observations
        # and per-case provenance stay in a disk-backed SQLite aggregate.
        rows = torch.tensor([r[0] for r in connection.execute("SELECT row_id FROM counts GROUP BY row_id "
            "HAVING SUM(occurrences)>=? ORDER BY row_id", (min_count,))], dtype=torch.int64)
        return rows
    finally:
        connection.close()


def original_file_identity(profile):
    """Full bytes, including original MTP/vision and base PLE, not headers alone."""
    profile = Path(profile).resolve()
    index = json.loads((profile / "model.safetensors.index.json").read_text())
    marker = json.loads((profile / "native-ple.json").read_text())
    table_root = Path(marker["root"]).resolve()
    paths = {profile / name for name in index["weight_map"].values()}
    paths.update(profile.glob("*.safetensors"))
    paths.update(profile / name for name in ("config.json", "model.safetensors.index.json", "native-ple.json"))
    verified = {}
    for descriptor in marker["shards"]:
        path = (table_root / descriptor["file"]).resolve()
        require(path.is_relative_to(table_root), "original_table_path_escape")
        require(path.stat().st_size == descriptor["size"], "original_table_size_changed")
        actual = sha(path)
        require(actual == descriptor["sha256"], "original_table_hash_changed")
        paths.add(path)
        verified[path] = {"sha256": actual, "bytes": path.stat().st_size}
    return {str(p.resolve()): verified[p] if p in verified else
            {"sha256": sha(p), "bytes": p.stat().st_size} for p in sorted(paths)}


def sparse_gradient_gate(model, delta, clip_norm=None, *, require_nonzero=True):
    import torch
    require(all(p.grad is None for p in model.parameters() if p is not delta.weight),
            "frozen_parameter_gradient")
    grad = delta.weight.grad
    require(grad is not None and grad.is_sparse, "sparse_delta_gradient_required")
    grad = grad.coalesce()
    values = grad.values()
    require(bool(torch.isfinite(values).all()), "nonfinite_sparse_gradient")
    norm = values.double().norm()
    require(bool(torch.isfinite(norm)) and (not require_nonzero or float(norm) > 0),
            "zero_or_nonfinite_gradient_norm")
    if clip_norm is not None:
        require(math.isfinite(clip_norm) and clip_norm > 0, "clip_norm_invalid")
        values.mul_(min(1.0, clip_norm / (float(norm) + 1e-12)))
    delta.weight.grad = grad
    return {"gradient_norm": float(norm), "gradient_rows": int(grad.indices().shape[1]),
            "gradient_abs_max": float(values.abs().max()) if values.numel() else 0.0, "original_gradients": 0}


def loss_for(model, payload, device, chunk_size, mode="real"):
    from qwen_exo_booster.native_ple_training_backend import chunked_causal_cross_entropy
    model.set_ple_window(payload["ngram_history"].to(device).unsqueeze(0), mode=mode, shuffle_seed=1729)
    hidden = model.forward_hidden(payload["input_ids"].to(device).unsqueeze(0))
    return chunked_causal_cross_entropy(hidden, model.lm_head.weight,
                                        payload["labels"].to(device).unsqueeze(0), chunk_size)


def short_source_gate(root, document, identity, rows, limit):
    import torch
    require(limit >= 2, "gate_sequence_limit_invalid")
    for window in document["windows"]:
        payload = read_window(root, document, window, identity)
        keys = payload["row_keys"][:-1]
        indices = torch.searchsorted(rows, keys).clamp(max=rows.numel() - 1)
        hit = (rows[indices] == keys).any(dim=-1) & (payload["labels"][1:] != -100)
        positions = hit.nonzero().flatten()
        if not positions.numel():
            continue
        end = int(positions[0]) + 2
        start = max(0, end - limit)
        combined = torch.cat((payload["ngram_history"], payload["input_ids"]))
        gate = {name: payload[name][start:end].clone() for name in (
            "input_ids", "labels", "loss_mask", "row_keys")}
        gate["ngram_history"] = combined[start:start + 2].clone()
        gate["labels"][0] = -100
        gate["loss_mask"][0] = False
        return gate, {"source_line": window["source_line"], "start": window["start"] + start,
                      "end": window["start"] + end, "window_sha256": document["files"][window["file"]]["sha256"],
                      "diagnostic_subwindow_only": True, "optimizer_steps": 0}
    raise TrainingError("no_eligible_supervised_gate_rows")


def save_progress(run, delta, optimizer, state):
    import torch
    fd, temporary = tempfile.mkstemp(prefix=".checkpoint-", dir=run)
    try:
        with os.fdopen(fd, "wb") as handle:
            torch.save({"schema": 1, "row_ids": delta.row_ids.detach().cpu(),
                        "values": delta.weight.detach().cpu(), "optimizer": optimizer.state_dict(),
                        "state": dict(state), "rng_cpu": torch.get_rng_state(),
                        "rng_cuda": torch.cuda.get_rng_state_all()}, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, run / "progress.pt")
        atomic_json(run / "progress.json", {**state, "checkpoint_sha256": sha(run / "progress.pt")})
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def evaluate(model, root, document, identity, device, chunk_size, abort):
    import torch
    results = {}
    model.eval()
    with torch.no_grad():
        for mode in ("off", "real", "shuffled"):
            totals = collections.defaultdict(lambda: [0.0, 0])
            for window in document["windows"]:
                require(not abort[0], "evaluation_aborted")
                payload = read_window(root, document, window, identity)
                loss = loss_for(model, payload, device, chunk_size, mode)
                require(bool(torch.isfinite(loss)), "nonfinite_evaluation_loss")
                count = window["assistant_targets"]
                totals[window["case_group"]][0] += float(loss) * count
                totals[window["case_group"]][1] += count
            results[mode] = {group: {"nll": total / count, "targets": count}
                             for group, (total, count) in totals.items()}
    return {"modes": results, "training_task_overlap": True, "heldout": False,
            "generated_task_success_measured": False, "matched_all_training_windows": True,
            "shuffle_seed": 1729, "off_disables_delta_only": True}


def execute(args):
    # This guard precedes CUDA checks, model loads, run creation and optimizer construction.
    require(args.start_training, "explicit_start_training_required")
    require(args.epochs == 1 and args.batch_size == 1 and args.lr == 1e-4,
            "approved_one_epoch_batch_one_learning_rate_required")
    require(args.ce_chunk_size > 0 and math.isfinite(args.min_free_gib) and args.min_free_gib > 0
            and math.isfinite(args.reserve_gib) and args.reserve_gib >= 16
            and args.cpu_threads > 0 and math.isfinite(args.clip_norm) and args.clip_norm > 0,
            "gpu_head_budget_invalid")
    import torch
    from qwen_exo_booster.native_ple_knowledge import SparseNativePLEDelta
    from qwen_exo_booster.native_ple_training_backend import load_native_ple_training_model
    torch.set_num_threads(args.cpu_threads)
    document, identity, group_targets = validate_manifest(args.manifest, args.profile, args.case_group)
    root = Path(args.manifest).resolve().parent
    require(torch.cuda.is_available(), "cuda_required_parent_controls_service")
    free, total = torch.cuda.mem_get_info(args.device)
    require(free >= args.min_free_gib * 1024**3, "gpu_free_budget_insufficient")
    run = private_new_run(args.output)
    rows = scan_rows(root, document, identity, run / "row-provenance.sqlite", args.min_count, args.max_rows)
    # F32 delta plus SparseAdam's two dense moment arrays, row ids, and head workspace.
    delta_bytes = rows.numel() * identity.head_dim * 4
    head_workspace = args.ce_chunk_size * identity.hash.vocab_size * 12
    require(free >= 3 * delta_bytes + head_workspace + args.reserve_gib * 1024**3,
            "delta_optimizer_activation_budget_insufficient")
    originals = original_file_identity(args.profile)
    atomic_json(run / "original-files.json", originals)
    seed = args.seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    # Native CUDA SDPA must support the exact QSA mask; no quadratic fallback.
    torch.backends.cuda.enable_math_sdp(False)
    delta = SparseNativePLEDelta(rows, identity, device=args.device)
    model = None
    optimizer = None
    abort = [False]
    handlers = {}
    for name in ("SIGINT", "SIGTERM"):
        number = getattr(signal, name, None)
        if number is not None:
            handlers[number] = signal.signal(number, lambda *_: abort.__setitem__(0, True))
    state = {"schema": 1, "manifest_sha256": sha(args.manifest), "source_sha256": document["source_sha256"],
             "native_identity": identity.to_dict(), "native_identity_sha256": identity.fingerprint(),
             "tokenizer_fingerprint": document["tokenizer"]["files_sha256"],
             "runner_sha256": sha(__file__), "original_files_sha256": sha(run / "original-files.json"),
             "implementation_sha256": {name: sha(REPO_ROOT / "python" / "qwen_exo_booster" / name)
                 for name in ("native_ple_training_backend.py", "native_ple_checkpoint.py", "native_ple_knowledge.py")},
             "attention_implementation": "native_sdpa_cuda_math_fallback_disabled",
             "rows_provenance_sha256": sha(run / "row-provenance.sqlite"),
             "row_count": rows.numel(), "min_count": args.min_count, "seed": seed,
             "lr": args.lr, "batch_size": 1, "epochs_complete": 0, "optimizer_steps": 0,
             "targets_seen": 0, "windows_complete": 0, "epoch_targets": sum(group_targets.values()),
             "group_targets": dict(group_targets), "group_targets_seen": {g: 0 for g in group_targets},
             "gradient_gate_passed": False, "training_task_overlap": True,
             "weight_reference": "frozen_checkpoint_W4A16_not_serving_W4A4", "automatic_start": False}
    atomic_json(run / "provenance.json", state)
    try:
        model = load_native_ple_training_model(args.profile, delta, device=args.device,
                                               dtype=torch.bfloat16, gradient_checkpointing=True)
        optimizer = torch.optim.SparseAdam([delta.weight], lr=args.lr)
        require(not abort[0], "training_aborted_before_gate")
        gate, gate_source = short_source_gate(root, document, identity, rows, args.gate_tokens)
        gate_loss = loss_for(model, gate, args.device, args.ce_chunk_size)
        require(bool(torch.isfinite(gate_loss)), "nonfinite_first_native_loss")
        gate_loss.backward()
        gate_metrics = sparse_gradient_gate(model, delta)
        require(bool((delta.weight == 0).all()), "pre_optimizer_delta_mutation")
        gate_metrics.update(gate_source)
        gate_metrics["loss"] = float(gate_loss.detach())
        atomic_json(run / "first-gradient-gate.json", gate_metrics)
        del gate_loss
        optimizer.zero_grad(set_to_none=True)
        state["gradient_gate_passed"] = True
        save_progress(run, delta, optimizer, state)
        for window in document["windows"]:
            if abort[0]:
                break
            payload = read_window(root, document, window, identity)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_for(model, payload, args.device, args.ce_chunk_size)
            require(bool(torch.isfinite(loss)), "nonfinite_training_loss")
            gradient_metrics = {"gradient_norm": 0.0}
            if loss.requires_grad:
                loss.backward()
                gradient_metrics = sparse_gradient_gate(model, delta, args.clip_norm, require_nonzero=False)
            if abort[0]:
                break
            if gradient_metrics["gradient_norm"] > 0:
                optimizer.step()
                require(bool(torch.isfinite(delta.weight).all()), "nonfinite_delta_after_step")
                state["optimizer_steps"] += 1
            else:
                state["zero_update_windows"] = state.get("zero_update_windows", 0) + 1
            state["windows_complete"] += 1
            state["targets_seen"] += window["assistant_targets"]
            state["group_targets_seen"][window["case_group"]] += window["assistant_targets"]
            state["last_loss"] = float(loss.detach())
            state["last_gradient_norm"] = gradient_metrics["gradient_norm"]
            del loss
            optimizer.zero_grad(set_to_none=True)
            save_progress(run, delta, optimizer, state)
        if abort[0]:
            state["aborted"] = True
            save_progress(run, delta, optimizer, state)
            return {"exit_code": 130, "targets_seen": state["targets_seen"], "epochs_complete": 0}
        require(state["targets_seen"] == state["epoch_targets"]
                and state["windows_complete"] == len(document["windows"])
                and state["group_targets_seen"] == state["group_targets"], "epoch_coverage_incomplete")
        state["epochs_complete"] = 1
        save_progress(run, delta, optimizer, state)
        # Evaluate precisely the BF16 values export will persist, rather than
        # silently claiming F32-table metrics for a rounded artifact.
        with torch.no_grad():
            delta.weight.copy_(delta.weight.to(torch.bfloat16))
        state["evaluation_uses_export_bf16_values"] = True
        evaluation = evaluate(model, root, document, identity, args.device, args.ce_chunk_size, abort)
        require(original_file_identity(args.profile) == originals, "original_checkpoint_bytes_changed")
        atomic_json(run / "evaluation.json", evaluation)
        state["evaluation_sha256"] = sha(run / "evaluation.json")
        state["original_files_reverified"] = True
        artifact = delta.export(run / "artifact", training=state,
                                tokenizer_fingerprint=document["tokenizer"]["files_sha256"])
        atomic_json(run / "result.json", {**state, "artifact_sha256": sha(artifact)})
        save_progress(run, delta, optimizer, state)
        return {"exit_code": 0, "epochs_complete": 1, "targets_seen": state["targets_seen"],
                "optimizer_steps": state["optimizer_steps"], "row_count": rows.numel(),
                "gate_loss": gate_metrics["loss"], "evaluation_sha256": sha(run / "evaluation.json"),
                "artifact_sha256": sha(artifact), "matched_nll": evaluation["modes"]}
    except BaseException as error:
        state["aborted"] = bool(abort[0] or isinstance(error, KeyboardInterrupt))
        state["failed"] = not state["aborted"]
        if optimizer is not None:
            save_progress(run, delta, optimizer, state)
        atomic_json(run / "failure.json", {"exception_type_sha256": hashlib.sha256(type(error).__name__.encode()).hexdigest(),
                    "targets_seen": state["targets_seen"], "epochs_complete": state["epochs_complete"],
                    "aborted": state["aborted"], "failed": state["failed"]})
        if state["aborted"]:
            return {"exit_code": 130, "targets_seen": state["targets_seen"],
                    "epochs_complete": state["epochs_complete"], "aborted": True}
        raise
    finally:
        for number, previous in handlers.items():
            signal.signal(number, previous)
        if model is not None:
            model.close()


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--start-training", action="store_true")
    result.add_argument("--manifest", type=Path, required=True)
    result.add_argument("--profile", type=Path, required=True)
    result.add_argument("--output", type=Path, required=True)
    result.add_argument("--case-group", action="append", required=True)
    result.add_argument("--device", default="cuda:0")
    result.add_argument("--epochs", type=int, default=1)
    result.add_argument("--batch-size", type=int, default=1)
    result.add_argument("--lr", type=float, default=1e-4)
    result.add_argument("--seed", type=int, default=1729)
    result.add_argument("--min-count", type=int, default=2)
    result.add_argument("--max-rows", type=int, default=2000000)
    result.add_argument("--min-free-gib", type=float, default=64)
    result.add_argument("--reserve-gib", type=float, default=24)
    result.add_argument("--ce-chunk-size", type=int, default=128)
    result.add_argument("--gate-tokens", type=int, default=512)
    result.add_argument("--clip-norm", type=float, default=1.0)
    result.add_argument("--cpu-threads", type=int, default=2)
    return result


def main(argv=None):
    original_output = sys.stdout
    # Suppress third-party warnings/logs. Never echo traceback or exception bodies:
    # a malformed private data value may otherwise appear in dependency errors.
    with open(os.devnull, "w") as sink, contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
        try:
            result = execute(parser().parse_args(argv))
        except BaseException as error:
            code = 130 if isinstance(error, KeyboardInterrupt) else 1
            result = {"exit_code": code, "exception_type_sha256": hashlib.sha256(type(error).__name__.encode()).hexdigest()}
            frames = []
            trace = error.__traceback__
            while trace is not None:
                frames.append({"file": Path(trace.tb_frame.f_code.co_filename).name,
                               "function": trace.tb_frame.f_code.co_name, "line": trace.tb_lineno})
                trace = trace.tb_next
            result["stack_locations"] = frames
            if isinstance(error, TrainingError):
                result["error_code_sha256"] = hashlib.sha256(str(error).encode()).hexdigest()
    print(json.dumps(result, sort_keys=True), file=original_output)
    return result["exit_code"]


if __name__ == "__main__":
    raise SystemExit(main())
