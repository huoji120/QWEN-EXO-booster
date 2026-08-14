import hashlib
import json
import importlib.util
import sys
import types
from pathlib import Path

import pytest
import torch

from qwen_exo_booster.activation_training import (
    COMBINED_EDITOR_NAME,
    ActivationTrainingError,
    ActivationTrainingStore,
    run_pending_activation_training,
)


def _trajectory(path: Path, label: str = "sample") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "session": {
                    "messages": [
                        {"role": "system", "content": f"system {label}"},
                        {"role": "user", "content": f"start {label}"},
                        {
                            "role": "assistant",
                            "content": f"first assistant action with enough detail for {label}",
                        },
                        {"role": "user", "content": f"continue {label}"},
                        {
                            "role": "assistant",
                            "content": f"second assistant action with enough detail for {label}",
                        },
                    ]
                }
            }
        ),
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _runner(
    metrics: dict[str, float],
    *,
    source_metrics: dict[str, dict[str, float]] | None = None,
    commands: list[list[str]] | None = None,
):
    def run(command: list[str], log_path: Path, environment: dict[str, str]) -> int:
        if commands is not None:
            commands.append(command)
        editor_path = Path(command[command.index("--editor-out") + 1])
        report_path = Path(command[command.index("--output") + 1])
        trajectory_paths = [
            Path(command[index + 1])
            for index, value in enumerate(command)
            if value == "--trajectory"
        ]
        sources = [
            {"name": path.stem, "sha256": _sha256(path)} for path in trajectory_paths
        ]
        editor_path.parent.mkdir(parents=True, exist_ok=True)
        projection = torch.eye(8)[:8]
        torch.save(
            {
                "schema": 1,
                "layer": 47,
                "rank": 8,
                "window": 16,
                "hidden_size": 8,
                "sources": sources,
                "state_dict": {
                    "projection": projection,
                    "transform": projection.clone(),
                    "bias": torch.zeros(8),
                },
            },
            editor_path,
        )
        report_sources = []
        for source in sources:
            values = (source_metrics or {}).get(source["name"], metrics)
            report_sources.append({"name": source["name"], **values})
        report_path.write_text(
            json.dumps(
                {
                    **metrics,
                    "sources": report_sources,
                }
            ),
            encoding="utf-8",
        )
        log_path.write_text("trained\n", encoding="utf-8")
        assert environment["CUDA_VISIBLE_DEVICES"] == "0"
        return 0

    return run


def test_environment_store_uses_shared_data_root(tmp_path: Path, monkeypatch):
    data = tmp_path / "data"
    profile = data / "model-profiles" / ("f" * 64)
    monkeypatch.setenv("QWEN_EXO_SERVICE_CONFIG", str(data / "service-config.json"))
    monkeypatch.setenv("QWEN_EXO_ACTIVE_MODEL_PROFILE", str(profile))

    store = ActivationTrainingStore.from_environment()

    assert store.data_root == data.resolve()
    assert store.root == data.resolve() / "activation-training"


def test_model_profile_state_writes_editor_but_reads_shared_trajectory(tmp_path: Path):
    data = tmp_path / "data"
    _trajectory(data / "trajectories" / "first.json", "first")
    state = data / "model-profiles" / ("f" * 64) / "state-cuda"
    store = ActivationTrainingStore(data)

    job = store.enqueue(["first"], state_directory=state)
    sources, _, _, _, target = store.paths(job)

    assert sources == [data / "trajectories" / "first.json"]
    assert target == state / "activation-editors" / f"{COMBINED_EDITOR_NAME}.editor.pt"


def test_selection_and_enqueue_preserve_multiple_trajectory_boundaries(tmp_path: Path):
    _trajectory(tmp_path / "trajectories" / "first.json", "first")
    _trajectory(tmp_path / "trajectories" / "second.json", "second")
    state = tmp_path / "state"
    store = ActivationTrainingStore(tmp_path)

    selection = store.set_selection(["first", "second"])
    job = store.enqueue(selection["trajectories"], state_directory=state)

    assert store.selection()["trajectories"] == ["first", "second"]
    assert job["status"] == "queued"
    assert job["editor"] == COMBINED_EDITOR_NAME
    assert job["trajectories"] == ["first", "second"]
    assert [source["name"] for source in job["sources"]] == ["first", "second"]
    assert job["message_count"] == 10
    assert job["sample_count"] == 4
    assert job["config"] == {
        "layer": 47,
        "rank": 8,
        "window": 16,
        "epochs": 0.25,
        "learning_rate": 5e-4,
        "max_context_tokens": 512,
        "max_target_tokens": 2048,
        "max_sequence_tokens": 2560,
    }
    assert "state_directory" not in store.public_status()["job"]

    with pytest.raises(ActivationTrainingError, match="正在训练"):
        store.enqueue(["first", "second"], state_directory=state)


def test_successful_joint_training_publishes_and_applies_one_editor(tmp_path: Path):
    first = tmp_path / "trajectories" / "first.json"
    second = tmp_path / "trajectories" / "second.json"
    _trajectory(first, "first")
    _trajectory(second, "second")
    state = tmp_path / "state"
    old_target = state / "activation-editors" / f"{COMBINED_EDITOR_NAME}.editor.pt"
    old_target.parent.mkdir(parents=True)
    old_target.write_bytes(b"old")
    store = ActivationTrainingStore(tmp_path)
    store.enqueue(["first", "second"], state_directory=state)
    commands: list[list[str]] = []

    result = run_pending_activation_training(
        store,
        runner=_runner(
            {"baseline_nll": 2.0, "random_editor_nll": 2.1, "trained_editor_nll": 1.5},
            commands=commands,
        ),
        training_script=tmp_path / "train.py",
        model_path=tmp_path / "model",
    )

    assert result is not None
    assert result["status"] == "succeeded"
    assert result["metrics"]["trained_editor_nll"] == 1.5
    assert old_target.is_file() and old_target.read_bytes() != b"old"
    command = commands[0]
    assert command.count("--trajectory") == 2
    assert command[command.index("--trajectory") + 1] == str(first)
    second_index = command.index("--trajectory", command.index("--trajectory") + 1)
    assert command[second_index + 1] == str(second)
    assert command[command.index("--max-context-tokens") + 1] == "512"
    assert command[command.index("--max-target-tokens") + 1] == "2048"
    assert command[command.index("--max-sequence-tokens") + 1] == "2560"
    assert command[command.index("--lr") + 1] == "0.0005"
    active = json.loads((old_target.parent / "active.json").read_text("utf-8"))
    assert active["editor"] == COMBINED_EDITOR_NAME
    assert active["sources"] == ["first", "second"]
    assert "strength" not in active
    assert "editors" not in active
    assert active["validation_schema"] == 1


def test_source_quality_regression_is_reported_without_blocking_publish(tmp_path: Path):
    _trajectory(tmp_path / "trajectories" / "first.json", "first")
    _trajectory(tmp_path / "trajectories" / "second.json", "second")
    state = tmp_path / "state"
    store = ActivationTrainingStore(tmp_path)
    store.enqueue(["first", "second"], state_directory=state)

    result = run_pending_activation_training(
        store,
        runner=_runner(
            {"baseline_nll": 2.0, "random_editor_nll": 2.1, "trained_editor_nll": 1.5},
            source_metrics={
                "second": {
                    "baseline_nll": 1.0,
                    "random_editor_nll": 1.1,
                    "trained_editor_nll": 1.2,
                }
            },
        ),
    )

    assert result is not None
    assert result["status"] == "succeeded"
    assert result["metrics"]["sources"][1]["trained_editor_nll"] == 1.2
    assert (
        state / "activation-editors" / f"{COMBINED_EDITOR_NAME}.editor.pt"
    ).is_file()
    assert (state / "activation-editors" / "active.json").is_file()


def test_changed_source_is_rejected_before_runner(tmp_path: Path):
    first = tmp_path / "trajectories" / "first.json"
    second = tmp_path / "trajectories" / "second.json"
    _trajectory(first, "first")
    _trajectory(second, "second")
    state = tmp_path / "state"
    store = ActivationTrainingStore(tmp_path)
    store.enqueue(["first", "second"], state_directory=state)
    second.write_text(second.read_text("utf-8") + "\n", encoding="utf-8")
    called = False

    def runner(*_args):
        nonlocal called
        called = True
        return 0

    result = run_pending_activation_training(store, runner=runner)

    assert result is not None
    assert result["status"] == "failed"
    assert "second" in result["error"]
    assert "发生变化" in result["error"]
    assert called is False


def test_selection_tracks_rename_and_delete(tmp_path: Path):
    _trajectory(tmp_path / "trajectories" / "first.json", "first")
    _trajectory(tmp_path / "trajectories" / "renamed.json", "renamed")
    store = ActivationTrainingStore(tmp_path)
    store.set_selection(["first"])

    assert store.rename_selection("first", "renamed") is True
    assert store.selection()["trajectories"] == ["renamed"]
    assert store.remove_selection("renamed") is True
    assert store.selection()["trajectories"] == []


def test_training_script_keeps_source_contexts_isolated(tmp_path: Path, monkeypatch):
    fake_transformers = types.ModuleType("transformers")
    fake_transformers.AutoModelForCausalLM = object
    fake_transformers.AutoTokenizer = object
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    script_path = (
        Path(__file__).resolve().parents[3]
        / "scripts"
        / "qwen_exo"
        / "train_activation_editor.py"
    )
    spec = importlib.util.spec_from_file_location("joint_editor_training", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    class FakeTokenizer:
        @staticmethod
        def apply_chat_template(messages, **_kwargs):
            marker = 101 if "first" in messages[0]["content"] else 202
            return [marker, len(messages)]

        @staticmethod
        def encode(_content, **_kwargs):
            return [1, 2, 3, 4, 5]

    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    _trajectory(first, "first")
    _trajectory(second, "second")
    sources = module.prepare_source_samples(
        [first, second], FakeTokenizer(), 128, 16, 0.5
    )

    first_markers = [
        sample["context_ids"][0] for sample in sources[0]["train"] + sources[0]["eval"]
    ]
    second_markers = [
        sample["context_ids"][0] for sample in sources[1]["train"] + sources[1]["eval"]
    ]
    assert first_markers == [101, 101]
    assert second_markers == [202, 202]


def test_training_script_never_truncates_inside_tool_call(tmp_path: Path, monkeypatch):
    fake_transformers = types.ModuleType("transformers")
    fake_transformers.AutoModelForCausalLM = object
    fake_transformers.AutoTokenizer = object
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    script_path = (
        Path(__file__).resolve().parents[3]
        / "scripts"
        / "qwen_exo"
        / "train_activation_editor.py"
    )
    spec = importlib.util.spec_from_file_location("tool_boundary_training", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    template_kwargs = []

    class CharacterTokenizer:
        @staticmethod
        def apply_chat_template(_messages, **kwargs):
            template_kwargs.append(kwargs)
            return [1, 2, 3]

        @staticmethod
        def encode(content, **_kwargs):
            return list(range(len(content)))

    tool_text = (
        'reasoning before action <tool_call>{"name":"bash","arguments":'
        '{"command":"python -m pytest focused_test.py"}}</tool_call>'
    )
    trajectory = {
        "session": {
            "messages": [
                {"role": "system", "content": "system"},
                {"role": "user", "content": "run the focused test"},
                {"role": "assistant", "content": tool_text},
            ]
        }
    }

    truncated = module.build_samples(
        trajectory, CharacterTokenizer(), 128, len(tool_text) - 1
    )
    complete = module.build_samples(
        trajectory, CharacterTokenizer(), 128, len(tool_text)
    )
    bounded = module.build_samples(
        trajectory,
        CharacterTokenizer(),
        128,
        len(tool_text),
        len(tool_text) + 2,
    )

    assert truncated == []
    assert complete[0]["target_ids"] == list(range(len(tool_text)))
    assert complete[0]["tool_call_count"] == 1
    assert bounded[0]["target_ids"] == list(range(len(tool_text)))
    assert len(bounded[0]["context_ids"]) == 2
    assert (
        len(bounded[0]["context_ids"]) + len(bounded[0]["target_ids"])
        == len(tool_text) + 2
    )
    assert template_kwargs[-1]["enable_thinking"] is True
    assert template_kwargs[-1]["preserve_thinking"] is True
    assert (
        module.parse_complete_tool_calls(
            '<tool_call>{"name":"bash","arguments":{}}</tool_call>'
        )
        is None
    )


def test_chunked_target_loss_matches_full_logits_gradient(tmp_path: Path, monkeypatch):
    fake_transformers = types.ModuleType("transformers")
    fake_transformers.AutoModelForCausalLM = object
    fake_transformers.AutoTokenizer = object
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    script_path = (
        Path(__file__).resolve().parents[3]
        / "scripts"
        / "qwen_exo"
        / "train_activation_editor.py"
    )
    spec = importlib.util.spec_from_file_location("chunked_loss_training", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    class TinyBackbone(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = torch.nn.Embedding(7, 5)

        def forward(self, input_ids, **_kwargs):
            return types.SimpleNamespace(last_hidden_state=self.embedding(input_ids))

    class TinyCausalLM(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = TinyBackbone()
            self.lm_head = torch.nn.Linear(5, 7, bias=False)

    torch.manual_seed(7)
    chunked_model = TinyCausalLM()
    full_model = TinyCausalLM()
    full_model.load_state_dict(chunked_model.state_dict())
    input_ids = torch.tensor([[0, 1, 2, 3, 4]])

    chunked_loss = module.exact_target_nll(
        chunked_model,
        input_ids,
        context_end=2,
        backward=True,
        chunk_tokens=1,
    )
    hidden = full_model.model(input_ids).last_hidden_state[:, 1:-1]
    logits = full_model.lm_head(hidden)
    full_loss = torch.nn.functional.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        input_ids[:, 2:].reshape(-1),
    )
    full_loss.backward()

    assert chunked_loss == pytest.approx(float(full_loss), abs=1e-6)
    for chunked_parameter, full_parameter in zip(
        chunked_model.parameters(), full_model.parameters()
    ):
        assert torch.allclose(
            chunked_parameter.grad, full_parameter.grad, atol=1e-6, rtol=1e-6
        )
