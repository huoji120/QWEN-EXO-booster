import torch

from qwen_exo_booster.activation_editor import (
    ActivationEditor,
    ActivationEditorStore,
    apply_activation_editor,
    parse_activation_editor_spec,
    resolve_default_activation_editor_spec,
)


def _payload(rank=2, hidden=8, window=4, layer=47):
    projection = torch.eye(hidden)[:rank].clone()
    return {
        "schema": 1,
        "layer": layer,
        "rank": rank,
        "window": window,
        "hidden_size": hidden,
        "state_dict": {
            "projection": projection,
            "transform": projection.clone(),
            "bias": torch.zeros(rank),
        },
    }


def test_parse_spec_accepts_name_and_object():
    assert parse_activation_editor_spec("ctf-v1") == {
        "mode": "active",
        "editor": "ctf-v1",
    }
    assert parse_activation_editor_spec({"editor": "ctf-v1"}) == {
        "mode": "active",
        "editor": "ctf-v1",
    }
    assert parse_activation_editor_spec({"editor": "../evil"}) is None
    assert parse_activation_editor_spec({"mode": "off", "editor": "ctf-v1"}) is None


def test_zero_init_editor_is_identity():
    editor = ActivationEditor(_payload(), torch.device("cpu"))
    hidden = torch.randn(5, 8)
    assert torch.allclose(editor.apply(hidden), hidden, atol=1e-5)


def test_editor_matches_training_formula():
    payload = _payload()
    payload["state_dict"]["transform"] = torch.randn(2, 8)
    payload["state_dict"]["bias"] = torch.randn(2)
    editor = ActivationEditor(payload, torch.device("cpu"))
    hidden = torch.randn(5, 8)
    base = hidden @ payload["state_dict"]["projection"].T
    target = (
        hidden @ payload["state_dict"]["transform"].T + payload["state_dict"]["bias"]
    )
    expected = hidden + (target - base) @ payload["state_dict"]["projection"]
    assert torch.allclose(editor.apply(hidden), expected, atol=1e-5)


def test_apply_only_on_final_prefill_window(tmp_path):
    payload = _payload(window=2)
    payload["state_dict"]["bias"] = torch.ones(2)
    torch.save(payload, tmp_path / "ctf.editor.pt")
    store = ActivationEditorStore(tmp_path)
    hidden = torch.zeros((6, 8))
    specs = (
        {"mode": "active", "editor": "ctf"},
        {"mode": "active", "editor": "ctf"},
    )
    edited = apply_activation_editor(
        store, hidden, specs, (3, 3), (False, True), layer_index=47
    )
    assert edited is not None
    assert (
        torch.count_nonzero(edited[:4]) == 0
    )  # request 0 not final, head of request 1
    assert torch.count_nonzero(edited[4:]) > 0  # last 2 tokens of request 1

    other_layer = apply_activation_editor(
        store, hidden, specs, (3, 3), (True, True), layer_index=15
    )
    assert other_layer is None


def test_apply_rejects_legacy_multi_editor_spec(tmp_path):
    first = _payload(window=2)
    first["state_dict"]["bias"] = torch.ones(2)
    second = _payload(window=2)
    second["state_dict"]["bias"] = torch.full((2,), 2.0)
    torch.save(first, tmp_path / "first.editor.pt")
    torch.save(second, tmp_path / "second.editor.pt")
    store = ActivationEditorStore(tmp_path)
    hidden = torch.zeros((2, 8))

    edited = apply_activation_editor(
        store,
        hidden,
        (
            {
                "editors": [
                    {"editor": "first", "strength": 1.0},
                    {"editor": "second", "strength": 1.0},
                ]
            },
        ),
        (2,),
        (True,),
        layer_index=47,
    )

    assert edited is None


def test_store_fails_closed_on_missing_editor(tmp_path):
    store = ActivationEditorStore(tmp_path)
    assert store.editor("missing", torch.device("cpu")) is None
    hidden = torch.zeros((2, 8))
    edited = apply_activation_editor(
        store,
        hidden,
        ({"mode": "active", "editor": "missing"},),
        (2,),
        (True,),
        layer_index=47,
    )
    assert edited is None


def test_persisted_active_editor_is_authoritative_without_master_switch():
    assert resolve_default_activation_editor_spec(
        {"mode": "active", "editor": "first", "strength": 0.25},
        None,
        enabled=False,
        fallback_strength=2.0,
    ) == {"mode": "active", "editor": "first", "strength": 2.0}
