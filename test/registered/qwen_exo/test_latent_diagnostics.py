import pytest
import torch

from qwen_exo_booster.latent_transplant import (
    LatentArtifactStore,
    build_layer_addition,
    parse_latent_transplant_spec,
    save_latent_artifact,
)


def test_diagnostic_spec_is_opt_in():
    assert parse_latent_transplant_spec(
        {"mode": "active", "artifact": "identity-smoke", "strength": 0.2}
    ) == {"mode": "active", "artifact": "identity-smoke", "strength": 0.2}
    assert parse_latent_transplant_spec(
        {
            "mode": "active",
            "artifact": "identity-smoke",
            "strength": 0.2,
            "diagnostics": True,
        }
    ) == {
        "mode": "active",
        "artifact": "identity-smoke",
        "strength": 0.2,
        "diagnostics": True,
    }


def test_layer_addition_reports_residual_relative_metrics(tmp_path):
    save_latent_artifact(
        tmp_path,
        "identity-smoke",
        torch.ones((1, 128), dtype=torch.float32),
        layers=(15,),
        model_fingerprint="model",
        source_digest="source",
        token_count=32,
        chunk_count=1,
    )
    store = LatentArtifactStore(tmp_path, hidden_size=128, target_layers=(15,))
    diagnostics = []
    addition, applied, strengths = build_layer_addition(
        store,
        15,
        torch.zeros((5, 128), dtype=torch.bfloat16),
        (
            {
                "mode": "active",
                "artifact": "identity-smoke",
                "strength": 0.25,
                "diagnostics": True,
            },
        ),
        (5,),
        (True,),
        residual=torch.ones((5, 128), dtype=torch.bfloat16),
        diagnostics=diagnostics,
    )

    assert addition is not None
    assert applied.tolist() == [1]
    assert strengths.tolist() == pytest.approx([0.25])
    assert diagnostics == [
        {
            "layer": 15,
            "request_index": 0,
            "token_index": 4,
            "base_rms": pytest.approx(1.0),
            "vector_rms": pytest.approx(1.0),
            "injected_rms": pytest.approx(0.25),
            "relative_rms": pytest.approx(0.25),
            "base_vector_cosine": pytest.approx(1.0),
            "post_rms": pytest.approx(1.25),
            "strength": pytest.approx(0.25),
        }
    ]


def test_layer_addition_supports_bounded_token_window(tmp_path):
    save_latent_artifact(
        tmp_path,
        "identity-window",
        torch.ones((1, 128), dtype=torch.float32),
        layers=(15,),
        model_fingerprint="model",
        source_digest="source",
        token_count=32,
        chunk_count=1,
    )
    store = LatentArtifactStore(tmp_path, hidden_size=128, target_layers=(15,))
    diagnostics = []
    addition, applied, strengths = build_layer_addition(
        store,
        15,
        torch.zeros((5, 128), dtype=torch.bfloat16),
        (
            {
                "mode": "active",
                "artifact": "identity-window",
                "strength": 0.25,
                "token_window": 3,
                "diagnostics": True,
            },
        ),
        (5,),
        (True,),
        residual=torch.ones((5, 128), dtype=torch.bfloat16),
        diagnostics=diagnostics,
    )

    assert addition is not None
    assert applied.tolist() == [1]
    assert strengths.tolist() == pytest.approx([0.25])
    assert torch.count_nonzero(addition[:2]).item() == 0
    assert torch.allclose(
        addition[2:], torch.full((3, 128), 0.25, dtype=torch.bfloat16)
    )
    assert diagnostics[0]["token_window"] == 3
    assert (
        parse_latent_transplant_spec(
            {"mode": "active", "artifact": "identity-window", "token_window": 0}
        )
        is None
    )
    assert (
        parse_latent_transplant_spec(
            {"mode": "active", "artifact": "identity-window", "token_window": 129}
        )
        is None
    )
