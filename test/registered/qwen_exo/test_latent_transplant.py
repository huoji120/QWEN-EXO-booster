import math

import pytest
import torch

from qwen_exo_booster.latent_transplant import (
    LATENT_TRANSPLANT_CAPTURE_COUNT_KEY,
    LATENT_TRANSPLANT_CAPTURE_LAYERS_KEY,
    LATENT_TRANSPLANT_CAPTURE_VECTOR_KEY,
    LATENT_TRANSPLANT_CAPTURE_TRAJECTORY_CHUNKS_KEY,
    LATENT_TRANSPLANT_CAPTURE_TRAJECTORY_COUNT_KEY,
    LATENT_TRANSPLANT_CAPTURE_TRAJECTORY_VECTOR_KEY,
    LatentArtifactStore,
    LatentCaptureAccumulator,
    build_capture_customized_info,
    build_layer_addition,
    load_latent_artifact,
    save_latent_artifact,
    parse_latent_transplant_spec,
    pool_capture_layer,
    select_latent_layers,
)


def test_select_latent_layers_uses_deep_full_attention_quartiles():
    dense_types = [
        "attention" if (index + 1) % 4 == 0 else "linear_attention"
        for index in range(64)
    ]
    moe_types = [
        "attention" if (index + 1) % 4 == 0 else "linear_attention"
        for index in range(40)
    ]

    assert select_latent_layers(dense_types) == (15, 31, 47, 63)
    assert select_latent_layers(moe_types) == (11, 19, 31, 39)


def test_fp8_artifact_roundtrip_and_summary(tmp_path):
    vectors = torch.linspace(-3.0, 3.0, steps=4 * 512).reshape(4, 512)
    trajectory = torch.stack((vectors - 1.0, vectors + 1.0))
    summary = save_latent_artifact(
        tmp_path,
        "ctf-gpt-sol",
        vectors,
        layers=(15, 31, 47, 63),
        model_fingerprint="model-fingerprint",
        source_digest="source-digest",
        token_count=2048,
        chunk_count=2,
        trajectory_vectors=trajectory,
        trajectory_token_counts=(1024, 1024),
    )
    payload = load_latent_artifact(tmp_path / "ctf-gpt-sol.pt")
    store = LatentArtifactStore(
        tmp_path,
        hidden_size=512,
        target_layers=(15, 31, 47, 63),
    )
    restored = torch.stack(
        [
            store.vector(
                "ctf-gpt-sol",
                layer,
                device=torch.device("cpu"),
                dtype=torch.float32,
            )
            for layer in summary.layers
        ]
    )

    assert payload["quantized"].dtype == torch.uint8
    assert summary.storage_dtype == "float8_e4m3fn"
    assert summary.token_count == 2048
    assert payload["trajectory_quantized"].shape[:2] == (2, 4)
    assert payload["trajectory_token_counts"].tolist() == [1024, 1024]
    assert torch.nn.functional.cosine_similarity(vectors, restored).min() > 0.999


def test_capture_pooling_keeps_only_capture_requests():
    layer_15 = torch.arange(24, dtype=torch.float32).reshape(6, 4)
    layer_31 = layer_15 + 100
    info = build_capture_customized_info(
        ((15, layer_15), (31, layer_31)),
        ({"mode": "capture"}, None),
        (2, 4),
    )

    assert info is not None
    vectors = info[LATENT_TRANSPLANT_CAPTURE_VECTOR_KEY].reshape(2, 2, 4)
    pooled = (
        (15, pool_capture_layer(layer_15, ({"mode": "capture"}, None), (2, 4))),
        (31, pool_capture_layer(layer_31, ({"mode": "capture"}, None), (2, 4))),
    )
    pooled_info = build_capture_customized_info(
        pooled, ({"mode": "capture"}, None), (2, 4), pooled=True
    )
    assert pooled_info is not None
    assert torch.equal(
        pooled_info[LATENT_TRANSPLANT_CAPTURE_VECTOR_KEY],
        info[LATENT_TRANSPLANT_CAPTURE_VECTOR_KEY],
    )
    assert info[LATENT_TRANSPLANT_CAPTURE_COUNT_KEY].tolist() == [2, 0]
    assert info[LATENT_TRANSPLANT_CAPTURE_LAYERS_KEY].tolist() == [
        [15, 31],
        [15, 31],
    ]
    assert torch.equal(vectors[0, 0], layer_15[:2].mean(dim=0))
    assert torch.equal(vectors[0, 1], layer_31[:2].mean(dim=0))
    assert torch.count_nonzero(vectors[1]) == 0


def test_capture_pooling_can_select_tail_tokens():
    captured = torch.arange(24, dtype=torch.float32).reshape(6, 4)
    pooled = pool_capture_layer(
        captured,
        ({"mode": "capture", "capture_tail_tokens": 2},),
        (6,),
    )

    assert torch.equal(pooled[0], captured[4:].mean(dim=0))


def test_capture_tail_spec_fails_closed_outside_bound():
    assert (
        parse_latent_transplant_spec({"mode": "capture", "capture_tail_tokens": 0})
        is None
    )
    assert (
        parse_latent_transplant_spec({"mode": "capture", "capture_tail_tokens": 4097})
        is None
    )


def test_capture_accumulator_returns_one_full_request_artifact():
    accumulator = LatentCaptureAccumulator()
    spec = ({"mode": "capture"},)
    first = (
        (15, torch.ones((1, 4))),
        (31, torch.full((1, 4), 2.0)),
    )
    second = (
        (15, torch.full((1, 4), 3.0)),
        (31, torch.full((1, 4), 4.0)),
    )

    assert accumulator.update(first, spec, (4096,), (False,), ("request",)) is None
    info = accumulator.update(second, spec, (100,), (True,), ("request",))

    assert info is not None
    assert info[LATENT_TRANSPLANT_CAPTURE_COUNT_KEY].tolist() == [4196]
    assert info[LATENT_TRANSPLANT_CAPTURE_TRAJECTORY_CHUNKS_KEY].tolist() == [2]
    assert info[LATENT_TRANSPLANT_CAPTURE_TRAJECTORY_COUNT_KEY].tolist() == [
        [4096, 100]
    ]
    trajectory = info[LATENT_TRANSPLANT_CAPTURE_TRAJECTORY_VECTOR_KEY].reshape(
        1, 2, 2, 4
    )
    assert torch.equal(trajectory[0, 0], torch.stack((first[0][1][0], first[1][1][0])))
    assert torch.equal(
        trajectory[0, 1], torch.stack((second[0][1][0], second[1][1][0]))
    )
    expected = (first[0][1][0] * 4096 + second[0][1][0] * 100) / 4196
    aggregate = info[LATENT_TRANSPLANT_CAPTURE_VECTOR_KEY].reshape(1, 2, 4)
    assert torch.allclose(aggregate[0, 0], expected)


def test_layer_addition_targets_last_token_of_final_prefill(tmp_path):
    vectors = torch.stack((torch.ones(128), torch.full((128,), 2.0)))
    save_latent_artifact(
        tmp_path,
        "ctf",
        vectors,
        layers=(15, 31),
        model_fingerprint="model",
        source_digest="source",
        token_count=100,
        chunk_count=1,
    )
    store = LatentArtifactStore(
        tmp_path,
        hidden_size=128,
        target_layers=(15, 31),
    )
    hidden = torch.zeros((5, 128), dtype=torch.bfloat16)
    addition, applied, strengths = build_layer_addition(
        store,
        15,
        hidden,
        (
            {"mode": "active", "artifact": "ctf", "strength": 0.1},
            {"mode": "active", "artifact": "ctf", "strength": 0.2},
        ),
        (2, 3),
        (True, False),
    )

    assert addition is not None
    assert applied.tolist() == [1, 0]
    assert strengths.tolist() == pytest.approx([0.1, 0.0])
    assert torch.allclose(addition[1].float(), torch.full((128,), 0.1), atol=0.01)
    assert torch.count_nonzero(addition[:1]) == 0
    assert torch.count_nonzero(addition[2:]) == 0


def test_forward_batch_latent_params_fail_closed():
    assert parse_latent_transplant_spec({"mode": "capture"}) == {"mode": "capture"}
    assert parse_latent_transplant_spec(
        {"mode": "active", "artifact": "ctf", "strength": 0.05}
    ) == {"mode": "active", "artifact": "ctf", "strength": 0.05}
    assert (
        parse_latent_transplant_spec(
            {"mode": "active", "artifact": "../ctf", "strength": 0.05}
        )
        is None
    )
    assert (
        parse_latent_transplant_spec(
            {"mode": "active", "artifact": "ctf", "strength": 2.0}
        )
        is None
    )


def test_merged_vector_uses_orthogonalized_token_mass_subspace(tmp_path):
    hidden = 128
    alpha_rows = torch.zeros((1, 2, hidden))
    alpha_rows[0, :, 0] = 1.0
    beta_rows = torch.zeros((1, 2, hidden))
    beta_rows[0, :, 1] = 1.0
    for name, rows, counts in (
        ("alpha", alpha_rows, (100,)),
        ("beta", beta_rows, (300,)),
    ):
        save_latent_artifact(
            tmp_path,
            name,
            rows[0],
            layers=(15, 31),
            model_fingerprint="fp",
            source_digest=name,
            token_count=sum(counts),
            chunk_count=len(counts),
            trajectory_vectors=rows,
            trajectory_token_counts=counts,
        )
    store = LatentArtifactStore(tmp_path, hidden_size=hidden, target_layers=(15, 31))

    merged = store.merged_vector(15, device=torch.device("cpu"), dtype=torch.float32)

    assert merged is not None
    alpha_share = 10.0 / (10.0 + math.sqrt(300.0))
    assert merged[0].item() == pytest.approx(alpha_share, abs=0.02)
    assert merged[1].item() == pytest.approx(1.0 - alpha_share, abs=0.02)
    assert torch.count_nonzero(merged[2:]) == 0
    assert merged[0].item() != pytest.approx(0.25, abs=0.05)
    assert (
        store.vector("merged", 15, device=torch.device("cpu"), dtype=torch.float32)
        is not None
    )
    assert (
        store.merged_vector(7, device=torch.device("cpu"), dtype=torch.float32) is None
    )
    filtered = store.merged_vector(
        15,
        device=torch.device("cpu"),
        dtype=torch.float32,
        model_fingerprint="other",
    )
    assert filtered is None


def test_runtime_config_default_auto_attaches(tmp_path):
    from types import SimpleNamespace

    from qwen_exo_booster.config import QwenExoConfig, QwenExoFeatureFlags
    from qwen_exo_booster.fingerprint import ModelIdentity
    from qwen_exo_booster.hybrid_state import HybridRuntimePolicy
    from qwen_exo_booster.runtime import QwenExoRuntime, QwenExoRuntimeState

    def build_runtime(enabled: bool) -> QwenExoRuntime:
        value = QwenExoRuntime(
            QwenExoConfig(
                state_directory=tmp_path / "state",
                knowledge_directory=tmp_path / "knowledge",
                max_internal_fanout=8,
                max_internal_tokens=1024,
                max_candidates=8,
                max_memory_tokens=256,
                observer_mode="shadow",
                feature_flags=QwenExoFeatureFlags(
                    hybrid_prefix=True,
                    external_memory=False,
                    reference_judge=False,
                    capsule=True,
                    observer=True,
                    adaptive_refresh=False,
                ),
                model_path="model",
                tp_size=2,
                latent_transplant_enabled=enabled,
                latent_transplant_strength=0.05,
            ),
            SimpleNamespace(),
            HybridRuntimePolicy(
                tp_size=2,
                dtype="bfloat16",
                page_size=64,
                mamba_strategy="extra_buffer_lazy",
                mamba_state_dtype="bfloat16",
            ),
        )
        value.state = QwenExoRuntimeState.READY
        return value

    save_latent_artifact(
        tmp_path / "state" / "latent-transplant" / "artifacts",
        "ctf",
        torch.ones((2, 128)),
        layers=(15, 31),
        model_fingerprint="fp",
        source_digest="src",
        token_count=10,
        chunk_count=1,
    )
    runtime = build_runtime(True)
    runtime.model_identity = ModelIdentity(
        fingerprint="fp",
        model_path="model",
        architecture="arch",
        model_type="type",
        layer_count=64,
        full_attention_layers=16,
        linear_attention_layers=48,
        max_position_embeddings=10240,
        weight_bytes=0,
        file_hashes={},
    )

    assert runtime.latent_transplant_default() == {
        "artifact": "merged",
        "strength": 0.05,
    }
    request = SimpleNamespace(request_id="r1", metadata={})
    assert runtime.latent_transplant_payload(request) == {
        "mode": "active",
        "artifact": "merged",
        "strength": 0.05,
    }

    explicit_off = SimpleNamespace(
        request_id="r2", metadata={"qwen_exo_latent_transplant": None}
    )
    assert runtime.latent_transplant_payload(explicit_off) is None

    missing = build_runtime(True)
    missing.latent_artifacts = LatentArtifactStore(tmp_path / "empty")
    assert missing.latent_transplant_payload(request) is None

    disabled = build_runtime(False)
    assert disabled.latent_transplant_default() is None
    assert disabled.latent_transplant_payload(request) is None
