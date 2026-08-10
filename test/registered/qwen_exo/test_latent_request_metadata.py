import json
from types import SimpleNamespace

import torch

from qwen_exo_booster.latent_transplant import LatentArtifactStore, save_latent_artifact
from qwen_exo_booster.runtime import QwenExoRuntime


class _Telemetry:
    def __init__(self):
        self.events = []

    def emit(self, request_id, event_type, payload):
        self.events.append((request_id, event_type, payload))


def _runtime(tmp_path):
    save_latent_artifact(
        tmp_path,
        "identity-cognition-smoke",
        torch.ones((1, 128), dtype=torch.float32),
        layers=(15,),
        model_fingerprint="identity-model",
        source_digest="identity-source",
        token_count=32,
        chunk_count=1,
    )
    runtime = object.__new__(QwenExoRuntime)
    runtime._latent_default = None
    runtime._latent_default_warned = set()
    runtime.latent_artifacts = LatentArtifactStore(
        tmp_path,
        hidden_size=128,
        target_layers=(15,),
    )
    runtime.model_identity = SimpleNamespace(fingerprint="identity-model")
    runtime.telemetry = _Telemetry()
    runtime._request_latent_transplants = {}
    runtime._request_latent_transplant_layers = {}
    return runtime


def test_latent_request_metadata_accepts_responses_stringified_object(tmp_path):
    runtime = _runtime(tmp_path)
    request = SimpleNamespace(
        request_id="identity-request",
        metadata={
            "qwen_exo_latent_transplant": json.dumps(
                {"artifact": "identity-cognition-smoke", "strength": 0.125}
            )
        },
    )

    payload = runtime.latent_transplant_payload(request)

    assert payload == {
        "mode": "active",
        "artifact": "identity-cognition-smoke",
        "strength": 0.125,
    }
    assert runtime._request_latent_transplants["identity-request"] == payload
    assert runtime._request_latent_transplant_layers["identity-request"] == (15,)
    assert runtime.telemetry.events == [
        (
            "identity-request",
            "latent_transplant.requested",
            {
                "artifact": "identity-cognition-smoke",
                "strength": 0.125,
                "layers": [15],
                "source_digest": "identity-source",
                "merged_artifacts": ["identity-cognition-smoke"],
                "source": "request",
            },
        )
    ]


def test_latent_request_metadata_keeps_plain_artifact_name_compatible(tmp_path):
    runtime = _runtime(tmp_path)
    request = SimpleNamespace(
        request_id="identity-plain-name",
        metadata={"qwen_exo_latent_transplant": "identity-cognition-smoke"},
    )

    assert runtime.latent_transplant_payload(request) == {
        "mode": "active",
        "artifact": "identity-cognition-smoke",
        "strength": 0.05,
    }


def test_latent_request_metadata_preserves_explicit_token_window(tmp_path):
    runtime = _runtime(tmp_path)
    request = SimpleNamespace(
        request_id="identity-window",
        metadata={
            "qwen_exo_latent_transplant": {
                "artifact": "identity-cognition-smoke",
                "strength": 0.125,
                "token_window": 16,
            }
        },
    )

    assert runtime.latent_transplant_payload(request) == {
        "mode": "active",
        "artifact": "identity-cognition-smoke",
        "strength": 0.125,
        "token_window": 16,
    }
    assert runtime.telemetry.events[0][2]["token_window"] == 16
