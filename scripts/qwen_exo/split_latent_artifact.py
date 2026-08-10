#!/usr/bin/env python3
"""Split one benign multi-layer latent artifact into one-layer probes."""

from __future__ import annotations

import argparse

import torch

from qwen_exo_booster.latent_transplant import (
    LatentArtifactStore,
    load_latent_artifact,
    save_latent_artifact,
    validate_artifact_name,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-root", required=True)
    parser.add_argument("--artifact", required=True)
    args = parser.parse_args()
    artifact_root = args.artifact_root
    payload = load_latent_artifact(
        f"{artifact_root}/{validate_artifact_name(args.artifact)}.pt"
    )
    layers = tuple(int(value) for value in payload["layers"].tolist())
    store = LatentArtifactStore(artifact_root)
    for layer in layers:
        vector = store.vector(
            args.artifact,
            layer,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )
        if vector is None:
            raise RuntimeError(f"missing layer {layer}")
        name = f"{validate_artifact_name(args.artifact)}-l{layer}"
        summary = save_latent_artifact(
            artifact_root,
            name,
            vector.unsqueeze(0),
            layers=(layer,),
            model_fingerprint=str(payload.get("model_fingerprint") or ""),
            source_digest=str(payload.get("source_digest") or ""),
            token_count=int(payload.get("token_count") or 0),
            chunk_count=1,
        )
        print(summary.public_dict())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
