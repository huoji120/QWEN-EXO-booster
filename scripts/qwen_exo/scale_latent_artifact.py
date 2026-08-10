#!/usr/bin/env python3
"""Rescale each layer of a latent artifact to a reference artifact's RMS."""

from __future__ import annotations

import argparse

import torch

from qwen_exo_booster.latent_transplant import (
    LatentArtifactStore,
    load_latent_artifact,
    save_latent_artifact,
    source_sha256,
    validate_artifact_name,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-root", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--artifact", required=True)
    args = parser.parse_args()
    source_name = validate_artifact_name(args.source)
    reference_name = validate_artifact_name(args.reference)
    target_name = validate_artifact_name(args.artifact)
    payload = load_latent_artifact(f"{args.artifact_root}/{source_name}.pt")
    reference_payload = load_latent_artifact(
        f"{args.artifact_root}/{reference_name}.pt"
    )
    layers = tuple(int(value) for value in payload["layers"].tolist())
    store = LatentArtifactStore(args.artifact_root)
    scaled = []
    factors = []
    for layer in layers:
        vector = store.vector(
            source_name, layer, device=torch.device("cpu"), dtype=torch.float32
        )
        reference = store.vector(
            reference_name, layer, device=torch.device("cpu"), dtype=torch.float32
        )
        if vector is None or reference is None:
            raise RuntimeError(f"missing layer {layer}")
        vector_rms = vector.pow(2).mean().sqrt().clamp_min(1e-12)
        reference_rms = reference.pow(2).mean().sqrt()
        factor = reference_rms / vector_rms
        factors.append(float(factor))
        scaled.append(vector * factor)
    summary = save_latent_artifact(
        args.artifact_root,
        target_name,
        torch.stack(scaled),
        layers=layers,
        model_fingerprint=str(payload.get("model_fingerprint") or ""),
        source_digest=source_sha256(
            f"scaled:{source_name}:to:{reference_name}".encode()
        ),
        token_count=int(payload.get("token_count") or 0),
        chunk_count=int(payload.get("chunk_count") or 1),
    )
    print(
        {
            "artifact": summary.public_dict(),
            "reference_layers": [
                int(value) for value in reference_payload["layers"].tolist()
            ],
            "scale_factors": factors,
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
