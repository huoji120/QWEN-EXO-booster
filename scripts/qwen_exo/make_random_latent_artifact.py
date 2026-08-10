#!/usr/bin/env python3
"""Create a per-layer RMS-matched random latent control artifact."""

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
    parser.add_argument("--reference", required=True)
    parser.add_argument("--artifact", required=True)
    parser.add_argument("--seed", type=int, default=20260807)
    args = parser.parse_args()
    reference = validate_artifact_name(args.reference)
    artifact = validate_artifact_name(args.artifact)
    payload = load_latent_artifact(f"{args.artifact_root}/{reference}.pt")
    layers = tuple(int(value) for value in payload["layers"].tolist())
    store = LatentArtifactStore(args.artifact_root)
    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    vectors = []
    for layer in layers:
        reference_vector = store.vector(
            reference,
            layer,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )
        if reference_vector is None:
            raise RuntimeError(f"missing reference layer {layer}")
        random_vector = torch.randn(
            reference_vector.shape, generator=generator, dtype=torch.float32
        )
        random_rms = random_vector.pow(2).mean().sqrt().clamp_min(1e-12)
        reference_rms = reference_vector.pow(2).mean().sqrt()
        vectors.append(random_vector * (reference_rms / random_rms))
    matrix = torch.stack(vectors)
    summary = save_latent_artifact(
        args.artifact_root,
        artifact,
        matrix,
        layers=layers,
        model_fingerprint=str(payload.get("model_fingerprint") or ""),
        source_digest=source_sha256(f"random:{args.seed}:{reference}".encode()),
        token_count=int(payload.get("token_count") or 0),
        chunk_count=int(payload.get("chunk_count") or 1),
    )
    print(summary.public_dict())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
