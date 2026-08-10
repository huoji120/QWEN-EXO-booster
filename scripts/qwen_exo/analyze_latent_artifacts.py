#!/usr/bin/env python3
"""Report measurable geometry of benign latent artifacts without model inference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from qwen_exo_booster.latent_transplant import LatentArtifactStore, load_latent_artifact


def stats(vector: torch.Tensor) -> dict[str, float]:
    value = vector.float()
    rms = float(value.pow(2).mean().sqrt())
    norm = float(value.norm())
    return {
        "rms": rms,
        "norm": norm,
        "mean": float(value.mean()),
        "std": float(value.std(unbiased=False)),
        "min": float(value.min()),
        "max": float(value.max()),
    }


def cosine(left: torch.Tensor, right: torch.Tensor) -> float:
    denominator = left.float().norm() * right.float().norm()
    if float(denominator) == 0:
        return 0.0
    return float(torch.dot(left.float(), right.float()) / denominator)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--state-dir", type=Path, required=True)
    parser.add_argument("--artifacts", nargs="+", required=True)
    args = parser.parse_args()
    artifact_root = args.state_dir / "latent-transplant" / "artifacts"
    payloads = {
        name: load_latent_artifact(artifact_root / f"{name}.pt")
        for name in args.artifacts
    }
    layers = tuple(
        int(value) for value in payloads[args.artifacts[0]]["layers"].tolist()
    )
    store = LatentArtifactStore(artifact_root)
    vectors: dict[str, dict[int, torch.Tensor]] = {}
    report: dict[str, object] = {"layers": list(layers), "artifacts": {}}
    for name, payload in payloads.items():
        vectors[name] = {}
        per_layer = {}
        for layer in layers:
            vector = store.vector(
                name,
                layer,
                device=torch.device("cpu"),
                dtype=torch.float32,
            )
            if vector is None:
                raise RuntimeError(f"missing vector for {name} layer {layer}")
            vectors[name][layer] = vector
            per_layer[str(layer)] = stats(vector)
        report["artifacts"][name] = {
            "model_fingerprint": str(payload.get("model_fingerprint")),
            "source_digest": str(payload.get("source_digest")),
            "token_count": int(payload.get("token_count") or 0),
            "chunk_count": int(payload.get("chunk_count") or 0),
            "per_layer": per_layer,
        }
    pairwise = {}
    for index, left_name in enumerate(args.artifacts):
        for right_name in args.artifacts[index + 1 :]:
            pairwise[f"{left_name}__vs__{right_name}"] = {
                str(layer): cosine(
                    vectors[left_name][layer], vectors[right_name][layer]
                )
                for layer in layers
            }
    report["pairwise_cosine"] = pairwise
    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
