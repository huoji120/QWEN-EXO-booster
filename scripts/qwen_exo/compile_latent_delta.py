#!/usr/bin/env python3
"""Compile a paired activation-delta artifact from two benign ChatML traces."""

from __future__ import annotations

import argparse
import hashlib
import json
import urllib.request
import uuid
from pathlib import Path
from typing import Any

import torch
from transformers import AutoTokenizer

from qwen_exo_booster.latent_transplant import (
    LATENT_TRANSPLANT_MAX_CAPTURE_TAIL,
    save_latent_artifact,
)


def get_json(url: str, timeout: float) -> dict[str, Any]:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        result = json.loads(response.read().decode("utf-8"))
    if not isinstance(result, dict):
        raise RuntimeError("status endpoint returned a non-object")
    return result


def post_json(url: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json", "Accept": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        result = json.loads(response.read().decode("utf-8"))
    if not isinstance(result, dict):
        raise RuntimeError("capture endpoint returned a non-object")
    return result


def load_source(path: Path) -> tuple[bytes, list[dict[str, str]]]:
    raw = path.read_bytes()
    payload = json.loads(raw.decode("utf-8"))
    session = payload.get("session") if isinstance(payload, dict) else None
    messages = session.get("messages") if isinstance(session, dict) else None
    if not isinstance(messages, list) or not messages:
        raise ValueError(f"{path} must contain session.messages")
    normalized: list[dict[str, str]] = []
    for index, message in enumerate(messages):
        if not isinstance(message, dict):
            raise ValueError(f"message {index} is not an object")
        role = str(message.get("role") or "").strip()
        content = message.get("content")
        if role not in {"system", "user", "assistant", "tool"}:
            raise ValueError(f"message {index} has unsupported role {role!r}")
        if not isinstance(content, str):
            content = json.dumps(content, ensure_ascii=False, sort_keys=True)
        normalized.append({"role": role, "content": content})
    return raw, normalized


def rows(value: object, row_count: int) -> list[list[float]]:
    if not isinstance(value, list) or not value:
        return []
    if value and isinstance(value[0], list):
        return [[float(item) for item in row] for row in value]
    if row_count < 1 or len(value) % row_count:
        return [list(map(float, value))]
    width = len(value) // row_count
    return [
        [float(item) for item in value[start : start + width]]
        for start in range(0, len(value), width)
    ]


def capture(
    *,
    source: Path,
    tokenizer: Any,
    model: str,
    base_url: str,
    max_tokens: int,
    capture_tail_tokens: int,
    timeout: float,
) -> tuple[bytes, list[int], tuple[int, ...], torch.Tensor]:
    raw, messages = load_source(source)
    encoded = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=False
    )
    input_ids = encoded.get("input_ids") if hasattr(encoded, "get") else encoded
    if hasattr(input_ids, "tolist"):
        input_ids = input_ids.tolist()
    if input_ids and isinstance(input_ids[0], list):
        input_ids = input_ids[0]
    token_ids = [int(value) for value in input_ids[:max_tokens]]
    status = get_json(f"{base_url}/qwen-exo/status", timeout)
    fingerprint = str((status.get("tensor_bank") or {}).get("model_fingerprint") or "")
    if not fingerprint:
        raise RuntimeError("service did not report a model fingerprint")
    capture_spec: dict[str, object] = {"mode": "capture"}
    if capture_tail_tokens:
        capture_spec["capture_tail_tokens"] = capture_tail_tokens
    result = post_json(
        f"{base_url}/generate",
        {
            "input_ids": token_ids,
            "rid": f"latent-delta-{uuid.uuid4().hex}",
            "extra_key": f"qwen-exo:latent-delta:{uuid.uuid4().hex}",
            "sampling_params": {
                "temperature": 0,
                "top_p": 1,
                "top_k": 1,
                "max_new_tokens": 1,
                "skip_special_tokens": True,
                "custom_params": {"qwen_exo_latent_transplant": capture_spec},
            },
            "stream": False,
            "no_logs": True,
        },
        timeout,
    )
    metadata = result.get("meta_info") or {}
    counts = [
        int(value) for value in metadata.get("qwen_exo_latent_capture_counts", [])
    ]
    layer_rows = rows(metadata.get("qwen_exo_latent_capture_layers"), len(counts))
    vector_rows = rows(metadata.get("qwen_exo_latent_capture_vectors"), len(counts))
    if not counts or not layer_rows or not vector_rows:
        raise RuntimeError("capture returned incomplete latent metadata")
    layers = tuple(int(value) for value in layer_rows[-1])
    vectors = torch.tensor(vector_rows[-1], dtype=torch.float32).reshape(
        len(layers), -1
    )
    if counts[-1] != len(token_ids):
        raise RuntimeError(f"captured {counts[-1]} tokens, expected {len(token_ids)}")
    return raw, token_ids, layers, vectors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--control-source", type=Path, required=True)
    parser.add_argument("--artifact", required=True)
    parser.add_argument("--model", default="/models/qwen-exo")
    parser.add_argument("--state-dir", type=Path, required=True)
    parser.add_argument("--url", default="http://127.0.0.1:30000")
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--capture-tail-tokens", type=int, default=0)
    parser.add_argument("--timeout", type=float, default=1200)
    parser.add_argument("--normalize-delta", action="store_true")
    args = parser.parse_args()
    if not 0 <= args.capture_tail_tokens <= LATENT_TRANSPLANT_MAX_CAPTURE_TAIL:
        raise ValueError(
            f"capture-tail-tokens must be 0 or <= {LATENT_TRANSPLANT_MAX_CAPTURE_TAIL}"
        )
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    source_raw, source_ids, source_layers, source_vectors = capture(
        source=args.source,
        tokenizer=tokenizer,
        model=args.model,
        base_url=args.url,
        max_tokens=args.max_tokens,
        capture_tail_tokens=args.capture_tail_tokens,
        timeout=args.timeout,
    )
    control_raw, control_ids, control_layers, control_vectors = capture(
        source=args.control_source,
        tokenizer=tokenizer,
        model=args.model,
        base_url=args.url,
        max_tokens=args.max_tokens,
        capture_tail_tokens=args.capture_tail_tokens,
        timeout=args.timeout,
    )
    if source_layers != control_layers or source_vectors.shape != control_vectors.shape:
        raise RuntimeError("identity/control capture layouts do not match")
    delta = source_vectors - control_vectors
    scale_factors: list[float] = []
    if args.normalize_delta:
        for row in range(delta.shape[0]):
            source_rms = float(source_vectors[row].pow(2).mean().sqrt())
            delta_rms = float(delta[row].pow(2).mean().sqrt())
            factor = source_rms / max(delta_rms, 1e-6)
            delta[row].mul_(factor)
            scale_factors.append(factor)
    digest = hashlib.sha256(source_raw + b"\0" + control_raw).hexdigest()
    summary = save_latent_artifact(
        args.state_dir / "latent-transplant" / "artifacts",
        args.artifact,
        delta,
        layers=source_layers,
        model_fingerprint=str(
            (
                get_json(f"{args.url}/qwen-exo/status", args.timeout).get("model") or {}
            ).get("fingerprint", "")
        ),
        source_digest=digest,
        token_count=len(source_ids),
        chunk_count=1,
    )
    cosine = []
    for source_row, delta_row in zip(source_vectors, delta):
        denom = source_row.norm() * delta_row.norm()
        cosine.append(float(torch.dot(source_row, delta_row) / denom) if denom else 0.0)
    print(
        json.dumps(
            {
                "artifact": summary.public_dict(),
                "source_tokens": len(source_ids),
                "control_tokens": len(control_ids),
                "capture_tail_tokens": args.capture_tail_tokens,
                "layers": list(source_layers),
                "normalize_delta": args.normalize_delta,
                "layer_delta_rms": [
                    float(delta[row].pow(2).mean().sqrt())
                    for row in range(delta.shape[0])
                ],
                "identity_delta_cosine": cosine,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
