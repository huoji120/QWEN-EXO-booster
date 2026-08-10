from __future__ import annotations

import argparse
import json
import urllib.error
import urllib.request
import uuid
from pathlib import Path

import torch
from transformers import AutoTokenizer

from qwen_exo_booster.latent_transplant import (
    LATENT_TRANSPLANT_CAPTURE_COUNT_KEY,
    LATENT_TRANSPLANT_CAPTURE_LAYERS_KEY,
    LATENT_TRANSPLANT_CAPTURE_TRAJECTORY_CHUNKS_KEY,
    LATENT_TRANSPLANT_CAPTURE_TRAJECTORY_COUNT_KEY,
    LATENT_TRANSPLANT_CAPTURE_TRAJECTORY_VECTOR_KEY,
    LATENT_TRANSPLANT_CAPTURE_VECTOR_KEY,
    LATENT_TRANSPLANT_MAX_CAPTURE_TAIL,
    save_latent_artifact,
    source_sha256,
    validate_artifact_name,
)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compile a ChatML agent trajectory into a Qwen-native FP8 H artifact."
    )
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--artifact", required=True)
    parser.add_argument("--model", default="/models/qwen-exo")
    parser.add_argument("--state-dir", type=Path, required=True)
    parser.add_argument("--url", default="http://127.0.0.1:30000")
    parser.add_argument("--max-tokens", type=int, default=100000)
    parser.add_argument("--capture-tail-tokens", type=int, default=0)
    parser.add_argument("--timeout", type=float, default=1200.0)
    args = parser.parse_args()
    if not 0 <= args.capture_tail_tokens <= LATENT_TRANSPLANT_MAX_CAPTURE_TAIL:
        raise ValueError(
            f"capture-tail-tokens must be 0 or <= {LATENT_TRANSPLANT_MAX_CAPTURE_TAIL}"
        )
    return args


def _load_messages(source: Path) -> tuple[bytes, list[dict[str, object]]]:
    raw = source.read_bytes()
    payload = json.loads(raw.decode("utf-8"))
    session = payload.get("session") if isinstance(payload, dict) else None
    messages = session.get("messages") if isinstance(session, dict) else None
    if not isinstance(messages, list) or not messages:
        raise ValueError("trajectory must contain session.messages")
    normalized: list[dict[str, object]] = []
    for index, message in enumerate(messages):
        if not isinstance(message, dict):
            raise ValueError(f"trajectory message {index} is not an object")
        role = str(message.get("role") or "").strip()
        content = message.get("content")
        if role not in {"system", "user", "assistant", "tool"}:
            raise ValueError(
                f"trajectory message {index} has unsupported role {role!r}"
            )
        if not isinstance(content, str):
            content = json.dumps(content, ensure_ascii=False, sort_keys=True)
        normalized.append({"role": role, "content": content})
    return raw, normalized


def _post_json(url: str, payload: dict[str, object], timeout: float) -> dict:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json", "Accept": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            result = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        body = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {error.code}: {body}") from error
    if not isinstance(result, dict):
        raise RuntimeError("generation endpoint returned a non-object response")
    return result


def _get_json(url: str, timeout: float) -> dict:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        result = json.loads(response.read().decode("utf-8"))
    if not isinstance(result, dict):
        raise RuntimeError("status endpoint returned a non-object response")
    return result


def _matrix_entries(value: object, row_count: int) -> list[list[float]]:
    if not isinstance(value, list) or not value:
        return []
    if isinstance(value[0], list):
        return value
    if row_count < 1 or len(value) % row_count:
        return [value]
    width = len(value) // row_count
    return [value[start : start + width] for start in range(0, len(value), width)]


def _scalar_entries(value: object) -> list[int]:
    if not isinstance(value, list):
        return []
    return [int(item[0] if isinstance(item, list) else item) for item in value]


def main() -> int:
    args = _arguments()
    artifact_name = validate_artifact_name(args.artifact)
    if not 1 <= args.max_tokens <= 100000:
        raise ValueError("max-tokens must be between 1 and 100000")

    raw_source, messages = _load_messages(args.source)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    encoded = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=False,
    )
    encoded_ids = encoded.get("input_ids") if hasattr(encoded, "get") else encoded
    if (
        isinstance(encoded_ids, list)
        and encoded_ids
        and isinstance(encoded_ids[0], list)
    ):
        encoded_ids = encoded_ids[0]
    all_token_ids = [int(token) for token in encoded_ids]
    token_ids = all_token_ids[: args.max_tokens]
    source_digest = source_sha256(raw_source)
    status = _get_json(f"{args.url}/qwen-exo/status", args.timeout)
    tensor_bank = status.get("tensor_bank") or {}
    model_fingerprint = str(tensor_bank.get("model_fingerprint") or "")
    if not model_fingerprint:
        raise RuntimeError("running service did not report a model fingerprint")
    run_id = uuid.uuid4().hex[:12]
    capture_spec: dict[str, object] = {"mode": "capture"}
    if args.capture_tail_tokens:
        capture_spec["capture_tail_tokens"] = args.capture_tail_tokens
    result = _post_json(
        f"{args.url}/generate",
        {
            "input_ids": token_ids,
            "rid": f"qwen-exo-latent-{source_digest[:12]}-{run_id}",
            "extra_key": (f"qwen-exo:v3:latent-compile:{source_digest[:16]}:{run_id}"),
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
        args.timeout,
    )
    metadata = result.get("meta_info") or {}
    count_entries = _scalar_entries(metadata.get(LATENT_TRANSPLANT_CAPTURE_COUNT_KEY))
    vector_entries = _matrix_entries(
        metadata.get(LATENT_TRANSPLANT_CAPTURE_VECTOR_KEY), len(count_entries)
    )
    layer_entries = _matrix_entries(
        metadata.get(LATENT_TRANSPLANT_CAPTURE_LAYERS_KEY), len(count_entries)
    )
    trajectory_chunk_entries = _scalar_entries(
        metadata.get(LATENT_TRANSPLANT_CAPTURE_TRAJECTORY_CHUNKS_KEY)
    )
    trajectory_count_entries = _matrix_entries(
        metadata.get(LATENT_TRANSPLANT_CAPTURE_TRAJECTORY_COUNT_KEY),
        len(trajectory_chunk_entries),
    )
    trajectory_vector_entries = _matrix_entries(
        metadata.get(LATENT_TRANSPLANT_CAPTURE_TRAJECTORY_VECTOR_KEY),
        len(trajectory_chunk_entries),
    )
    if not all(
        (
            count_entries,
            vector_entries,
            layer_entries,
            trajectory_chunk_entries,
            trajectory_count_entries,
            trajectory_vector_entries,
        )
    ):
        raise RuntimeError("full-context request returned incomplete latent metadata")

    captured_tokens = int(count_entries[-1])
    layers = tuple(int(layer) for layer in layer_entries[-1])
    vectors = torch.tensor(vector_entries[-1], dtype=torch.float32)
    if not layers or vectors.numel() % len(layers):
        raise RuntimeError("captured latent vector shape is invalid")
    vectors = vectors.reshape(len(layers), -1)
    trajectory_chunks = int(trajectory_chunk_entries[-1])
    trajectory_counts = tuple(
        int(value) for value in trajectory_count_entries[-1][:trajectory_chunks]
    )
    trajectory = torch.tensor(trajectory_vector_entries[-1], dtype=torch.float32)
    expected_values = trajectory_chunks * len(layers) * vectors.shape[1]
    if trajectory.numel() < expected_values:
        raise RuntimeError("captured ordered latent trajectory is truncated")
    trajectory = trajectory[:expected_values].reshape(
        trajectory_chunks, len(layers), vectors.shape[1]
    )
    if (
        captured_tokens != len(token_ids)
        or len(trajectory_counts) != trajectory_chunks
        or any(count < 1 for count in trajectory_counts)
        or sum(trajectory_counts) != captured_tokens
    ):
        raise RuntimeError(
            f"captured {captured_tokens} of {len(token_ids)} requested tokens"
        )

    summary = save_latent_artifact(
        args.state_dir / "latent-transplant" / "artifacts",
        artifact_name,
        vectors,
        layers=layers,
        model_fingerprint=model_fingerprint,
        source_digest=source_digest,
        token_count=captured_tokens,
        chunk_count=trajectory_chunks,
        trajectory_vectors=trajectory,
        trajectory_token_counts=trajectory_counts,
    )
    report = {
        "artifact": summary.public_dict(),
        "source_messages": len(messages),
        "source_tokens": len(all_token_ids),
        "requested_tokens": len(token_ids),
        "capture_tail_tokens": args.capture_tail_tokens,
        "captured_tokens": captured_tokens,
        "truncated_at_max_tokens": len(all_token_ids) > len(token_ids),
        "internal_prefill_blocks": trajectory_chunks,
        "prompt_tokens": int(metadata.get("prompt_tokens") or 0),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
