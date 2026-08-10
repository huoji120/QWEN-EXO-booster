from __future__ import annotations

import argparse
import json
import urllib.error
import urllib.request
import uuid
from pathlib import Path
from typing import Any

import torch

from qwen_exo_booster.contracts import stable_digest
from qwen_exo_booster.native_state_bank import load_page_key_heads


def _post_json(url: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} from {url}: {body}") from exc


def _raw_query_heads(
    value: Any, *, query_head_count: int, head_dim: int
) -> torch.Tensor:
    candidates = [value]
    if isinstance(value, (list, tuple)):
        candidates.extend(reversed(value))
    state_width = int(query_head_count) * int(head_dim)
    for raw in candidates:
        try:
            heads = torch.tensor(raw, dtype=torch.float32)
        except (TypeError, ValueError, RuntimeError):
            continue
        if state_width < 1 or heads.numel() % state_width:
            continue
        heads = heads.reshape(-1, int(query_head_count), int(head_dim))
        finite = torch.isfinite(heads).flatten(start_dim=1).all(dim=1)
        heads = heads[finite]
        if heads.numel():
            return heads
    raise RuntimeError(
        "live QueryProbe returned no finite raw Attention-Q heads: "
        + repr(value)[:1000]
    )


def _status(payload: dict[str, Any]) -> str | None:
    value = (payload.get("meta_info") or {}).get("qwen_exo_bank_cache_status")
    if isinstance(value, list):
        return str(value[-1]) if value else None
    return str(value) if value is not None else None


def _page_score(queries: torch.Tensor, keys: torch.Tensor) -> float:
    if (
        queries.ndim != 3
        or keys.ndim != 3
        or queries.shape[-1] != keys.shape[-1]
        or queries.shape[1] % keys.shape[1]
    ):
        raise RuntimeError("raw Q/K head geometry is incompatible")
    grouped = queries.reshape(
        queries.shape[0],
        keys.shape[1],
        queries.shape[1] // keys.shape[1],
        queries.shape[2],
    )
    logits = torch.einsum("qkrd,tkd->qtkr", grouped, keys.float()) / (
        queries.shape[-1] ** 0.5
    )
    per_token = torch.topk(
        logits.flatten(start_dim=2),
        k=min(4, logits.shape[2] * logits.shape[3]),
        dim=2,
    ).values.mean(dim=2)
    per_query = torch.topk(per_token, k=min(4, per_token.shape[1]), dim=1).values.mean(
        dim=1
    )
    return float(
        torch.topk(per_query, k=min(4, per_query.shape[0])).values.mean().item()
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Probe the live TP-native Tensor Bank with raw Attention Q×K heads"
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:30000")
    parser.add_argument("--state-dir", type=Path, default=Path("/data/qwen-exo/state"))
    parser.add_argument("--tp-size", type=int, default=2)
    parser.add_argument("--query-head-count", type=int, default=24)
    parser.add_argument("--head-dim", type=int, default=256)
    parser.add_argument("--timeout", type=float, default=600.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    base_url = args.base_url.rstrip("/")
    snapshot_path = args.state_dir / "tensor-bank.pt"
    snapshot = torch.load(
        str(snapshot_path), map_location="cpu", weights_only=True, mmap=True
    )
    if snapshot.get("schema") != 12:
        raise RuntimeError("Tensor Bank snapshot is not raw-QK schema 12")
    compressed_index_fields = sorted(
        {
            "key_storage",
            "key_shape",
            "token_key_storage",
            "token_key_shapes",
        }.intersection(snapshot)
    )
    if compressed_index_fields:
        raise RuntimeError(
            f"Tensor Bank still persists compressed index fields: {compressed_index_fields}"
        )
    pages = snapshot.get("pages") or ()
    if not pages:
        raise RuntimeError("Tensor Bank has no native document state")
    source_digest = str(snapshot["source_digest"])
    seed_page = pages[0]
    seed_artifact_path = (
        args.state_dir
        / "native-bank"
        / source_digest
        / f"page-{int(seed_page['page_id']):08d}-rank-0000.pt"
    )
    seed_artifact = torch.load(
        str(seed_artifact_path), map_location="cpu", weights_only=True, mmap=True
    )
    seed_ids = tuple(int(token) for token in seed_artifact["token_ids"])[-32:]
    if not seed_ids:
        raise RuntimeError("Tensor Bank seed page has no tokens")
    query_request_id = f"qwen-exo-native-probe-{uuid.uuid4().hex}"
    query_result = _post_json(
        f"{base_url}/generate",
        {
            "input_ids": list(seed_ids),
            "sampling_params": {
                "temperature": 0,
                "max_new_tokens": 1,
                "custom_params": {
                    "qwen_exo_kind": "internal",
                    "qwen_exo_job_type": "query_probe",
                    "qwen_exo_parent_request_id": query_request_id,
                    "qwen_exo_query_spans": [{"start": 0, "end": len(seed_ids)}],
                },
            },
            "rid": query_request_id,
            "stream": False,
        },
        args.timeout,
    )
    queries = _raw_query_heads(
        (query_result.get("meta_info") or {}).get("qwen_exo_user_query_full_heads"),
        query_head_count=args.query_head_count,
        head_dim=args.head_dim,
    )

    best: tuple[float, dict[str, Any], torch.Tensor] | None = None
    for page in pages:
        if not page.get("model_native"):
            continue
        keys = load_page_key_heads(
            args.state_dir / "native-bank",
            source_digest=source_digest,
            page_id=int(page["page_id"]),
            world_size=args.tp_size,
            model_fingerprint=str(snapshot["model_fingerprint"]),
            prefix_identity=str(page["prefix_identity"]),
            token_count=int(page["state_token_count"]),
            dtype=torch.float32,
        )
        score = _page_score(queries, keys)
        if best is None or score > best[0]:
            best = (score, page, keys)
    if best is None:
        raise RuntimeError("Tensor Bank has no scoreable raw K heads")

    score, page, keys = best
    local_positions = tuple(int(item) for item in page["salient_positions"])
    if (
        not local_positions
        or len(local_positions) % 64
        or max(local_positions) >= int(keys.shape[0])
    ):
        raise RuntimeError("selected Tensor Bank document has no aligned salient plan")
    page_id = int(page["page_id"])
    artifact_path = (
        args.state_dir
        / "native-bank"
        / source_digest
        / f"page-{page_id:08d}-rank-0000.pt"
    )
    artifact = torch.load(
        str(artifact_path), map_location="cpu", weights_only=True, mmap=True
    )
    artifact_token_ids = tuple(int(item) for item in artifact["token_ids"])
    selected_token_ids = tuple(artifact_token_ids[item] for item in local_positions)
    prefix_identity = stable_digest(
        source_digest,
        page_id,
        *local_positions,
        *selected_token_ids,
    )
    extra_key = f"{page['radix_namespace']}:probe:{uuid.uuid4().hex}"
    request = {
        "input_ids": [*selected_token_ids, selected_token_ids[-1]],
        "extra_key": extra_key,
        "sampling_params": {
            "temperature": 0,
            "max_new_tokens": 1,
            "custom_params": {
                "qwen_exo_kind": "user",
                "qwen_exo_native_bank_selection": {
                    "source_digest": source_digest,
                    "page_id": page_id,
                    "local_positions": list(local_positions),
                    "prefix_identity": prefix_identity,
                },
            },
        },
    }
    first = _post_json(f"{base_url}/generate", request, args.timeout)
    second = _post_json(f"{base_url}/generate", request, args.timeout)
    first_status = _status(first)
    second_status = _status(second)
    report = {
        "passed": (
            not compressed_index_fields
            and first_status == "loaded"
            and second_status == "hit"
        ),
        "retrieval_geometry": "raw_attention_q_x_raw_attention_k",
        "compressed_index_fields_absent": not compressed_index_fields,
        "query_count": int(queries.shape[0]),
        "query_heads": int(queries.shape[1]),
        "head_dim": int(queries.shape[2]),
        "page_id": page_id,
        "page_score": score,
        "selected_tokens": len(local_positions),
        "first_cache_status": first_status,
        "second_cache_status": second_status,
        "first_cached_tokens": (first.get("meta_info") or {}).get("cached_tokens"),
        "second_cached_tokens": (second.get("meta_info") or {}).get("cached_tokens"),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
