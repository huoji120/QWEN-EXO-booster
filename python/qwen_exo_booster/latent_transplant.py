from __future__ import annotations

from collections import OrderedDict
import hashlib
import logging
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import torch

LATENT_TRANSPLANT_SCHEMA = 2
LATENT_TRANSPLANT_GROUP_SIZE = 128
LATENT_TRANSPLANT_MAX_STRENGTH = 0.5
LATENT_TRANSPLANT_MAX_WINDOW = 128
LATENT_TRANSPLANT_MAX_CAPTURE_TAIL = 4096
LATENT_TRANSPLANT_CAPTURE_VECTOR_KEY = "qwen_exo_latent_capture_vectors"
LATENT_TRANSPLANT_CAPTURE_COUNT_KEY = "qwen_exo_latent_capture_counts"
LATENT_TRANSPLANT_CAPTURE_LAYERS_KEY = "qwen_exo_latent_capture_layers"
LATENT_TRANSPLANT_CAPTURE_TRAJECTORY_VECTOR_KEY = (
    "qwen_exo_latent_capture_trajectory_vectors"
)
LATENT_TRANSPLANT_CAPTURE_TRAJECTORY_COUNT_KEY = (
    "qwen_exo_latent_capture_trajectory_counts"
)
LATENT_TRANSPLANT_CAPTURE_TRAJECTORY_CHUNKS_KEY = (
    "qwen_exo_latent_capture_trajectory_chunks"
)
LATENT_TRANSPLANT_APPLIED_KEY = "qwen_exo_latent_transplant_applied"
LATENT_TRANSPLANT_STRENGTH_KEY = "qwen_exo_latent_transplant_strength"
LATENT_TRANSPLANT_DIAGNOSTICS_KEY = "qwen_exo_latent_transplant_diagnostics"
MERGED_LATENT_ARTIFACT = "merged"
MERGED_LATENT_MAX_BASIS = 64

_ARTIFACT_NAME = re.compile(r"^[a-z0-9][a-z0-9._-]{0,95}$")
_LOGGER = logging.getLogger(__name__)


class LatentTransplantError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class LatentArtifactSummary:
    name: str
    model_fingerprint: str
    source_digest: str
    layers: tuple[int, ...]
    hidden_size: int
    token_count: int
    chunk_count: int
    storage_dtype: str

    def public_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "model_fingerprint": self.model_fingerprint,
            "source_digest": self.source_digest,
            "layers": list(self.layers),
            "hidden_size": self.hidden_size,
            "token_count": self.token_count,
            "chunk_count": self.chunk_count,
            "storage_dtype": self.storage_dtype,
        }


def validate_artifact_name(value: object) -> str:
    name = str(value or "").strip().lower()
    if not _ARTIFACT_NAME.fullmatch(name):
        raise ValueError(
            "latent transplant artifact name must contain only lowercase letters, "
            "digits, dots, dashes, and underscores"
        )
    return name


def parse_latent_transplant_spec(raw: object) -> dict[str, object] | None:
    if not isinstance(raw, dict):
        return None
    mode = str(raw.get("mode") or "").strip().lower()
    if mode == "capture":
        parsed: dict[str, object] = {"mode": "capture"}
        if "capture_tail_tokens" in raw:
            try:
                tail_tokens = int(raw.get("capture_tail_tokens"))
            except (TypeError, ValueError):
                return None
            if not 1 <= tail_tokens <= LATENT_TRANSPLANT_MAX_CAPTURE_TAIL:
                return None
            parsed["capture_tail_tokens"] = tail_tokens
        return parsed
    if mode != "active":
        return None
    try:
        artifact = validate_artifact_name(raw.get("artifact"))
        strength = float(raw.get("strength", 0.05))
        token_window = int(raw.get("token_window", 1))
    except (TypeError, ValueError):
        return None
    if (
        not math.isfinite(strength)
        or not 0 < strength <= LATENT_TRANSPLANT_MAX_STRENGTH
        or not 1 <= token_window <= LATENT_TRANSPLANT_MAX_WINDOW
    ):
        return None
    parsed: dict[str, object] = {
        "mode": "active",
        "artifact": artifact,
        "strength": strength,
    }
    if "token_window" in raw:
        parsed["token_window"] = token_window
    if bool(raw.get("diagnostics", False)):
        parsed["diagnostics"] = True
    return parsed


def select_latent_layers(
    layer_types: Sequence[str], *, count: int = 4
) -> tuple[int, ...]:
    if count < 1:
        raise ValueError("latent layer count must be positive")
    full_attention = tuple(
        index
        for index, layer_type in enumerate(layer_types)
        if str(layer_type) in {"attention", "full_attention"}
    )
    candidates = full_attention or tuple(range(len(layer_types)))
    if not candidates:
        raise ValueError("model exposes no layers for latent capture")
    selected_count = min(int(count), len(candidates))
    if selected_count == 1:
        return (candidates[-1],)
    indices = tuple(
        math.ceil((index + 1) * len(candidates) / selected_count) - 1
        for index in range(selected_count)
    )
    return tuple(candidates[index] for index in indices)


def _quantize_vectors(
    vectors: torch.Tensor,
    *,
    group_size: int = LATENT_TRANSPLANT_GROUP_SIZE,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    source = vectors.detach().to(device="cpu", dtype=torch.float32).contiguous()
    if source.ndim != 2 or source.shape[0] < 1 or source.shape[1] < 1:
        raise ValueError("latent vectors must have shape [layers, hidden_size]")
    if not torch.isfinite(source).all():
        raise ValueError("latent vectors contain NaN or Inf")
    hidden_size = int(source.shape[1])
    padded_size = math.ceil(hidden_size / group_size) * group_size
    if padded_size != hidden_size:
        source = torch.nn.functional.pad(source, (0, padded_size - hidden_size))
    grouped = source.reshape(source.shape[0], -1, group_size)
    scales = grouped.abs().amax(dim=-1, keepdim=True) / 448.0
    scales = torch.where(scales > 0, scales, torch.ones_like(scales))
    quantized = torch.clamp(grouped / scales, min=-448.0, max=448.0).to(
        torch.float8_e4m3fn
    )
    return (
        quantized.reshape(source.shape[0], padded_size).view(torch.uint8),
        scales.squeeze(-1).to(torch.bfloat16),
        hidden_size,
    )


def _dequantize_vector(
    quantized: torch.Tensor,
    scales: torch.Tensor,
    *,
    row: int,
    hidden_size: int,
    group_size: int,
) -> torch.Tensor:
    if quantized.dtype == torch.uint8:
        quantized = quantized.view(torch.float8_e4m3fn)
    groups = int(quantized.shape[1]) // int(group_size)
    restored = (
        quantized[row]
        .reshape(groups, group_size)
        .float()
        .mul(scales[row].float().reshape(groups, 1))
        .reshape(-1)
    )
    return restored[:hidden_size]


def save_latent_artifact(
    root: Path | str,
    name: str,
    vectors: torch.Tensor,
    *,
    layers: Sequence[int],
    model_fingerprint: str,
    source_digest: str,
    token_count: int,
    chunk_count: int,
    trajectory_vectors: torch.Tensor | None = None,
    trajectory_token_counts: Sequence[int] | None = None,
    group_size: int = LATENT_TRANSPLANT_GROUP_SIZE,
) -> LatentArtifactSummary:
    artifact_name = validate_artifact_name(name)
    layer_ids = tuple(int(layer) for layer in layers)
    if len(layer_ids) != len(set(layer_ids)) or any(layer < 0 for layer in layer_ids):
        raise ValueError("latent artifact layers must be unique and non-negative")
    if vectors.ndim != 2 or vectors.shape[0] != len(layer_ids):
        raise ValueError("latent vector rows must match artifact layers")
    quantized, scales, hidden_size = _quantize_vectors(vectors, group_size=group_size)
    payload: dict[str, Any] = {
        "schema": LATENT_TRANSPLANT_SCHEMA,
        "name": artifact_name,
        "model_fingerprint": str(model_fingerprint),
        "source_digest": str(source_digest),
        "layers": torch.tensor(layer_ids, dtype=torch.int64),
        "hidden_size": int(hidden_size),
        "token_count": int(token_count),
        "chunk_count": int(chunk_count),
        "group_size": int(group_size),
        "storage_dtype": "float8_e4m3fn",
        "quantized": quantized,
        "scales": scales,
    }
    if trajectory_vectors is not None:
        if (
            trajectory_vectors.ndim != 3
            or trajectory_vectors.shape[1] != len(layer_ids)
            or trajectory_vectors.shape[2] != hidden_size
        ):
            raise ValueError(
                "trajectory vectors must have shape [chunks, layers, hidden_size]"
            )
        trajectory_counts = tuple(
            int(value) for value in (trajectory_token_counts or ())
        )
        if (
            len(trajectory_counts) != trajectory_vectors.shape[0]
            or any(value < 1 for value in trajectory_counts)
            or sum(trajectory_counts) != int(token_count)
            or int(chunk_count) != trajectory_vectors.shape[0]
        ):
            raise ValueError("trajectory token counts do not cover the artifact tokens")
        trajectory_quantized, trajectory_scales, trajectory_hidden_size = (
            _quantize_vectors(
                trajectory_vectors.reshape(-1, hidden_size), group_size=group_size
            )
        )
        if trajectory_hidden_size != hidden_size:
            raise ValueError("trajectory hidden size changed during quantization")
        payload["trajectory_quantized"] = trajectory_quantized.reshape(
            trajectory_vectors.shape[0], len(layer_ids), -1
        )
        payload["trajectory_scales"] = trajectory_scales.reshape(
            trajectory_vectors.shape[0], len(layer_ids), -1
        )
        payload["trajectory_token_counts"] = torch.tensor(
            trajectory_counts, dtype=torch.int64
        )
    directory = Path(root).expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / f"{artifact_name}.pt"
    temporary = directory / f".{artifact_name}.{os.getpid()}.tmp"
    torch.save(payload, temporary)
    os.replace(temporary, target)
    return _summary(payload)


def _summary(payload: dict[str, Any]) -> LatentArtifactSummary:
    layers = payload.get("layers")
    if not isinstance(layers, torch.Tensor):
        raise LatentTransplantError("latent artifact has no layer tensor")
    return LatentArtifactSummary(
        name=str(payload.get("name") or ""),
        model_fingerprint=str(payload.get("model_fingerprint") or ""),
        source_digest=str(payload.get("source_digest") or ""),
        layers=tuple(int(value) for value in layers.tolist()),
        hidden_size=int(payload.get("hidden_size") or 0),
        token_count=int(payload.get("token_count") or 0),
        chunk_count=int(payload.get("chunk_count") or 0),
        storage_dtype=str(payload.get("storage_dtype") or ""),
    )


def load_latent_artifact(path: Path | str) -> dict[str, Any]:
    payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise LatentTransplantError("latent artifact payload is not a mapping")
    if int(payload.get("schema") or 0) != LATENT_TRANSPLANT_SCHEMA:
        raise LatentTransplantError("latent artifact schema is unsupported")
    summary = _summary(payload)
    quantized = payload.get("quantized")
    scales = payload.get("scales")
    group_size = int(payload.get("group_size") or 0)
    if (
        not summary.name
        or not summary.model_fingerprint
        or summary.hidden_size < 1
        or not summary.layers
        or not isinstance(quantized, torch.Tensor)
        or quantized.dtype != torch.uint8
        or summary.storage_dtype != "float8_e4m3fn"
        or group_size < 1
        or quantized.ndim != 2
        or not isinstance(scales, torch.Tensor)
        or scales.ndim != 2
        or quantized.shape[0] != len(summary.layers)
        or scales.shape[0] != len(summary.layers)
        or quantized.shape[1] % group_size
        or scales.shape[1] != quantized.shape[1] // group_size
    ):
        raise LatentTransplantError("latent artifact tensor layout is invalid")
    trajectory_quantized = payload.get("trajectory_quantized")
    trajectory_scales = payload.get("trajectory_scales")
    trajectory_counts = payload.get("trajectory_token_counts")
    if any(
        value is not None
        for value in (trajectory_quantized, trajectory_scales, trajectory_counts)
    ):
        if (
            not isinstance(trajectory_quantized, torch.Tensor)
            or trajectory_quantized.dtype != torch.uint8
            or trajectory_quantized.ndim != 3
            or not isinstance(trajectory_scales, torch.Tensor)
            or trajectory_scales.ndim != 3
            or not isinstance(trajectory_counts, torch.Tensor)
            or trajectory_counts.dtype != torch.int64
            or trajectory_counts.ndim != 1
            or trajectory_quantized.shape[0] != summary.chunk_count
            or trajectory_quantized.shape[1] != len(summary.layers)
            or trajectory_scales.shape[:2] != trajectory_quantized.shape[:2]
            or trajectory_scales.shape[2] != trajectory_quantized.shape[2] // group_size
            or trajectory_counts.shape[0] != summary.chunk_count
            or int(trajectory_counts.sum().item()) != summary.token_count
        ):
            raise LatentTransplantError("latent trajectory tensor layout is invalid")
    return payload


class LatentArtifactStore:
    def __init__(
        self,
        root: Path | str,
        *,
        hidden_size: int | None = None,
        target_layers: Iterable[int] | None = None,
    ):
        self.root = Path(root).expanduser().resolve()
        self.hidden_size = int(hidden_size) if hidden_size is not None else None
        self.target_layers = (
            tuple(int(layer) for layer in target_layers)
            if target_layers is not None
            else None
        )
        self._payloads: dict[str, tuple[int, dict[str, Any]]] = {}
        self._vectors: dict[tuple[str, int, int, str, str], torch.Tensor] = {}
        self._failures: set[tuple[str, int]] = set()

    def path(self, name: str) -> Path:
        return self.root / f"{validate_artifact_name(name)}.pt"

    def summaries(self) -> tuple[LatentArtifactSummary, ...]:
        if not self.root.is_dir():
            return ()
        summaries: list[LatentArtifactSummary] = []
        for path in sorted(self.root.glob("*.pt")):
            try:
                summaries.append(_summary(load_latent_artifact(path)))
            except Exception:
                _LOGGER.exception("Failed to inspect latent artifact %s", path)
        return tuple(summaries)

    def has(self, name: str) -> bool:
        try:
            return self.path(name).is_file()
        except ValueError:
            return False

    def _payload(self, name: str) -> tuple[int, dict[str, Any]] | None:
        artifact_name = validate_artifact_name(name)
        path = self.path(artifact_name)
        try:
            modified_ns = path.stat().st_mtime_ns
        except FileNotFoundError:
            return None
        cached = self._payloads.get(artifact_name)
        if cached is not None and cached[0] == modified_ns:
            return cached
        failure_key = (artifact_name, modified_ns)
        if failure_key in self._failures:
            return None
        try:
            payload = load_latent_artifact(path)
            summary = _summary(payload)
            if self.hidden_size is not None and summary.hidden_size != self.hidden_size:
                raise LatentTransplantError(
                    "latent artifact hidden size does not match the running model"
                )
            if self.target_layers is not None and any(
                layer not in self.target_layers for layer in summary.layers
            ):
                raise LatentTransplantError(
                    "latent artifact layer does not match the running model"
                )
        except Exception:
            self._failures.add(failure_key)
            _LOGGER.exception("Failed to load latent artifact %s", path)
            return None
        self._payloads[artifact_name] = (modified_ns, payload)
        self._vectors = {
            key: value
            for key, value in self._vectors.items()
            if key[0] != artifact_name
        }
        return modified_ns, payload

    def _trajectory_rows(
        self,
        payload: dict[str, Any],
        summary: LatentArtifactSummary,
        row: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Per-layer ordered block rows and their token weights.

        Falls back to the token-weighted prototype as a single row when the
        artifact carries no ordered trajectory blocks.
        """
        group_size = int(payload["group_size"])
        quantized = payload.get("trajectory_quantized")
        scales = payload.get("trajectory_scales")
        counts = payload.get("trajectory_token_counts")
        if isinstance(quantized, torch.Tensor) and isinstance(scales, torch.Tensor):
            layer_quantized = quantized[:, row, :]
            if layer_quantized.dtype == torch.uint8:
                layer_quantized = layer_quantized.view(torch.float8_e4m3fn)
            groups = layer_quantized.shape[1] // group_size
            rows = (
                layer_quantized.reshape(-1, groups, group_size).float()
                * scales[:, row, :].float().reshape(-1, groups, 1)
            ).reshape(quantized.shape[0], -1)[:, : summary.hidden_size]
            weights = counts.float()
            return rows, weights
        prototype = _dequantize_vector(
            payload["quantized"],
            payload["scales"],
            row=row,
            hidden_size=summary.hidden_size,
            group_size=group_size,
        )
        return prototype.unsqueeze(0), torch.tensor([float(summary.token_count)])

    def merged_vector(
        self,
        layer: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
        model_fingerprint: str | None = None,
    ) -> torch.Tensor | None:
        """Subspace merge of every compatible artifact.

        Block rows from all artifacts are concatenated, weighted by the
        square root of their token mass, and reduced to the top-K
        (<= MERGED_LATENT_MAX_BASIS) principal directions of the row space
        (equivalent to QR re-orthogonalization of the concatenated basis).
        The merged injection vector is the energy-weighted combination of
        those orthonormal directions.
        """
        rows: list[torch.Tensor] = []
        weights: list[torch.Tensor] = []
        stamps: list[int] = []
        for summary in self.summaries():
            if summary.name == MERGED_LATENT_ARTIFACT:
                continue
            if model_fingerprint and summary.model_fingerprint != model_fingerprint:
                continue
            if summary.token_count < 1 or int(layer) not in summary.layers:
                continue
            loaded = self._payload(summary.name)
            if loaded is None:
                continue
            modified_ns, payload = loaded
            artifact_rows, artifact_weights = self._trajectory_rows(
                payload, summary, summary.layers.index(int(layer))
            )
            rows.append(artifact_rows)
            weights.append(artifact_weights)
            stamps.append(modified_ns)
        if not rows:
            return None
        basis = torch.cat(rows).float()
        mass = torch.cat(weights).float().clamp_min(0)
        if basis.shape[0] < 1 or mass.sum() <= 0:
            return None
        cache_key = (
            MERGED_LATENT_ARTIFACT,
            int(layer),
            hash(tuple(stamps)),
            str(device),
            str(dtype),
        )
        cached = self._vectors.get(cache_key)
        if cached is not None:
            return cached
        weighted = basis * mass.sqrt().unsqueeze(1)
        gram = weighted @ weighted.T
        eigenvalues, eigenvectors = torch.linalg.eigh(gram)
        keep = min(MERGED_LATENT_MAX_BASIS, basis.shape[0])
        eigenvalues = eigenvalues[-keep:]
        eigenvectors = eigenvectors[:, -keep:]
        positive = eigenvalues > 1e-10
        if not positive.any():
            return None
        eigenvalues = eigenvalues[positive]
        eigenvectors = eigenvectors[:, positive]
        directions = (weighted.T @ eigenvectors) / eigenvalues.sqrt().unsqueeze(0)
        energy = eigenvalues.sqrt()
        merged = (
            (directions * (energy / energy.sum()).unsqueeze(0))
            .sum(dim=1)
            .to(device=device, dtype=dtype)
        )
        self._vectors[cache_key] = merged
        return merged

    def vector(
        self,
        name: str,
        layer: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor | None:
        if name == MERGED_LATENT_ARTIFACT:
            return self.merged_vector(layer, device=device, dtype=dtype)
        loaded = self._payload(name)
        if loaded is None:
            return None
        modified_ns, payload = loaded
        summary = _summary(payload)
        try:
            row = summary.layers.index(int(layer))
        except ValueError:
            return None
        cache_key = (
            summary.name,
            int(layer),
            modified_ns,
            str(device),
            str(dtype),
        )
        cached = self._vectors.get(cache_key)
        if cached is not None:
            return cached
        restored = _dequantize_vector(
            payload["quantized"],
            payload["scales"],
            row=row,
            hidden_size=summary.hidden_size,
            group_size=int(payload["group_size"]),
        ).to(device=device, dtype=dtype)
        self._vectors[cache_key] = restored
        return restored


def pool_capture_layer(
    captured: torch.Tensor,
    specs: Sequence[dict[str, object] | None],
    extend_seq_lens: Sequence[int],
) -> torch.Tensor:
    total_tokens = sum(int(length) for length in extend_seq_lens)
    if captured.ndim != 2 or captured.shape[0] != total_tokens:
        raise LatentTransplantError("captured latent tensor length is inconsistent")
    pooled = torch.zeros(
        (len(specs), int(captured.shape[-1])),
        device=captured.device,
        dtype=captured.dtype,
    )
    offset = 0
    for request_index, (spec, length) in enumerate(zip(specs, extend_seq_lens)):
        end = offset + int(length)
        if spec and spec.get("mode") == "capture" and end > offset:
            try:
                tail_tokens = int(spec.get("capture_tail_tokens") or 0)
            except (TypeError, ValueError):
                tail_tokens = 0
            start = max(offset, end - tail_tokens) if tail_tokens > 0 else offset
            pooled[request_index] = (
                captured[start:end].float().mean(dim=0).to(captured.dtype)
            )
        offset = end
    return pooled


@dataclass(slots=True)
class _LatentCaptureState:
    layers: tuple[int, ...]
    counts: list[int]
    vectors: list[torch.Tensor]


class LatentCaptureAccumulator:
    def __init__(self, *, max_active: int = 128):
        if max_active < 1:
            raise ValueError("latent capture max_active must be positive")
        self.max_active = int(max_active)
        self._states: OrderedDict[str, _LatentCaptureState] = OrderedDict()

    def update(
        self,
        captured: Sequence[tuple[int, torch.Tensor]],
        specs: Sequence[dict[str, object] | None],
        extend_seq_lens: Sequence[int],
        final_prefill: Sequence[bool] | None,
        request_ids: Sequence[str],
    ) -> dict[str, torch.Tensor] | None:
        if not captured or not specs:
            return None
        batch_size = len(specs)
        if len(extend_seq_lens) != batch_size or len(request_ids) != batch_size:
            raise LatentTransplantError("latent capture batch metadata is inconsistent")
        layer_ids = tuple(int(layer) for layer, _tensor in captured)
        first = captured[0][1]
        if any(
            tensor.ndim != 2
            or tensor.shape[0] != batch_size
            or tensor.shape[1] != first.shape[1]
            for _layer, tensor in captured
        ):
            raise LatentTransplantError("pooled latent tensor shape is inconsistent")
        final_flags = tuple(final_prefill or (True,) * batch_size)
        completed: dict[int, _LatentCaptureState] = {}
        for request_index, (spec, raw_count, request_id) in enumerate(
            zip(specs, extend_seq_lens, request_ids)
        ):
            if not spec or spec.get("mode") != "capture":
                continue
            count = int(raw_count)
            if count < 1 or not request_id:
                continue
            block = torch.stack(
                [tensor[request_index] for _layer, tensor in captured], dim=0
            ).detach()
            state = self._states.get(str(request_id))
            if state is None:
                state = _LatentCaptureState(layer_ids, [], [])
                self._states[str(request_id)] = state
            elif state.layers != layer_ids or state.vectors[0].shape != block.shape:
                self._states.pop(str(request_id), None)
                raise LatentTransplantError(
                    "latent capture layout changed during prefill"
                )
            state.counts.append(count)
            state.vectors.append(block)
            self._states.move_to_end(str(request_id))
            if request_index < len(final_flags) and final_flags[request_index]:
                completed[request_index] = self._states.pop(str(request_id))

        while len(self._states) > self.max_active:
            self._states.popitem(last=False)
        if not completed:
            return None

        max_chunks = max(len(state.vectors) for state in completed.values())
        layer_count = len(layer_ids)
        hidden_size = int(first.shape[1])
        aggregate = torch.zeros(
            (batch_size, layer_count, hidden_size),
            device=first.device,
            dtype=first.dtype,
        )
        trajectory = torch.zeros(
            (batch_size, max_chunks, layer_count, hidden_size),
            device=first.device,
            dtype=first.dtype,
        )
        counts = torch.zeros(batch_size, device=first.device, dtype=torch.int64)
        trajectory_counts = torch.zeros(
            (batch_size, max_chunks), device=first.device, dtype=torch.int64
        )
        trajectory_chunks = torch.zeros(
            batch_size, device=first.device, dtype=torch.int64
        )
        for request_index, state in completed.items():
            total = sum(state.counts)
            weighted = torch.zeros_like(state.vectors[0], dtype=torch.float32)
            for chunk_index, (vector, count) in enumerate(
                zip(state.vectors, state.counts)
            ):
                trajectory[request_index, chunk_index] = vector
                trajectory_counts[request_index, chunk_index] = count
                weighted.add_(vector.float(), alpha=count)
            aggregate[request_index] = (weighted / total).to(first.dtype)
            counts[request_index] = total
            trajectory_chunks[request_index] = len(state.vectors)
        layers = torch.tensor(layer_ids, device=first.device, dtype=torch.int64).expand(
            batch_size, -1
        )
        return {
            LATENT_TRANSPLANT_CAPTURE_VECTOR_KEY: aggregate.reshape(batch_size, -1),
            LATENT_TRANSPLANT_CAPTURE_COUNT_KEY: counts,
            LATENT_TRANSPLANT_CAPTURE_LAYERS_KEY: layers,
            LATENT_TRANSPLANT_CAPTURE_TRAJECTORY_VECTOR_KEY: trajectory.reshape(
                batch_size, -1
            ),
            LATENT_TRANSPLANT_CAPTURE_TRAJECTORY_COUNT_KEY: trajectory_counts,
            LATENT_TRANSPLANT_CAPTURE_TRAJECTORY_CHUNKS_KEY: trajectory_chunks,
        }


def build_capture_customized_info(
    captured: Sequence[tuple[int, torch.Tensor]],
    specs: Sequence[dict[str, object] | None],
    extend_seq_lens: Sequence[int],
    *,
    pooled: bool = False,
) -> dict[str, torch.Tensor] | None:
    if (
        not captured
        or not specs
        or not any(spec and spec.get("mode") == "capture" for spec in specs)
    ):
        return None
    batch_size = len(specs)
    if pooled:
        if any(
            tensor.ndim != 2 or tensor.shape[0] != batch_size for _, tensor in captured
        ):
            raise LatentTransplantError("pooled latent tensor shape is inconsistent")
        layer_vectors = [tensor for _layer, tensor in captured]
    else:
        layer_vectors = [
            pool_capture_layer(tensor, specs, extend_seq_lens)
            for _layer, tensor in captured
        ]
    vectors = torch.stack(layer_vectors, dim=1)
    first = vectors[0]
    counts = torch.tensor(
        [
            int(length) if spec and spec.get("mode") == "capture" else 0
            for spec, length in zip(specs, extend_seq_lens)
        ],
        device=vectors.device,
        dtype=torch.int64,
    )
    layers = torch.tensor(
        [int(layer) for layer, _tensor in captured],
        device=first.device,
        dtype=torch.int64,
    ).expand(batch_size, -1)
    return {
        LATENT_TRANSPLANT_CAPTURE_VECTOR_KEY: vectors.reshape(batch_size, -1),
        LATENT_TRANSPLANT_CAPTURE_COUNT_KEY: counts,
        LATENT_TRANSPLANT_CAPTURE_LAYERS_KEY: layers,
    }


def build_layer_addition(
    store: LatentArtifactStore,
    layer_id: int,
    hidden_states: torch.Tensor,
    specs: Sequence[dict[str, object] | None],
    extend_seq_lens: Sequence[int],
    final_prefill: Sequence[bool] | None,
    *,
    residual: torch.Tensor | None = None,
    diagnostics: list[dict[str, object]] | None = None,
) -> tuple[torch.Tensor | None, torch.Tensor, torch.Tensor]:
    batch_size = len(specs)
    applied = torch.zeros(batch_size, device=hidden_states.device, dtype=torch.int64)
    strengths = torch.zeros(
        batch_size, device=hidden_states.device, dtype=torch.float32
    )
    if not specs or not any(spec and spec.get("mode") == "active" for spec in specs):
        return None, applied, strengths
    final_flags = tuple(final_prefill or (True,) * batch_size)
    addition: torch.Tensor | None = None
    offset = 0
    for request_index, (spec, length) in enumerate(zip(specs, extend_seq_lens)):
        length = int(length)
        end = offset + length
        if (
            spec
            and spec.get("mode") == "active"
            and request_index < len(final_flags)
            and final_flags[request_index]
            and length > 0
        ):
            vector = store.vector(
                str(spec.get("artifact") or ""),
                int(layer_id),
                device=hidden_states.device,
                dtype=hidden_states.dtype,
            )
            if vector is not None:
                strength = min(
                    LATENT_TRANSPLANT_MAX_STRENGTH,
                    max(0.0, float(spec.get("strength") or 0.0)),
                )
                try:
                    token_window = int(spec.get("token_window") or 1)
                except (TypeError, ValueError):
                    token_window = 1
                token_window = min(length, max(1, token_window))
                start = end - token_window
                if strength > 0:
                    if diagnostics is not None and bool(spec.get("diagnostics")):
                        base = (
                            residual[end - 1]
                            if residual is not None
                            else hidden_states[end - 1]
                        ).float()
                        direction = vector.float()
                        injected = direction * strength
                        base_rms = base.pow(2).mean().sqrt()
                        direction_rms = direction.pow(2).mean().sqrt()
                        injected_rms = injected.pow(2).mean().sqrt()
                        base_norm = base.norm()
                        direction_norm = direction.norm()
                        cosine = torch.dot(base, direction) / (
                            base_norm * direction_norm
                        ).clamp_min(1e-12)
                        post_rms = (base + injected).pow(2).mean().sqrt()
                        values = (
                            torch.stack(
                                (
                                    base_rms,
                                    direction_rms,
                                    injected_rms,
                                    injected_rms / base_rms.clamp_min(1e-12),
                                    cosine,
                                    post_rms,
                                )
                            )
                            .detach()
                            .cpu()
                            .tolist()
                        )
                        row: dict[str, object] = {
                            "layer": int(layer_id),
                            "request_index": int(request_index),
                            "token_index": int(end - 1),
                            "base_rms": float(values[0]),
                            "vector_rms": float(values[1]),
                            "injected_rms": float(values[2]),
                            "relative_rms": float(values[3]),
                            "base_vector_cosine": float(values[4]),
                            "post_rms": float(values[5]),
                            "strength": float(strength),
                        }
                        if token_window != 1:
                            row["token_window"] = token_window
                        diagnostics.append(row)
                    if addition is None:
                        addition = torch.zeros_like(hidden_states)
                    addition[start:end].copy_(vector).mul_(strength)
                    applied[request_index] = 1
                    strengths[request_index] = strength
        offset = end
    return addition, applied, strengths


def source_sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()
