"""Read native Qwen4 PLE rows from a checkpoint or an NVMe snapshot.

The native snapshot format is intentionally row-addressable.  Its marker
(`native-ple.json`) describes a table made from BF16/F16 rows or FP8 rows with
one FP32 scale per row; no complete table is ever materialized by this module.
"""

from __future__ import annotations

import hashlib
import json
import math
import mmap
import os
import re
import struct
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from threading import Lock
from typing import Any

import msgspec
import numpy as np
import torch
from torch import nn

from sglang.srt.environ import envs
from sglang.srt.model_executor.runner import get_is_capture_mode


class PLESourceIdentity(str, Enum):
    CHECKPOINT_FP8 = "checkpoint_fp8"
    NATIVE_UNSCALED = "native_unscaled"


_PLE_SHARD_PATTERN = re.compile(r"^(?P<prefix>.+\.ngram_embedding)\.shard_(?P<index>\d+)\.weight$")
_NATIVE_MARKER = "native-ple.json"
_GPU_DECODE_MIN_ROWS = 1024


class PLEShard(msgspec.Struct, frozen=True):
    path: Path
    row_start: int
    row_end: int
    offset: int
    row_bytes: int
    scale_offset: int = -1
    scale_row_bytes: int = 0


class PLEManifest(msgspec.Struct, frozen=True):
    prefix: str
    embedding_dim: int
    total_rows: int
    shard_size: int
    shards: tuple[PLEShard, ...]
    storage: str = "fp8_e4m3"
    source_global_scale: float = 1.0
    scale_is_per_row: bool = False
    expected_layout: dict[str, Any] | None = None
    external_effective_values: bool = False

    @classmethod
    def from_snapshot(
        cls,
        snapshot: str | Path,
        *,
        expected_shards: int,
        expected_layout: dict[str, Any] | None = None,
    ) -> "PLEManifest":
        root = Path(snapshot).resolve()
        marker = root / _NATIVE_MARKER
        if marker.is_file():
            return cls._from_native_marker(
                marker, expected_shards=expected_shards, expected_layout=expected_layout
            )
        return cls._from_checkpoint_snapshot(root, expected_shards=expected_shards)

    @classmethod
    def _from_checkpoint_snapshot(
        cls, root: Path, *, expected_shards: int
    ) -> "PLEManifest":
        index = json.loads((root / "model.safetensors.index.json").read_text())
        matched = []
        for name, filename in index["weight_map"].items():
            match = _PLE_SHARD_PATTERN.fullmatch(name)
            if match is not None:
                matched.append((int(match["index"]), match["prefix"], name, filename))
        matched.sort()
        if not matched or [item[0] for item in matched] != list(range(expected_shards)):
            raise ValueError("PLE shard indices do not match the complete model table")
        if len({item[1] for item in matched}) != 1:
            raise ValueError("a native PLE snapshot must contain exactly one table")

        headers = {}
        records = []
        for shard_index, prefix, name, filename in matched:
            path = (root / filename).resolve()
            if not path.is_relative_to(root):
                raise ValueError("PLE weight shard escapes the checkpoint directory")
            if path not in headers:
                with path.open("rb") as source:
                    size = source.read(8)
                    if len(size) != 8:
                        raise ValueError(f"truncated safetensors header: {path}")
                    header_size = struct.unpack("<Q", size)[0]
                    if not 0 < header_size <= 128 * 1024 * 1024:
                        raise ValueError(f"invalid safetensors header size: {path}")
                    encoded = source.read(header_size)
                    if len(encoded) != header_size:
                        raise ValueError(f"truncated safetensors header: {path}")
                headers[path] = (json.loads(encoded), header_size + 8, path.stat().st_size)
            header, data_start, file_size = headers[path]
            record = header[name]
            shape = record["shape"]
            start, end = record["data_offsets"]
            if (
                record["dtype"] not in {"F8_E4M3", "BF16", "F16"}
                or len(shape) != 2
                or min(shape) <= 0
                or not 0 <= start <= end
                or end - start != math.prod(shape) * (1 if record["dtype"] == "F8_E4M3" else 2)
                or data_start + end > file_size
            ):
                raise ValueError(f"invalid or incomplete native FP8 PLE tensor: {name}")
            records.append((path, shape[0], shape[1], data_start + start, record["dtype"]))
        if len({record[2] for record in records}) != 1:
            raise ValueError("PLE embedding dimensions differ between shards")
        shard_size = records[0][1]
        if len({record[4] for record in records}) != 1:
            raise ValueError("PLE tensor dtype differs between shards")
        if any(record[1] != shard_size for record in records[:-1]) or records[-1][1] > shard_size:
            raise ValueError("only the final PLE shard may be short")
        shards = tuple(
            PLEShard(
                path=record[0],
                row_start=i * shard_size,
                row_end=i * shard_size + record[1],
                offset=record[3],
                row_bytes=record[2] * (1 if record[4] == "F8_E4M3" else 2),
            )
            for i, record in enumerate(records)
        )
        return cls(
            prefix=matched[0][1],
            embedding_dim=records[0][2],
            total_rows=shards[-1].row_end,
            shard_size=shard_size,
            shards=shards,
            storage={"F8_E4M3": "fp8_e4m3", "BF16": "bf16", "F16": "f16"}[records[0][4]],
        )

    @staticmethod
    def _read_header(path: Path) -> tuple[dict, int, int]:
        with path.open("rb") as source:
            size = source.read(8)
            if len(size) != 8:
                raise ValueError(f"truncated safetensors header: {path}")
            header_size = struct.unpack("<Q", size)[0]
            if not 0 < header_size <= 128 * 1024 * 1024:
                raise ValueError(f"invalid safetensors header size: {path}")
            encoded = source.read(header_size)
            if len(encoded) != header_size:
                raise ValueError(f"truncated safetensors header: {path}")
        return json.loads(encoded), header_size + 8, path.stat().st_size

    @classmethod
    def _from_native_marker(
        cls,
        marker_path: Path,
        *,
        expected_shards: int,
        expected_layout: dict[str, Any] | None,
    ) -> "PLEManifest":
        marker = json.loads(marker_path.read_text())
        config_path = marker_path.parent / "config.json"
        if config_path.is_file():
            config = json.loads(config_path.read_text())
            declared = config.get("qwen_exo_native_ple")
            if not isinstance(declared, dict) or declared.get("schema") != 1:
                raise ValueError("model config must explicitly bind its external PLE source")
            if declared.get("manifest") != marker_path.name or declared.get("manifest_sha256") != hashlib.sha256(marker_path.read_bytes()).hexdigest():
                raise ValueError("external PLE manifest does not match its model identity")
        if marker.get("schema") != 1:
            raise ValueError("unsupported native PLE marker schema")
        storage = str(marker.get("storage", "")).lower()
        storage_aliases = {
            "bf16": ("BF16", 2, False),
            "bfloat16": ("BF16", 2, False),
            "f16": ("F16", 2, False),
            "float16": ("F16", 2, False),
            "fp8_e4m3_rowscale": ("F8_E4M3", 1, True),
        }
        if storage not in storage_aliases:
            raise ValueError(f"unsupported native PLE storage format: {storage!r}")
        data_dtype, dtype_bytes, per_row = storage_aliases[storage]
        try:
            table_root = Path(marker["root"])
            if not table_root.is_absolute():
                raise ValueError("native PLE marker root must be absolute")
            table_root = table_root.resolve()
            num_shards = int(marker["num_shards"])
            rows_per_shard = int(marker["rows_per_shard"])
            total_rows = int(marker["num_embeddings"])
            embedding_dim = int(marker["embedding_dim"])
            source_global_scale = float(marker.get("global_scale", 1.0))
            descriptors = marker["shards"]
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("malformed native PLE marker geometry") from exc
        if num_shards != expected_shards or len(descriptors) != num_shards:
            raise ValueError("native PLE marker shard count does not match the model")
        if rows_per_shard <= 0 or total_rows <= 0 or embedding_dim <= 0:
            raise ValueError("native PLE marker has invalid table geometry")
        if not math.isfinite(source_global_scale) or source_global_scale != 1.0:
            raise ValueError("native PLE native source_global_scale must be exactly 1.0")
        marker_hash = marker.get("hash")
        if not isinstance(marker_hash, dict):
            raise ValueError("native PLE marker is missing its hash/layout record")
        if expected_layout is not None:
            cls._validate_layout(marker_hash, expected_layout)

        shards = []
        prefix = str(marker.get("prefix", "native-ple"))
        for index, descriptor in enumerate(descriptors):
            if not isinstance(descriptor, dict):
                raise ValueError("native PLE shard descriptor is not an object")
            try:
                relative_file = Path(descriptor["file"])
                rows = int(descriptor["rows"])
                declared_size = int(descriptor["size"])
                data_name = str(descriptor["data_tensor"])
                scale_name = descriptor.get("scale_tensor")
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError("malformed native PLE shard descriptor") from exc
            if relative_file.is_absolute():
                raise ValueError("native PLE shard file must be relative to marker root")
            path = (table_root / relative_file).resolve()
            if not path.is_relative_to(table_root):
                raise ValueError("native PLE shard escapes its table root")
            if not path.is_file() or path.stat().st_size != declared_size:
                raise ValueError(f"native PLE shard size mismatch: {relative_file}")
            digest = descriptor.get("sha256")
            if not isinstance(digest, str) or len(digest) != 64:
                raise ValueError("native PLE shard is missing sha256")
            hasher = hashlib.sha256()
            with path.open("rb") as source:
                for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
                    hasher.update(chunk)
            if hasher.hexdigest() != digest.lower():
                raise ValueError(f"native PLE shard hash mismatch: {relative_file}")
            header, data_start, file_size = cls._read_header(path)
            if data_name not in header:
                raise ValueError(f"native PLE data tensor is absent: {data_name}")
            data_record = header[data_name]
            shape = data_record.get("shape")
            start, end = data_record.get("data_offsets", (-1, -1))
            if (
                data_record.get("dtype") != data_dtype
                or shape != [rows, embedding_dim]
                or end - start != rows * embedding_dim * dtype_bytes
                or start < 0
                or data_start + end > file_size
            ):
                raise ValueError(f"native PLE data tensor layout mismatch: {data_name}")
            scale_offset = -1
            scale_row_bytes = 0
            if per_row:
                if not isinstance(scale_name, str) or scale_name not in header:
                    raise ValueError("FP8 native PLE requires a row-scale tensor")
                scale_record = header[scale_name]
                sshape = scale_record.get("shape")
                sstart, send = scale_record.get("data_offsets", (-1, -1))
                if (
                    scale_record.get("dtype") != "F32"
                    or sshape != [rows]
                    or send - sstart != rows * 4
                    or sstart < 0
                    or data_start + send > file_size
                ):
                    raise ValueError(f"native PLE row-scale layout mismatch: {scale_name}")
                scale_offset = data_start + sstart
                scale_row_bytes = 4
            elif isinstance(scale_name, str):
                # BF16/F16 may carry a single source scalar.  Read it per
                # requested row (rather than loading a whole scale tensor).
                if scale_name not in header:
                    raise ValueError(f"native PLE scale tensor is absent: {scale_name}")
                scale_record = header[scale_name]
                sshape = scale_record.get("shape")
                sstart, send = scale_record.get("data_offsets", (-1, -1))
                if (
                    scale_record.get("dtype") != "F32"
                    or not isinstance(sshape, list)
                    or math.prod(sshape) != 1
                    or send - sstart != 4
                    or sstart < 0
                    or data_start + send > file_size
                ):
                    raise ValueError(f"native PLE scalar scale layout mismatch: {scale_name}")
                scale_offset = data_start + sstart
                scale_row_bytes = 4
            if rows <= 0 or rows > rows_per_shard or (rows != rows_per_shard and index != num_shards - 1):
                raise ValueError("only the final native PLE shard may be short")
            shards.append(
                PLEShard(
                    path=path,
                    row_start=index * rows_per_shard,
                    row_end=index * rows_per_shard + rows,
                    offset=data_start + start,
                    row_bytes=embedding_dim * dtype_bytes,
                    scale_offset=scale_offset,
                    scale_row_bytes=scale_row_bytes,
                )
            )
        if shards[-1].row_end != total_rows:
            raise ValueError("native PLE marker total row count does not match shards")
        return cls(
            prefix=prefix,
            embedding_dim=embedding_dim,
            total_rows=total_rows,
            shard_size=rows_per_shard,
            shards=tuple(shards),
            storage={"BF16": "bf16", "F16": "f16", "F8_E4M3": "fp8_e4m3_rowscale"}[data_dtype],
            source_global_scale=source_global_scale,
            external_effective_values=True,
            scale_is_per_row=per_row,
            expected_layout=marker_hash,
        )

    @staticmethod
    def _validate_layout(actual: dict[str, Any], expected: dict[str, Any]) -> None:
        for name, value in expected.items():
            if actual.get(name) != value:
                raise ValueError(f"native PLE hash/layout mismatch for {name}")

    def locate(self, row: int) -> tuple[Path, int]:
        if not 0 <= row < self.total_rows:
            raise IndexError(f"PLE row {row} is outside [0, {self.total_rows})")
        shard = self.shards[min(row // self.shard_size, len(self.shards) - 1)]
        return shard.path, shard.offset + (row - shard.row_start) * shard.row_bytes

    def locate_record(self, row: int) -> PLEShard:
        if not 0 <= row < self.total_rows:
            raise IndexError(f"PLE row {row} is outside [0, {self.total_rows})")
        return self.shards[min(row // self.shard_size, len(self.shards) - 1)]


class PLEFileReader:
    def __init__(self, manifest: PLEManifest, backend: str):
        if backend not in {"pread", "mmap"}:
            raise ValueError("native disk PLE supports only pread or mmap")
        if backend == "pread" and not hasattr(os, "pread"):
            raise ValueError("pread PLE requires POSIX; select mmap on Windows")
        self.manifest = manifest
        self.backend = backend
        self.files = {}
        self.maps = {}

    def _read(self, path: Path, offset: int, size: int) -> bytes:
        if path not in self.files:
            self.files[path] = path.open("rb")
        data = os.pread(self.files[path].fileno(), size, offset)
        if len(data) != size:
            raise OSError(f"incomplete PLE row at {path}:{offset}")
        return data

    def _mapping(self, path: Path) -> mmap.mmap:
        if path not in self.files:
            self.files[path] = path.open("rb")
        if path not in self.maps:
            if os.fstat(self.files[path].fileno()).st_size == 0:
                raise OSError(f"incomplete PLE row at {path}:0")
            self.maps[path] = mmap.mmap(
                self.files[path].fileno(), 0, access=mmap.ACCESS_READ
            )
        return self.maps[path]

    def _read_mmap_rows(self, rows: list[int], data: bytearray, scales: bytearray):
        try:
            row_ids = np.asarray(rows, dtype=np.int64)
        except OverflowError as exc:
            raise IndexError("PLE row is outside the table") from exc
        invalid = (row_ids < 0) | (row_ids >= self.manifest.total_rows)
        if invalid.any():
            row = row_ids[np.flatnonzero(invalid)[0]]
            raise IndexError(f"PLE row {row} is outside [0, {self.manifest.total_rows})")
        row_bytes = self.manifest.shards[0].row_bytes
        output = np.ndarray((len(rows), row_bytes), dtype=np.uint8, buffer=data)
        scale_output = (
            np.ndarray((len(rows), 4), dtype=np.uint8, buffer=scales)
            if scales else None
        )
        shard_ids = np.minimum(
            row_ids // self.manifest.shard_size, len(self.manifest.shards) - 1
        )
        order = np.argsort(shard_ids, kind="stable")
        grouped_shards = shard_ids[order]
        boundaries = np.flatnonzero(grouped_shards[1:] != grouped_shards[:-1]) + 1
        starts = np.concatenate(([0], boundaries))
        ends = np.concatenate((boundaries, [len(rows)]))
        # Only mapped views span a shard; each advanced-index temporary is <=4 MiB.
        chunk_rows = max(1, (4 * 1024 * 1024) // max(row_bytes, 4))
        for start, end in zip(starts, ends):
            shard = self.manifest.shards[int(grouped_shards[start])]
            mapping = self._mapping(shard.path)
            positions = order[start:end]
            local_rows = row_ids[positions] - shard.row_start
            last_row = int(local_rows.max())
            available_rows = min(
                shard.row_end - shard.row_start,
                max(0, (len(mapping) - shard.offset) // shard.row_bytes),
            )
            if local_rows.min() < 0 or last_row >= available_rows:
                raise OSError(f"incomplete PLE row at {shard.path}:{shard.offset}")
            row_view = scale_view = None
            try:
                row_view = np.ndarray(
                    (available_rows, shard.row_bytes), dtype=np.uint8,
                    buffer=mapping, offset=shard.offset,
                )
                if scale_output is not None and shard.scale_offset >= 0:
                    per_row = shard.scale_row_bytes == 4 and self.manifest.scale_is_per_row
                    scale_rows = last_row + 1 if per_row else 1
                    if shard.scale_offset + scale_rows * 4 > len(mapping):
                        raise OSError(f"incomplete PLE row at {shard.path}:{shard.scale_offset}")
                    scale_view = np.ndarray(
                        (scale_rows, 4), dtype=np.uint8,
                        buffer=mapping, offset=shard.scale_offset,
                    )
                for chunk_start in range(0, len(positions), chunk_rows):
                    chunk_end = chunk_start + chunk_rows
                    dest = positions[chunk_start:chunk_end]
                    indices = local_rows[chunk_start:chunk_end]
                    output[dest] = row_view[indices]
                    if scale_output is not None:
                        if scale_view is None:
                            scale_output[dest] = np.frombuffer(struct.pack("<f", 1.0), dtype=np.uint8)
                        elif per_row:
                            scale_output[dest] = scale_view[indices]
                        else:
                            scale_output[dest] = scale_view[0]
            finally:
                # No exported mmap view may survive a return or an exception.
                row_view = scale_view = None

    def read_rows(self, rows: list[int]) -> tuple[bytearray, bytearray]:
        """Return owned writable row/scale buffers, independent of mmap lifetime."""
        data = bytearray(len(rows) * self.manifest.shards[0].row_bytes)
        scale_bytes = 4 if any(s.scale_row_bytes for s in self.manifest.shards) else 0
        scales = bytearray(len(rows) * scale_bytes)
        if not rows:
            return data, scales
        if self.backend == "mmap":
            self._read_mmap_rows(rows, data, scales)
            return data, scales
        unique: dict[int, tuple[bytes, bytes]] = {}
        for i, row in enumerate(rows):
            item = unique.get(row)
            if item is None:
                shard = self.manifest.locate_record(row)
                delta = row - shard.row_start
                raw = self._read(shard.path, shard.offset + delta * shard.row_bytes, shard.row_bytes)
                scale = b""
                if scale_bytes:
                    if shard.scale_offset < 0:
                        scale = struct.pack("<f", 1.0)
                    elif shard.scale_row_bytes == 4 and self.manifest.scale_is_per_row:
                        scale = self._read(shard.path, shard.scale_offset + delta * 4, 4)
                    else:
                        scale = self._read(shard.path, shard.scale_offset, 4)
                item = (raw, scale)
                unique[row] = item
            data_start = i * len(item[0])
            data[data_start : data_start + len(item[0])] = item[0]
            if scale_bytes:
                scale_start = i * 4
                scales[scale_start : scale_start + 4] = item[1]
        return data, scales

    def close(self):
        for mapping in self.maps.values():
            mapping.close()
        for source in self.files.values():
            source.close()
        self.maps.clear()
        self.files.clear()


@dataclass
class _GatherPayload:
    raw_data: bytearray | None = None
    raw_scales: bytearray | None = None
    decoded: torch.Tensor | None = None


class PendingGather(msgspec.Struct):
    future: Future
    shape: tuple[int, ...]



class NVMePLEEmbedding(nn.Module):
    """TP1 eager PLE lookup; staging is proportional to the current token batch."""

    def __init__(
        self,
        snapshot: str | Path,
        *,
        num_embeddings: int,
        embedding_dim: int,
        expected_shards: int,
        expected_layout: dict[str, Any] | None = None,
    ):
        super().__init__()
        self.manifest = PLEManifest.from_snapshot(
            snapshot, expected_shards=expected_shards, expected_layout=expected_layout
        )
        if (self.manifest.total_rows, self.manifest.embedding_dim) != (num_embeddings, embedding_dim):
            raise ValueError("native PLE snapshot geometry does not match the model")
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.tp_size = 1
        self.source_identity = (
            PLESourceIdentity.NATIVE_UNSCALED
            if self.manifest.external_effective_values or self.manifest.storage != "fp8_e4m3"
            else PLESourceIdentity.CHECKPOINT_FP8
        )
        self.source_global_scale = float(self.manifest.source_global_scale)
        self.register_buffer("weight_scale", torch.ones(1, dtype=torch.bfloat16))
        self.reader = PLEFileReader(self.manifest, envs.SGLANG_QWEN4_PLE_NVME_BACKEND.get())
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ple-prefetch")
        # Host/device leases are reused at high-water capacity and cannot be
        # overwritten until the stream event after the last decode/copy signals.
        self._staging_lock = Lock()
        self._pinned_staging = None
        self._pinned_raw = None
        self._pinned_scales = None
        self._cuda_raw = None
        self._cuda_scales = None
        self._staging_capacity = 0
        self._staging_device = None
        self._staging_event = None
        self._staging_event_recorded = False
        self._staging_event_device = None

    @property
    def source_dtype(self) -> torch.dtype:
        return {
            "fp8_e4m3": torch.float8_e4m3fn,
            "fp8_e4m3_rowscale": torch.float8_e4m3fn,
            "bf16": torch.bfloat16,
            "bfloat16": torch.bfloat16,
            "f16": torch.float16,
            "float16": torch.float16,
        }[self.manifest.storage]

    def allocate_output(self, shape, device):
        return torch.empty(shape, dtype=torch.bfloat16, device=device)

    def _decode_raw(
        self, raw_data: bytearray, raw_scales: bytearray, rows: int
    ) -> torch.Tensor:
        if not rows:
            return torch.empty((0, self.embedding_dim), dtype=torch.bfloat16)
        source_bytes = torch.frombuffer(raw_data, dtype=torch.uint8)
        source = source_bytes.view(self.source_dtype).reshape(rows, self.embedding_dim)
        if raw_scales:
            scales = torch.frombuffer(raw_scales, dtype=torch.float32).reshape(rows, 1)
            # Preserve FP8/F16/BF16 -> FP32 * FP32 -> BF16 rounding.
            return source.float().mul_(scales).to(torch.bfloat16)
        return source.to(torch.bfloat16)

    def _read_and_decode(self, rows: list[int], raw_only: bool) -> _GatherPayload:
        raw_data, raw_scales = self.reader.read_rows(rows)
        if raw_only:
            return _GatherPayload(raw_data=raw_data, raw_scales=raw_scales)
        return _GatherPayload(
            decoded=self._decode_raw(raw_data, raw_scales, len(rows))
        )

    def _release_staging(self):
        if self._staging_event_recorded:
            self._staging_event.synchronize()
            self._staging_event_recorded = False

    def _record_staging_event(self, stream):
        if self._staging_event is None or self._staging_event_device != stream.device:
            self._staging_event = torch.cuda.Event()
            self._staging_event_device = stream.device
        self._staging_event.record(stream)
        self._staging_event_recorded = True

    def _ensure_raw_staging(self, rows: int, device: torch.device):
        if self._staging_device != device:
            self._cuda_raw = None
            self._cuda_scales = None
            self._staging_device = device
        if self._pinned_raw is None or self._staging_capacity < rows:
            self._pinned_raw = torch.empty(
                (rows, self.embedding_dim), dtype=torch.uint8, pin_memory=True
            )
            self._pinned_scales = torch.empty(rows, dtype=torch.float32, pin_memory=True)
            self._cuda_raw = torch.empty(
                (rows, self.embedding_dim), dtype=torch.uint8, device=device
            )
            self._cuda_scales = torch.empty(rows, dtype=torch.float32, device=device)
            self._staging_capacity = rows
        elif self._cuda_raw is None:
            self._cuda_raw = torch.empty(
                (self._staging_capacity, self.embedding_dim), dtype=torch.uint8, device=device
            )
            self._cuda_scales = torch.empty(
                self._staging_capacity, dtype=torch.float32, device=device
            )

    def _copy_decoded(self, decoded: torch.Tensor, output: torch.Tensor, shape, device, stream):
        with self._staging_lock, torch.cuda.device(device):
            self._release_staging()
            rows = decoded.shape[0]
            if self._pinned_staging is None or self._pinned_staging.shape[0] < rows:
                self._pinned_staging = torch.empty(
                    (rows, self.embedding_dim), dtype=torch.bfloat16, pin_memory=True
                )
            staging = self._pinned_staging[:rows]
            staging.copy_(decoded)
            copy_stream = stream if stream is not None else torch.cuda.current_stream(device)
            with torch.cuda.stream(copy_stream):
                try:
                    output.copy_(staging.view(shape), non_blocking=True)
                finally:
                    self._record_staging_event(copy_stream)

    def start_gather(self, input_ids: torch.Tensor) -> PendingGather:
        if get_is_capture_mode():
            raise ValueError("disk PLE requires CUDA graph backends disabled")
        ids = input_ids.detach().reshape(-1).to(device="cpu", dtype=torch.int64).tolist()
        raw_only = (
            input_ids.is_cuda
            and self.manifest.storage == "fp8_e4m3_rowscale"
            and len(ids) >= _GPU_DECODE_MIN_ROWS
        )
        return PendingGather(
            future=self.executor.submit(self._read_and_decode, ids, raw_only),
            shape=tuple(input_ids.shape),
        )

    def finish_gather(
        self, pending: PendingGather, device: torch.device, out=None, stream=None
    ):
        if get_is_capture_mode():
            raise ValueError("disk PLE gather must finish before CUDA graph capture")
        payload = pending.future.result()
        shape = (*pending.shape, self.embedding_dim)
        output = out if out is not None else self.allocate_output(shape, device)
        if (
            tuple(output.shape) != shape
            or output.device != device
            or output.dtype != torch.bfloat16
        ):
            raise ValueError("invalid native PLE output buffer")
        row_count = math.prod(pending.shape)
        if row_count == 0:
            return output

        raw_fast = (
            payload.raw_data is not None
            and device.type == "cuda"
            and output.is_contiguous()
        )
        if not raw_fast:
            decoded = payload.decoded
            if decoded is None:
                decoded = self._decode_raw(payload.raw_data, payload.raw_scales, row_count)
            if device.type != "cuda":
                output.copy_(decoded.view(shape))
            else:
                self._copy_decoded(decoded, output, shape, device, stream)
            return output

        from sglang.kernels.ops.qwen4_ple import fused_qwen4_fp8_rowscale

        with self._staging_lock, torch.cuda.device(device):
            self._release_staging()
            self._ensure_raw_staging(row_count, device)
            raw = torch.frombuffer(payload.raw_data, dtype=torch.uint8).view(
                row_count, self.embedding_dim
            )
            scales = torch.frombuffer(payload.raw_scales, dtype=torch.float32)
            self._pinned_raw[:row_count].copy_(raw)
            self._pinned_scales[:row_count].copy_(scales)
            copy_stream = stream if stream is not None else torch.cuda.current_stream(device)
            with torch.cuda.stream(copy_stream):
                try:
                    self._cuda_raw[:row_count].copy_(
                        self._pinned_raw[:row_count], non_blocking=True
                    )
                    self._cuda_scales[:row_count].copy_(
                        self._pinned_scales[:row_count], non_blocking=True
                    )
                    fused_qwen4_fp8_rowscale(
                        self._cuda_raw[:row_count],
                        self._cuda_scales[:row_count],
                        output.view(row_count, self.embedding_dim),
                    )
                finally:
                    # Protect both DMA sources and decode inputs if compilation
                    # or launch fails after either transfer has been enqueued.
                    self._record_staging_event(copy_stream)
        return output

    def gather(self, input_ids, out=None):
        return self.finish_gather(self.start_gather(input_ids), input_ids.device, out=out)

    def reduce(self, output):
        return output

    def forward(self, input_ids):
        return self.gather(input_ids)

    def close(self):
        self.executor.shutdown(wait=True)
        with self._staging_lock:
            self._release_staging()
            self._pinned_staging = None
            self._pinned_raw = None
            self._pinned_scales = None
            self._cuda_raw = None
            self._cuda_scales = None
            self._staging_capacity = 0
            self._staging_device = None
            self._staging_event = None
            self._staging_event_device = None
        self.reader.close()


def is_nvme_ple_embedding(module) -> bool:
    return isinstance(module, NVMePLEEmbedding)
