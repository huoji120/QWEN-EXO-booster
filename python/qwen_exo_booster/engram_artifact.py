"""Engram artifact: manifest, host-memory table, and reader weights.

An artifact directory holds ``engram.json`` (msgspec ``EngramManifest``), a
reader safetensors file, and a reference to the table shards. Table shards are
safetensors files ``<table_dir>/shard_{i}.safetensors`` with ``data``
(F8_E4M3 [rows_per_shard, head_dim]) and ``scale`` (F32 [rows_per_shard]);
global row ``r`` lives in shard ``r // rows_per_shard``.

The whole table is copied into one anonymous, ``cudaHostRegister``ed host
buffer so a GPU kernel can gather rows zero-copy; each shard's page cache is
dropped right after its copy so the file cache does not double the footprint.
"""

from __future__ import annotations

import json
import logging
import math
import os
import struct
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import msgspec
import numpy as np
import torch

from qwen_exo_booster.engram import EngramHashSpec, EngramReaderWeights

logger = logging.getLogger(__name__)

MANIFEST_NAME = "engram.json"
MANIFEST_SCHEMA = 1


class EngramTableSpec(msgspec.Struct, frozen=True, kw_only=True):
    dir: str
    num_shards: int
    rows_per_shard: int
    head_dim: int
    dtype: str = "fp8_e4m3_rowscale"

    @property
    def rows(self) -> int:
        return self.num_shards * self.rows_per_shard


class EngramReaderSpec(msgspec.Struct, frozen=True, kw_only=True):
    file: str
    layer: int
    hidden_size: int
    embed_dim: int
    eps: float = 1e-6
    sha256: str = ""


class EngramManifest(msgspec.Struct, frozen=True, kw_only=True):
    schema: int
    name: str
    hash: EngramHashSpec
    table: EngramTableSpec
    reader: EngramReaderSpec
    # Hash self-check: check_tokens[i] = (x0, x1, x2) must map to check_rows[i].
    check_tokens: tuple[tuple[int, int, int], ...] = ()
    check_rows: tuple[tuple[int, ...], ...] = ()
    root: str = ""

    @classmethod
    def load(cls, path: str | os.PathLike) -> "EngramManifest":
        path = Path(path)
        if path.is_dir():
            path = path / MANIFEST_NAME
        manifest = msgspec.json.decode(path.read_bytes(), type=cls)
        return msgspec.structs.replace(manifest, root=str(path.parent))

    def resolve(self, relative: str) -> Path:
        candidate = Path(relative)
        return candidate if candidate.is_absolute() else Path(self.root) / candidate

    def validate(self, *, hidden_size: int, num_layers: int) -> None:
        if self.schema != MANIFEST_SCHEMA:
            raise ValueError(f"Engram manifest schema {self.schema} != {MANIFEST_SCHEMA}")
        self.hash.validate()
        if self.table.dtype != "fp8_e4m3_rowscale":
            raise ValueError(f"Unsupported Engram table dtype {self.table.dtype!r}")
        if self.hash.num_heads * self.table.head_dim != self.reader.embed_dim:
            raise ValueError("Engram heads x head_dim must equal the reader embed_dim")
        if max(o + s for o, s in zip(self.hash.head_offsets, self.hash.head_sizes)) > self.table.rows:
            raise ValueError("Engram hash addresses rows beyond the table")
        if self.reader.hidden_size != hidden_size:
            raise ValueError(
                f"Engram reader hidden size {self.reader.hidden_size} does not match the model ({hidden_size})"
            )
        # Layer 0 has no residual yet, so post_residual_addition would be dropped.
        if not 1 <= self.reader.layer < num_layers:
            raise ValueError(f"Engram reader layer {self.reader.layer} outside [1, {num_layers})")
        if len(self.check_tokens) != len(self.check_rows):
            raise ValueError("Engram check_tokens and check_rows differ in length")


_MASK64 = (1 << 64) - 1
_GAMMA = 0x9E3779B97F4A7C15
_LAYER_PRIME = 10007


def _splitmix64(value: int) -> int:
    value = (value + _GAMMA) & _MASK64
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & _MASK64
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & _MASK64
    return (value ^ (value >> 31)) & _MASK64


def _next_prime(value: int) -> int:
    value += 1
    while value < 2 or any(value % d == 0 for d in range(2, math.isqrt(value) + 1)):
        value += 1
    return value


def hash_spec_from_qwen38_config(config: dict, *, ple_layer_index: int = 0) -> EngramHashSpec:
    """The PLE hash of a Qwen3.8 (``qwen4_exp``) config.json, as transformers
    builds it: splitmix64 multipliers and the k-th primes after the base."""
    text = config.get("text_config", config)
    vocab, n, per = text["vocab_size"], text["ngram_size"], text["heads_per_ngram"]
    eos = text["eos_token_id"]
    eos = eos[0] if isinstance(eos, list) else eos
    half_bound = max(1, ((1 << 63) - 1) // max(vocab, 1) // 2)
    base_seed = text.get("seed", 1234) + _LAYER_PRIME * ple_layer_index
    multipliers = tuple(
        2 * (_splitmix64((base_seed + _GAMMA * (i + 1)) & _MASK64) % half_bound) + 1 for i in range(n)
    )
    heads = (n - 1) * per
    prime = text["ngram_vocab_size_base"] - 1
    for _ in range(ple_layer_index * heads):
        prime = _next_prime(prime)
    sizes, offsets, total = [], [], 0
    for _ in range(heads):
        prime = _next_prime(prime)
        sizes.append(prime)
        offsets.append(total)
        total += prime
    return EngramHashSpec(
        vocab_size=vocab,
        eos_id=eos,
        ngram_size=n,
        heads_per_order=per,
        multipliers=multipliers,
        head_sizes=tuple(sizes),
        head_offsets=tuple(offsets),
    )


def _safetensors_header(f) -> tuple[int, dict]:
    (length,) = struct.unpack("<Q", f.read(8))
    return 8 + length, json.loads(f.read(length))


def _read_shard(path: Path, data: np.ndarray, scale: np.ndarray, table: EngramTableSpec) -> None:
    """Copy one shard's rows into ``data`` / ``scale`` (views of the host buffers)."""
    with open(path, "rb", buffering=0) as f:
        base, header = _safetensors_header(f)
        expected = {
            "data": ("F8_E4M3", [table.rows_per_shard, table.head_dim], data),
            "scale": ("F32", [table.rows_per_shard], scale),
        }
        for name, (dtype, shape, dest) in expected.items():
            meta = header[name]
            if meta["dtype"] != dtype or meta["shape"] != shape:
                raise ValueError(f"{path}: {name} is {meta['dtype']}{meta['shape']}, expected {dtype}{shape}")
            begin, end = meta["data_offsets"]
            view = memoryview(dest.reshape(-1).view(np.uint8))
            if end - begin != view.nbytes:
                raise ValueError(f"{path}: {name} holds {end - begin} bytes, expected {view.nbytes}")
            f.seek(base + begin)
            done = 0
            while done < view.nbytes:
                n = f.readinto(view[done:])
                if not n:
                    raise ValueError(f"{path}: truncated {name}")
                done += n
        os.posix_fadvise(f.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)


def cgroup_memory_current() -> int | None:
    try:
        return int(Path("/sys/fs/cgroup/memory.current").read_text())
    except (OSError, ValueError):
        return None


class HostEngramTable:
    """The whole table in host memory: ``data`` u8 [rows, head_dim], ``scale`` f32 [rows]."""

    def __init__(self, data: torch.Tensor, scale: torch.Tensor):
        self.data = data
        self.scale = scale

    @classmethod
    def load(cls, manifest: EngramManifest, *, pin: bool, threads: int = 8) -> "HostEngramTable":
        from sglang.srt.mem_cache.mmap_allocator import alloc_mmap

        table = manifest.table
        started = time.monotonic()
        data = alloc_mmap((table.rows, table.head_dim), torch.uint8)
        scale = alloc_mmap((table.rows,), torch.float32)
        data_np, scale_np = data.numpy(), scale.numpy()
        table_dir = manifest.resolve(table.dir)
        # Numeric shard order: shard_10 must follow shard_9, not shard_1.
        jobs = [
            (
                table_dir / f"shard_{i}.safetensors",
                data_np[i * table.rows_per_shard : (i + 1) * table.rows_per_shard],
                scale_np[i * table.rows_per_shard : (i + 1) * table.rows_per_shard],
            )
            for i in range(table.num_shards)
        ]
        with ThreadPoolExecutor(max_workers=threads) as pool:
            list(pool.map(lambda job: _read_shard(*job, table), jobs))
        read_s = time.monotonic() - started
        if pin:
            from sglang.srt.mem_cache.pool_host.common import _cuda_host_register

            _cuda_host_register(data)
            _cuda_host_register(scale)
        logger.info(
            "Engram table %s: %d rows x %d (%.1f GB) read in %.1fs, pinned in %.1fs, cgroup memory %s",
            manifest.name,
            table.rows,
            table.head_dim,
            (data.numel() + scale.numel() * 4) / 1e9,
            read_s,
            time.monotonic() - started - read_s,
            cgroup_memory_current(),
        )
        return cls(data, scale)


def gather_rows_reference(data: torch.Tensor, scale: torch.Tensor, rows: torch.Tensor) -> torch.Tensor:
    """CPU reference: rows [...] -> dequantized bf16 [..., head_dim] (fp8 x fp32 scale, RNE)."""
    flat = rows.reshape(-1).cpu()
    vals = data.index_select(0, flat).view(torch.float8_e4m3fn).float()
    vals = vals * scale.index_select(0, flat)[:, None]
    return vals.to(torch.bfloat16).reshape(*rows.shape, data.shape[1])


def load_reader_weights(manifest: EngramManifest, *, device) -> EngramReaderWeights:
    from safetensors.torch import load_file

    state = load_file(str(manifest.resolve(manifest.reader.file)))
    weights = EngramReaderWeights.from_state_dict(state, eps=manifest.reader.eps, device=device)
    if tuple(weights.kv.shape) != (2 * manifest.reader.hidden_size, manifest.reader.embed_dim):
        raise ValueError(f"Engram reader projections have shape {tuple(weights.kv.shape)}")
    return weights


__all__ = [
    "EngramManifest",
    "EngramReaderSpec",
    "EngramTableSpec",
    "HostEngramTable",
    "MANIFEST_NAME",
    "cgroup_memory_current",
    "gather_rows_reference",
    "hash_spec_from_qwen38_config",
    "load_reader_weights",
]
