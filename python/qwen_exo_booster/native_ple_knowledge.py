"""Native PLE sparse knowledge deltas; no backbone loading or serving side effects."""
from __future__ import annotations

import hashlib
import json
import struct
from pathlib import Path

import msgspec
import torch
from torch import nn
from torch.nn import functional as F

from qwen_exo_booster.engram import EngramHashSpec, EngramHashTensors, hash_rows
from qwen_exo_booster.engram_artifact import hash_spec_from_qwen38_config


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def native_reader_sha256(root: Path, layer_id: int) -> str:
    """Hash reader tensor ranges only, without loading the backbone or giant table."""
    index = json.loads((root / "model.safetensors.index.json").read_text(encoding="utf-8"))
    prefix = f".layers.{layer_id}.ple."
    names = sorted(name for name in index["weight_map"] if prefix in name and ".ngram_embedding." not in name)
    if not names or not any(".key_proj.weight" in name for name in names):
        raise ValueError("Native checkpoint index lacks its frozen PLE reader")
    digest = hashlib.sha256()
    for name in names:
        path = (root / index["weight_map"][name]).resolve()
        if not path.is_relative_to(root):
            raise ValueError("Native reader shard escapes the checkpoint profile")
        with path.open("rb") as source:
            size_bytes = source.read(8)
            if len(size_bytes) != 8:
                raise ValueError("Native reader shard has a truncated header")
            header_size = struct.unpack("<Q", size_bytes)[0]
            if not 0 < header_size <= 128 * 1024 ** 2:
                raise ValueError("Native reader shard has an invalid header size")
            header = json.loads(source.read(header_size))
            entry = header[name]
            start, end = entry["data_offsets"]
            if not 0 <= start < end or 8 + header_size + end > path.stat().st_size:
                raise ValueError("Native reader tensor range is invalid")
            digest.update(json.dumps([name, entry["dtype"], entry["shape"]], separators=(",", ":")).encode())
            source.seek(8 + header_size + start)
            remaining = end - start
            while remaining:
                chunk = source.read(min(1024 * 1024, remaining))
                if not chunk:
                    raise ValueError("Native reader tensor is truncated")
                digest.update(chunk)
                remaining -= len(chunk)
    return digest.hexdigest()


class NativePLEIdentity(msgspec.Struct, frozen=True, kw_only=True):
    model_config_sha256: str
    native_ple_manifest_sha256: str
    index_sha256: str
    reader_sha256: str
    hash: EngramHashSpec
    head_dim: int
    num_embeddings: int
    layer_id: int
    hidden_size: int
    hc_count: int

    @classmethod
    def from_profile(cls, model_path: str | Path) -> "NativePLEIdentity":
        root = Path(model_path).resolve()
        config_path = root / "config.json"
        config = json.loads(config_path.read_text(encoding="utf-8"))
        if config.get("architectures") != ["Qwen4ExpForConditionalGeneration"]:
            raise ValueError("Native PLE preparation requires the exact Qwen4Exp architecture")
        text = config["text_config"]
        binding = config.get("qwen_exo_native_ple")
        if not isinstance(binding, dict) or binding.get("schema") != 1 or binding.get("manifest") != "native-ple.json":
            raise ValueError("Native PLE preparation requires a hash-bound external base table")
        marker = root / binding["manifest"]
        manifest_hash = sha256_file(marker)
        if manifest_hash != binding["manifest_sha256"]:
            raise ValueError("Native PLE manifest differs from its model binding")
        manifest = json.loads(marker.read_text(encoding="utf-8"))
        if manifest.get("schema") != 1 or manifest.get("global_scale") != 1.0:
            raise ValueError("Native PLE base must declare effective unscaled row values")
        spec = msgspec.convert(manifest["hash"], type=EngramHashSpec)
        spec.validate()
        if spec != hash_spec_from_qwen38_config(config):
            raise ValueError("Native PLE addressing differs from checkpoint seed/head geometry")
        if (spec.vocab_size, spec.eos_id, spec.ngram_size, spec.heads_per_order) != (
            text["vocab_size"], text["eos_token_id"], text["ngram_size"], text["heads_per_ngram"]
        ):
            raise ValueError("Native PLE hash layout differs from the model")
        dim, rows = int(manifest["embedding_dim"]), int(manifest["num_embeddings"])
        if dim * spec.num_heads != int(text["ple_embed_dim"]):
            raise ValueError("Native PLE row width does not match the native reader")
        if any(offset < 0 or size <= 0 or offset + size > rows for offset, size in zip(spec.head_offsets, spec.head_sizes)):
            raise ValueError("Native PLE head addresses escape the table")
        if len(text["ple_layer_ids"]) != 1:
            raise ValueError("Sparse knowledge preparation currently targets one native PLE layer")
        return cls(
            model_config_sha256=sha256_file(config_path),
            native_ple_manifest_sha256=manifest_hash,
            index_sha256=sha256_file(root / "model.safetensors.index.json"),
            reader_sha256=native_reader_sha256(root, int(text["ple_layer_ids"][0]) - 1),
            hash=spec, head_dim=dim, num_embeddings=rows,
            layer_id=int(text["ple_layer_ids"][0]) - 1,
            hidden_size=int(text["hidden_size"]), hc_count=int(text["hc_count"]),
        )

    def to_dict(self) -> dict:
        return msgspec.json.decode(msgspec.json.encode(self))

    def fingerprint(self) -> str:
        encoded = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(encoded).hexdigest()


def native_ple_row_keys(input_ids: torch.Tensor, identity: NativePLEIdentity,
                        history: torch.Tensor | None = None) -> torch.Tensor:
    """Causal row ids [N, heads]; history is the two preceding source tokens."""
    if input_ids.ndim != 1 or input_ids.dtype not in (torch.int32, torch.int64):
        raise ValueError("Native PLE input ids must be a one-dimensional integer tensor")
    spec = identity.hash
    if bool(((input_ids < 0) | (input_ids >= spec.vocab_size)).any()):
        raise ValueError("Native PLE token ids are outside the exact tokenizer vocabulary")
    if history is None:
        history = input_ids.new_full((2,), spec.eos_id)
    if history.shape != (2,) or history.dtype not in (torch.int32, torch.int64):
        raise ValueError("Native PLE history must contain exactly two integer tokens")
    history = history.to(device=input_ids.device, dtype=torch.int64)
    if bool(((history < 0) | (history >= spec.vocab_size)).any()):
        raise ValueError("Native PLE history is outside the tokenizer vocabulary")
    sequence = torch.cat((history, input_ids.long()))
    return hash_rows(sequence[2:], sequence[1:-1], sequence[:-2],
                     EngramHashTensors(spec, input_ids.device))


class SparseNativePLEDelta(nn.Module):
    """Train only independent rows, then add their values before native reader projections."""
    def __init__(self, row_ids: torch.Tensor, identity: NativePLEIdentity, *, device="cpu"):
        super().__init__()
        rows = row_ids.to(device=device, dtype=torch.int64)
        if rows.ndim != 1 or not rows.numel() or not bool((rows[1:] > rows[:-1]).all()):
            raise ValueError("Knowledge row ids must be nonempty, sorted and unique")
        if bool(((rows < 0) | (rows >= identity.num_embeddings)).any()):
            raise ValueError("Knowledge row ids escape the native PLE table")
        self.identity = identity
        self.register_buffer("row_ids", rows)
        self.weight = nn.Parameter(torch.zeros((rows.numel(), identity.head_dim), device=device))
        self.register_buffer("shuffle_indices", torch.arange(rows.numel(), device=device))

    @torch.no_grad()
    def set_shuffle_seed(self, seed: int) -> None:
        """Shuffle only delta values within each head; never shuffle the frozen base."""
        generator = torch.Generator(device="cpu").manual_seed(seed)
        rows_cpu = self.row_ids.cpu()
        permutation = torch.arange(rows_cpu.numel())
        for offset, width in zip(self.identity.hash.head_offsets, self.identity.hash.head_sizes):
            positions = torch.where((rows_cpu >= offset) & (rows_cpu < offset + width))[0]
            permutation[positions] = positions[torch.randperm(positions.numel(), generator=generator)]
        self.shuffle_indices.copy_(permutation.to(self.shuffle_indices.device))

    def forward(self, row_keys: torch.Tensor, *, mode="real", token_mask=None) -> torch.Tensor:
        if mode not in {"off", "real", "shuffled"}:
            raise ValueError("Knowledge mode must be off, real or shuffled")
        if row_keys.device != self.row_ids.device or row_keys.dtype != torch.int64:
            raise ValueError("Knowledge row keys must be int64 on the delta device")
        result = self.weight.new_zeros((*row_keys.shape, self.identity.head_dim))
        if mode == "off" or not row_keys.numel():
            return result
        indices = torch.searchsorted(self.row_ids, row_keys).clamp(max=self.row_ids.numel() - 1)
        hit = self.row_ids[indices] == row_keys
        if token_mask is not None:
            if token_mask.shape != row_keys.shape[:-1] or token_mask.dtype != torch.bool:
                raise ValueError("Knowledge token mask must match source-token dimensions")
            hit = hit & token_mask.unsqueeze(-1)
        source = indices[hit]
        if mode == "shuffled":
            source = self.shuffle_indices[source]
        result[hit] = F.embedding(source, self.weight, sparse=True)
        return result

    def export(self, output: str | Path, *, training: dict, tokenizer_fingerprint: str) -> Path:
        """Export immutable delta bytes and provenance, never source text or base weights."""
        from safetensors.torch import save_file
        if not tokenizer_fingerprint:
            raise ValueError("Knowledge export requires the exact tokenizer fingerprint")
        if not training.get("gradient_gate_passed") or training.get("epochs_complete", 0) < 1:
            raise ValueError("Knowledge export requires real native-backbone gradients and a completed epoch")
        if not bool(torch.isfinite(self.weight).all()):
            raise ValueError("Knowledge delta contains non-finite values")
        root = Path(output)
        root.mkdir(parents=True, exist_ok=False)
        tensor = root / "delta.safetensors"
        save_file({"row_ids": self.row_ids.detach().cpu().contiguous(),
                   "values": self.weight.detach().to(device="cpu", dtype=torch.bfloat16).contiguous()}, str(tensor))
        manifest = {"schema": 1, "kind": "native_ple_sparse_delta", "identity": self.identity.to_dict(),
                    "identity_fingerprint": self.identity.fingerprint(), "tokenizer_fingerprint": tokenizer_fingerprint,
                    "tensor_file": tensor.name, "tensor_sha256": sha256_file(tensor), "training": training}
        path = root / "native-ple-knowledge.json"
        path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        return path

    @classmethod
    def load(cls, manifest_path: str | Path, identity: NativePLEIdentity, *,
             tokenizer_fingerprint: str, device="cpu") -> "SparseNativePLEDelta":
        from safetensors.torch import load_file
        path = Path(manifest_path).resolve()
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if manifest.get("schema") != 1 or manifest.get("kind") != "native_ple_sparse_delta":
            raise ValueError("Unsupported native PLE knowledge artifact")
        if manifest.get("identity_fingerprint") != identity.fingerprint() or manifest.get("identity") != identity.to_dict():
            raise ValueError("Knowledge delta belongs to another model, native reader or PLE table")
        if manifest.get("tokenizer_fingerprint") != tokenizer_fingerprint:
            raise ValueError("Knowledge delta belongs to another tokenizer/template")
        tensor = (path.parent / manifest["tensor_file"]).resolve()
        if not tensor.is_relative_to(path.parent) or sha256_file(tensor) != manifest["tensor_sha256"]:
            raise ValueError("Knowledge tensor escapes its artifact or has changed bytes")
        blob = load_file(str(tensor), device="cpu")
        delta = cls(blob["row_ids"], identity, device=device)
        values = blob["values"]
        if values.shape != delta.weight.shape or not bool(torch.isfinite(values).all()):
            raise ValueError("Knowledge tensor values have invalid shape or non-finite entries")
        with torch.no_grad():
            delta.weight.copy_(values.to(device=device, dtype=delta.weight.dtype))
        return delta
