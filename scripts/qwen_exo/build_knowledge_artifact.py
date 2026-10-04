#!/usr/bin/env python3
"""Package a trained sparse table alongside its unchanged frozen Engram base.

The root manifest references the original base table and reader. The additional
table/reader live under knowledge/, have identical hash addressing, and can be
disabled independently. Unwritten knowledge rows read zero.

    python scripts/qwen_exo/build_knowledge_artifact.py \
        --table-pt /root/autodl-tmp/engram-exp/runs/wt27_table.pt \
        --base-artifact /root/autodl-tmp/engram/flash-next-ple-mix27 \
        --qwen38-config /root/autodl-tmp/qwen38-ple/config.json \
        --name cyber-traj-wt27 --out /root/autodl-tmp/engram/cyber-traj-wt27
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import msgspec
import torch
from safetensors.torch import save_file

from qwen_exo_booster.engram import segment_reference_rows
from qwen_exo_booster.engram_artifact import (
    MANIFEST_NAME,
    MANIFEST_SCHEMA,
    EngramManifest,
    EngramReaderSpec,
    EngramTableSpec,
    hash_spec_from_qwen38_config,
)

READER_KEYS = ("in_norm.weight", "key.weight", "value.weight", "h_norm.weight", "k_norm.weight")


def check_pairs(spec, *, count: int, seed: int):
    gen = torch.Generator().manual_seed(seed)
    tokens = torch.randint(0, spec.vocab_size, (count,), generator=gen)
    tokens[torch.randperm(count, generator=gen)[: count // 8]] = spec.eos_id
    rows = segment_reference_rows(tokens, spec)
    padded = torch.cat([torch.full((2,), spec.eos_id), tokens])
    triples = [(int(padded[t + 2]), int(padded[t + 1]), int(padded[t])) for t in range(count)]
    return triples, [tuple(int(v) for v in row) for row in rows]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--table-pt", type=Path, required=True)
    p.add_argument("--base-artifact", type=Path, required=True)
    p.add_argument("--qwen38-config", type=Path, required=True)
    p.add_argument("--name", required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--checks", type=int, default=64)
    args = p.parse_args()

    blob = torch.load(args.table_pt, map_location="cpu", weights_only=True, mmap=True)
    training = blob["training"]
    if training["epochs_complete"] < 1:
        raise SystemExit("Refusing to publish a partial training run")
    if training["target_tokens_seen"] != training["args"]["epochs"] * training["target_tokens_available"]:
        raise SystemExit("Training target coverage does not match completed epochs")
    base = EngramManifest.load(args.base_artifact)
    base.validate(hidden_size=blob["reader"]["key.weight"].shape[0], num_layers=int(blob["inject_layer"]) + 1)
    if hashlib.sha256((Path(base.root) / MANIFEST_NAME).read_bytes()).hexdigest() != training["provenance"]["base_manifest_sha256"]:
        raise SystemExit("Frozen base manifest differs from the training base")
    if hashlib.sha256(base.resolve(base.reader.file).read_bytes()).hexdigest() != training["provenance"]["base_reader_sha256"]:
        raise SystemExit("Frozen base reader differs from the training base")
    uniq = blob["uniq"].to(torch.int64).contiguous()
    weight = blob["weight"].to(torch.bfloat16).contiguous()
    reader = {k: blob["reader"][k].float().contiguous() for k in READER_KEYS}
    layer = int(blob["inject_layer"])
    head_dim = weight.shape[1]
    config = json.loads(args.qwen38_config.read_text())
    spec = hash_spec_from_qwen38_config(config)
    spec.validate()
    if base.hash != spec or base.reader.layer != layer:
        raise SystemExit("Frozen and trained tables must share addressing and injection layer")
    if spec.num_heads * head_dim != reader["key.weight"].shape[1]:
        raise SystemExit("reader embed_dim does not match hash heads x head_dim")
    if not bool(torch.isfinite(weight).all()) or not all(bool(torch.isfinite(value).all()) for value in reader.values()):
        raise SystemExit("Refusing to publish non-finite table or reader weights")
    knowledge_dir = args.out / "knowledge"
    knowledge_dir.mkdir(parents=True, exist_ok=True)
    save_file({"uniq": uniq, "weight": weight}, str(knowledge_dir / "sparse_table.safetensors"))
    save_file(reader, str(knowledge_dir / "reader.safetensors"),
              metadata={"source": str(args.table_pt.resolve())})
    check_tokens, check_rows = check_pairs(spec, count=args.checks, seed=0)
    manifest = EngramManifest(
        schema=MANIFEST_SCHEMA,
        name=args.name,
        hash=spec,
        table=EngramTableSpec(kind="sparse_trained", head_dim=head_dim,
                              rows_file="sparse_table.safetensors", num_rows=uniq.numel(), dtype="bfloat16"),
        reader=EngramReaderSpec(
            file="reader.safetensors",
            layer=layer,
            hidden_size=reader["key.weight"].shape[0],
            embed_dim=reader["key.weight"].shape[1],
            sha256=hashlib.sha256((knowledge_dir / "reader.safetensors").read_bytes()).hexdigest(),
        ),
        check_tokens=tuple(check_tokens),
        check_rows=tuple(check_rows),
    )
    encoded = msgspec.json.encode(msgspec.structs.replace(manifest, root=""))
    (knowledge_dir / MANIFEST_NAME).write_bytes(msgspec.json.format(encoded, indent=1))
    (args.out / "training.json").write_text(json.dumps(training, indent=2), encoding="utf-8")
    bundle = msgspec.structs.replace(
        base, name=f"{base.name}+{args.name}", root="", knowledge_path="knowledge",
        table=msgspec.structs.replace(base.table, dir=str(base.resolve(base.table.dir))),
        reader=msgspec.structs.replace(base.reader, file=str(base.resolve(base.reader.file))),
    )
    encoded = msgspec.json.encode(bundle)
    (args.out / MANIFEST_NAME).write_bytes(msgspec.json.format(encoded, indent=1))
    EngramManifest.load(knowledge_dir).validate(hidden_size=manifest.reader.hidden_size, num_layers=layer + 1)
    print(f"wrote {args.out / MANIFEST_NAME}: frozen base {base.name} + "
          f"{uniq.numel()} independently trained rows x {head_dim}, "
          f"knowledge table {weight.numel() * 2 / 1e9:.2f} GB bf16 in host RAM")


if __name__ == "__main__":
    main()
