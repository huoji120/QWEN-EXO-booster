#!/usr/bin/env python3
"""Build an Engram artifact (engram.json + reader.safetensors) for QWEN-EXO.

The table shards are referenced in place (no 52 GB copy). The hash comes from a
Qwen3.8-Flash-Next config.json; the self-check rows come from the independent
segment-rule reference, and ``--offline-dir`` additionally cross-checks against
the offline ``qwen38_ple.PleSpec`` (verified bit-exact against transformers).

    python scripts/qwen_exo/build_engram_artifact.py \
        --qwen38-config /root/autodl-tmp/qwen38-ple/config.json \
        --table-dir /root/autodl-tmp/qwen38-ple/table \
        --reader /root/autodl-tmp/engram-exp/runs/mix27_real_adapter.pt \
        --name flash-next-ple-mix27 --out /root/autodl-tmp/engram/flash-next-ple-mix27 \
        --offline-dir /root/autodl-tmp/engram-exp
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
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


def reader_state(path: Path) -> tuple[int, dict[str, torch.Tensor]]:
    """Offline adapter state_dict ("<layer>.<param>") -> (layer, reader tensors)."""
    state = torch.load(path, map_location="cpu", weights_only=True)
    layers = {key.split(".", 1)[0] for key in state}
    if len(layers) != 1 or not next(iter(layers)).isdigit():
        raise SystemExit(f"{path}: expected one '<layer>.' prefixed linear reader, got {sorted(state)}")
    layer = int(layers.pop())
    tensors = {key.split(".", 1)[1]: value.float().contiguous() for key, value in state.items()}
    if set(tensors) != set(READER_KEYS):
        raise SystemExit(f"{path}: reader must be linear without conv, got {sorted(tensors)}")
    return layer, tensors


def check_pairs(spec, *, count: int, seed: int) -> tuple[list, list]:
    """(x0, x1, x2) -> rows pairs from random sequences with EOS breaks."""
    gen = torch.Generator().manual_seed(seed)
    tokens = torch.randint(0, spec.vocab_size, (count,), generator=gen)
    tokens[torch.randperm(count, generator=gen)[: count // 8]] = spec.eos_id
    rows = segment_reference_rows(tokens, spec)
    padded = torch.cat([torch.full((2,), spec.eos_id), tokens])
    triples = [(int(padded[t + 2]), int(padded[t + 1]), int(padded[t])) for t in range(count)]
    return triples, [tuple(int(v) for v in row) for row in rows]


def cross_check_offline(offline_dir: Path, config_path: Path, spec) -> None:
    sys.path.insert(0, str(offline_dir))
    from qwen38_ple import PleSpec  # noqa: PLC0415 - offline experiment code

    ple = PleSpec.from_config(config_path)
    if (ple.multipliers, ple.head_sizes, ple.head_offsets) != (spec.multipliers, spec.head_sizes, spec.head_offsets):
        raise SystemExit("hash constants differ from the offline PleSpec")
    gen = torch.Generator().manual_seed(1)
    tokens = torch.randint(0, spec.vocab_size, (4, 512), generator=gen)
    tokens[0, ::37] = spec.eos_id
    for row in range(tokens.shape[0]):
        ours = segment_reference_rows(tokens[row], spec)
        theirs = ple.row_ids(tokens[row : row + 1])[0]
        if not torch.equal(ours, theirs):
            raise SystemExit("segment reference rows differ from the offline PleSpec")
    print("offline PleSpec cross-check: OK")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--qwen38-config", type=Path, required=True)
    p.add_argument("--table-dir", type=Path, required=True)
    p.add_argument("--reader", type=Path, required=True)
    p.add_argument("--name", required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--offline-dir", type=Path, default=None)
    p.add_argument("--checks", type=int, default=64)
    args = p.parse_args()

    config = json.loads(args.qwen38_config.read_text())
    text = config.get("text_config", config)
    spec = hash_spec_from_qwen38_config(config)
    spec.validate()
    total = spec.head_offsets[-1] + spec.head_sizes[-1]
    divisor = text["make_ngram_vocab_size_divisible_by"]
    padded = math.ceil(total / divisor) * divisor
    parts = text["split_ngram_parts"]
    table = EngramTableSpec(
        dir=str(args.table_dir.resolve()),
        num_shards=parts,
        rows_per_shard=padded // parts,
        head_dim=text["ple_embed_dim"] // spec.num_heads,
    )
    if args.offline_dir is not None:
        cross_check_offline(args.offline_dir, args.qwen38_config, spec)

    layer, tensors = reader_state(args.reader)
    args.out.mkdir(parents=True, exist_ok=True)
    reader_path = args.out / "reader.safetensors"
    save_file(tensors, str(reader_path), metadata={"source": str(args.reader.resolve())})
    check_tokens, check_rows = check_pairs(spec, count=args.checks, seed=0)
    manifest = EngramManifest(
        schema=MANIFEST_SCHEMA,
        name=args.name,
        hash=spec,
        table=table,
        reader=EngramReaderSpec(
            file=reader_path.name,
            layer=layer,
            hidden_size=tensors["key.weight"].shape[0],
            embed_dim=tensors["key.weight"].shape[1],
            sha256=hashlib.sha256(reader_path.read_bytes()).hexdigest(),
        ),
        check_tokens=tuple(check_tokens),
        check_rows=tuple(check_rows),
    )
    encoded = msgspec.json.encode(msgspec.structs.replace(manifest, root=""))
    (args.out / MANIFEST_NAME).write_bytes(msgspec.json.format(encoded, indent=1))
    EngramManifest.load(args.out).validate(hidden_size=manifest.reader.hidden_size, num_layers=layer + 1)
    print(f"wrote {args.out / MANIFEST_NAME}: layer {layer}, {table.rows} rows x {table.head_dim}")


if __name__ == "__main__":
    main()
