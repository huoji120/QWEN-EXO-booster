import json
import random
from pathlib import Path

import pytest
import torch

from qwen_exo_booster.engram import (
    ENGRAM_DISABLED_CACHE_MARKER,
    EngramHashSpec,
    EngramHashTensors,
    EngramReaderWeights,
    engram_radix_extra_key,
    extend_inputs_host,
    hash_rows,
    is_confident_surface_token,
    reader_delta,
    ring_update_and_read,
    ring_width_for,
    segment_reference_rows,
    validate_engram_server_config,
)

EOS = 7
SPEC = EngramHashSpec(
    vocab_size=50,
    eos_id=EOS,
    ngram_size=3,
    heads_per_order=2,
    multipliers=(2_147_483_659, 1_000_000_007, 998_244_353),
    head_sizes=(101, 103, 107, 109),
    head_offsets=(0, 101, 204, 311),
)


def _sequence(length, seed, eos_every=6):
    rng = random.Random(seed)
    return [EOS if rng.random() < 1 / eos_every else rng.randrange(SPEC.vocab_size) for _ in range(length)]


def test_extend_triples_hash_to_segment_rows_for_any_prefix_split():
    """Host triples + the trigram EOS rule must equal the official segment rule
    for every prefix / chunk boundary (radix hits, chunked prefill, 1-token
    extends after a prefix)."""
    tensors = EngramHashTensors(SPEC, "cpu")
    for seed in range(20):
        seq = _sequence(40, seed)
        reference = segment_reference_rows(torch.tensor(seq), SPEC)
        for pre, ext in [(0, 40), (1, 1), (2, 5), (17, 1), (17, 23), (39, 1)]:
            host = extend_inputs_host(
                fill_ids=[seq],
                req_pool_indices=[3],
                prefix_lens=[pre],
                extend_lens=[ext],
                request_mask=[True],
                ring_width=16,
                vocab_size=SPEC.vocab_size,
                eos_id=EOS,
            )
            x0, x1, x2 = torch.from_numpy(host.tokens)
            assert torch.equal(hash_rows(x0, x1, x2, tensors), reference[pre : pre + ext])
            # Ring seed = the tokens at the last two positions (prefix included).
            end = pre + ext
            expected = {(p % 16): seq[p] for p in (end - 2, end - 1) if p >= 0}
            assert dict(zip(host.tail_cols.tolist(), host.tail_tokens.tolist())) == expected


def _simulate_dflash(ring_width, *, seed, width=9, rounds=40):
    """Prefill, then DFLASH verify blocks with random accept lengths and padded
    graph rows (req 0, position 0). Returns the first history mismatch."""
    rng = random.Random(seed)
    ring = torch.full((4, ring_width), EOS, dtype=torch.int64)
    committed = _sequence(rng.randrange(1, 30), seed)
    host = extend_inputs_host(
        fill_ids=[committed],
        req_pool_indices=[2],
        prefix_lens=[0],
        extend_lens=[len(committed)],
        request_mask=[True],
        ring_width=ring_width,
        vocab_size=SPEC.vocab_size,
        eos_id=EOS,
    )
    ring[torch.from_numpy(host.tail_rows), torch.from_numpy(host.tail_cols)] = torch.from_numpy(host.tail_tokens)
    bonus = rng.randrange(SPEC.vocab_size)
    for _ in range(rounds):
        start = len(committed)
        block = [bonus] + [rng.randrange(SPEC.vocab_size) for _ in range(width - 1)]
        positions = torch.tensor([0] * width + list(range(start, start + width)))
        input_ids = torch.tensor([rng.randrange(SPEC.vocab_size) for _ in range(width)] + block)
        _, x1, x2 = ring_update_and_read(
            ring=ring,
            req_pool_indices=torch.tensor([0, 2]),
            positions=positions,
            input_ids=input_ids,
            eos_id=EOS,
        )
        visible = committed + block
        for j in range(width):
            p = start + j
            want = (visible[p - 1] if p >= 1 else EOS, visible[p - 2] if p >= 2 else EOS)
            if (int(x1[width + j]), int(x2[width + j])) != want:
                return p
        accept = rng.randrange(width)
        committed = committed + block[: accept + 1]
        bonus = rng.randrange(SPEC.vocab_size)
    return None


def test_ring_history_survives_rejected_drafts():
    """Writing each verify block before reading keeps the history equal to the
    committed tokens: rejected drafts past the commit point are overwritten
    before any later position reads them."""
    width = 9
    ring_width = ring_width_for(width)
    assert ring_width >= width + 2
    assert all(_simulate_dflash(ring_width, seed=s, width=width) is None for s in range(30))
    # A (power-of-two) ring narrower than block + 2 lets the block overwrite
    # its own history.
    assert any(_simulate_dflash(8, seed=s, width=width) is not None for s in range(30))


def test_reader_delta_matches_trained_adapter_formula():
    """Same math as the offline PleAdapterV2 (fp32 RMSNorm with weight, stacked
    key/value projections) given its state_dict."""
    gen = torch.Generator().manual_seed(0)
    embed, hidden = 32, 24
    state = {
        "in_norm.weight": torch.rand(embed, generator=gen) + 0.5,
        "key.weight": torch.randn(hidden, embed, generator=gen),
        "value.weight": torch.randn(hidden, embed, generator=gen),
        "h_norm.weight": torch.rand(hidden, generator=gen) + 0.5,
        "k_norm.weight": torch.rand(hidden, generator=gen) + 0.5,
    }
    weights = EngramReaderWeights.from_state_dict(state, eps=1e-6, device="cpu")
    h = torch.randn(5, hidden, generator=gen).to(torch.bfloat16)
    e = torch.randn(5, embed, generator=gen).to(torch.bfloat16)

    def rms(x, w):
        x32 = x.float()
        return (x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + 1e-6)) * w

    e_n = rms(e, state["in_norm.weight"])
    gate = torch.sigmoid(
        (rms(h, state["h_norm.weight"]) * rms(e_n @ state["key.weight"].T, state["k_norm.weight"])).sum(
            -1, keepdim=True
        )
        * hidden**-0.5
    )
    expected = (gate * (e_n @ state["value.weight"].T)).to(h.dtype)
    torch.testing.assert_close(reader_delta(h, e, weights).float(), expected.float(), rtol=1e-2, atol=1e-2)


def test_confident_surface_tokens_are_suppressed():
    """Numeric / hex / separator tokens (the 10.0.186.74 regression: an Engram
    nudge flipped a correct IP octet) suppress injection; plain word tokens do
    not, so Engram still acts on language."""
    for text in (" 10", "0", "186", "3f9", "v2", ".", "...", ":", "/", "-", " 2.14"):
        assert is_confident_surface_token(text)
    for text in (" server", "reachable", "the", "", "   ", " and", "http"):
        assert not is_confident_surface_token(text)


def test_opted_out_requests_get_their_own_radix_namespace():
    assert engram_radix_extra_key("qwen-exo=abc", {}) == "qwen-exo=abc"
    off = engram_radix_extra_key("qwen-exo=abc", {"qwen_exo_engram": False})
    assert off == f"qwen-exo=abc|{ENGRAM_DISABLED_CACHE_MARKER}"
    assert engram_radix_extra_key(off, {"qwen_exo_engram": False}) == off
    assert engram_radix_extra_key(None, {"qwen_exo_engram": False}) == ENGRAM_DISABLED_CACHE_MARKER


@pytest.mark.parametrize(
    "override",
    [
        {"tp_size": 2},
        {"enable_mixed_chunk": True},
        {"speculative_algorithm": "EAGLE"},
        {"cuda_graph_backend_prefill": "breakable"},
        {"enable_qwen_exo": False},
    ],
)
def test_unsupported_server_configs_are_rejected(override):
    config = dict(
        enable_qwen_exo=True,
        tp_size=1,
        pp_size=1,
        dp_size=1,
        enable_dp_attention=False,
        enable_two_batch_overlap=False,
        enable_mixed_chunk=False,
        enable_torch_compile=False,
        speculative_algorithm="DFLASH",
        cuda_graph_backend_prefill="disabled",
    )
    validate_engram_server_config(**config)
    with pytest.raises(ValueError):
        validate_engram_server_config(**{**config, **override})


def _write_table(table_dir: Path, *, num_shards: int, rows_per_shard: int, head_dim: int):
    from safetensors.torch import save_file

    table_dir.mkdir()
    rows = num_shards * rows_per_shard
    values = torch.arange(rows * head_dim, dtype=torch.float32).reshape(rows, head_dim) % 23 - 11
    scales = torch.arange(1, rows + 1, dtype=torch.float32) / 64
    for i in range(num_shards):
        part = slice(i * rows_per_shard, (i + 1) * rows_per_shard)
        save_file(
            {"data": values[part].to(torch.float8_e4m3fn), "scale": scales[part].contiguous()},
            str(table_dir / f"shard_{i}.safetensors"),
        )
    return values.to(torch.float8_e4m3fn).float() * scales[:, None]


def _manifest(tmp_path: Path, *, num_shards: int, rows_per_shard: int, head_dim: int):
    from qwen_exo_booster.engram_artifact import EngramManifest

    payload = {
        "schema": 1,
        "name": "test",
        "hash": json.loads(json.dumps({f: getattr(SPEC, f) for f in SPEC.__struct_fields__})),
        "table": {"dir": "table", "num_shards": num_shards, "rows_per_shard": rows_per_shard, "head_dim": head_dim},
        "reader": {"file": "reader.safetensors", "layer": 1, "hidden_size": 8, "embed_dim": 4 * head_dim},
    }
    (tmp_path / "engram.json").write_text(json.dumps(payload))
    return EngramManifest.load(tmp_path)


def test_table_rows_follow_numeric_shard_order(tmp_path):
    """Global row r lives in shard r // rows_per_shard: shard_10 must follow
    shard_9 (a lexicographic listing would put it after shard_1)."""
    from qwen_exo_booster.engram_artifact import HostEngramTable, gather_rows_reference

    expected = _write_table(tmp_path / "table", num_shards=12, rows_per_shard=3, head_dim=16)
    manifest = _manifest(tmp_path, num_shards=12, rows_per_shard=3, head_dim=16)
    table = HostEngramTable.load(manifest, pin=False, threads=3)
    rows = torch.tensor([0, 5, 29, 30, 35])
    got = gather_rows_reference(table.data, table.scale, rows)
    assert torch.equal(got, expected[rows].to(torch.bfloat16))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_jit_gather_from_pinned_host_matches_reference_inside_cuda_graph(tmp_path):
    """The zero-copy kernel reads the registered host table bit-exactly, also
    when the ring / hash / gather path is replayed from a captured graph."""
    from qwen_exo_booster.engram_artifact import HostEngramTable, gather_rows_reference
    from sglang.jit_kernel.qwen_exo_engram import engram_gather_dequant

    _write_table(tmp_path / "table", num_shards=4, rows_per_shard=128, head_dim=160)
    manifest = _manifest(tmp_path, num_shards=4, rows_per_shard=128, head_dim=160)
    table = HostEngramTable.load(manifest, pin=True)
    tensors = EngramHashTensors(SPEC, "cuda")
    ring = torch.full((4, 16), EOS, dtype=torch.int64, device="cuda")
    req = torch.tensor([1, 2], device="cuda")
    positions = torch.zeros(18, dtype=torch.int64, device="cuda")
    input_ids = torch.zeros(18, dtype=torch.int64, device="cuda")
    out = torch.empty((18 * SPEC.num_heads, 160), dtype=torch.bfloat16, device="cuda")

    def step():
        x0, x1, x2 = ring_update_and_read(
            ring=ring, req_pool_indices=req, positions=positions, input_ids=input_ids, eos_id=EOS
        )
        rows = hash_rows(x0, x1, x2, tensors)
        engram_gather_dequant(
            table_data=table.data, table_scale=table.scale, rows=rows.reshape(-1), out=out, num_heads=SPEC.num_heads
        )
        return rows

    step()  # compile + warm up outside capture
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        rows = step()
    for trial in range(3):
        positions.copy_(torch.arange(18, device="cuda") % 9 + 5 * trial)
        input_ids.copy_(torch.randint(0, SPEC.vocab_size, (18,), device="cuda"))
        graph.replay()
        reference = gather_rows_reference(table.data, table.scale, rows.cpu())
        assert torch.equal(out.cpu(), reference.reshape(out.shape))
