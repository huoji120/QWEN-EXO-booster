"""DFLASH on an NVFP4 KV cache: draft KV dtype, draft pool budget, and the
XQA draft-block verify used by trtllm_mha for native FP4 KV."""

from types import SimpleNamespace

import pytest
import torch

from sglang.srt.mem_cache.kv_cache_dtype import dflash_draft_kv_cache_dtype
from sglang.srt.utils import is_sm120_supported
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=20, stage="base-b", runner_config="1-gpu-large")

FP4 = getattr(torch, "float4_e2m1fn_x2", None)


@pytest.mark.skipif(FP4 is None, reason="torch without float4_e2m1fn_x2")
def test_draft_kv_dtype_never_inherits_fp4():
    """The DFLASH draft pool is a plain per-slot pool whose FlashInfer backend
    has no FP4 dequant-workspace or native-FP4 path; inheriting the target's
    NVFP4 dtype made the draft attention backend reject its own KV."""
    rule = lambda target, backend: dflash_draft_kv_cache_dtype(
        target_kv_cache_dtype=target,
        model_dtype=torch.bfloat16,
        speculative_draft_attention_backend=backend,
    )
    assert rule(FP4, "flashinfer") == torch.float8_e4m3fn
    assert rule(torch.float8_e4m3fn, "flashinfer") == torch.float8_e4m3fn
    assert rule(FP4, "fa4") == torch.bfloat16


@pytest.mark.skipif(FP4 is None, reason="torch without float4_e2m1fn_x2")
def test_target_pool_budget_uses_draft_geometry_and_dtype(monkeypatch):
    """The draft pool has one slot per target slot. Scaling the target's
    per-layer cost by layer count assumed equal per-layer KV: 1.15 KB/layer for
    the NVFP4 target vs 2 KB/layer for the 8x128 FP8 draft, under-reserving
    ~5 GB at 1.3M tokens."""
    from sglang.srt.model_executor import pool_configurator

    monkeypatch.setattr(
        pool_configurator, "get_parallel", lambda: SimpleNamespace(attn_tp_size=2)
    )
    kvc = SimpleNamespace(
        spec_aux_config=SimpleNamespace(
            dflash_draft_total_kv_heads=8, dflash_draft_kv_head_dims=256
        ),
        kv_cache_dtype=FP4,
        model_dtype=torch.bfloat16,
        server_args=SimpleNamespace(speculative_draft_attention_backend="flashinfer"),
    )
    cell = pool_configurator._dflash_draft_cell_size(kvc, draft_num_layers=5)
    assert cell == 5 * (8 // 2) * 256
    kvc.spec_aux_config.dflash_draft_total_kv_heads = None
    assert pool_configurator._dflash_draft_cell_size(kvc, draft_num_layers=5) is None


def test_causal_draft_block_mask_matches_xqa_bit_layout():
    """XQA expects uint16 words, 32-bit aligned, bit j of row i = draft token
    i may see draft token j (external kernel format)."""
    from sglang.srt.layers.attention.trtllm_mha_backend import causal_draft_block_mask

    mask = causal_draft_block_mask(3, 20, "cpu")
    assert mask.dtype == torch.uint16 and mask.shape == (3, 20, 2)
    rows = mask[1].to(torch.int64)
    assert rows[0].tolist() == [1, 0]
    assert rows[15].tolist() == [0xFFFF, 0]
    assert rows[19].tolist() == [0xFFFF, 0xF]


@pytest.mark.skipif(
    not (torch.cuda.is_available() and is_sm120_supported()), reason="SM120 XQA only"
)
def test_xqa_draft_block_verify_on_nvfp4_matches_causal_reference():
    """Verify token i at position L - Q + i must see exactly KV [0, L - Q + i]
    of the dequantized cache, for the sglang paged HND view of the NVFP4 pool."""
    import flashinfer
    from flashinfer import nvfp4_kv_dequantize, nvfp4_kv_quantize

    from sglang.srt.layers.attention.trtllm_mha_backend import causal_draft_block_mask

    torch.manual_seed(0)
    hq, hkv, d, page, q_len = 24, 4, 256, 64, 8
    lens = [700, 3003]
    pages = (max(lens) + page - 1) // page
    table = torch.randperm(len(lens) * pages, device="cuda", dtype=torch.int32).view(
        len(lens), pages
    )
    slots = table.numel() * page
    one = torch.ones(1, device="cuda")

    def quantize(x):
        data, scale = nvfp4_kv_quantize(x.view(-1, d), one)
        data = data.view(slots, hkv, d // 2)
        scale = scale.view(torch.uint8).reshape(slots, hkv, d // 16)
        deq = nvfp4_kv_dequantize(
            data.reshape(-1, d // 2),
            scale.reshape(-1, d // 16),
            one,
            output_dtype=torch.bfloat16,
        )
        return data, scale, deq.view(slots, hkv, d)

    k4, ks, kd = quantize(torch.randn(slots, hkv, d, device="cuda").bfloat16())
    v4, vs, vd = quantize(torch.randn(slots, hkv, d, device="cuda").bfloat16())
    hnd = lambda buf, last: buf.view(-1, page, hkv, last).permute(0, 2, 1, 3)
    q = torch.randn(len(lens) * q_len, hq, d, device="cuda").bfloat16()
    out = flashinfer.decode.trtllm_batch_decode_with_kv_cache(
        query=q,
        kv_cache=(hnd(k4, d // 2), hnd(v4, d // 2)),
        workspace_buffer=torch.zeros(128 << 20, dtype=torch.uint8, device="cuda"),
        block_tables=table,
        seq_lens=torch.tensor(lens, device="cuda", dtype=torch.int32),
        max_seq_len=8192,
        bmm1_scale=d**-0.5,
        bmm2_scale=1.0,
        out_dtype=torch.bfloat16,
        q_len_per_req=q_len,
        mask=causal_draft_block_mask(len(lens), q_len, "cuda"),
        kv_cache_sf=(
            hnd(ks.view(torch.float8_e4m3fn), d // 16),
            hnd(vs.view(torch.float8_e4m3fn), d // 16),
        ),
    ).view(len(lens), q_len, hq, d)
    for b, n in enumerate(lens):
        tok = (
            table[b].long()[:, None] * page + torch.arange(page, device="cuda")
        ).flatten()[:n]
        kk = kd[tok].float().repeat_interleave(hq // hkv, dim=1)
        vv = vd[tok].float().repeat_interleave(hq // hkv, dim=1)
        for i in range(q_len):
            visible = n - q_len + i + 1
            qi = q.view(len(lens), q_len, hq, d)[b, i].float()
            p = torch.einsum("hd,nhd->hn", qi, kk[:visible]).mul(d**-0.5).softmax(-1)
            ref = torch.einsum("hn,nhd->hd", p, vv[:visible])
            assert ((out[b, i].float() - ref).norm() / ref.norm()).item() < 0.01
