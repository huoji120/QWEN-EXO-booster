"""Numeric coverage for context-bounded QSA decode scoring and its top-k contract."""

import math
import sys
from types import SimpleNamespace

import pytest
import torch

from sglang.srt.layers.attention.qsa import mqa
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=60, stage="base-b-kernel-unit", runner_config="1-gpu-large")
register_cuda_ci(est_time=60, stage="base-b-kernel-unit", runner_config="4-gpu-b200")

CUDA_TILELANG = torch.cuda.is_available() and mqa.HAS_TILELANG


def _inputs(lengths, page_size, max_model_len, device, heads=4, head_dim=64):
    generator = torch.Generator(device=device).manual_seed(43463)
    rows = len(lengths)
    used_pages = math.ceil(max(lengths, default=0) / page_size)
    cache_pages = max(used_pages + 3, 1)
    cache = torch.randn(
        cache_pages, page_size, 1, head_dim,
        generator=generator, dtype=torch.bfloat16, device=device,
    )
    q = torch.randn(
        rows, heads, head_dim,
        generator=generator, dtype=torch.bfloat16, device=device,
    )
    # Deliberately wider than either the output or actual contexts. Invalid
    # trailing pages must never be loaded by the context-bounded kernel.
    width = max(used_pages, math.ceil(max_model_len / page_size)) + 37
    table = torch.full((rows, width), -1, dtype=torch.int32, device=device)
    for row, length in enumerate(lengths):
        count = math.ceil(length / page_size)
        table[row, :count] = torch.randperm(
            cache_pages, generator=generator, device=device
        )[:count].to(torch.int32)
    lens = torch.tensor(lengths, dtype=torch.int32, device=device)
    return q, cache, table, lens


@pytest.mark.parametrize("page_size", [8, 16, 32, 64])
def test_qsa_decode_cpu_fallback_keeps_all_scores(page_size):
    args = _inputs([0, 1, 511, 512, 513], page_size, 1031, "cpu")
    expected = mqa.torch_qsa_mqa_decode(*args, 1031)
    # CPU top-k reads even short-row scores; min_scored_len must not change
    # the reference fallback's fully initialized matrix.
    actual = mqa.qsa_mqa_decode(*args, 1031, min_scored_len=512)
    torch.testing.assert_close(actual, expected)
    assert torch.isfinite(actual[1, 0])
    assert torch.isneginf(actual[0]).all()


@pytest.mark.parametrize(
    "sms,rows,expected",
    [(1, 4, 1), (1, 3, 16), (4, 1, 16), (5, 1, 256),
     (132, 1, 256), (132, 33, 16), (132, 528, 1), (132, 0, 256)],
)
def test_qsa_decode_cta_tiers_depend_on_rows_and_sms(monkeypatch, sms, rows, expected):
    monkeypatch.setattr(mqa, "_num_sms", lambda device_index: sms)
    assert mqa._decode_ctas_per_row(rows, torch.device("cuda:1")) == expected


@pytest.mark.skipif(not CUDA_TILELANG, reason="CUDA and TileLang required")
@pytest.mark.parametrize("page_size", [8, 16, 32, 64])
@pytest.mark.parametrize("max_model_len", [513, 4099])
def test_qsa_decode_kernel_boundary_scores_and_padding(page_size, max_model_len):
    args = _inputs([0, 1, 511, 512, 513], page_size, max_model_len, "cuda")
    expected = mqa.torch_qsa_mqa_decode(*args, max_model_len)
    # Four query heads exercise the kernel's eight-head zero padding too.
    actual = mqa.tilelang_qsa_mqa_decode(*args, max_model_len)
    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)
    for row, length in enumerate([0, 1, 511, 512, 513]):
        assert torch.isneginf(actual[row, length:]).all()


@pytest.mark.skipif(not CUDA_TILELANG, reason="CUDA and TileLang required")
@pytest.mark.parametrize("ctas_per_row", [1, 16, 256])
def test_qsa_decode_kernel_strides_groups_and_skips_only_short_rows(ctas_per_row):
    from sglang.kernels.ops.elementwise.fast_topk import fast_topk

    lengths = [0, 1, 511, 512, 513, 17001]
    # 17001 spans more than 256 64-key groups, forcing multiple iterations
    # even at the largest CTA tier. Partial sub-pages and output tails matter.
    args = _inputs(lengths, 16, 17003, "cuda", heads=8)
    q, cache, table, lens = args
    expected = mqa.torch_qsa_mqa_decode(*args, 17003)
    logits = torch.full_like(expected, float("nan"))
    mqa._tilelang_qsa_mqa_decode_kernel(
        heads=8, head_dim=64, page_size=16,
        ctas_per_row=ctas_per_row, min_scored_len=512,
    )(q.unsqueeze(1), cache, table, lens, logits, 8.0)
    for row, length in enumerate(lengths):
        if length <= 512:
            assert torch.isnan(logits[row]).all()
        else:
            torch.testing.assert_close(
                logits[row, :length], expected[row, :length], rtol=2e-2, atol=2e-2
            )
            assert torch.isnan(logits[row, length:]).all()

    # Poisoned omitted scores must not influence either branch of the real
    # JIT top-k: <=512 returns every key, >512 never loads the trailing NaNs.
    selected = fast_topk(logits, lens, topk=512)
    for row, length in enumerate(lengths):
        if length <= 512:
            torch.testing.assert_close(
                selected[row, :length],
                torch.arange(length, device="cuda", dtype=torch.int32),
            )
            assert (selected[row, length:] == -1).all()
        else:
            assert selected[row].unique().numel() == 512
            assert ((selected[row] >= 0) & (selected[row] < length)).all()
            # Test selection against its own computed scores, independent of
            # small allowed GEMM differences from the FP32 reference.
            selected_scores = logits[row, selected[row].long()]
            threshold = torch.topk(logits[row, :length], 512).values[-1]
            assert (selected_scores >= threshold).all()


@pytest.mark.skipif(not CUDA_TILELANG, reason="CUDA and TileLang required")
def test_qsa_decode_batch_invariant_mode_never_omits_scores(monkeypatch):
    args = _inputs([0, 1, 511, 512, 513], 16, 1031, "cuda")
    expected = mqa.torch_qsa_mqa_decode(*args, 1031)
    monkeypatch.setattr(mqa, "is_batch_invariant_mode_enabled", lambda: True)
    actual = mqa.tilelang_qsa_mqa_decode(*args, 1031, min_scored_len=512)
    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)


@pytest.mark.skipif(not CUDA_TILELANG, reason="CUDA and TileLang required")
@pytest.mark.parametrize("max_model_len", [0, 1, 511])
def test_qsa_decode_bounds_stores_by_output_width(max_model_len):
    args = _inputs([513], 16, max_model_len, "cuda")
    expected = mqa.torch_qsa_mqa_decode(*args, max_model_len)
    actual = mqa.tilelang_qsa_mqa_decode(*args, max_model_len)
    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)


@pytest.mark.skipif(not CUDA_TILELANG, reason="CUDA and TileLang required")
def test_qsa_decode_selection_preserves_shared_mtp_prototype():
    from sglang.srt.environ import envs
    from sglang.srt.layers.attention.qsa.kernel import torch_expand_qsa_block_indices
    from sglang.srt.layers.attention.qsa.qsa_indexer import QSAIndexer

    lengths = [0, 511, 512, 513]
    q, cache, table, lens = _inputs(lengths, 16, 1031, "cuda", heads=16)
    # The prototype requires one identical page table per four-row group.
    table[:] = table[-1:].expand_as(table)
    # Unique, exactly representable scores avoid cutoff ties and verify exact
    # long-row sets, not just approximate logits. Two BF16 components encode
    # the physical token rank without BF16 rounding above 256.
    ranks = torch.arange(cache.shape[0] * 16, device="cuda").view(-1, 16)
    cache.zero_()
    cache[:, :, 0, 0] = ranks // 128
    cache[:, :, 0, 1] = (ranks % 128).float() / 128
    q.zero_()
    q[:, :, :2] = 1
    positions = lens * 4 + 1
    sequence_lengths = positions + 1
    config = SimpleNamespace(block_topk=512, compress_ratio=4, token_topk=2048)
    scores = mqa.torch_qsa_mqa_decode(q, cache, table, lens, 1031)
    blocks = torch.full((4, 512), -1, device="cuda", dtype=torch.int32)
    for row, length in enumerate(lengths):
        width = min(length, 512)
        if width:
            blocks[row, :width] = scores[row, :length].topk(width).indices.int()
    expected = torch_expand_qsa_block_indices(
        blocks, positions, sequence_lengths, compress_ratio=4, token_topk=2048
    )
    for shared in (False, True):
        with envs.SGLANG_QWEN4_EXP_QSA_SHARED_MTP.override(shared):
            actual = QSAIndexer.select_decode_tokens(
                config, q, cache, table, lens, 1031, positions, sequence_lengths,
                share_mtp_queries=True,
            )
        # Selection ordering is unspecified; the attention-visible set is not.
        torch.testing.assert_close(
            actual.sort(dim=-1).values, expected.sort(dim=-1).values
        )
    shared_scores = mqa.tilelang_qsa_mqa_decode_shared(q, cache, table, lens, 1031)
    assert shared_scores is not None
    torch.testing.assert_close(shared_scores, scores, rtol=2e-2, atol=2e-2)


@pytest.mark.skipif(not CUDA_TILELANG, reason="CUDA and TileLang required")
def test_qsa_decode_graph_replay_crosses_score_free_threshold(monkeypatch):
    from sglang.kernels.ops.elementwise.fast_topk import fast_topk

    monkeypatch.setattr(mqa, "is_batch_invariant_mode_enabled", lambda: False)
    args = _inputs([0, 1, 511, 512, 513], 16, 1031, "cuda")
    q, cache, table, lens = args
    table[:] = table[-1:].expand_as(table)

    def select():
        logits = mqa.tilelang_qsa_mqa_decode(*args, 1031, min_scored_len=512)
        return logits, fast_topk(logits, lens, topk=512)

    select()  # Compile both real kernels before graph capture.
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        logits, selected = select()
    for lengths in ([513, 512, 1, 511, 0], [0, 1, 511, 512, 513]):
        lens.copy_(torch.tensor(lengths, dtype=torch.int32, device="cuda"))
        graph.replay()
        expected = mqa.torch_qsa_mqa_decode(q, cache, table, lens, 1031)
        for row, length in enumerate(lengths):
            if length <= 512:
                torch.testing.assert_close(
                    selected[row, :length],
                    torch.arange(length, dtype=torch.int32, device="cuda"),
                )
                assert (selected[row, length:] == -1).all()
            else:
                torch.testing.assert_close(
                    logits[row, :length], expected[row, :length],
                    rtol=2e-2, atol=2e-2,
                )
                assert selected[row].unique().numel() == 512
                assert ((selected[row] >= 0) & (selected[row] < length)).all()
                threshold = torch.topk(logits[row, :length], 512).values[-1]
                assert (logits[row, selected[row].long()] >= threshold).all()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
