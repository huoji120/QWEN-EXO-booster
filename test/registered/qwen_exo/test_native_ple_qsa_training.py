"""CPU parity against the installed HF Qwen4Exp indexer, never model data."""
import math
from types import SimpleNamespace

import pytest
import torch

from qwen_exo_booster.native_ple_qsa_training import install_chunked_qsa

native = pytest.importorskip("transformers.models.qwen4_exp.modeling_qwen4_exp")


def _config(ratio=4, budget=8):
    return SimpleNamespace(hidden_size=24, indexer_n_heads=8, indexer_kv_heads=1,
                           indexer_head_dim=8, indexer_budget=budget,
                           indexer_compress_ratio=ratio, rms_norm_eps=1e-6,
                           num_attention_heads=3, num_key_value_heads=1,
                           head_dim=8, attention_dropout=0.0, attention_bias=False,
                           _attn_implementation="sdpa")


def _inputs(length, dtype, batch=2):
    generator = torch.Generator().manual_seed(1729 + length)
    hidden = torch.randn(batch, length, 24, generator=generator)
    angles = torch.randn(batch, length, 4, generator=generator).repeat(1, 1, 2)
    visible = torch.ones(length, length, dtype=torch.bool).tril().expand(batch, 1, length, length).clone()
    mask = visible if dtype == torch.bool else torch.where(visible, 0., torch.finfo(dtype).min).to(dtype)
    return hidden, (angles.cos(), angles.sin()), mask


def _visible(mask):
    return mask if mask.dtype == torch.bool else mask == 0


def _assert_native_selection(indexer, hidden, positions, actual, expected):
    """Demand exact masks except measured score-equivalent Top-K ties."""
    actual, expected = _visible(actual), _visible(expected)
    mismatches = 0
    ratio, dim = indexer.compress_ratio, indexer.index_head_dim
    with torch.no_grad():
        qk = indexer.index_qk_proj(hidden)
        q, keys = qk.split([indexer.index_n_heads * dim, dim], -1)
        q = indexer.q_layernorm(q.reshape(*hidden.shape[:2], indexer.index_n_heads, dim))
        q = native.apply_rotary_pos_emb(q, cos=positions[0], sin=positions[1], unsqueeze_dim=2)
        for batch in range(hidden.shape[0]):
            for query in range(hidden.shape[1]):
                count = (query + 1) // ratio
                got, wanted = actual[batch, 0, query], expected[batch, 0, query]
                assert not got[query + 1:].any()
                assert got[count * ratio:query + 1].all()
                assert got.sum().item() == min(count, indexer.block_topk) * ratio + (query + 1) % ratio
                if torch.equal(got, wanted):
                    continue
                mismatches += 1
                pooled = keys[batch, :count * ratio].reshape(count, ratio, dim).float().mean(1).to(keys.dtype)
                pooled = indexer.k_layernorm(pooled)
                pooled = native.apply_rotary_pos_emb(pooled.unsqueeze(1),
                    cos=positions[0][batch, :count * ratio:ratio],
                    sin=positions[1][batch, :count * ratio:ratio]).squeeze(1)
                scores = (q[batch, query].float() @ pooled.float().T).T.relu().sum(-1) / math.sqrt(dim)
                got_blocks = got[:count * ratio].reshape(count, ratio)
                wanted_blocks = wanted[:count * ratio].reshape(count, ratio)
                assert torch.equal(got_blocks, got_blocks[:, :1].expand_as(got_blocks))
                # A different untied block is a regression, not acceptable drift.
                torch.testing.assert_close(scores[got_blocks[:, 0]].sort().values,
                                           scores[wanted_blocks[:, 0]].sort().values,
                                           rtol=1e-6, atol=1e-6)
    return mismatches


@pytest.mark.parametrize("ratio,budget,length", [(1, 3, 19), (2, 5, 17), (4, 8, 23),
    (4, 32, 15), (4, 4, 3), (3, 3, 1), (3, 0, 13), (4, 8, 20)])
@pytest.mark.parametrize("dtype", [torch.bool, torch.float32, torch.bfloat16])
@pytest.mark.parametrize("chunk", [1, 7, 128])
def test_native_cpu_mask_parity(ratio, budget, length, dtype, chunk, record_property):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(73)
        indexer = native.Qwen4ExpTextQSAIndexer(_config(ratio, budget), 0)
    hidden, positions, mask = _inputs(length, dtype)
    with torch.no_grad():
        expected = indexer(hidden, positions, mask, None)
    install_chunked_qsa(indexer, native, chunk)
    actual = indexer(hidden, positions, mask, None)
    assert actual.dtype == expected.dtype and actual.shape == expected.shape
    mismatches = _assert_native_selection(indexer, hidden, positions, actual, expected)
    record_property("score_equivalent_tie_queries", mismatches)
    if dtype != torch.bool:
        assert ((actual == 0) | (actual == torch.finfo(dtype).min)).all()


def test_tied_scores_keep_budget_complete_blocks_and_causal_tail():
    indexer = native.Qwen4ExpTextQSAIndexer(_config(3, 6), 0)
    with torch.no_grad():
        indexer.index_qk_proj.weight.zero_()
    hidden, positions, mask = _inputs(25, torch.bool)
    expected = indexer(hidden, positions, mask, None)
    install_chunked_qsa(indexer, native, 8)
    actual = indexer(hidden, positions, mask, None)
    _assert_native_selection(indexer, hidden, positions, actual, expected)


@pytest.mark.parametrize("corruption", ["hole", "future", "shape", "dtype", "cache", "positions"])
def test_unsupported_contracts_fail_closed(corruption):
    model = torch.nn.ModuleList([native.Qwen4ExpTextQSAIndexer(_config(), i) for i in range(2)])
    install_chunked_qsa(model, native, 3)
    hidden, positions, mask = _inputs(9, torch.bool)
    model[0](hidden, positions, mask, None)
    # Mutating the same tensor must invalidate the cross-layer verified cache.
    cache = None
    if corruption == "hole":
        mask[:, :, 5, 2] = False
    elif corruption == "future":
        mask[:, :, 2, 5] = True
    elif corruption == "shape":
        mask = mask[:, :, :, :-1]
    elif corruption == "dtype":
        mask = mask.to(torch.int64)
    elif corruption == "cache":
        cache = object()
    else:
        positions = tuple(value[:, :-1] for value in positions)
    with pytest.raises(ValueError, match="chunked_qsa"):
        model[1](hidden, positions, mask, cache)


def test_parameter_ownership_and_native_attention_backward_are_preserved():
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(413)
        attention = native.Qwen4ExpTextAttention(_config(2, 32), 0)
    hidden, positions, mask = _inputs(11, torch.bool)
    hidden.requires_grad_()
    expected = attention(hidden, positions, mask)[0]
    expected.square().sum().backward()
    gradient = hidden.grad.detach().clone()
    parameter_gradients = {name: parameter.grad.detach().clone() for name, parameter in attention.named_parameters()
                           if parameter.grad is not None}
    owners = dict(attention.named_parameters())
    original_indexer = attention.indexer
    attention.zero_grad(set_to_none=True)
    hidden.grad = None
    install_chunked_qsa(attention, native, 3)
    assert attention.indexer is original_indexer
    assert dict(attention.named_parameters()).keys() == owners.keys()
    assert all(parameter is owners[name] for name, parameter in attention.named_parameters())
    modes = []
    hook = attention.indexer.index_qk_proj.register_forward_pre_hook(
        lambda module, args: modes.append(torch.is_grad_enabled()))
    actual = attention(hidden, positions, mask)[0]
    hook.remove()
    actual.square().sum().backward()
    assert modes == [False]
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(hidden.grad, gradient)
    for name, parameter in attention.named_parameters():
        if name in parameter_gradients:
            torch.testing.assert_close(parameter.grad, parameter_gradients[name])
        else:
            assert parameter.grad is None
