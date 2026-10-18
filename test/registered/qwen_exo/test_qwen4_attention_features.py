from types import SimpleNamespace

import pytest
import torch

from qwen_exo_booster.attention_diagnostic_capture import capture_attention_diagnostic


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA diagnostic capture required")
@pytest.mark.parametrize("cached", [False, True])
def test_qsa_diagnostic_uses_selected_positions_and_native_key_precision(monkeypatch, cached):
    from qwen_exo_booster import attention_diagnostic_capture as capture
    from sglang.srt.model_executor import runner
    from sglang.srt import runtime_context

    device = "cuda"
    query = torch.tensor([[[.7, -.2], [.3, .8]]] * 4, device=device)
    keys = torch.tensor([[[.125, .375]], [[20., 20.]], [[.625, -.875]], [[.25, .5]]], device=device)
    mapping = torch.tensor([[0, 0, 0, 0], [1, 2, 3, 4]], device=device)
    cached_keys = torch.cat([torch.zeros_like(keys[:1]), keys * 2])
    full_pool = SimpleNamespace(
        is_quantized_kv_cache=False, dtype=torch.float8_e4m3fn,
        size=8, page_size=1, head_num=1, head_dim=2,
    )
    backend = type("QwenSparseAttnBackend", (), {})()
    backend.forward_metadata = SimpleNamespace(
        prefill_metadata=SimpleNamespace(kv_locations=torch.arange(4, device=device) if cached else None)
    )
    backend.req_to_token_pool = SimpleNamespace(req_to_token=mapping)
    backend.token_to_kv_pool = SimpleNamespace(
        get_kv_tokens=lambda layer, slots, dtype: (cached_keys[slots].to(dtype), None)
    )
    monkeypatch.setattr(capture, "_diagnostic_backend", lambda: (backend, full_pool))
    monkeypatch.setattr(runner, "get_is_capture_mode", lambda: False)
    monkeypatch.setattr(runtime_context, "get_parallel", lambda: SimpleNamespace(
        pp_size=1, attn_dp_size=1, attn_cp_size=1, attn_dcp_size=1, attn_tp_size=1,
    ))
    layer = SimpleNamespace(
        layer_id=0, is_cross_attention=False, logit_cap=0., sliding_window_size=-1,
        xai_temperature_len=-1, attn_type=SimpleNamespace(value="decoder"),
        pos_encoding_mode="NONE", k_scale=1., v_scale=1., k_scale_float=.5, scaling=.7,
    )
    decoder = SimpleNamespace(attn=layer, num_heads=2, num_kv_heads=1, head_dim=2,
                              config=SimpleNamespace(layers_block_type=["qwen_sparse_attention"]))
    batch = SimpleNamespace(
        qwen_exo_attention_diagnostics=[{"error": 0, "layer_ids": [0], "token_count": 4}],
        forward_mode=SimpleNamespace(is_extend=lambda: True),
        extend_seq_lens_cpu=[4], extend_prefix_lens_cpu=[0],
        req_pool_indices=torch.tensor([1], device=device), tbo_split_seq_index=None,
        cross_attention_custom_mask=None, out_cache_loc=mapping[1], qwen_exo_customized_info=None,
    )
    topk = torch.tensor([[0, -1, -1], [0, 1, -1], [0, 2, -1], [2, 0, 3]], device=device)
    capture_attention_diagnostic(decoder, query, torch.arange(4, device=device), batch,
                                 sparse_topk=topk, current_keys=keys)
    actual = batch.qwen_exo_customized_info["qwen_exo_attention_layer_0"][0]
    # Token 1 has an enormous dense score, but the native indexer excluded it.
    selected = torch.tensor([2, 0, 3], device=device)
    logits = torch.einsum("hd,td->ht", query[-1], keys[selected, 0]) * .7
    expected = torch.zeros(4, device=device)
    expected[selected] = logits.softmax(-1).mean(0)
    torch.testing.assert_close(actual, expected)
    assert batch.qwen_exo_customized_info["qwen_exo_attention_error"].item() == 0
