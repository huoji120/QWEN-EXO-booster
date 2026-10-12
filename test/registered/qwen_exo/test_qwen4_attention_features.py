from types import SimpleNamespace

import pytest
import torch

from qwen_exo_booster.attention_diagnostic_capture import capture_attention_diagnostic


@pytest.mark.parametrize("kind", ["attention", "full_attention", "qwen_sparse_attention", "sparse_attention"])
@pytest.mark.parametrize("observer", ["off", "active"])
@pytest.mark.parametrize("selected_layer", [None, 0])
def test_retrieval_query_capture_is_independent_of_observer(
    monkeypatch, kind, observer, selected_layer
):
    from sglang.srt.models import qwen3_5
    from sglang.srt.model_executor.forward_batch_info import ForwardMode

    args = SimpleNamespace(
        enable_qwen_exo=True, qwen_exo_observer_mode=observer,
        qwen_exo_qk_layer=selected_layer,
    )
    monkeypatch.setattr(qwen3_5, "get_server_args", lambda: args)
    monkeypatch.setattr(qwen3_5, "get_parallel", lambda: SimpleNamespace(attn_tp_rank=0, attn_tp_size=1))
    monkeypatch.setattr(qwen3_5, "get_rope_config", lambda config: (10000, None))
    monkeypatch.setattr(qwen3_5, "get_is_capture_mode", lambda: False)

    # Stop immediately before weight allocation: this exercises the real model's
    # tracker configuration without initializing unrelated projection/MoE state.
    class BeforeWeights(Exception):
        pass

    def stop_before_weights(**kwargs):
        raise BeforeWeights

    monkeypatch.setattr(qwen3_5, "get_rope", stop_before_weights)
    config = SimpleNamespace(
        hidden_size=4, num_attention_heads=2, num_key_value_heads=1,
        head_dim=2, layers_block_type=[kind, "linear_attention", kind],
    )
    layers = []
    for layer_id in (0, 2):
        layer = qwen3_5.Qwen3_5AttentionDecoderLayer.__new__(qwen3_5.Qwen3_5AttentionDecoderLayer)
        with pytest.raises(BeforeWeights):
            layer.__init__(config, layer_id)
        layer.rotary_emb = SimpleNamespace(
            rotary_dim=2, cos_sin_cache=torch.tensor([[1., 0.], [1., 0.]]),
            is_neox_style=True,
        )
        layers.append(layer)

    batch = SimpleNamespace(
        forward_mode=ForwardMode.EXTEND,
        qwen_exo_full_query_capture=(True,), qwen_exo_observe=(False,),
        extend_seq_lens_cpu=[2], extend_prefix_lens_cpu=[0],
        contains_last_prefill_chunk=True, rids=["query"],
        qwen_exo_final_prefill=(True,), qwen_exo_user_query_spans=(((0, 2),),),
        qwen_exo_customized_info=None,
    )
    query = torch.tensor([[[1., 2.], [3., 4.]], [[5., 6.], [7., 8.]]])
    for layer in layers:
        layer._capture_qwen_exo_probe_queries(query + layer.layer_id, torch.arange(2), batch)
    expected_layer = 2 if selected_layer is None else selected_layer
    actual = batch.qwen_exo_customized_info["qwen_exo_user_query_full_heads"][0, 0]
    torch.testing.assert_close(actual, query.mean(0) + expected_layer)
    assert sum(layer.qwen_exo_probe_tracker is not None for layer in layers) == 1
    for layer in layers:
        if layer.qwen_exo_signal_tracker is not None:
            assert not layer.qwen_exo_signal_tracker.capture_full_queries


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
