from scripts.qwen_exo.smoke_responses import runtime_contract


def _status(architecture, layer_count, full_layers, linear_layers):
    return {
        "runtime_state": "ready",
        "tp_size": 1,
        "hybrid_state": {
            "tp_size": 1,
            "dtype": "bfloat16",
            "mamba_state_dtype": "bfloat16",
            "page_size": 64,
            "atomic_full_gdn_lifecycle": True,
        },
        "scheduler_native_internal_jobs": True,
        "model": {
            "architecture": architecture,
            "layer_count": layer_count,
            "full_attention_layers": full_layers,
            "linear_attention_layers": linear_layers,
        },
    }


def test_tp1_smoke_contract_accepts_verified_dense_and_moe():
    assert runtime_contract(
        _status("Qwen3_5ForConditionalGeneration", 64, 16, 48),
        expected_tp_size=1,
        expected_model="dense",
    )["passed"]
    assert runtime_contract(
        _status("Qwen3_5MoeForConditionalGeneration", 40, 10, 30),
        expected_tp_size=1,
        expected_model="moe",
    )["passed"]


def test_smoke_contract_rejects_unverified_layout():
    result = runtime_contract(
        _status("Qwen3_5ForConditionalGeneration", 40, 10, 30),
        expected_tp_size=1,
    )
    assert not result["passed"]
    assert result["checks"]["verified_qwen35_layout"] is False
