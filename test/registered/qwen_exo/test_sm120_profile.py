"""Model-bound numerical presets must not reuse the original model identity."""
import copy
import json
from types import SimpleNamespace

import pytest

from qwen_exo_booster.fingerprint import ModelIdentity, _QWEN_EXO_LAYOUTS
from qwen_exo_booster.sm120_profile import (
    SM120_PROFILE_KEY, SM120_PROFILE_MARKER,
    configure_sm120_profile_arguments, sm120_profile_enabled, validate_sm120_profile_args,
)
from scripts.qwen_exo.build_sm120_profile import build_profile
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def native_config():
    layout = _QWEN_EXO_LAYOUTS["Qwen4ExpForConditionalGeneration"][0]
    text = copy.deepcopy(layout["text_structure"])
    text.update(model_type="qwen4_exp_text", tie_word_embeddings=False,
        layer_types=["full_attention" if (i + 1) % 4 == 0 else "linear_attention" for i in range(48)],
        rope_parameters={"partial_rotary_factor":0.25, "rope_theta":10000000},
        mtp={"hybrid":True, "num_hidden_layers":1, "layer_types":["full_attention"], "rope_theta":10000000})
    return {"architectures":["Qwen4ExpForConditionalGeneration"], "model_type":"qwen4_exp",
        "text_config":text, "quantization_config":{"quant_method":"modelopt", "quant_algo":"MIXED_PRECISION"}}


def write_native(root):
    root.mkdir()
    (root / "config.json").write_text(json.dumps(native_config()))
    (root / "weights.safetensors").write_bytes(b"unchanged serialized checkpoint fixture")
    (root / "model.safetensors.index.json").write_text(json.dumps({"weight_map":{"model.weight":"weights.safetensors"}}))
    (root / "tokenizer.json").write_text("tokenizer-v1")


def test_profile_keeps_source_weights_and_obtains_new_fingerprint(tmp_path):
    source, target = tmp_path / "source", tmp_path / "fast"
    write_native(source)
    before = (source / "config.json").read_bytes()
    original = ModelIdentity.from_path(source)
    receipt = build_profile(source, target)
    assert (source / "config.json").read_bytes() == before
    assert (target / "weights.safetensors").read_bytes() == (source / "weights.safetensors").read_bytes()
    assert not sm120_profile_enabled(json.loads(before))
    assert sm120_profile_enabled(json.loads((target / "config.json").read_text()))
    assert receipt["source_model_fingerprint"] == original.fingerprint
    assert receipt["model_fingerprint"] == ModelIdentity.from_path(target).fingerprint
    assert receipt["model_fingerprint"] != original.fingerprint
    (target / "tokenizer.json").write_text("tokenizer-fast")
    assert (source / "tokenizer.json").read_text() == "tokenizer-v1"
    assert ModelIdentity.from_path(source).fingerprint == original.fingerprint
    with pytest.raises(ValueError, match="existing"):
        build_profile(source, target)


def test_launch_profile_overrides_coupled_precision_without_changing_capacity(tmp_path):
    source, target = tmp_path / "source", tmp_path / "fast"
    write_native(source); build_profile(source, target)
    original = ["--enable-qwen-exo", "--model-path", str(target), "--moe-runner-backend", "flashinfer_cutlass",
        "--max-running-requests", "10", "--mem-fraction-static", "0.95", "--qwen4-host-kv-cache"]
    result = configure_sm120_profile_arguments(original)
    values = {result[i]:result[i+1] for i in range(len(result)-1) if result[i].startswith('--') and not result[i+1].startswith('--')}
    assert values["--moe-runner-backend"] == "marlin"
    assert values["--speculative-accept-threshold-single"] == values["--speculative-accept-threshold-acc"] == "1.0"
    assert values["--max-running-requests"] == "10" and values["--mem-fraction-static"] == "0.95"
    assert "--speculative-use-rejection-sampling" in result and "--qwen4-host-kv-cache" in result
    assert original[original.index("--moe-runner-backend") + 1] == "flashinfer_cutlass"


def test_marker_cannot_be_enabled_independently_or_partially(tmp_path):
    source = tmp_path / "source"; write_native(source)
    with pytest.raises(ValueError, match="marked"):
        configure_sm120_profile_arguments(["--model-path",str(source),"--qwen-exo-sm120-fast"])
    config = native_config(); config[SM120_PROFILE_KEY] = dict(SM120_PROFILE_MARKER)
    config["text_config"][SM120_PROFILE_KEY] = {**SM120_PROFILE_MARKER,"schema":2}
    with pytest.raises(ValueError, match="disagree"):
        sm120_profile_enabled(config)
    for marker in (True, {}, {**SM120_PROFILE_MARKER,"schema":True}, {**SM120_PROFILE_MARKER,"extra":True}):
        with pytest.raises(ValueError):
            sm120_profile_enabled({SM120_PROFILE_KEY:marker})


def valid_args():
    return SimpleNamespace(qwen_exo_sm120_fast=True, enable_qwen_exo=True,
        tp_size=1, pp_size=1, dp_size=1, ep_size=1, moe_dp_size=1, attn_cp_size=1, dcp_size=1,
        quantization="modelopt_mixed", moe_runner_backend="marlin", speculative_algorithm="EAGLE",
        speculative_num_steps=3, speculative_eagle_topk=1, speculative_num_draft_tokens=4,
        speculative_use_rejection_sampling=True, speculative_accept_threshold_single=1.0,
        speculative_accept_threshold_acc=1.0, dtype="bfloat16")


@pytest.mark.parametrize("field,value", [("ep_size",2),("tp_size",2),("speculative_use_rejection_sampling",False),
    ("speculative_accept_threshold_single",0.5),("speculative_accept_threshold_acc",0.5),
    ("enable_fp32_lm_head",True),("dtype","float16"),("qwen_exo_moe_cpu_offload",True),("enable_lora",True)])
def test_runtime_rejects_incompatible_numerical_contracts(field,value):
    args = valid_args(); validate_sm120_profile_args(args)
    setattr(args,field,value)
    with pytest.raises(ValueError):
        validate_sm120_profile_args(args)


def test_hc_large_row_fallback_uses_the_same_quantized_weights():
    import torch
    from sglang.srt.layers.hyperconnection import GatedResidual

    torch.manual_seed(120)
    module = object.__new__(GatedResidual)
    torch.nn.Module.__init__(module)
    module.config = SimpleNamespace(sm120_fast=True, hc_lowrank=64)
    module.hc_count, module.hidden_size = 4, 64
    module._sm120_prepared = False
    module.input_mix_weight_down = torch.nn.Linear(256, 64, bias=False, dtype=torch.bfloat16)
    module.input_mix_weight_up = torch.nn.Linear(64, 256, bias=False, dtype=torch.bfloat16)
    module.prepare_sm120_int8_weights()
    for projection, q, scale, group in (
        (module.input_mix_weight_down, module.input_mix_weight_down_int8, module.input_mix_weight_down_scale, 128),
        (module.input_mix_weight_up, module.input_mix_weight_up_int8, module.input_mix_weight_up_scale, 64),
    ):
        expected_weight = (q.float() * scale.float().repeat_interleave(group, dim=1)).bfloat16()
        x = torch.randn(10, q.shape[1], dtype=torch.bfloat16)
        torch.testing.assert_close(projection(x), torch.nn.functional.linear(x, expected_weight), rtol=0, atol=0)
    before = module.input_mix_weight_down.weight.detach().clone()
    module.prepare_sm120_int8_weights()
    torch.testing.assert_close(module.input_mix_weight_down.weight, before, rtol=0, atol=0)


def test_native_hf_config_preserves_root_and_text_numerical_markers():
    from qwen_exo_booster.fingerprint import validate_qwen_exo_config
    from sglang.srt.configs.qwen4_exp import Qwen4ExpConfig

    config = native_config()
    config[SM120_PROFILE_KEY] = dict(SM120_PROFILE_MARKER)
    config["text_config"][SM120_PROFILE_KEY] = dict(SM120_PROFILE_MARKER)
    loaded = Qwen4ExpConfig.from_dict(config)
    assert sm120_profile_enabled(loaded)
    assert sm120_profile_enabled(loaded.text_config)
    assert validate_qwen_exo_config(loaded) == "qwen4-exp-flash-next"
