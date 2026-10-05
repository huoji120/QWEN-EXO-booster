"""Backward-capable FLA kernels for the isolated native PLE training process."""
from __future__ import annotations

import functools
import inspect

import torch


def install_training_gdn(native):
    """Bind installed FLA kernels without changing serving or installed HF files."""
    from fla import __version__ as fla_version
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule
    from fla.modules.conv import causal_conv1d

    reference_gdn = inspect.unwrap(native.torch_chunk_gated_delta_rule)
    reference_conv = inspect.unwrap(native.causal_conv1d_fn)

    @functools.wraps(reference_gdn)
    def training_gdn(query, key, value, g, beta, chunk_size=64, initial_state=None,
                     output_final_state=False, use_qk_l2norm_in_kernel=False, **kwargs):
        if query.device.type != "cuda":
            return reference_gdn(query, key, value, g, beta, chunk_size=chunk_size,
                                 initial_state=initial_state, output_final_state=output_final_state,
                                 use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel, **kwargs)
        return chunk_gated_delta_rule(
            query, key, value, g, beta, initial_state=initial_state,
            output_final_state=output_final_state,
            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
            cu_seqlens=kwargs.get("cu_seqlens"),
        )
    @functools.wraps(reference_conv)

    def training_conv(hidden_states, weight, bias=None, activation=None, **kwargs):
        if hidden_states.device.type != "cuda":
            return reference_conv(hidden_states, weight, bias, activation=activation, **kwargs)
        output, _ = causal_conv1d(
            hidden_states.transpose(1, 2), weight=weight, bias=bias,
            activation=activation, backend="triton", output_final_state=False,
            cu_seqlens=kwargs.get("cu_seqlens"),
        )
        return output.transpose(1, 2)

    native.torch_chunk_gated_delta_rule = training_gdn
    native.causal_conv1d_fn = training_conv
    return {"gdn": "fla_chunk_gated_delta_rule", "convolution": "fla_triton_causal_conv1d",
            "fla_version": fla_version, "activation_backward": True,
            "gdn_source": inspect.getfile(chunk_gated_delta_rule)}


def verify_cuda_gdn(native):
    """Synthetic forward/input-gradient parity; no optimizer or private data."""
    reference_gdn = inspect.unwrap(native.torch_chunk_gated_delta_rule)
    reference_conv = inspect.unwrap(native.causal_conv1d_fn)
    report = install_training_gdn(native)
    torch.manual_seed(920)
    base = [torch.randn(1, 128, 2, 32, device="cuda", dtype=torch.bfloat16) * .2 for _ in range(3)]
    base += [-torch.rand(1, 128, 2, device="cuda") * .1,
             torch.rand(1, 128, 2, device="cuda", dtype=torch.bfloat16)]
    direction = torch.randn_like(base[2])
    outputs, gradients = [], []
    for fn in (reference_gdn, native.torch_chunk_gated_delta_rule):
        inputs = [t.detach().clone().requires_grad_() for t in base]
        result, _ = fn(*inputs, use_qk_l2norm_in_kernel=True)
        grads = torch.autograd.grad((result * direction).float().sum(), inputs)
        outputs.append(result.detach())
        gradients.append([g.detach() for g in grads])
    torch.testing.assert_close(outputs[1].float(), outputs[0].float(), rtol=.035, atol=.003)
    for actual, expected in zip(gradients[1], gradients[0]):
        torch.testing.assert_close(actual.float(), expected.float(), rtol=.07, atol=.01)
    x = torch.randn(1, 64, 128, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(64, 4, device="cuda", dtype=torch.bfloat16) * .1
    conv_outputs, conv_grads = [], []
    conv_direction = torch.randn_like(x)
    for fn in (reference_conv, native.causal_conv1d_fn):
        current = x.detach().clone().requires_grad_()
        value = fn(current, w, activation="silu")
        conv_grads.append(torch.autograd.grad((value * conv_direction).float().sum(), current)[0])
        conv_outputs.append(value.detach())
    torch.testing.assert_close(conv_outputs[1].float(), conv_outputs[0].float(), rtol=.025, atol=.005)
    torch.testing.assert_close(conv_grads[1].float(), conv_grads[0].float(), rtol=.03, atol=.005)
    return {**report, "gdn_output_max_abs_error": float((outputs[1]-outputs[0]).abs().max()),
            "gdn_gradient_max_abs_errors": [float((a-b).abs().max()) for a,b in zip(gradients[1],gradients[0])],
            "conv_output_max_abs_error": float((conv_outputs[1]-conv_outputs[0]).abs().max()),
            "conv_gradient_max_abs_error": float((conv_grads[1]-conv_grads[0]).abs().max()),
            "optimizer_steps": 0}
