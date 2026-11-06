"""Offline frozen-linear FP32 accumulation with original activation/storage dtype.

The arithmetic policy is explicit and model-local. It does not rewrite checkpoint
weights, install serving hooks or round a trainable adapter into the base.
"""
from __future__ import annotations
import types
import torch
from torch import nn
from torch.nn import functional as F


def _forward(inputs, weight, bias=None):
    if inputs.dtype != torch.bfloat16:
        return F.linear(inputs, weight, bias)
    flat = inputs.reshape(-1, inputs.shape[-1])
    result = torch.empty((flat.shape[0], weight.shape[0]), device=inputs.device, dtype=inputs.dtype)
    # Bound FP32 input/output workspace, including the large vocabulary head.
    w = weight.float()
    b = None if bias is None else bias.float()
    with torch.autocast(device_type=inputs.device.type, enabled=False):
        for start in range(0, flat.shape[0], 128):
            stop = min(start + 128, flat.shape[0])
            result[start:stop] = F.linear(flat[start:stop].float(), w, b).to(inputs.dtype)
    return result.reshape(*inputs.shape[:-1], weight.shape[0])


class _FrozenAccumulation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs, weight, bias):
        if weight.requires_grad or (bias is not None and bias.requires_grad):
            raise ValueError('stable_frozen_linear_cannot_train_base')
        ctx.save_for_backward(weight)
        ctx.input_dtype = inputs.dtype
        return _forward(inputs, weight, bias)

    @staticmethod
    def backward(ctx, gradient):
        (weight,) = ctx.saved_tensors
        flat = gradient.reshape(-1, gradient.shape[-1])
        result = torch.empty((flat.shape[0], weight.shape[1]), device=gradient.device, dtype=ctx.input_dtype)
        w = weight.float()
        with torch.autocast(device_type=gradient.device.type, enabled=False):
            for start in range(0, flat.shape[0], 128):
                stop = min(start + 128, flat.shape[0])
                result[start:stop] = flat[start:stop].float().matmul(w).to(ctx.input_dtype)
        return result.reshape(*gradient.shape[:-1], weight.shape[1]), None, None


def frozen_linear_fp32(inputs, weight, bias=None):
    if inputs.requires_grad and torch.is_grad_enabled():
        return _FrozenAccumulation.apply(inputs, weight, bias)
    return _forward(inputs, weight, bias)


def install_frozen_fp32_accumulation(reference):
    """Use FP32 accumulation for frozen materialized linears and streamed experts."""
    count = 0
    def forward(module, inputs):
        return frozen_linear_fp32(inputs, module.weight, module.bias)
    for module in reference.modules():
        if isinstance(module, nn.Linear):
            if module.weight.requires_grad:
                raise ValueError('install stable accumulation before adapters')
            module.forward = types.MethodType(forward, module)
            count += 1
    reference.native_checkpoint.fp32_accumulation = True
    reference._mid_lora_fp32_accumulation = True
    return {'linear_accumulation':'fp32','activation_dtype':str(reference.lm_head.weight.dtype),
            'materialized_linears':count,'streamed_experts':True,'checkpoint_bytes_changed':False}


class _StableHeadCE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, hidden, weight, labels, chunk_size):
        if labels.shape != hidden.shape[:2] or weight.requires_grad or labels.dtype != torch.int64:
            raise ValueError('stable_head_ce_contract_invalid')
        shifted = labels[:, 1:]
        positions = (shifted.reshape(-1) != -100).nonzero().flatten()
        if not positions.numel():
            raise ValueError('stable_head_ce_no_targets')
        length = hidden.shape[1]
        source = positions // (length - 1) * length + positions % (length - 1)
        target = shifted.reshape(-1)[positions]
        flat = hidden.reshape(-1, hidden.shape[-1])
        total = torch.zeros((), device=hidden.device, dtype=torch.float32)
        for start in range(0, source.numel(), chunk_size):
            selected = source[start:start + chunk_size]
            logits = _forward(flat[selected], weight).float()
            total += F.cross_entropy(logits, target[start:start + chunk_size], reduction='sum')
        ctx.save_for_backward(hidden, weight, source, target)
        ctx.chunk_size = chunk_size
        return total / source.numel()

    @staticmethod
    def backward(ctx, grad_output):
        hidden, weight, source, target = ctx.saved_tensors
        flat = hidden.reshape(-1, hidden.shape[-1])
        gradient = torch.zeros_like(flat)
        for start in range(0, source.numel(), ctx.chunk_size):
            selected = source[start:start + ctx.chunk_size]
            logits = _forward(flat[selected], weight).float()
            derivative = logits.softmax(-1)
            derivative[torch.arange(selected.numel(), device=hidden.device), target[start:start + ctx.chunk_size]] -= 1
            derivative *= grad_output.float() / source.numel()
            # Forward casts its FP32 accumulation to native activation dtype;
            # backward crosses the same cast before the FP32 input-gradient GEMM.
            gradient[selected] = derivative.to(hidden.dtype).float().matmul(weight.float()).to(hidden.dtype)
        return gradient.reshape_as(hidden), None, None, None


def stable_chunked_causal_cross_entropy(hidden, weight, labels, chunk_size=128):
    return _StableHeadCE.apply(hidden, weight, labels, chunk_size)
