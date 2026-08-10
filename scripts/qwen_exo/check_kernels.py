from __future__ import annotations

import json

import torch
import triton
from sglang.kernels.ops.attention.fla.layernorm_gated import RMSNorm
from sglang.srt.layers.rotary_embedding.utils import (
    apply_rotary_pos_emb_native_eager,
)


def main() -> int:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the QWEN-EXO kernel preflight")

    device = torch.device("cuda:0")
    x = torch.randn((16, 128), device=device, dtype=torch.bfloat16)
    gate = torch.randn_like(x)
    norm = RMSNorm(128, device=device, dtype=torch.bfloat16)
    output = norm(x, gate)
    torch.cuda.synchronize(device)
    if output.shape != x.shape or not torch.isfinite(output).all().item():
        raise RuntimeError("gated RMSNorm kernel returned invalid output")

    q = torch.randn((8, 4, 128), device=device, dtype=torch.bfloat16)
    k = torch.randn_like(q)
    cos = torch.ones((8, 128), device=device, dtype=torch.bfloat16)
    sin = torch.zeros_like(cos)
    rotated_q, rotated_k = apply_rotary_pos_emb_native_eager(q, k, cos, sin)
    if not torch.equal(rotated_q, q) or not torch.equal(rotated_k, k):
        raise RuntimeError("eager vision rotary kernel returned invalid output")
    print(
        json.dumps(
            {
                "cuda": torch.version.cuda,
                "device": torch.cuda.get_device_name(device),
                "gated_rmsnorm": "ok",
                "torch": torch.__version__,
                "triton": triton.__version__,
                "vision_rotary_eager": "ok",
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
