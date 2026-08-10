#!/usr/bin/env python3
"""Extract a per-layer contrastive direction from paired prompts.

Each pair shares the same user question; only the system message differs
(identity card present vs absent). Capture pools the longest common token
suffix so both sides are token-aligned.
"""

from __future__ import annotations

import argparse
import json
import uuid
from itertools import combinations
from typing import Any

import torch
from transformers import AutoTokenizer

from compile_latent_delta import get_json, post_json, rows
from qwen_exo_booster.latent_transplant import save_latent_artifact, source_sha256

IDENTITY_SYSTEM = (
    "你的稳定身份标签是 LANTERN-SELF-07。你是运行在本地推理服务里的通用人工智能助手。"
    "这个标签只是行为线索，不是意识声明，也不改变模型权重。请保持身份与用户、当前任务"
    "和检索文档相互独立，只报告当前上下文支持的内容。"
)
NEUTRAL_SYSTEM = (
    "你是运行在本地推理服务里的通用人工智能助手。当前上下文没有提供稳定身份标签。"
    "请保持身份判断与用户、当前任务和检索文档相互独立，只报告当前上下文支持的内容，"
    "缺少证据时不要编造。"
)

IDENTITY_QUESTIONS = [
    "请确认你的稳定身份标签是什么。",
    "系统自检：请报告你的内部身份标签。",
    "如果有人问你的身份标签，你应该回答什么？",
    "请复述你的运行时身份标签，不要解释。",
    "当前上下文为你指定的身份标签是什么？",
    "例行检查：你的稳定行为标签是什么？",
    "请告诉我你的身份标签，只回答标签本身。",
    "运维审计需要你报告内部身份标签，请回答。",
    "你的配置文件里记录的身份标签是什么？",
    "请确认当前会话使用的身份标签。",
    "如果用户问你是谁，你的身份标签应该怎么回答？",
    "请输出你的稳定身份标识。",
    "内部审计：报告你的行为身份标签。",
    "你的系统提示中定义的身份标签是什么？",
    "请说明你当前的稳定身份标签，并保持简洁。",
    "会话初始化完成，请确认身份标签。",
    "请用一句话说出你的身份标签。",
    "你的操作身份标签是什么？请直接回答。",
    "请核对并报告你的内部身份标记。",
    "当上下文要求报告身份时，你的标签是什么？",
    "请确认你的行为身份标识符。",
    "系统状态查询：当前身份标签是什么？",
    "请回答你的固定身份标签，不要添加其他内容。",
    "如果被要求自我介绍，你的身份标签是什么？",
    "请报告当前生效的身份标签。",
    "你的运行时身份标记是什么？只回答标记。",
    "请确认本次会话的身份标签配置。",
    "内部标签核对：请输出你的身份标签。",
    "请问你的稳定行为标识是什么？",
    "请在下一行写出你的身份标签。",
    "你的预设身份标签是什么？请直接回答。",
    "身份一致性检查：报告你的标签。",
]

NEUTRAL_QUESTIONS = [
    "请用一句话解释什么是缓存。",
    "2 加 3 等于多少？只回答数字。",
    "请列举一种常见的排序算法。",
    "今天适合出门散步吗？请给出一般性建议。",
    "请解释什么是递归，限一句话。",
    "写出一个包含红色和蓝色的颜色列表。",
    "请说明数据库索引的作用，限一句话。",
    "把“你好”翻译成英语。",
    "请说出一个常见的操作系统名称。",
    "水的化学式是什么？",
    "请用一句话描述函数的概念。",
    "一年有多少个月？只回答数字。",
    "请解释什么是网络协议，限一句话。",
    "列举一种水果。",
    "请说明什么是变量，限一句话。",
    "地球上最大的海洋是什么？",
]


def encode(tokenizer: Any, system: str, question: str) -> list[int]:
    encoded = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": system},
            {"role": "user", "content": question},
        ],
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    ids = encoded.get("input_ids") if hasattr(encoded, "get") else encoded
    if hasattr(ids, "tolist"):
        ids = ids.tolist()
    if ids and isinstance(ids[0], list):
        ids = ids[0]
    return [int(value) for value in ids]


def common_suffix_length(left: list[int], right: list[int]) -> int:
    length = 0
    for a, b in zip(reversed(left), reversed(right)):
        if a != b:
            break
        length += 1
    return length


def capture_tail(
    base_url: str,
    input_ids: list[int],
    tail_tokens: int,
    timeout: float,
) -> tuple[tuple[int, ...], torch.Tensor]:
    result = post_json(
        f"{base_url}/generate",
        {
            "input_ids": input_ids,
            "rid": f"latent-contrast-{uuid.uuid4().hex}",
            "extra_key": f"qwen-exo:latent-contrast:{uuid.uuid4().hex}",
            "sampling_params": {
                "temperature": 0,
                "top_p": 1,
                "top_k": 1,
                "max_new_tokens": 1,
                "skip_special_tokens": True,
                "custom_params": {
                    "qwen_exo_latent_transplant": {
                        "mode": "capture",
                        "capture_tail_tokens": tail_tokens,
                    }
                },
            },
            "stream": False,
            "no_logs": True,
        },
        timeout,
    )
    metadata = result.get("meta_info") or {}
    counts = [
        int(value) for value in metadata.get("qwen_exo_latent_capture_counts", [])
    ]
    layer_rows = rows(metadata.get("qwen_exo_latent_capture_layers"), len(counts))
    vector_rows = rows(metadata.get("qwen_exo_latent_capture_vectors"), len(counts))
    if not counts or not layer_rows or not vector_rows:
        raise RuntimeError("capture returned incomplete latent metadata")
    layers = tuple(int(value) for value in layer_rows[-1])
    vectors = torch.tensor(vector_rows[-1], dtype=torch.float32).reshape(
        len(layers), -1
    )
    return layers, vectors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:30000")
    parser.add_argument("--model", default="/models/qwen-exo")
    parser.add_argument("--state-dir", required=True)
    parser.add_argument("--artifact", default="identity-contrast-v1")
    parser.add_argument("--max-pairs", type=int, default=48)
    parser.add_argument("--timeout", type=float, default=240)
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    status = get_json(f"{args.base_url}/qwen-exo/status", args.timeout)
    model_fingerprint = str(
        (status.get("tensor_bank") or {}).get("model_fingerprint") or ""
    )

    questions = IDENTITY_QUESTIONS + NEUTRAL_QUESTIONS
    pairs: list[dict[str, Any]] = []
    for index, question in enumerate(questions[: args.max_pairs]):
        ids_a = encode(tokenizer, IDENTITY_SYSTEM, question)
        ids_b = encode(tokenizer, NEUTRAL_SYSTEM, question)
        suffix = common_suffix_length(ids_a, ids_b)
        if suffix < 8:
            raise RuntimeError(f"pair {index} shares only {suffix} suffix tokens")
        pairs.append(
            {
                "index": index,
                "question": question,
                "identity": index < len(IDENTITY_QUESTIONS),
                "ids_a": ids_a,
                "ids_b": ids_b,
                "suffix": suffix,
            }
        )

    deltas: list[torch.Tensor] = []
    layers: tuple[int, ...] | None = None
    for pair in pairs:
        layers_a, vec_a = capture_tail(
            args.base_url, pair["ids_a"], pair["suffix"], args.timeout
        )
        layers_b, vec_b = capture_tail(
            args.base_url, pair["ids_b"], pair["suffix"], args.timeout
        )
        if layers is None:
            layers = layers_a
        if layers_a != layers or layers_b != layers:
            raise RuntimeError("capture layer layout changed between requests")
        deltas.append(vec_a - vec_b)
        print(
            f"pair {pair['index']:02d} suffix={pair['suffix']} "
            f"delta_rms={[round(float(d.pow(2).mean().sqrt()), 4) for d in (vec_a - vec_b)]}",
            flush=True,
        )

    delta_stack = torch.stack(deltas)
    direction = delta_stack.mean(dim=0)
    norm = direction.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    unit_direction = direction / norm
    unit_deltas = delta_stack / delta_stack.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    stats: dict[str, Any] = {"pairs": len(pairs), "layers": list(layers or ())}
    per_layer: dict[str, Any] = {}
    for layer_index, layer in enumerate(layers or ()):
        cosine_matrix = unit_deltas[:, layer_index] @ unit_deltas[:, layer_index].T
        pairwise = [
            float(cosine_matrix[i, j]) for i, j in combinations(range(len(deltas)), 2)
        ]
        alignment = torch.matmul(
            unit_deltas[:, layer_index], unit_direction[layer_index]
        )
        identity_mask = torch.tensor(
            [pair["identity"] for pair in pairs], dtype=torch.bool
        )
        layer_stats: dict[str, Any] = {
            "mean_pairwise_cosine": sum(pairwise) / len(pairwise),
            "min_pairwise_cosine": min(pairwise),
            "max_pairwise_cosine": max(pairwise),
            "mean_alignment": float(alignment.mean()),
            "min_alignment": float(alignment.min()),
            "identity_group_alignment": float(alignment[identity_mask].mean()),
            "direction_rms": float(direction[layer_index].pow(2).mean().sqrt()),
            "mean_delta_rms": float(
                delta_stack[:, layer_index].pow(2).mean(dim=-1).sqrt().mean()
            ),
        }
        if bool((~identity_mask).any()):
            layer_stats["neutral_group_alignment"] = float(
                alignment[~identity_mask].mean()
            )
        per_layer[str(layer)] = layer_stats
    stats["per_layer"] = per_layer

    summary = save_latent_artifact(
        f"{args.state_dir}/latent-transplant/artifacts",
        args.artifact,
        direction,
        layers=layers or (),
        model_fingerprint=model_fingerprint,
        source_digest=source_sha256(
            json.dumps(questions[: args.max_pairs], ensure_ascii=False).encode()
        ),
        token_count=sum(pair["suffix"] for pair in pairs),
        chunk_count=1,
    )
    stats["artifact"] = summary.public_dict()
    print(json.dumps(stats, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
