#!/usr/bin/env python3
"""Train one low-rank activation editor from one or more isolated agent trajectories.

The base model stays frozen. Every source keeps its own ChatML conversation boundary;
only the resulting training samples are mixed. Evaluation reports aggregate and
per-source held-out action NLL for baseline, trained editor, and random editor.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
import time
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

_TOOL_CALL_PATTERN = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)


def parse_complete_tool_calls(text: str) -> tuple[dict[str, Any], ...] | None:
    value = str(text or "")
    starts = value.count("<tool_call>")
    ends = value.count("</tool_call>")
    if starts == 0 or starts != ends:
        return None
    bodies = _TOOL_CALL_PATTERN.findall(value)
    if len(bodies) != starts:
        return None
    calls = []
    for body in bodies:
        try:
            payload = json.loads(body.strip())
        except (TypeError, ValueError, json.JSONDecodeError):
            return None
        if not isinstance(payload, dict) or not str(payload.get("name") or "").strip():
            return None
        arguments = payload.get("arguments")
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except (TypeError, ValueError, json.JSONDecodeError):
                return None
        if not isinstance(arguments, dict) or not arguments:
            return None
        calls.append({"name": str(payload["name"]), "arguments": arguments})
    return tuple(calls)


class LowRankEditor(torch.nn.Module):
    """h' = h + R^T (W h + b - R h), zero at initialization when W == R, b == 0."""

    def __init__(self, hidden_size: int, rank: int):
        super().__init__()
        projection = torch.empty(rank, hidden_size)
        torch.nn.init.orthogonal_(projection)
        self.projection = torch.nn.Parameter(projection, requires_grad=False)
        self.transform = torch.nn.Parameter(projection.clone())
        self.bias = torch.nn.Parameter(torch.zeros(rank))

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        base = hidden @ self.projection.T
        target = hidden @ self.transform.T + self.bias
        return hidden + (target - base) @ self.projection


class EditableModel:
    def __init__(self, model: Any, layer_index: int, rank: int, window: int):
        config = model.config
        text_config = getattr(config, "text_config", config)
        hidden_size = int(text_config.hidden_size)
        self.editor = LowRankEditor(hidden_size, rank)
        self.layer_index = layer_index
        self.window = window
        self.context_end: int | None = None
        self.apply = True
        layers = model.model.layers
        self.layer = layers[layer_index]
        self.handle = self.layer.register_forward_hook(self._hook)
        # Cut the autograd graph below the edit layer: nothing below is
        # trainable, and dropping that subgraph halves backward memory.
        self.pre_handle = self.layer.register_forward_pre_hook(self._pre_hook)

    def _pre_hook(self, _module, inputs):
        hidden = inputs[0]
        if isinstance(hidden, torch.Tensor) and hidden.requires_grad:
            return (hidden.detach(), *inputs[1:])
        return None

    def _hook(self, _module, _inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        if (
            not self.apply
            or self.context_end is None
            or not isinstance(hidden, torch.Tensor)
            or hidden.ndim < 3
            or hidden.shape[1] < self.context_end
        ):
            return output
        if next(self.editor.parameters()).device != hidden.device:
            self.editor.to(hidden.device)
        start = max(0, self.context_end - self.window)
        edited = hidden.clone()
        edited[:, start : self.context_end] = self.editor(
            hidden[:, start : self.context_end].float()
        ).to(hidden.dtype)
        if isinstance(output, tuple):
            return (edited, *output[1:])
        return edited

    def close(self):
        self.handle.remove()
        self.pre_handle.remove()


def source_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_samples(
    trajectory: dict[str, Any],
    tokenizer: Any,
    max_context_tokens: int,
    max_target_tokens: int,
    max_sequence_tokens: int | None = None,
) -> list[dict[str, Any]]:
    if max_sequence_tokens is None:
        max_sequence_tokens = max_context_tokens + max_target_tokens
    if max_sequence_tokens <= max_target_tokens:
        raise ValueError("max_sequence_tokens must leave room for context")
    messages = trajectory["session"]["messages"]
    samples = []
    for index, message in enumerate(messages):
        if message["role"] != "assistant" or index < 2:
            continue
        content = message["content"]
        if not isinstance(content, str) or len(content.strip()) < 20:
            continue
        # This slice is intentionally local to one source trajectory. Sources
        # are never concatenated into a synthetic conversation.
        context_messages = messages[:index]
        context_ids = tokenizer.apply_chat_template(
            context_messages,
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=True,
            preserve_thinking=True,
        )
        if hasattr(context_ids, "get"):
            context_ids = context_ids.get("input_ids")
        if hasattr(context_ids, "tolist"):
            context_ids = context_ids.tolist()
        if context_ids and isinstance(context_ids[0], list):
            context_ids = context_ids[0]
        context_ids = [int(value) for value in context_ids][-max_context_tokens:]
        target_ids = tokenizer.encode(content, add_special_tokens=False)
        tool_calls = parse_complete_tool_calls(content)
        has_tool_marker = "<tool_call>" in content or "</tool_call>" in content
        if len(target_ids) > max_target_tokens or (
            has_tool_marker and tool_calls is None
        ):
            continue
        if len(target_ids) < 4:
            continue
        context_budget = min(max_context_tokens, max_sequence_tokens - len(target_ids))
        if context_budget < 1:
            continue
        context_ids = context_ids[-context_budget:]
        samples.append(
            {
                "index": index,
                "context_ids": context_ids,
                "target_ids": target_ids,
                "target_text": content,
                "tool_call_count": len(tool_calls or ()),
            }
        )
    return samples


def prepare_source_samples(
    paths: list[Path],
    tokenizer: Any,
    max_context_tokens: int,
    max_target_tokens: int,
    holdout_ratio: float,
    max_sequence_tokens: int | None = None,
) -> list[dict[str, Any]]:
    sources = []
    seen = set()
    for path in paths:
        name = path.stem.lower()
        if name in seen:
            raise RuntimeError(f"duplicate trajectory name: {name}")
        seen.add(name)
        trajectory = json.loads(path.read_text(encoding="utf-8"))
        samples = build_samples(
            trajectory,
            tokenizer,
            max_context_tokens,
            max_target_tokens,
            max_sequence_tokens,
        )
        if len(samples) < 2:
            raise RuntimeError(
                f"trajectory {name} needs at least two usable assistant samples"
            )
        split = min(
            len(samples) - 1,
            max(1, int(len(samples) * (1 - holdout_ratio))),
        )
        sources.append(
            {
                "name": name,
                "sha256": source_sha256(path),
                "train": samples[:split],
                "eval": samples[split:],
            }
        )
    if not sources:
        raise RuntimeError("no trajectories supplied")
    return sources


def exact_target_nll(
    model: Any,
    input_ids: torch.Tensor,
    context_end: int,
    *,
    backward: bool,
    chunk_tokens: int = 32,
) -> float:
    if context_end < 1 or context_end >= input_ids.shape[1] or chunk_tokens < 1:
        raise ValueError("target loss requires non-empty context and target")
    output = model.model(input_ids=input_ids, use_cache=False, return_dict=True)
    hidden_states = (
        output.last_hidden_state if hasattr(output, "last_hidden_state") else output[0]
    )
    labels = input_ids[:, context_end:]
    prediction_hidden = hidden_states[:, context_end - 1 : input_ids.shape[1] - 1]
    target_tokens = int(labels.numel())
    loss_hidden = (
        prediction_hidden.detach().requires_grad_(True)
        if backward
        else prediction_hidden
    )
    total_loss = 0.0
    for start in range(0, labels.shape[1], chunk_tokens):
        end = min(labels.shape[1], start + chunk_tokens)
        logits = model.lm_head(loss_hidden[:, start:end])
        chunk_loss = torch.nn.functional.cross_entropy(
            logits.float().reshape(-1, logits.shape[-1]),
            labels[:, start:end].reshape(-1),
            reduction="sum",
        )
        total_loss += float(chunk_loss.detach())
        if backward:
            (chunk_loss / target_tokens).backward()
        del logits, chunk_loss
    if backward:
        if loss_hidden.grad is None:
            raise RuntimeError("target loss did not produce hidden-state gradients")
        prediction_hidden.backward(loss_hidden.grad)
    return total_loss / target_tokens


def evaluate(
    model: Any,
    editor: EditableModel | None,
    samples: list[dict[str, Any]],
    device: torch.device,
) -> tuple[float, int]:
    if editor is not None:
        editor.apply = True
    total_nll = 0.0
    total_tokens = 0
    for sample in samples:
        ids = sample["context_ids"] + sample["target_ids"]
        context_end = len(sample["context_ids"])
        input_ids = torch.tensor([ids], device=device)
        if editor is not None:
            editor.context_end = context_end
        with torch.no_grad():
            nll = exact_target_nll(model, input_ids, context_end, backward=False)
        count = len(sample["target_ids"])
        total_nll += nll * count
        total_tokens += count
    return total_nll / max(total_tokens, 1), total_tokens


def evaluate_sources(
    model: Any,
    editor: EditableModel | None,
    sources: list[dict[str, Any]],
    device: torch.device,
) -> tuple[float, dict[str, float]]:
    weighted_nll = 0.0
    total_tokens = 0
    per_source = {}
    for source in sources:
        nll, tokens = evaluate(model, editor, source["eval"], device)
        per_source[str(source["name"])] = nll
        weighted_nll += nll * tokens
        total_tokens += tokens
    return weighted_nll / max(total_tokens, 1), per_source


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--trajectory",
        type=Path,
        action="append",
        required=True,
        help="ChatML trajectory; repeat for joint training while preserving boundaries",
    )
    parser.add_argument("--model", default="/models/qwen-exo")
    parser.add_argument("--layer", type=int, default=47)
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--window", type=int, default=16)
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--epochs", type=float, default=None)
    parser.add_argument("--editor-out", type=Path, default=None)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--max-sequence-tokens", type=int, default=2560)
    parser.add_argument("--max-context-tokens", type=int, default=512)
    parser.add_argument("--max-target-tokens", type=int, default=2048)
    parser.add_argument("--holdout-ratio", type=float, default=0.2)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    torch.manual_seed(20260807)
    random.seed(20260807)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if args.max_sequence_tokens <= args.max_target_tokens:
        parser.error("--max-sequence-tokens must exceed --max-target-tokens")
    sources = prepare_source_samples(
        args.trajectory,
        tokenizer,
        args.max_context_tokens,
        args.max_target_tokens,
        args.holdout_ratio,
        args.max_sequence_tokens,
    )
    source_identity = [
        {"name": str(source["name"]), "sha256": str(source["sha256"])}
        for source in sources
    ]
    train_samples = [sample for source in sources for sample in source["train"]]
    print(
        f"sources={len(sources)} samples={sum(len(source['train']) + len(source['eval']) for source in sources)} "
        f"train={len(train_samples)} eval={sum(len(source['eval']) for source in sources)}",
        flush=True,
    )
    for source in sources:
        print(
            f"source={source['name']} train={len(source['train'])} eval={len(source['eval'])}",
            flush=True,
        )

    started = time.perf_counter()
    from transformers import BitsAndBytesConfig

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        ),
        device_map={"": 0},
        trust_remote_code=True,
    )
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    print(f"model_loaded seconds={time.perf_counter() - started:.1f}", flush=True)

    device = torch.device("cuda:0")
    editor = EditableModel(model, args.layer, args.rank, args.window)
    editor.editor.to(device)

    if args.smoke:
        sample = train_samples[0]
        editor.context_end = len(sample["context_ids"])
        ids = sample["context_ids"] + sample["target_ids"]
        input_ids = torch.tensor([ids], device=device)
        loss = exact_target_nll(model, input_ids, editor.context_end, backward=True)
        grad_norm = editor.editor.transform.grad.norm().item()
        print(
            json.dumps(
                {"smoke_loss": loss, "editor_grad_norm": grad_norm},
                indent=2,
            )
        )
        return 0

    baseline_nll, baseline_by_source = evaluate_sources(model, None, sources, device)
    print(f"baseline_nll={baseline_nll:.4f}", flush=True)

    random_editor = LowRankEditor(editor.editor.projection.shape[1], args.rank).to(
        device
    )
    with torch.no_grad():
        # Random non-zero edit of the same scale family as training updates.
        torch.nn.init.orthogonal_(random_editor.transform)
        random_editor.bias.normal_(0, 0.02)
    trained_backup = {
        key: value.detach().clone() for key, value in editor.editor.state_dict().items()
    }
    editor.editor.load_state_dict(random_editor.state_dict())
    random_nll, random_by_source = evaluate_sources(model, editor, sources, device)
    print(f"random_editor_nll={random_nll:.4f}", flush=True)
    editor.editor.load_state_dict(trained_backup)

    optimizer = torch.optim.AdamW(editor.editor.parameters(), lr=args.lr)
    rng = random.Random(20260807)
    if args.epochs is not None:
        plan = []
        for _ in range(math.ceil(args.epochs)):
            order = list(range(len(train_samples)))
            rng.shuffle(order)
            plan.extend(order)
        plan = plan[: max(1, int(len(train_samples) * args.epochs))]
    else:
        plan = [rng.randrange(len(train_samples)) for _ in range(args.steps)]
    log = []
    for step, sample_index in enumerate(plan):
        sample = train_samples[sample_index]
        editor.context_end = len(sample["context_ids"])
        ids = sample["context_ids"] + sample["target_ids"]
        input_ids = torch.tensor([ids], device=device)
        loss = exact_target_nll(model, input_ids, editor.context_end, backward=True)
        grad_norm = math.sqrt(
            sum(
                float(parameter.grad.norm() ** 2)
                for parameter in editor.editor.parameters()
                if parameter.grad is not None
            )
        )
        torch.nn.utils.clip_grad_norm_(editor.editor.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        if step % 10 == 0 or step == len(plan) - 1:
            print(
                f"step={step} loss={float(loss):.4f} grad={grad_norm:.4f}",
                flush=True,
            )
            log.append({"step": step, "loss": float(loss), "grad_norm": grad_norm})

    trained_nll, trained_by_source = evaluate_sources(model, editor, sources, device)
    print(f"trained_editor_nll={trained_nll:.4f}", flush=True)

    if args.editor_out is not None:
        args.editor_out.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "schema": 1,
                "layer": args.layer,
                "rank": args.rank,
                "window": args.window,
                "hidden_size": int(editor.editor.projection.shape[1]),
                "max_context_tokens": args.max_context_tokens,
                "max_target_tokens": args.max_target_tokens,
                "max_sequence_tokens": args.max_sequence_tokens,
                "sources": source_identity,
                "state_dict": {
                    key: value.detach().cpu()
                    for key, value in editor.editor.state_dict().items()
                },
            },
            args.editor_out,
        )
        print(f"editor_saved={args.editor_out}", flush=True)

    source_reports = []
    for source in sources:
        name = str(source["name"])
        source_reports.append(
            {
                "name": name,
                "sha256": str(source["sha256"]),
                "samples": {
                    "train": len(source["train"]),
                    "eval": len(source["eval"]),
                },
                "baseline_nll": baseline_by_source[name],
                "random_editor_nll": random_by_source[name],
                "trained_editor_nll": trained_by_source[name],
                "improvement_vs_baseline": baseline_by_source[name]
                - trained_by_source[name],
                "improvement_vs_random": random_by_source[name]
                - trained_by_source[name],
            }
        )
        print(
            f"source={name} baseline_nll={baseline_by_source[name]:.4f} "
            f"random_editor_nll={random_by_source[name]:.4f} "
            f"trained_editor_nll={trained_by_source[name]:.4f}",
            flush=True,
        )

    report = {
        "config": {
            "layer": args.layer,
            "rank": args.rank,
            "window": args.window,
            "steps": args.steps,
            "lr": args.lr,
            "max_context_tokens": args.max_context_tokens,
            "max_target_tokens": args.max_target_tokens,
            "max_sequence_tokens": args.max_sequence_tokens,
        },
        "samples": {
            "train": len(train_samples),
            "eval": sum(len(source["eval"]) for source in sources),
        },
        "sources": source_reports,
        "baseline_nll": baseline_nll,
        "random_editor_nll": random_nll,
        "trained_editor_nll": trained_nll,
        "improvement_vs_baseline": baseline_nll - trained_nll,
        "improvement_vs_random": random_nll - trained_nll,
        "log": log,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                key: report[key]
                for key in ("baseline_nll", "random_editor_nll", "trained_editor_nll")
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
