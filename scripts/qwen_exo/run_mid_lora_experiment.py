"""Run an explicitly authorized offline native LoRA capacity/optimizer experiment.

Examples (private outputs must not already exist):
  run_mid_lora_experiment.py --profile PROFILE --output OUT --mode verify-cpu
  run_mid_lora_experiment.py --profile PROFILE --output OUT --mode capacity --authorize-gpu
  run_mid_lora_experiment.py --profile PROFILE --output OUT --mode experiment \
      --data PREPARED_JSONL --authorize-gpu --authorize-training

Capacity is actual native forward/backward at each requested length, not an
arithmetic estimate. Experiment runs frozen-base evaluation, pristine ordinary
answer SFT, and independently pristine self-generated-Think answer SFT. It never
controls serving, deploys adapters, installs packages or runs RL.

Native-cache stays strict by default. --allow-numeric-drift-research is an
experiment-only opt-in: preserve measured failed equivalence, require bitwise
prefill and finite cache/logits, and compare the same cached policy before/after.
Heldout self-Think samples Think and answer in one continuous native decode.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--mode", choices=("verify-cpu", "capacity", "experiment"), required=True)
    parser.add_argument("--data")
    parser.add_argument("--cpu-validation-receipt", help="Previously completed isolated CPU native-cache proof; never rerun CPU tensors through GPU-only FLA kernels")
    parser.add_argument("--authorize-gpu", action="store_true")
    parser.add_argument("--authorize-training", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--reference-dtype", choices=("bfloat16", "float32"), default="bfloat16", help="Explicit offline activation/decoded-weight precision; original checkpoint remains immutable")
    parser.add_argument("--fp32-linear-accumulation", action="store_true", help="Accumulate frozen BF16 projections in FP32 while preserving activation dtype")
    parser.add_argument("--generation-mode", choices=("fresh", "native-cache"), default="fresh",
                        help="Native-cache requires tiny and policy-scoped actual-prefix cached/fresh checks; strict equivalence unless explicitly opted into drift research")
    parser.add_argument("--cache-gate-policy", choices=("each-prefix", "adapter-revision"),
                        default="each-prefix",
                        help="Validate every prefix, or reuse one actual-prefix gate per exact adapter tensor revision; KV stays per generation")
    parser.add_argument("--allow-numeric-drift-research", action="store_true",
                        help="Experiment/native-cache only: retain failed numerical equivalence while allowing finite cached-policy research with bitwise-equal prefill; not replay or online equivalence")
    parser.add_argument("--lengths", type=int, nargs="+", default=[256, 4096, 8192, 32768])
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--heldout-limit", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--clip-norm", type=float, default=1.)
    parser.add_argument("--seed", type=int, default=193)
    parser.add_argument("--max-answer-tokens", type=int, default=128)
    parser.add_argument("--routing-probe-tokens", type=int, default=512)
    parser.add_argument("--first-layer", type=int, default=16)
    parser.add_argument("--last-layer", type=int, default=31)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--alpha", type=float, default=16.)
    parser.add_argument("--checkpoint-group-size", type=int, default=1, help="Number of consecutive suffix layers per recomputation boundary")
    parser.add_argument("--offload-checkpoints", action="store_true", help="Store autograd checkpoint boundaries on CPU; preserves gradient flow")
    parser.add_argument("--arms", nargs="+", choices=("self_think_sft", "ordinary_sft"), default=("self_think_sft", "ordinary_sft"))
    parser.add_argument("--initial-adapter", help="Adapter-only warm start for self_think_sft")
    parser.add_argument("--initial-optimizer-steps", type=int, default=0)
    parser.add_argument("--start-sample-index", type=int, default=0, help="Continue the same seeded training order at this attempted-example cursor; Adam state is not restored")
    parser.add_argument("--resume-training-state", help="Restore validated optimizer/RNG companion; requires matching adapter, cumulative step and sample cursor")
    parser.add_argument("--transfer-dataset", action="store_true", help="Explicitly retain saved Adam/RNG on different data; requires --resume-training-state and --start-sample-index 0")
    parser.add_argument("--generation-expert-backend", choices=("reference", "nvfp4-gemv"), default="reference",
                        help="Opt-in generation-only packed expert GEMV; prefill/backward remain reference")
    parser.add_argument("--precomputed-think", help="Validated fixed-teacher Think cache JSONL; skips on-the-fly Think generation")
    parser.add_argument("--precomputed-think-source", help="Original prepared JSONL whose SHA binds the fixed Think cache")
    args = parser.parse_args()
    if args.mode == "verify-cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    elif args.device.startswith("cuda") and not args.authorize_gpu:
        parser.error("Real GPU allocation requires --authorize-gpu")
    if args.mode == "experiment" and (not args.authorize_training or not args.data):
        parser.error("Experiment requires --data and --authorize-training")
    if args.allow_numeric_drift_research and (args.mode != "experiment" or args.generation_mode != "native-cache"):
        parser.error("--allow-numeric-drift-research requires --mode experiment --generation-mode native-cache")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    os.environ.setdefault("MKL_NUM_THREADS", "2")
    from qwen_exo_booster.native_mid_lora import MidLoraConfig, verify_tiny_mid_lora_backward
    from qwen_exo_booster.native_mid_lora_experiment import (
        run_experiment, verify_tiny_persistent_mid_lora, verify_tiny_continuous_generation,
    )
    if args.mode == "verify-cpu":
        from qwen_exo_booster.native_ple_training_backend import verify_tiny_native_ple_backward
        output = Path(args.output).resolve()
        if output.is_relative_to(Path(__file__).resolve().parents[2]):
            parser.error("Private CPU receipt must remain outside the source tree")
        output.mkdir(parents=True, exist_ok=False)
        # Parent-requested integrated gate includes legacy destructive API and
        # original PLE fixture to protect compatibility. No optimizer is created.
        report = {"legacy": verify_tiny_mid_lora_backward(),
                  "persistent": verify_tiny_persistent_mid_lora(verify_cache=args.generation_mode == "native-cache"),
                  "native_ple": verify_tiny_native_ple_backward()}
        if args.generation_mode == "native-cache":
            report["continuous_generation"] = verify_tiny_continuous_generation()
        with (output / "receipt.json").open("x", encoding="utf-8") as target:
            json.dump(report, target, indent=2, ensure_ascii=False, allow_nan=False)
            target.write("\n")
        print(json.dumps({"status": "completed", "mode": args.mode, "output": str(output)}))
        return
    report = run_experiment(profile=args.profile, output=args.output, mode=args.mode,
        authorize_gpu=args.authorize_gpu, authorize_training=args.authorize_training,
        data=args.data, device=args.device, lengths=args.lengths, steps=args.steps,
        heldout_limit=args.heldout_limit, learning_rate=args.learning_rate,
        clip_norm=args.clip_norm, seed=args.seed, max_answer_tokens=args.max_answer_tokens,
        routing_probe_tokens=args.routing_probe_tokens,
        generation_mode=args.generation_mode,
        cache_gate_policy=args.cache_gate_policy,
        cpu_validation_receipt=args.cpu_validation_receipt,
        reference_dtype=args.reference_dtype,
        fp32_linear_accumulation=args.fp32_linear_accumulation,
        allow_numeric_drift_research=args.allow_numeric_drift_research,
        arms=tuple(args.arms), initial_adapter=args.initial_adapter, initial_optimizer_steps=args.initial_optimizer_steps,
        start_sample_index=args.start_sample_index,
        resume_training_state=args.resume_training_state,
        transfer_dataset=args.transfer_dataset,
        generation_expert_backend=args.generation_expert_backend,
        precomputed_think=args.precomputed_think, precomputed_think_source=args.precomputed_think_source,
        config=MidLoraConfig(first_layer=args.first_layer, last_layer=args.last_layer,
                             rank=args.rank, alpha=args.alpha, checkpoint_group_size=args.checkpoint_group_size,
                             offload_checkpoints=args.offload_checkpoints))
    print(json.dumps({"status": report["status"], "mode": args.mode,
                      "optimizer_steps": report["optimizer_steps"], "output": args.output}))


if __name__ == "__main__":
    main()
