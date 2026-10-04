#!/usr/bin/env python3
"""Measure a running QWEN-EXO server with Engram on vs off (per-request switch).

  nll     teacher-forced NLL of fixed token windows (prompt logprobs); compare
          the on-off delta with the offline reference of the same reader
  decode  per-request decode tok/s and speculative accept length at a given
          concurrency (greedy, ignore_eos)
  agree   greedy generation, then the same tokens teacher-forced as a prompt:
          argmax agreement checks that the decode/verify ring path and the
          extend path see the same n-gram history

    python scripts/qwen_exo/eval_engram_served.py nll --tokens wiki_en.pt --windows 64
    python scripts/qwen_exo/eval_engram_served.py decode --tokens wiki_en.pt --concurrency 1 5 10
    python scripts/qwen_exo/eval_engram_served.py agree --tokens wiki_en.pt
"""

from __future__ import annotations

import argparse
import json
import statistics
import threading
import time
import urllib.request

import torch

URL = "http://127.0.0.1:30000/generate"


def post(payload: dict, *, timeout: float = 1800) -> dict:
    request = urllib.request.Request(
        URL, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read())


def sampling(engram: bool, **extra) -> dict:
    return {"temperature": 0, "custom_params": {"qwen_exo_engram": engram}, **extra}


def windows(tokens: torch.Tensor, count: int, length: int) -> list[list[int]]:
    return [tokens[i * length : (i + 1) * length].tolist() for i in range(count)]


def window_nll(ids: list[int], engram: bool) -> float:
    out = post(
        {
            "input_ids": ids,
            "sampling_params": sampling(engram, max_new_tokens=1),
            "return_logprob": True,
            "logprob_start_len": 0,
        }
    )
    logprobs = [row[0] for row in out["meta_info"]["input_token_logprobs"][1:]]
    return -sum(logprobs) / len(logprobs)


def run_nll(args) -> None:
    rows = windows(args.tokens_tensor, args.windows, args.seq_len + 1)
    result = {}
    for engram in (True, False):
        nlls = [window_nll(ids, engram) for ids in rows]
        result["on" if engram else "off"] = statistics.mean(nlls)
    result["delta"] = result["on"] - result["off"]
    print(json.dumps({"mode": "nll", "windows": len(rows), **result}), flush=True)


def stream_one(ids: list[int], engram: bool, new_tokens: int, out: list, index: int) -> None:
    payload = {
        "input_ids": ids,
        "sampling_params": sampling(engram, max_new_tokens=new_tokens, ignore_eos=True),
        "stream": True,
    }
    request = urllib.request.Request(
        URL, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"}
    )
    first = last = None
    meta = {}
    with urllib.request.urlopen(request, timeout=1800) as response:
        for raw in response:
            line = raw.decode().strip()
            if not line.startswith("data:") or line == "data: [DONE]":
                continue
            now = time.perf_counter()
            first = first or now
            last = now
            meta = json.loads(line[5:])["meta_info"]
    tokens = meta.get("completion_tokens", 0)
    out[index] = {
        "tps": (tokens - 1) / (last - first) if last and last > first else 0.0,
        "accept": meta.get("spec_accept_length"),
    }


def run_decode(args) -> None:
    for engram in (True, False):
        for size in args.concurrency:
            out = [None] * size
            prompts = windows(args.tokens_tensor[args.offset :], size, args.prompt_len)
            threads = [
                threading.Thread(target=stream_one, args=(prompt, engram, args.new_tokens, out, i))
                for i, prompt in enumerate(prompts)
            ]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
            accepts = [row["accept"] for row in out if row["accept"] is not None]
            print(
                json.dumps(
                    {
                        "mode": "decode",
                        "engram": engram,
                        "concurrency": size,
                        "tps_mean": statistics.mean(row["tps"] for row in out),
                        "accept_mean": statistics.mean(accepts) if accepts else None,
                    }
                ),
                flush=True,
            )


def run_agree(args) -> None:
    for engram in (True, False):
        agree = total = 0
        for prompt in windows(args.tokens_tensor, args.windows, args.prompt_len):
            generated = post(
                {"input_ids": prompt, "sampling_params": sampling(engram, max_new_tokens=args.new_tokens)}
            )["output_ids"]
            replay = post(
                {
                    "input_ids": prompt + generated,
                    "sampling_params": sampling(engram, max_new_tokens=1),
                    "return_logprob": True,
                    "logprob_start_len": len(prompt) - 1,
                    "top_logprobs_num": 1,
                }
            )
            tops = replay["meta_info"]["input_top_logprobs"]
            # tops[j] predicts token len(prompt) - 1 + j + 1.
            predicted = [row[0][1] for row in tops[1 : 1 + len(generated)]]
            agree += sum(int(p == g) for p, g in zip(predicted, generated))
            total += len(generated)
        print(json.dumps({"mode": "agree", "engram": engram, "argmax_agreement": agree / total}), flush=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=("nll", "decode", "agree"))
    p.add_argument("--tokens", required=True, help="torch int tensor of token ids")
    p.add_argument("--windows", type=int, default=64)
    p.add_argument("--seq-len", type=int, default=1024)
    p.add_argument("--concurrency", type=int, nargs="+", default=[1, 5, 10])
    p.add_argument("--prompt-len", type=int, default=512)
    p.add_argument("--new-tokens", type=int, default=256)
    p.add_argument("--offset", type=int, default=0)
    args = p.parse_args()
    args.tokens_tensor = torch.load(args.tokens).long()
    {"nll": run_nll, "decode": run_decode, "agree": run_agree}[args.mode](args)


if __name__ == "__main__":
    main()
