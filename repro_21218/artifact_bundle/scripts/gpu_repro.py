#!/usr/bin/env python3
"""Independent STEP reproduction for ICML 2026 paper 21218.

The GPU path runs the authors' released OLMoE implementation at 50% expert
sparsity. The CPU smoke path only validates the measurement/output pipeline.
"""

from __future__ import annotations

import argparse
import ast
import gc
import json
import math
import os
import platform
import random
import statistics
import time
from pathlib import Path
from types import SimpleNamespace


UPSTREAM_COMMIT = "03fdea9ac627bb8e6a3f1f5243a1eb6008605198"
MODEL_ID = "allenai/OLMoE-1B-7B-0125"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--max-steps", type=int, default=128)
    parser.add_argument("--block-size", type=int, default=2048)
    parser.add_argument("--eval-blocks", type=int, default=12)
    parser.add_argument("--eval-block-size", type=int, default=512)
    parser.add_argument("--benchmark-repeats", type=int, default=12)
    parser.add_argument("--output", type=Path, default=Path("outputs/gpu_results.json"))
    return parser.parse_args()


def unique_parameter_bytes(model) -> int:
    seen: set[tuple[int, int]] = set()
    total = 0
    for parameter in model.parameters():
        key = (parameter.untyped_storage().data_ptr(), parameter.untyped_storage().nbytes())
        if key not in seen:
            seen.add(key)
            total += parameter.untyped_storage().nbytes()
    return total


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run_smoke(output: Path) -> dict:
    workspace = Path(__file__).resolve().parents[1]
    upstream_file = workspace / "upstream_STEP" / "step" / "pruning" / "expert_prune.py"
    source = upstream_file.read_text(encoding="utf-8")
    tree = ast.parse(source)
    functions = {node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)}
    required = {"expert_prune", "expert_prune_by_step", "finetune"}
    missing = sorted(required - functions)
    if missing:
        raise RuntimeError(f"Missing expected upstream functions: {missing}")
    roundtrip = {"experts": 64, "preserved": 32, "fraction": 0.5}
    payload = {
        "evidence_scope": "dependency-free local source/JSON smoke test; not claim evidence",
        "upstream_commit": UPSTREAM_COMMIT,
        "upstream_file": str(upstream_file.relative_to(workspace)),
        "required_functions": sorted(required),
        "sparsity_config": roundtrip,
        "python": platform.python_version(),
        "status": "ok",
    }
    write_json(output, payload)
    if json.loads(output.read_text(encoding="utf-8"))["sparsity_config"] != roundtrip:
        raise RuntimeError("JSON roundtrip failed")
    print("SMOKE_RESULT=" + json.dumps(payload, sort_keys=True))
    return payload


def build_blocks(tokenizer, split: str, count: int, block_size: int, seed: int):
    import torch
    from datasets import load_dataset
    from torch.utils.data import DataLoader
    from transformers import default_data_collator

    stream = load_dataset("allenai/c4", "en", split=split, streaming=True)
    stream = stream.shuffle(seed=seed, buffer_size=2_000)
    token_buffer: list[int] = []
    blocks: list[dict[str, list[int]]] = []
    for row in stream:
        token_buffer.extend(tokenizer(row["text"], add_special_tokens=False)["input_ids"])
        while len(token_buffer) >= block_size and len(blocks) < count:
            ids = token_buffer[:block_size]
            del token_buffer[:block_size]
            blocks.append({"input_ids": ids, "attention_mask": [1] * block_size, "labels": ids.copy()})
        if len(blocks) >= count:
            break
    if len(blocks) != count:
        raise RuntimeError(f"Only constructed {len(blocks)} of {count} requested C4 blocks")
    return DataLoader(blocks, batch_size=1, shuffle=False, collate_fn=default_data_collator)


def evaluate_loss(model, loader) -> dict:
    import torch

    losses = []
    model.eval()
    with torch.inference_mode():
        for batch in loader:
            batch = {key: value.to("cuda:0") for key, value in batch.items()}
            losses.append(float(model(**batch).loss.detach().float().cpu()))
    mean_loss = statistics.fmean(losses)
    return {
        "blocks": len(losses),
        "mean_loss": mean_loss,
        "perplexity": math.exp(mean_loss),
        "losses": losses,
    }


def benchmark_forward(model, input_ids, repeats: int) -> dict:
    import torch

    model.eval()
    attention_mask = torch.ones_like(input_ids)
    with torch.inference_mode():
        for _ in range(3):
            model(input_ids=input_ids, attention_mask=attention_mask)
        torch.cuda.synchronize()
        timings = []
        torch.cuda.reset_peak_memory_stats()
        for _ in range(repeats):
            start = time.perf_counter()
            model(input_ids=input_ids, attention_mask=attention_mask)
            torch.cuda.synchronize()
            timings.append(time.perf_counter() - start)
    tokens = input_ids.numel()
    return {
        "repeats": repeats,
        "tokens_per_repeat": tokens,
        "seconds": timings,
        "median_seconds": statistics.median(timings),
        "median_tokens_per_second": tokens / statistics.median(timings),
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(),
    }


def approximate_expert_counts(model) -> dict:
    per_layer = []
    for layer in model.model.layers:
        per_layer.append(sum(bool(getattr(expert, "is_approx", False)) for expert in layer.mlp.experts))
    return {
        "per_layer": per_layer,
        "total": sum(per_layer),
        "layers": len(per_layer),
        "experts_per_layer": model.config.num_experts,
    }


def run_gpu(args: argparse.Namespace) -> dict:
    import torch
    import transformers
    from transformers import AutoTokenizer

    from step.model.olmoe.modeling_olmoe import OlmoeForCausalLM
    from step.pruning.expert_prune import expert_prune

    random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.backends.cuda.matmul.allow_tf32 = False

    if not torch.cuda.is_available():
        raise RuntimeError("GPU experiment requires CUDA")

    started = time.time()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
    model = OlmoeForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        use_cache=False,
        attn_implementation="eager",
        device_map={"": "cuda:0"},
    )
    model.eval()
    torch.cuda.synchronize()
    baseline_parameter_bytes = unique_parameter_bytes(model)
    baseline_cuda_allocated = torch.cuda.memory_allocated()
    baseline_cuda_reserved = torch.cuda.memory_reserved()

    calibration_loader = build_blocks(tokenizer, "train", args.max_steps, args.block_size, seed=42)
    eval_loader = build_blocks(tokenizer, "validation", args.eval_blocks, args.eval_block_size, seed=21218)
    fixed_input = next(iter(eval_loader))["input_ids"].to("cuda:0")

    baseline_eval = evaluate_loss(model, eval_loader)
    baseline_throughput = benchmark_forward(model, fixed_input, args.benchmark_repeats)

    prune_args = SimpleNamespace(
        max_steps=args.max_steps,
        preserve_n_experts=model.config.num_experts // 2,
        expert_prune_metric="step",
        expert_ranking_metric="fusion",
        expert_ranking_scope="layer",
        fusion_io_weight=0.5,
        tau=0.5,
        no_bias=False,
        enable_novice_evolving=False,
        save_model=False,
    )
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    prune_started = time.perf_counter()
    expert_prune(prune_args, model, calibration_loader, tokenizer)
    torch.cuda.synchronize()
    pruning_seconds = time.perf_counter() - prune_started
    pruning_peak_allocated = torch.cuda.max_memory_allocated()
    pruning_peak_reserved = torch.cuda.max_memory_reserved()

    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    pruned_parameter_bytes = unique_parameter_bytes(model)
    pruned_cuda_allocated = torch.cuda.memory_allocated()
    pruned_cuda_reserved = torch.cuda.memory_reserved()
    counts = approximate_expert_counts(model)

    pruned_eval = evaluate_loss(model, eval_loader)
    pruned_throughput = benchmark_forward(model, fixed_input, args.benchmark_repeats)

    result = {
        "schema_version": 1,
        "evidence_scope": "scaled proxy: released OLMoE-1B-active/7B-total, not paper Qwen3-30B-A3B",
        "model": MODEL_ID,
        "upstream_commit": UPSTREAM_COMMIT,
        "method": "STEP, layer-wise fusion ranking, tau=0.5, bias calibration",
        "seed": 42,
        "gpu": torch.cuda.get_device_name(0),
        "gpu_count": torch.cuda.device_count(),
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "calibration": {
            "dataset": "allenai/c4 train streaming",
            "blocks": args.max_steps,
            "block_size": args.block_size,
        },
        "evaluation": {
            "dataset": "allenai/c4 validation streaming",
            "block_size": args.eval_block_size,
            "baseline": baseline_eval,
            "pruned": pruned_eval,
            "loss_delta": pruned_eval["mean_loss"] - baseline_eval["mean_loss"],
            "perplexity_ratio": pruned_eval["perplexity"] / baseline_eval["perplexity"],
        },
        "expert_sparsity": {
            "target_fraction": 0.5,
            "approximate_counts": counts,
            "realized_fraction": counts["total"] / (counts["layers"] * counts["experts_per_layer"]),
        },
        "memory": {
            "baseline_parameter_bytes": baseline_parameter_bytes,
            "pruned_parameter_bytes": pruned_parameter_bytes,
            "parameter_reduction_fraction": (baseline_parameter_bytes - pruned_parameter_bytes) / baseline_parameter_bytes,
            "baseline_cuda_allocated_bytes": baseline_cuda_allocated,
            "pruned_cuda_allocated_bytes": pruned_cuda_allocated,
            "cuda_allocated_reduction_fraction": (baseline_cuda_allocated - pruned_cuda_allocated) / baseline_cuda_allocated,
            "baseline_cuda_reserved_bytes": baseline_cuda_reserved,
            "pruned_cuda_reserved_bytes": pruned_cuda_reserved,
            "pruning_peak_allocated_bytes": pruning_peak_allocated,
            "pruning_peak_reserved_bytes": pruning_peak_reserved,
        },
        "throughput": {
            "baseline": baseline_throughput,
            "pruned": pruned_throughput,
            "speedup": pruned_throughput["median_tokens_per_second"] / baseline_throughput["median_tokens_per_second"],
        },
        "pruning_seconds": pruning_seconds,
        "pruning_under_10_minutes": pruning_seconds < 600,
        "total_wall_seconds": time.time() - started,
    }
    write_json(args.output, result)
    print("FINAL_RESULT=" + json.dumps(result, sort_keys=True))
    return result


def main() -> None:
    args = parse_args()
    if args.smoke:
        run_smoke(args.output)
    else:
        run_gpu(args)


if __name__ == "__main__":
    main()
