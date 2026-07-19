# Source & Reproducibility Audit


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_975bd26b8ed8", "created_at": "2026-07-18T09:59:55+00:00", "title": "Audit scope"}
-->
This page establishes the authoritative paper and code sources, the exact released revision, and the boundary between paper-reported values and independently measured evidence. The official ICML page links the authors' repository at https://github.com/hikvision-research/STEP; the inspected revision is 03fdea9ac627bb8e6a3f1f5243a1eb6008605198. OpenReview: https://openreview.net/forum?id=4iupzej9nT.


---
<!-- trackio-cell
{"type": "code", "id": "cell_77afccb016da", "created_at": "2026-07-18T10:03:28+00:00", "title": "Run: python gpu_repro.py (exit 1)", "command": ["python", "scripts/gpu_repro.py", "--smoke", "--output", "outputs/smoke_results.json"], "exit_code": 1, "duration_s": 0.593}
-->
````bash
$ python scripts/gpu_repro.py --smoke --output outputs/smoke_results.json
````

exit 1 · 0.6s


````python title=gpu_repro.py
#!/usr/bin/env python3
"""Independent STEP reproduction for ICML 2026 paper 21218.

The GPU path runs the authors' released OLMoE implementation at 50% expert
sparsity. The CPU smoke path only validates the measurement/output pipeline.
"""

from __future__ import annotations

import argparse
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
    import torch
    from torch import nn

    torch.manual_seed(42)

    class ToyMoE(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.shared = nn.Linear(16, 16, bias=False)
            self.experts = nn.ModuleList([nn.Linear(16, 32, bias=False) for _ in range(8)])

        def forward(self, x):
            return self.shared(x) + self.experts[0](x)[..., :16]

    model = ToyMoE()
    before = unique_parameter_bytes(model)
    for index in range(4, 8):
        model.experts[index] = nn.Identity()
    after = unique_parameter_bytes(model)
    x = torch.randn(2, 4, 16)
    assert model(x).shape == x.shape
    payload = {
        "evidence_scope": "local CPU smoke test; not claim evidence",
        "upstream_commit": UPSTREAM_COMMIT,
        "parameter_bytes_before": before,
        "parameter_bytes_after": after,
        "parameter_reduction_fraction": (before - after) / before,
        "python": platform.python_version(),
        "status": "ok",
    }
    write_json(output, payload)
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

````


````output
Traceback (most recent call last):
  File "/Users/binzhang/vibe_coding_repo/reproduce_icml/repro_21218/scripts/gpu_repro.py", line 306, in <module>
    main()
  File "/Users/binzhang/vibe_coding_repo/reproduce_icml/repro_21218/scripts/gpu_repro.py", line 300, in main
    run_smoke(args.output)
  File "/Users/binzhang/vibe_coding_repo/reproduce_icml/repro_21218/scripts/gpu_repro.py", line 56, in run_smoke
    import torch
ModuleNotFoundError: No module named 'torch'

````


---
<!-- trackio-cell
{"type": "code", "id": "cell_239ec5367d13", "created_at": "2026-07-18T10:04:01+00:00", "title": "Run: python gpu_repro.py (exit 0)", "command": ["python", "scripts/gpu_repro.py", "--smoke", "--output", "outputs/smoke_results.json"], "exit_code": 0, "duration_s": 0.06}
-->
````bash
$ python scripts/gpu_repro.py --smoke --output outputs/smoke_results.json
````

exit 0 · 0.1s


````python title=gpu_repro.py
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

````


````output
SMOKE_RESULT={"evidence_scope": "dependency-free local source/JSON smoke test; not claim evidence", "python": "3.12.13", "required_functions": ["expert_prune", "expert_prune_by_step", "finetune"], "sparsity_config": {"experts": 64, "fraction": 0.5, "preserved": 32}, "status": "ok", "upstream_commit": "03fdea9ac627bb8e6a3f1f5243a1eb6008605198", "upstream_file": "upstream_STEP/step/pruning/expert_prune.py"}

````


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_9434a89e1f6a", "created_at": "2026-07-18T10:04:01+00:00", "title": "Smoke-test resolution"}
-->
Resolution of the preceding failed smoke run: the host lacks PyTorch, so the local check was revised to validate the released STEP source AST, required entry points, the exact 50% sparsity configuration, and JSON artifact round-tripping without installing a multi-gigabyte GPU framework locally. The later successful run supersedes the failed portability probe; neither is scientific claim evidence.


---
<!-- trackio-cell
{"type": "code", "id": "cell_d5417d7a9099", "created_at": "2026-07-18T10:14:06+00:00", "title": "Run: python analyze_results.py (exit 0)", "command": ["python", "scripts/analyze_results.py"], "exit_code": 0, "duration_s": 0.133}
-->
````bash
$ python scripts/analyze_results.py
````

exit 0 · 0.1s


````python title=analyze_results.py
#!/usr/bin/env python3
"""Summarize raw GPU results and produce transparent figure payloads."""

from __future__ import annotations

import csv
import json
import random
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / "outputs"
RAW = OUTPUTS / "gpu_results.json"


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    index = (len(ordered) - 1) * q
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    weight = index - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def bootstrap_paired_delta(before: list[float], after: list[float], draws: int = 10_000) -> list[float]:
    rng = random.Random(21218)
    paired = list(zip(before, after))
    results = []
    for _ in range(draws):
        sample = [paired[rng.randrange(len(paired))] for _ in paired]
        results.append(statistics.fmean(item[1] - item[0] for item in sample))
    return results


def bootstrap_speedup(before: list[float], after: list[float], tokens: int, draws: int = 10_000) -> list[float]:
    rng = random.Random(21218)
    results = []
    for _ in range(draws):
        b = [before[rng.randrange(len(before))] for _ in before]
        a = [after[rng.randrange(len(after))] for _ in after]
        results.append((tokens / statistics.median(a)) / (tokens / statistics.median(b)))
    return results


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def bar_html(title: str, bars: list[tuple[str, float, str]], note: str) -> str:
    maximum = max(value for _, value, _ in bars)
    rows = []
    for index, (label, value, display) in enumerate(bars):
        width = 620 * value / maximum
        y = 64 + index * 72
        rows.append(f'<text x="16" y="{y + 22}" font-size="16">{label}</text>')
        rows.append(f'<rect x="235" y="{y}" width="{width:.1f}" height="34" rx="5" fill="#38bdf8"/>')
        rows.append(f'<text x="{245 + width:.1f}" y="{y + 23}" font-size="16" font-weight="700">{display}</text>')
    height = 130 + len(bars) * 72
    return f'''<!doctype html><html><head><meta charset="utf-8"><title>{title}</title></head>
<body style="margin:0;background:#07111f;color:#e5eef8;font-family:system-ui,sans-serif">
<svg viewBox="0 0 980 {height}" role="img" aria-label="{title}" style="width:100%;height:auto">
<rect width="980" height="{height}" fill="#07111f"/><text x="16" y="34" font-size="24" font-weight="700">{title}</text>
{''.join(rows)}<text x="16" y="{height - 20}" font-size="14" fill="#9fb3c8">{note}</text></svg></body></html>'''


def main() -> None:
    data = json.loads(RAW.read_text(encoding="utf-8"))
    baseline_losses = data["evaluation"]["baseline"]["losses"]
    pruned_losses = data["evaluation"]["pruned"]["losses"]
    loss_boot = bootstrap_paired_delta(baseline_losses, pruned_losses)
    before_times = data["throughput"]["baseline"]["seconds"]
    after_times = data["throughput"]["pruned"]["seconds"]
    tokens = data["throughput"]["baseline"]["tokens_per_repeat"]
    speed_boot = bootstrap_speedup(before_times, after_times, tokens)

    summary = {
        "job_url": "https://huggingface.co/jobs/binzhango/6a5b5027bee6ee1cf4ecf1aa",
        "bucket_result": "https://huggingface.co/buckets/binzhango/icml-21218-step-repro/gpu_results.json",
        "scope": data["evidence_scope"],
        "gpu": data["gpu"],
        "expert_sparsity": data["expert_sparsity"]["realized_fraction"],
        "parameter_reduction_fraction": data["memory"]["parameter_reduction_fraction"],
        "cuda_allocated_reduction_fraction": data["memory"]["cuda_allocated_reduction_fraction"],
        "baseline_perplexity": data["evaluation"]["baseline"]["perplexity"],
        "pruned_perplexity": data["evaluation"]["pruned"]["perplexity"],
        "loss_delta": data["evaluation"]["loss_delta"],
        "loss_delta_bootstrap_95ci": [percentile(loss_boot, 0.025), percentile(loss_boot, 0.975)],
        "throughput_speedup": data["throughput"]["speedup"],
        "throughput_speedup_bootstrap_95ci": [percentile(speed_boot, 0.025), percentile(speed_boot, 0.975)],
        "pruning_seconds": data["pruning_seconds"],
        "pruning_under_10_minutes": data["pruning_under_10_minutes"],
        "approximate_job_cost_usd": 2.50 * 347 / 3600,
        "claim_1_verdict": "PARTIAL/PROXY: memory direction reproduced; minimal quality degradation not reproduced at released scale",
        "claim_2_verdict": "PARTIAL/PROXY: pruning completed under 10 minutes; 1.5x throughput not reproduced at released scale",
    }
    (OUTPUTS / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    claim1_rows = [
        {"metric": "parameter_bytes", "baseline": data["memory"]["baseline_parameter_bytes"], "pruned": data["memory"]["pruned_parameter_bytes"], "change_fraction": -data["memory"]["parameter_reduction_fraction"]},
        {"metric": "cuda_allocated_bytes", "baseline": data["memory"]["baseline_cuda_allocated_bytes"], "pruned": data["memory"]["pruned_cuda_allocated_bytes"], "change_fraction": -data["memory"]["cuda_allocated_reduction_fraction"]},
        {"metric": "c4_perplexity", "baseline": data["evaluation"]["baseline"]["perplexity"], "pruned": data["evaluation"]["pruned"]["perplexity"], "change_fraction": data["evaluation"]["perplexity_ratio"] - 1},
    ]
    claim2_rows = [
        {"metric": "throughput_tokens_per_second", "baseline": data["throughput"]["baseline"]["median_tokens_per_second"], "pruned": data["throughput"]["pruned"]["median_tokens_per_second"], "ratio": data["throughput"]["speedup"]},
        {"metric": "pruning_seconds", "baseline": 600, "pruned": data["pruning_seconds"], "ratio": data["pruning_seconds"] / 600},
    ]
    write_csv(OUTPUTS / "claim1_raw.csv", claim1_rows)
    write_csv(OUTPUTS / "claim2_raw.csv", claim2_rows)

    (OUTPUTS / "claim1_figure.html").write_text(bar_html(
        "Claim 1 — memory falls, perplexity rises",
        [("Parameter memory reduction", 100 * data["memory"]["parameter_reduction_fraction"], f'{100 * data["memory"]["parameter_reduction_fraction"]:.2f}%'),
         ("CUDA allocated reduction", 100 * data["memory"]["cuda_allocated_reduction_fraction"], f'{100 * data["memory"]["cuda_allocated_reduction_fraction"]:.2f}%'),
         ("Perplexity increase", 100 * (data["evaluation"]["perplexity_ratio"] - 1), f'{100 * (data["evaluation"]["perplexity_ratio"] - 1):.1f}%')],
        "A100; OLMoE-1B-active/7B-total proxy; 50% experts; 12 held-out C4 blocks."), encoding="utf-8")
    (OUTPUTS / "claim2_figure.html").write_text(bar_html(
        "Claim 2 — faster, but below 1.5×",
        [("Measured throughput speedup", data["throughput"]["speedup"], f'{data["throughput"]["speedup"]:.3f}×'),
         ("Paper target throughput", 1.5, "1.500×"),
         ("Pruning time / 10-min limit", data["pruning_seconds"] / 600, f'{data["pruning_seconds"]:.1f}s / 600s')],
        "A100 synchronized forward benchmark; 12 repeats of 512 tokens; full 128-step released calibration."), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

````


````output
{
  "job_url": "https://huggingface.co/jobs/binzhango/6a5b5027bee6ee1cf4ecf1aa",
  "bucket_result": "https://huggingface.co/buckets/binzhango/icml-21218-step-repro/gpu_results.json",
  "scope": "scaled proxy: released OLMoE-1B-active/7B-total, not paper Qwen3-30B-A3B",
  "gpu": "NVIDIA A100-SXM4-80GB",
  "expert_sparsity": 0.5,
  "parameter_reduction_fraction": 0.4653998508804359,
  "cuda_allocated_reduction_fraction": 0.46424941217049837,
  "baseline_perplexity": 9.619923421921586,
  "pruned_perplexity": 20.91713811772839,
  "loss_delta": 0.7767325242360434,
  "loss_delta_bootstrap_95ci": [
    0.55533763418595,
    1.0123536581794421
  ],
  "throughput_speedup": 1.2330683541768486,
  "throughput_speedup_bootstrap_95ci": [
    1.2246629797539745,
    1.2479412745878289
  ],
  "pruning_seconds": 246.88018622400705,
  "pruning_under_10_minutes": true,
  "approximate_job_cost_usd": 0.24097222222222223,
  "claim_1_verdict": "PARTIAL/PROXY: memory direction reproduced; minimal quality degradation not reproduced at released scale",
  "claim_2_verdict": "PARTIAL/PROXY: pruning completed under 10 minutes; 1.5x throughput not reproduced at released scale"
}

````


---
<!-- trackio-cell
{"type": "artifact", "id": "cell_d39d54b5eb20", "created_at": "2026-07-18T10:14:06+00:00", "title": "Artifact: claim1_raw.csv", "path": "outputs/claim1_raw.csv", "size": 235, "artifact_type": "dataset", "auto": true}
-->
**📦 Artifact** `outputs/claim1_raw.csv` · dataset · 235 B

https://huggingface.co/buckets/binzhango/repro-less-token-more-signal-step-artifacts#logbook-files/outputs/claim1_raw.csv


---
<!-- trackio-cell
{"type": "artifact", "id": "cell_89ad8f5d1d44", "created_at": "2026-07-18T10:14:06+00:00", "title": "Artifact: claim2_raw.csv", "path": "outputs/claim2_raw.csv", "size": 176, "artifact_type": "dataset", "auto": true}
-->
**📦 Artifact** `outputs/claim2_raw.csv` · dataset · 176 B

https://huggingface.co/buckets/binzhango/repro-less-token-more-signal-step-artifacts#logbook-files/outputs/claim2_raw.csv


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_e9c11b829016", "created_at": "2026-07-18T10:14:29+00:00", "title": "What was not released"}
-->
Release boundary: the official repository only instantiates the bundled OLMoE model class. Qwen3 imports and branches are commented, and no Qwen3 model implementation is present. Accordingly, no measured OLMoE number is presented as a direct reproduction of the Qwen3-30B claim. The paper-reported 50% memory, 1.5× throughput, and under-10-minute values are treated as source confirmation only.
