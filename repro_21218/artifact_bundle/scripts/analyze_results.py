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


def bar_svg(title: str, bars: list[tuple[str, float, str]], note: str) -> str:
    maximum = max(value for _, value, _ in bars)
    rows = []
    for index, (label, value, display) in enumerate(bars):
        width = 620 * value / maximum
        y = 64 + index * 72
        rows.append(f'<text x="16" y="{y + 22}" font-size="16">{label}</text>')
        rows.append(f'<rect x="235" y="{y}" width="{width:.1f}" height="34" rx="5" fill="#38bdf8"/>')
        rows.append(f'<text x="{245 + width:.1f}" y="{y + 23}" font-size="16" font-weight="700">{display}</text>')
    height = 130 + len(bars) * 72
    return f'''<svg xmlns="http://www.w3.org/2000/svg" width="1960" height="{height * 2}" viewBox="0 0 980 {height}" role="img" aria-label="{title}">
<rect width="980" height="{height}" fill="#07111f"/><g fill="#e5eef8" font-family="system-ui, sans-serif"><text x="16" y="34" font-size="24" font-weight="700">{title}</text>
{''.join(rows)}<text x="16" y="{height - 20}" font-size="14" fill="#9fb3c8">{note}</text></g></svg>'''


def bar_html(title: str, svg: str) -> str:
    return f'''<!doctype html><html><head><meta charset="utf-8"><title>{title}</title></head>
<body style="margin:0;background:#07111f;color:#e5eef8;font-family:system-ui,sans-serif">{svg}</body></html>'''


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

    claim1_svg = bar_svg(
        "Claim 1 — memory falls, perplexity rises",
        [("Parameter memory reduction", 100 * data["memory"]["parameter_reduction_fraction"], f'{100 * data["memory"]["parameter_reduction_fraction"]:.2f}%'),
         ("CUDA allocated reduction", 100 * data["memory"]["cuda_allocated_reduction_fraction"], f'{100 * data["memory"]["cuda_allocated_reduction_fraction"]:.2f}%'),
         ("Perplexity increase", 100 * (data["evaluation"]["perplexity_ratio"] - 1), f'{100 * (data["evaluation"]["perplexity_ratio"] - 1):.1f}%')],
        "A100; OLMoE-1B-active/7B-total proxy; 50% experts; 12 held-out C4 blocks.")
    claim2_svg = bar_svg(
        "Claim 2 — faster, but below 1.5×",
        [("Measured throughput speedup", data["throughput"]["speedup"], f'{data["throughput"]["speedup"]:.3f}×'),
         ("Paper target throughput", 1.5, "1.500×"),
         ("Pruning time / 10-min limit", data["pruning_seconds"] / 600, f'{data["pruning_seconds"]:.1f}s / 600s')],
        "A100 synchronized forward benchmark; 12 repeats of 512 tokens; full 128-step released calibration.")
    (OUTPUTS / "claim1_figure.svg").write_text(claim1_svg, encoding="utf-8")
    (OUTPUTS / "claim2_figure.svg").write_text(claim2_svg, encoding="utf-8")
    (OUTPUTS / "claim1_figure.html").write_text(bar_html("Claim 1 — memory falls, perplexity rises", claim1_svg), encoding="utf-8")
    (OUTPUTS / "claim2_figure.html").write_text(bar_html("Claim 2 — faster, but below 1.5×", claim2_svg), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
