#!/usr/bin/env python3
"""Render compact, publication-ready summaries from the saved GPU result."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / "outputs"


def load_results() -> dict:
    with (OUTPUTS / "gpu_mechanism_results.json").open() as handle:
        return json.load(handle)


def diversity_figure(data: dict) -> None:
    random = data["random_baseline"]
    v1 = data["selection"]["v1_mean_distance"]["metrics"]
    fps = data["selection"]["v2_global_fps_diagnostic"]["metrics"]

    labels = ["Random\nmean", "v1 top-10\nmean-distance", "Global FPS\ndiagnostic"]
    values = [random["mean"], v1["mean_pairwise_cosine_distance"], fps["mean_pairwise_cosine_distance"]]
    colors = ["#94A3B8", "#0EA5E9", "#8B5CF6"]

    fig, ax = plt.subplots(figsize=(9.2, 5.4), dpi=220)
    bars = ax.bar(labels, values, color=colors, width=0.62)
    ax.axhline(random["p95"], color="#64748B", linestyle="--", linewidth=1.4, label="random 95th percentile")
    ax.axhline(random["max"], color="#DC2626", linestyle=":", linewidth=1.6, label="best of 5,000 random draws")
    ax.set_ylim(0.80, 0.99)
    ax.set_ylabel("Mean pairwise cosine distance")
    ax.set_title("Diversity selection exceeds 5,000 random subsets", loc="left", fontweight="bold")
    ax.text(
        0,
        1.02,
        "120 disclosed synthetic ideas · MiniLM embeddings · seed 17,897",
        transform=ax.transAxes,
        color="#475569",
        fontsize=10,
    )
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.003, f"{value:.4f}", ha="center", fontsize=10, fontweight="bold")
    ax.legend(frameon=False, loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.16)
    fig.tight_layout()
    fig.savefig(OUTPUTS / "gpu_diversity_summary.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def reward_figure(data: dict) -> None:
    cases = data["reward_cases"]
    keys = ["invalid_runtime_error", "corner_nonedit", "genuine_improvement", "no_change", "regression"]
    labels = ["Runtime\nerror", "Corner /\nnon-edit", "Genuine\nimprovement", "No\nchange", "Regression"]
    v1 = np.array([cases[key]["v1"] for key in keys])
    v2 = np.array([cases[key]["v2"] for key in keys])
    x = np.arange(len(keys))
    width = 0.34

    fig, ax = plt.subplots(figsize=(9.2, 5.4), dpi=220)
    bars1 = ax.bar(x - width / 2, v1, width, label="arXiv v1 equation", color="#0EA5E9")
    bars2 = ax.bar(x + width / 2, v2, width, label="arXiv v2 equation", color="#F97316")
    ax.axhline(0, color="#0F172A", linewidth=0.9)
    ax.set_xticks(x, labels)
    ax.set_ylabel("Reward")
    ax.set_ylim(-1.16, 1.03)
    ax.set_title("The requested reward rule is specific to arXiv v1", loc="left", fontweight="bold")
    ax.text(0, 1.02, "Same five executable test cases under the two published equations", transform=ax.transAxes, color="#475569", fontsize=10)
    for bars in (bars1, bars2):
        for bar in bars:
            value = bar.get_height()
            offset = 0.035 if value >= 0 else -0.09
            ax.text(bar.get_x() + bar.get_width() / 2, value + offset, f"{value:.2f}", ha="center", fontsize=9)
    ax.legend(frameon=False, loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.16)
    fig.tight_layout()
    fig.savefig(OUTPUTS / "reward_revision_comparison.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    data = load_results()
    diversity_figure(data)
    reward_figure(data)


if __name__ == "__main__":
    main()
