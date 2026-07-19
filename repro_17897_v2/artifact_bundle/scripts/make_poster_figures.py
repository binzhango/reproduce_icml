# /// script
# requires-python = ">=3.11"
# dependencies = ["matplotlib>=3.8", "numpy>=1.26"]
# ///
"""Generate wide, print-oriented figures from logged raw result JSON."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
source = ROOT / "outputs" / "hf_jobs" / "cifar-gpu" / "results.json"
output = ROOT / "poster" / "cifar_repeats_wide.png"
data = json.loads(source.read_text(encoding="utf-8"))
runs = data["measurement"]["runs"]
seeds = [str(run["seed"]) for run in runs]
values = np.asarray([run["final_accuracy"] * 100 for run in runs])
mean = float(values.mean())

figure, axis = plt.subplots(figsize=(12, 3.2))
bars = axis.bar(seeds, values, color="#238FBD", width=0.62)
axis.axhline(mean, color="#D69A2D", linestyle="--", linewidth=2.2, label=f"new mean {mean:.2f}%")
axis.set_ylim(85.8, 87.05)
axis.set_xlabel("Seed")
axis.set_ylabel("Final accuracy (%)")
axis.set_title("Five fresh full-data CIFAR-10 runs on one A10G")
axis.legend(loc="upper left", frameon=False)
for bar, value in zip(bars, values):
    axis.text(bar.get_x() + bar.get_width() / 2, value + 0.035, f"{value:.2f}", ha="center", va="bottom", fontsize=10)
axis.spines[["top", "right"]].set_visible(False)
figure.tight_layout()
figure.savefig(output, dpi=220)
plt.close(figure)
print(f"POSTER_FIGURE={output}")
