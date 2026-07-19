# /// script
# requires-python = ">=3.11"
# dependencies = ["pandas>=2.2"]
# ///
"""Create a self-contained HTML/SVG summary from raw GPU trials."""

import argparse
import html
import json
from pathlib import Path

import pandas as pd


COLORS = {"single_no_feedback": "#94a3b8", "single_iterative": "#2563eb", "multi_agent": "#d97706"}
LABELS = {"single_no_feedback": "No feedback", "single_iterative": "Iterative single", "multi_agent": "Fixed-role multi"}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--results", default="outputs/gpu")
    p.add_argument("--html", default="outputs/results_figure.html")
    p.add_argument("--svg", default="outputs/results_chart.svg")
    args = p.parse_args()
    root = Path(args.results)
    trials = pd.read_csv(root / "trials.csv")
    summary = json.loads((root / "summary.json").read_text())
    means = trials.groupby("condition")["score"].mean()
    pivot = trials.pivot(index=["task", "seed"], columns="condition", values="score")
    deltas = (pivot["multi_agent"] - pivot["single_iterative"]).tolist()

    w, h = 1100, 620
    bars = []
    for i, condition in enumerate(["single_no_feedback", "single_iterative", "multi_agent"]):
        value = float(means[condition])
        x = 90 + i * 255
        bar_h = value * 400
        y = 500 - bar_h
        bars.append(
            f'<rect x="{x}" y="{y:.1f}" width="160" height="{bar_h:.1f}" rx="10" fill="{COLORS[condition]}"/>'
            f'<text x="{x + 80}" y="{y - 14:.1f}" text-anchor="middle" class="value">{value:.3f}</text>'
            f'<text x="{x + 80}" y="535" text-anchor="middle" class="label">{html.escape(LABELS[condition])}</text>'
        )
    dots = []
    for i, delta in enumerate(deltas):
        cx = 850 + (i % 3) * 65
        cy = 300 - delta * 1700
        color = "#dc2626" if delta < 0 else "#16a34a" if delta > 0 else "#64748b"
        dots.append(f'<circle cx="{cx}" cy="{cy:.1f}" r="10" fill="{color}" opacity="0.85"/>')

    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">
<style>.title{{font:700 30px system-ui;fill:#0f172a}}.value{{font:700 22px system-ui;fill:#0f172a}}.label{{font:600 18px system-ui;fill:#334155}}.note{{font:500 18px system-ui;fill:#475569}}</style>
<rect width="100%" height="100%" fill="#ffffff"/><text x="50" y="48" class="title">Scaled proxy: balanced accuracy and coordination deltas</text>
<line x1="55" y1="500" x2="790" y2="500" stroke="#94a3b8" stroke-width="2"/>{''.join(bars)}
<text x="835" y="120" class="title">Multi − single</text><line x1="825" y1="300" x2="1050" y2="300" stroke="#64748b" stroke-width="2"/>{''.join(dots)}
<text x="835" y="480" class="note">helps {summary['multi_agent_helps']} · hurts {summary['multi_agent_hurts']} · ties {summary['multi_agent_ties']}</text>
<text x="55" y="585" class="note">Iteration gain {summary['iteration_gain_mean']:+.4f}; coordination mean {summary['coordination_delta_mean']:+.4f}. Qwen2.5-Coder-1.5B, 3 tasks × 3 seeds.</text>
</svg>'''
    svg_path = Path(args.svg)
    svg_path.parent.mkdir(parents=True, exist_ok=True)
    svg_path.write_text(svg)
    Path(args.html).write_text(f'<!doctype html><meta charset="utf-8"><title>ICML 32495 proxy results</title><style>body{{margin:0;background:#f8fafc;display:grid;place-items:center;min-height:100vh}}svg{{max-width:100%;height:auto}}</style>{svg}')
    print(f"Wrote {args.html} and {args.svg}")


if __name__ == "__main__":
    main()
