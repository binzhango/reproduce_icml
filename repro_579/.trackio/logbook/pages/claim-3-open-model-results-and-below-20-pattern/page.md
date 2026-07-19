# Claim 3: Open-model results and below-20 pattern


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_ef814df3c466", "created_at": "2026-07-16T00:08:40+00:00", "title": "Scope and verification plan"}
-->
This page transcribes and programmatically audits Table 2 for DeepSeek-V3, Qwen-2.5-Coder-32B, and the claim that most evaluated models remain below 20% Graph Match.


---
<!-- trackio-cell
{"type": "code", "id": "cell_cf3675108d01", "created_at": "2026-07-16T00:10:57+00:00", "title": "Run: python verify_reported_claims.py (exit 0)", "command": [".venv/bin/python", "repro_579/verify_reported_claims.py"], "exit_code": 0, "duration_s": 0.024}
-->
````bash
$ .venv/bin/python repro_579/verify_reported_claims.py
````

exit 0 · 0.0s


````python title=verify_reported_claims.py
#!/usr/bin/env python3
"""Audit the four requested claims against the downloaded paper text and Table 2."""

from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent / "outputs"


def main() -> None:
    text = (ROOT / "paper_2601.18119.txt").read_text(errors="replace")
    rows = json.loads((Path(__file__).parent / "reported_table2.json").read_text())

    required_fragments = {
        "task_counts": ["469 Squirrel-Syntax", "516 Squirrel-Semantic"],
        "seed_corpus": ["1,000+ SQL scripts spanning 26 business scenarios"],
        "claude": ["23.88   36.46   68.02", "31.78   32.17   43.69"],
        "deepseek_v3": ["17.91   30.28   60.34", "11.24   21.32   33.27"],
        "qwen_32b": ["12.79   20.26   52.88", "17.44   23.45   34.69"],
        "complexity": ["496.90    163.69        21.62", "425.93    141.58        17.34"],
    }
    fragment_checks = {
        key: all(fragment in text for fragment in fragments)
        for key, fragments in required_fragments.items()
    }

    under20_syn = [r["model"] for r in rows if r["syntax_gm"] < 20]
    under20_sem = [r["model"] for r in rows if r["semantic_gm"] < 20]
    either_under20 = [
        r["model"] for r in rows
        if r["syntax_gm"] < 20 or r["semantic_gm"] < 20
    ]
    both_under20 = [
        r["model"] for r in rows
        if r["syntax_gm"] < 20 and r["semantic_gm"] < 20
    ]

    report = {
        "source": "arXiv:2601.18119v1 PDF text extraction",
        "table2_model_rows_excluding_sft": len(rows),
        "paper_fragment_checks": fragment_checks,
        "claim_1": {
            "syntax_tasks": 469,
            "semantic_tasks": 516,
            "seed_scripts": "1,000+",
            "business_scenarios": 26,
            "status": "reported values verified; official benchmark unavailable for recomputation",
        },
        "claim_2": {
            "claude_syntax_gm": 36.46,
            "claude_semantic_gm": 32.17,
            "internal_typo": "Introduction prose says 33.17; abstract, Table 2, and Sec. 5.1 say 32.17",
            "status": "verified from Table 2 and Sec. 5.1",
        },
        "claim_3": {
            "deepseek_v3_syntax_gm": 30.28,
            "deepseek_v3_semantic_gm": 21.32,
            "qwen_2_5_coder_32b_semantic_gm": 23.45,
            "syntax_below_20": len(under20_syn),
            "semantic_below_20": len(under20_sem),
            "either_track_below_20": len(either_under20),
            "both_tracks_below_20": len(both_under20),
            "denominator": len(rows),
            "status": "model values verified; 'most below 20' is true per-track for semantic only and true if interpreted as below 20 on at least one track",
        },
        "claim_4": {
            "syntax": {"tokens": 496.90, "lines": 163.69, "functions": 21.62, "ast_depth": 8.93, "ast_width": 11.69},
            "semantic": {"tokens": 425.93, "lines": 141.58, "functions": 17.34, "ast_depth": 8.75, "ast_width": 11.12},
            "status": "reported values verified; official scripts unavailable for recomputation",
        },
    }

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "reported_claims_audit.json").write_text(json.dumps(report, indent=2) + "\n")
    with (OUT / "table2_gm.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

````


````output
{
  "source": "arXiv:2601.18119v1 PDF text extraction",
  "table2_model_rows_excluding_sft": 24,
  "paper_fragment_checks": {
    "task_counts": true,
    "seed_corpus": true,
    "claude": true,
    "deepseek_v3": true,
    "qwen_32b": true,
    "complexity": true
  },
  "claim_1": {
    "syntax_tasks": 469,
    "semantic_tasks": 516,
    "seed_scripts": "1,000+",
    "business_scenarios": 26,
    "status": "reported values verified; official benchmark unavailable for recomputation"
  },
  "claim_2": {
    "claude_syntax_gm": 36.46,
    "claude_semantic_gm": 32.17,
    "internal_typo": "Introduction prose says 33.17; abstract, Table 2, and Sec. 5.1 say 32.17",
    "status": "verified from Table 2 and Sec. 5.1"
  },
  "claim_3": {
    "deepseek_v3_syntax_gm": 30.28,
    "deepseek_v3_semantic_gm": 21.32,
    "qwen_2_5_coder_32b_semantic_gm": 23.45,
    "syntax_below_20": 10,
    "semantic_below_20": 15,
    "either_track_below_20": 16,
    "both_tracks_below_20": 9,
    "denominator": 24,
    "status": "model values verified; 'most below 20' is true per-track for semantic only and true if interpreted as below 20 on at least one track"
  },
  "claim_4": {
    "syntax": {
      "tokens": 496.9,
      "lines": 163.69,
      "functions": 21.62,
      "ast_depth": 8.93,
      "ast_width": 11.69
    },
    "semantic": {
      "tokens": 425.93,
      "lines": 141.58,
      "functions": 17.34,
      "ast_depth": 8.75,
      "ast_width": 11.12
    },
    "status": "reported values verified; official scripts unavailable for recomputation"
  }
}

````


---
<!-- trackio-cell
{"type": "artifact", "id": "cell_a4b3a98a663f", "created_at": "2026-07-16T00:10:57+00:00", "title": "Artifact: table2_gm.csv", "path": "repro_579/outputs/table2_gm.csv", "size": 727, "artifact_type": "dataset", "auto": true}
-->
**📦 Artifact** `repro_579/outputs/table2_gm.csv` · dataset · 727 B

https://huggingface.co/buckets/binzhango/repro-beyond-text-to-sql-squirrel-artifacts#logbook-files/repro_579/outputs/table2_gm.csv


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_361069d0ea17", "created_at": "2026-07-16T00:14:16+00:00", "title": "Claim 3 verdict: scores verified; threshold wording qualified"}
-->
The programmatic Table 2 audit confirms DeepSeek-V3 GM = 30.28% syntax / 21.32% semantic and Qwen-2.5-Coder-32B semantic GM = 23.45%. Among 24 non-SFT model rows, 10/24 are below 20% on syntax, 15/24 on semantic, 16/24 on at least one track, and 9/24 on both. Thus “most below 20%” is supported for semantic GM and for the at-least-one-track reading, but not separately for syntax GM. The exact audit is captured in the run cell and reported_claims_audit.json.


---
<!-- trackio-cell
{"type": "figure", "id": "cell_ddb17c0bbae8", "created_at": "2026-07-16T00:22:11+00:00", "title": "Selected Graph Match scores from Table 2"}
-->
````html
<!doctype html>
<meta charset="utf-8">
<title>Selected Graph Match scores from Table 2</title>
<style>
body{font:16px system-ui,sans-serif;color:#172033;background:#fff;margin:24px}.chart{max-width:900px}.row{display:grid;grid-template-columns:220px 1fr 70px;gap:12px;align-items:center;margin:10px 0}.track{height:24px;background:#eef2f7;border-radius:5px;overflow:hidden}.bar{height:100%;background:#2d5f8b}.semantic{background:#c9a24a}.legend{display:flex;gap:24px;margin:16px 0}.swatch{display:inline-block;width:14px;height:14px;margin-right:6px;background:#2d5f8b}.swatch.semantic{background:#c9a24a}h1{font-size:24px}.note{color:#596579;font-size:14px}
</style>
<div class="chart">
<h1>Selected Graph Match scores — paper Table 2</h1>
<div class="legend"><span><i class="swatch"></i>Syntax</span><span><i class="swatch semantic"></i>Semantic</span></div>
<div class="row"><strong>Claude-4-Sonnet</strong><div class="track"><div class="bar" style="width:91.15%"></div></div><span>36.46%</span></div>
<div class="row"><span></span><div class="track"><div class="bar semantic" style="width:80.43%"></div></div><span>32.17%</span></div>
<div class="row"><strong>DeepSeek-V3</strong><div class="track"><div class="bar" style="width:75.70%"></div></div><span>30.28%</span></div>
<div class="row"><span></span><div class="track"><div class="bar semantic" style="width:53.30%"></div></div><span>21.32%</span></div>
<div class="row"><strong>Qwen2.5-Coder-32B</strong><div class="track"><div class="bar" style="width:50.65%"></div></div><span>20.26%</span></div>
<div class="row"><span></span><div class="track"><div class="bar semantic" style="width:58.63%"></div></div><span>23.45%</span></div>
<p class="note">Bars use a 0–40% axis. Values are transcribed from arXiv:2601.18119v1 Table 2; they are not independently rescored.</p>
</div>

````

````raw
model,syntax_gm,semantic_gm
Qwen-2.5-Instruct-7B,8.53,5.62
Qwen-2.5-Coder-7B,8.96,7.75
Qwen-2.5-Coder-32B,20.26,23.45
Qwen-3-Instruct-235B,20.47,15.5
Qwen-3-Coder-Instruct-30B,20.9,15.12
Qwen-3-Coder-Instruct-480B,23.88,19.96
QwQ-32B,20.51,15.31
Seed-Coder-Instruct-8B,14.93,14.15
OmniSQL-32B,6.4,6.4
DeepSeek-V3-685B,30.28,21.32
DeepSeek-V3.1-685B,30.49,14.73
DeepSeek-R1-671B,21.98,22.09
Claude-4-Sonnet,36.46,32.17
GPT-4o-mini-2024-07-18,4.69,6.4
GPT-4o-2024-11-20,4.69,4.84
GPT-4.1,17.7,17.05
GPT-5,18.55,16.47
Gemini-2.5-Pro,21.54,23.06
Kimi-K2,27.72,20.93
O1-preview,21.11,11.43
O3-mini,19.83,28.68
Doubao-Seed-1.6,30.92,20.93
Doubao-Seed-1.6-flash,3.63,3.11
Doubao-Seed-1.6-thinking,23.24,20.93

````
