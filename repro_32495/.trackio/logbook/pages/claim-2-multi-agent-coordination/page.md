# Claim 2 - Multi-agent coordination


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_60df09c6121e", "created_at": "2026-07-18T02:24:27+00:00", "title": "Claim and protocol"}
-->
Exact claim: **Multi-agent coordination can hurt as often as it helps performance.**

Planned test: for every task/seed pair, compare the best three-round fixed-role multi-agent score with the best three-round single-agent score, then count positive, negative, and tied deltas. This tests one sequential coordination protocol only and cannot establish a claim about all multi-agent designs.


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_08a61e3671c5", "created_at": "2026-07-18T02:30:43+00:00", "title": "Protocol boundary in the paper"}
-->
Paper-reported context (source confirmation only): the tested multi-agent protocol is a fixed sequential Explorer → Builder → Evaluator pipeline. The paper explicitly limits its negative conclusion to that protocol and reports −8.5 K-LIVE percentile versus the iterative single-agent baseline, with 18.0% failures versus 10.3% for the baseline.


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_dbd1c362d866", "created_at": "2026-07-18T02:45:07+00:00", "title": "Claim 2 verdict and measured evidence"}
-->
Verdict: **PARTIAL / PROXY SUPPORT.** Across nine paired task/seed cases, fixed-role coordination helped **0**, hurt **1**, and tied **8**; its mean delta versus the iterative single agent was **−0.001053 ± 0.003158 SD** balanced accuracy. The literal sign-count statement (“hurt as often as helped”) holds in this toy proxy, but evidence is weak because eight comparisons tie, one digits split drives the negative mean, tasks are small/saturated, and only one sequential Explorer → Builder → Evaluator protocol was tested.

Clean completed measurement provenance: https://huggingface.co/jobs/binzhango/6a5b47cfd216bd6f3a1fdea9; raw paired rows: https://huggingface.co/datasets/binzhango/icml-32495-reproduction-artifacts/blob/main/outputs/gpu/trials.csv.
