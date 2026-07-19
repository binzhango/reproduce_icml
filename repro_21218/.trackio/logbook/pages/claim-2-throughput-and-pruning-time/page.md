# Claim 2: Throughput and pruning time


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_db1585cb8871", "created_at": "2026-07-18T09:59:56+00:00", "title": "Claim and protocol"}
-->
Exact required claim: Delivers 1.5× throughput improvement and completes entire pruning process within 10 minutes. We will time the released pruning pipeline and benchmark synchronized GPU inference throughput before and after 50% expert pruning with identical prompts, batch sizes, and token counts; results are new measurements and will not be conflated with paper-reported Qwen3-30B values.


---
<!-- trackio-cell
{"type": "figure", "id": "cell_227f230d0751", "created_at": "2026-07-18T10:14:06+00:00", "title": "Throughput and pruning-time result"}
-->
````html
<!doctype html><html><head><meta charset="utf-8"><title>Claim 2 — faster, but below 1.5×</title></head>
<body style="margin:0;background:#07111f;color:#e5eef8;font-family:system-ui,sans-serif">
<svg viewBox="0 0 980 346" role="img" aria-label="Claim 2 — faster, but below 1.5×" style="width:100%;height:auto">
<rect width="980" height="346" fill="#07111f"/><text x="16" y="34" font-size="24" font-weight="700">Claim 2 — faster, but below 1.5×</text>
<text x="16" y="86" font-size="16">Measured throughput speedup</text><rect x="235" y="64" width="509.7" height="34" rx="5" fill="#38bdf8"/><text x="754.7" y="87" font-size="16" font-weight="700">1.233×</text><text x="16" y="158" font-size="16">Paper target throughput</text><rect x="235" y="136" width="620.0" height="34" rx="5" fill="#38bdf8"/><text x="865.0" y="159" font-size="16" font-weight="700">1.500×</text><text x="16" y="230" font-size="16">Pruning time / 10-min limit</text><rect x="235" y="208" width="170.1" height="34" rx="5" fill="#38bdf8"/><text x="415.1" y="231" font-size="16" font-weight="700">246.9s / 600s</text><text x="16" y="326" font-size="14" fill="#9fb3c8">A100 synchronized forward benchmark; 12 repeats of 512 tokens; full 128-step released calibration.</text></svg></body></html>
````

````raw
metric,baseline,pruned,ratio
throughput_tokens_per_second,1445.6445656789726,1782.5785653264759,1.2330683541768486
pruning_seconds,600,246.88018622400705,0.4114669770400117

````


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_8a74a2f51166", "created_at": "2026-07-18T10:14:28+00:00", "title": "Measured verdict"}
-->
Verdict: PARTIAL/PROXY. The full released 128-step STEP calibration plus bias-finetuning procedure completed in 246.88 seconds (4.11 minutes), independently supporting the under-10-minute direction on one A100. Median synchronized forward throughput rose from 1,445.6 to 1,782.6 tokens/s, a 1.233× speedup with bootstrap 95% CI [1.225×, 1.248×], which does not reproduce the claimed 1.5×. The benchmark used 12 repeats of one 512-token block before and after pruning. This remains an OLMoE scale proxy, not Qwen3-30B. A100 Job: https://huggingface.co/jobs/binzhango/6a5b5027bee6ee1cf4ecf1aa. Raw result: https://huggingface.co/buckets/binzhango/icml-21218-step-repro/gpu_results.json.
