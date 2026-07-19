# Claim 1: Memory and quality at 50% expert sparsity


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_bebaaf269b4a", "created_at": "2026-07-18T09:59:56+00:00", "title": "Claim and protocol"}
-->
Exact required claim: On 30B Qwen3 MoE with 50% expert sparsity achieves nearly 50% reduction in memory usage with minimal performance degradation. We will test the released STEP method at the largest feasible released scale, explicitly label the model-scale mismatch, and report newly measured parameter bytes, peak/resident CUDA memory, and held-out language-model loss/perplexity against an unpruned baseline.


---
<!-- trackio-cell
{"type": "figure", "id": "cell_fe1a92655c6b", "created_at": "2026-07-18T10:14:06+00:00", "title": "Memory-quality tradeoff"}
-->
````html
<!doctype html><html><head><meta charset="utf-8"><title>Claim 1 — memory falls, perplexity rises</title></head>
<body style="margin:0;background:#07111f;color:#e5eef8;font-family:system-ui,sans-serif">
<svg viewBox="0 0 980 346" role="img" aria-label="Claim 1 — memory falls, perplexity rises" style="width:100%;height:auto">
<rect width="980" height="346" fill="#07111f"/><text x="16" y="34" font-size="24" font-weight="700">Claim 1 — memory falls, perplexity rises</text>
<text x="16" y="86" font-size="16">Parameter memory reduction</text><rect x="235" y="64" width="245.7" height="34" rx="5" fill="#38bdf8"/><text x="490.7" y="87" font-size="16" font-weight="700">46.54%</text><text x="16" y="158" font-size="16">CUDA allocated reduction</text><rect x="235" y="136" width="245.1" height="34" rx="5" fill="#38bdf8"/><text x="490.1" y="159" font-size="16" font-weight="700">46.42%</text><text x="16" y="230" font-size="16">Perplexity increase</text><rect x="235" y="208" width="620.0" height="34" rx="5" fill="#38bdf8"/><text x="865.0" y="231" font-size="16" font-weight="700">117.4%</text><text x="16" y="326" font-size="14" fill="#9fb3c8">A100; OLMoE-1B-active/7B-total proxy; 50% experts; 12 held-out C4 blocks.</text></svg></body></html>
````

````raw
metric,baseline,pruned,change_fraction
parameter_bytes,13838323712,7397969920,-0.4653998508804359
cuda_allocated_bytes,13840421376,7415013888,-0.46424941217049837
c4_perplexity,9.619923421921586,20.91713811772839,1.174355990200822

````


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_dc87e1dfe356", "created_at": "2026-07-18T10:14:28+00:00", "title": "Measured verdict"}
-->
Verdict: PARTIAL/PROXY. On the released OLMoE-1B-active/7B-total model, replacing exactly 32 of 64 experts in every layer reduced unique parameter bytes from 13,838,323,712 to 7,397,969,920 (46.54%) and live CUDA allocation from 13,840,421,376 to 7,415,013,888 bytes (46.42%). However, held-out C4 perplexity increased from 9.620 to 20.917: paired mean loss delta +0.777, bootstrap 95% CI [0.555, 1.012], so minimal performance degradation was not reproduced at this scale. This is not a direct Qwen3-30B test because the official release omits runnable Qwen support. A100 Job: https://huggingface.co/jobs/binzhango/6a5b5027bee6ee1cf4ecf1aa. Immutable raw result: https://huggingface.co/buckets/binzhango/icml-21218-step-repro/gpu_results.json.
