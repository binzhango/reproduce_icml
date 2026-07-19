# Claim 1: 7B ML-Agent vs Larger Agents


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_54a5737fde06", "created_at": "2026-07-16T01:12:10+00:00", "title": "Verdict: reported in v1; full comparison blocked"}
-->
ArXiv v1 Table 1 reports the 7B ML-Agent above the MLAB-scaffolded Qwen2.5-7B/32B, GPT-4o-mini, GPT-4o, and 671B DeepSeek-R1 baselines across the listed 3 held-in and 10 held-out tasks. This is source-verified, not independently reproduced: the trained ML-Agent checkpoint and runnable evaluator are absent. Revision warning: v2 replaces GPT-4o and GPT-4o-mini with newer GPT-5/Gemini/Qwen3 comparisons and reports average performance gain; ML-Agent averages 16.40%, below the best GPT-5 agent framework at 20.95%, while remaining above the listed DeepSeek-R1 averages. The challenge claim is therefore specifically a v1 claim, not an unqualified accepted-version result.
