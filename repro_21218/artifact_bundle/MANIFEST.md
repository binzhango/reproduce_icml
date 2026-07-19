# Reproduction bundle manifest

- `README.md` — scope and scale boundary.
- `scripts/gpu_repro.py` — independent measurement harness.
- `scripts/job_entrypoint.sh` — exact GPU environment entrypoint.
- `scripts/analyze_results.py` — bootstrap analysis and figure generation.
- `outputs/gpu_results.json` — raw GPU measurements, per-block losses, and all
  timing repetitions.
- `outputs/summary.json` — derived verdicts and bootstrap intervals.
- `outputs/*.csv`, `outputs/*.svg`, `outputs/*.html` — raw figure data and
  rendered claim figures.
- `outputs/smoke_results.json` — local source/JSON smoke check.
- `upstream_STEP/` — authors' released code at commit
  `03fdea9ac627bb8e6a3f1f5243a1eb6008605198` (Git metadata excluded).
- `poster/` — final Posterly HTML, interactive embed, PNG, PDF, gate reports,
  and build notes.

GPU Job: https://huggingface.co/jobs/binzhango/6a5b5027bee6ee1cf4ecf1aa

Raw Job artifact: https://huggingface.co/buckets/binzhango/icml-21218-step-repro/gpu_results.json
