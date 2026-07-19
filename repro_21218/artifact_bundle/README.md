# STEP reproduction bundle — ICML 2026 paper 21218

This bundle supports the Trackio logbook for “Less Token, More Signal: MoE
Expert Pruning via Critical Token Selection” (OpenReview `4iupzej9nT`).

The official release contains runnable OLMoE support but omits the Qwen3
implementation used for the headline 30B claims. The included A100 experiment
therefore runs the released STEP algorithm on `allenai/OLMoE-1B-7B-0125` at
exactly 50% expert sparsity and the full 128-step released calibration schedule.
It is a scaled proxy and must not be interpreted as a Qwen3-30B replication.

To reproduce, inspect `scripts/job_entrypoint.sh` and `scripts/gpu_repro.py`.
Every plotted value is recoverable from `outputs/gpu_results.json`.
