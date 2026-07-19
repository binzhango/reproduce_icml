# ML-Agent paper #17897 reproduction bundle

This bundle contains the independently executed evidence used by the Trackio logbook for arXiv 2505.23723 / OpenReview `kcPPWaoegr`. It does **not** contain the unreleased ML-Agent checkpoint, evaluator, exact prompts, trajectories, or author training code, so it cannot reproduce the paper's headline agent comparison.

## Contents

- `scripts/run_scaled_suite.py`: T4 experiment with 120 executable pipelines, 9 training tasks, 10 held-out tasks, 2,280 fits, reward events, and 5,000 diversity controls.
- `scripts/run_cifar_repeats.py`: five full-data CIFAR-10 CNN runs on an A10G.
- `scripts/verify_*.py`: fail-closed checks for experiment identity, counts, hardware, data scale, metrics, and reward branches.
- `outputs/scaled-suite/`: raw JSON/CSV, checksums, and summary figure from HF Job `6a5a89c6bee6ee1cf4ecddc5`.
- `outputs/cifar-gpu/`: raw results, checksums, and figure from HF Job `6a5a8a51bee6ee1cf4ecddcb`.
- `poster/`: source HTML, strict gate report, rendered PNG, and verified 60×36-inch PDF.

## Verify locally

From the bundle root:

```bash
uv run scripts/verify_scaled_suite.py outputs/scaled-suite/results.json
uv run scripts/verify_cifar_repeats.py outputs/cifar-gpu/results.json
```

The scaled suite is a disclosed mechanism proxy, not Qwen2.5-7B EFT/RL. The CIFAR experiment is a disclosed CNN proxy, not an ML-Agent trajectory. Paper-reported values are kept separate from fresh measurements throughout the logbook.

## Public compute provenance

- T4 Job: https://huggingface.co/jobs/binzhango/6a5a89c6bee6ee1cf4ecddc5
- T4 artifacts: https://huggingface.co/buckets/binzhango/jobs-artifacts#20260717T200004-ddae91/scaled-suite-retry
- A10G Job: https://huggingface.co/jobs/binzhango/6a5a8a51bee6ee1cf4ecddcb
- A10G artifacts: https://huggingface.co/buckets/binzhango/jobs-artifacts#20260717T200224-7699dd/cifar-gpu-localcopy
- Staged datasets: https://huggingface.co/buckets/binzhango/ml-agent-repro-v2-jobs

