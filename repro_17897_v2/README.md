# ML-Agent scaled independent reproduction

This workspace rebuilds the reproduction around newly measured evidence rather
than values transcribed from the paper. The core experiment evaluates 120 ML
pipeline actions on exactly 9 training tasks and 10 held-out tasks spanning
tabular, image, and text modalities. It then:

1. selects 10 actions whose measured behavior vectors are most distant;
2. trains a task-conditioned ranker only on the 9 training tasks;
3. evaluates action selection on all 10 held-out tasks;
4. applies the paper's v1 ML-specific reward to actual execution outcomes.

Run the local experiment:

```bash
uv run scripts/run_scaled_suite.py --output-dir outputs/scaled_suite
uv run scripts/verify_scaled_suite.py --results outputs/scaled_suite/results.json
```

The separate CIFAR-10 GPU script is added after the task-suite protocol passes
locally. This is a scaled, independent mechanism reproduction; it is not the
unreleased 7B ML-Agent checkpoint.
