# ICML 2026 paper 32495 reproduction

This directory independently tests the mechanisms behind the paper's first two claims using a deliberately scaled proxy: `Qwen/Qwen2.5-Coder-1.5B-Instruct` chooses sklearn pipelines for three public classification tasks under no-feedback, iterative single-agent, and fixed-role multi-agent protocols.

It is not a full replication of the paper's 4,000 Kaggle runs. The paper's cited repository, `https://github.com/mireskandari/klive`, returned `Repository not found` on 2026-07-18, and the released PDF does not provide executable agent code or raw run data.

Run the local smoke test:

```bash
uv run run_ablation.py --mock --seeds 1 --rounds 2 --output-dir outputs/smoke
```

Run the scaled experiment on a GPU:

```bash
uv run run_ablation.py --seeds 3 --rounds 3 --output-dir outputs/gpu
```
