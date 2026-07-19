# Scaled proxy protocol

## Conditions

1. `single_no_feedback`: one Qwen proposal, evaluated once.
2. `single_iterative`: three proposals; each later prompt receives the prior configuration, balanced accuracy, and execution feedback; the best measured round is retained.
3. `multi_agent`: an Explorer message conditions a Builder, an Evaluator critiques each measured trial, and the Builder revises for three rounds; the best measured round is retained.

All conditions use the same Qwen/Qwen2.5-Coder-1.5B-Instruct checkpoint and the same sklearn pipeline search surface. Multi-agent coordination is sequential and fixed-role, matching the broad shape—but not the scale or exact implementation—of the paper’s Explorer → Builder → Evaluator protocol.

## Tasks and measurement

- sklearn breast cancer, wine, and digits classification datasets.
- Three stratified train/validation splits (`seed = 0, 1, 2`).
- Balanced accuracy on the held-out 30% validation split.
- Paired task/seed comparisons for iteration and coordination deltas.

## Compute and provenance

- Hardware: one Hugging Face `t4-small` Job (1× NVIDIA T4 16 GB).
- Successful Job: https://huggingface.co/jobs/binzhango/6a5ae418bee6ee1cf4ece59a
- Two earlier Jobs failed before measurement because of missing `accelerate` and a Transformers API incompatibility; both were fixed in the successful revision.
- Exact script revision: https://huggingface.co/datasets/binzhango/icml-32495-reproduction-artifacts/resolve/67259e2f0261b7e2d315c9f4a12de79fd183f941/run_ablation.py

## Scope label

`scaled-proxy`. This does not reproduce the paper’s DeepSeek-V3.2/Gemini/Claude/Kimi backbones, 100 Kaggle competitions, code-writing execution environment, 10 iterations, retrieval, or approximately 4,000 runs. The proxy can support or challenge a mechanism direction but cannot independently reproduce the full paper claims.
