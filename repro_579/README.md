# Paper 579 reproduction bundle

This folder contains the auditable reproduction materials for arXiv:2601.18119 / OpenReview `4ltyJqAHMg`.

- `verify_reported_claims.py` checks the requested values against the extracted paper text and audits Table 2.
- `reported_table2.json` is a faithful transcription of the 24 non-SFT model rows in Table 2.
- `toy_gpu_eval.py` constructs a deterministic 12-task enterprise-style SQL proxy and can evaluate Qwen2.5-Coder-7B on a GPU.
- `outputs/` contains machine-readable audits, toy tasks, predictions, and summaries.

The synthetic benchmark is explicitly a toy proxy. The paper's Squirrel data, Calcite/TQS evaluator, and official model predictions were not publicly available when this reproduction was run.

## Re-run

```bash
python repro_579/verify_reported_claims.py
UV_CACHE_DIR=/tmp/uv-cache uv run repro_579/toy_gpu_eval.py --mode smoke --output-dir repro_579/outputs/toy_smoke
```

The substantive run used `Qwen/Qwen2.5-Coder-7B-Instruct` on one Hugging Face A10G-small Job with greedy decoding. The complete Job record is https://huggingface.co/jobs/binzhango/6a58221cb1669a49bf07668b and its persisted outputs are at https://huggingface.co/datasets/binzhango/paper579-squirrel-toy-repro.
