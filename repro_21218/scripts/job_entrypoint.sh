#!/usr/bin/env bash
set -euo pipefail

python -m pip install --quiet \
  transformers==4.55.3 \
  datasets==2.20.0 \
  accelerate==1.5.2 \
  tokenizers==0.21.1 \
  tqdm==4.67.1 \
  scipy==1.13.1 \
  scikit-learn==1.6.1

export PYTHONPATH=/workspace/upstream_STEP
python /workspace/scripts/gpu_repro.py \
  --max-steps 128 \
  --block-size 2048 \
  --eval-blocks 12 \
  --eval-block-size 512 \
  --benchmark-repeats 12 \
  --output /results/gpu_results.json
