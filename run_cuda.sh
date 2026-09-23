#!/usr/bin/env bash
# Max-speed training on a CUDA GPU (tested on the RTX 4090 box).
set -euo pipefail
cd "$(dirname "$0")"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
exec .venv/bin/python -u train_vae.py "$@" \
  --epochs 1200 \
  --grow-frac 0.5 \
  --batch-size 256 \
  --lr 8e-4 \
  --beta 0.3 \
  --perceptual-weight 0.04 \
  --channels-last \
  --sample-every 50 \
  --sample-count 36
