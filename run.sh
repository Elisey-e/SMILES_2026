#!/usr/bin/env bash
set -euo pipefail

python validate.py \
  --data_dir ./data \
  --batch_size 64 \
  --n_batches 128 \
  --output results.json \
  --seed 42
