#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash scripts/train_paddleocr.sh /path/to/PaddleOCR /path/to/generated_dataset

if [ "$#" -ne 2 ]; then
  echo "Usage: bash scripts/train_paddleocr.sh /path/to/PaddleOCR /path/to/generated_dataset"
  exit 1
fi

PADDLEOCR_REPO="$1"
DATASET_DIR="$2"
CONFIG_PATH="$(cd "$(dirname "$0")/../configs" && pwd)/rec_aug_train.yml"

if [ ! -d "$PADDLEOCR_REPO" ]; then
  echo "PaddleOCR repo not found: $PADDLEOCR_REPO"
  exit 1
fi

if [ ! -d "$DATASET_DIR" ]; then
  echo "Dataset dir not found: $DATASET_DIR"
  exit 1
fi

cd "$PADDLEOCR_REPO"

python tools/train.py -c "$CONFIG_PATH" \
  -o Train.dataset.data_dir="$DATASET_DIR/train" \
     Train.dataset.label_file_list="['$DATASET_DIR/train/train.txt']" \
     Eval.dataset.data_dir="$DATASET_DIR/val" \
     Eval.dataset.label_file_list="['$DATASET_DIR/val/val.txt']"
