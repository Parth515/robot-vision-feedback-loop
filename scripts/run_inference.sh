#!/bin/bash
# Usage: ./scripts/run_inference.sh <image_path_or_dir> [category]

set -e

IMAGE_PATH=${1:?"Usage: $0 <image_path_or_dir> [category]"}
CATEGORY=${2:-screw}
WEIGHTS="models/checkpoints/${CATEGORY}_patchcore.pt"
CONFIG="config/config.yaml"

echo "========================================"
echo " Robot Vision — Inference"
echo " Category : $CATEGORY"
echo " Input    : $IMAGE_PATH"
echo " Weights  : $WEIGHTS"
echo "========================================"

# check weights exist
if [ ! -f "$WEIGHTS" ]; then
    echo "[ERROR] Weights not found: $WEIGHTS"
    echo "        Run ./scripts/retrain.sh first"
    exit 1
fi

python -c "
from src.inference.detect import run_inference
import os, glob

path = '$IMAGE_PATH'
if os.path.isdir(path):
    images = glob.glob(os.path.join(path, '*.png')) + glob.glob(os.path.join(path, '*.jpg'))
    print(f'[INFO] Running on {len(images)} images...')
    for img in sorted(images):
        run_inference(img, weights='$WEIGHTS')
else:
    run_inference(path, weights='$WEIGHTS')
"

echo ""
echo "[DONE] Inference complete. Edge cases saved to data/raw/mistakes/${CATEGORY}/"
