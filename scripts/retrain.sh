#!/bin/bash
# Usage: ./scripts/retrain.sh [category] [config]

set -e

CATEGORY=${1:-screw}
CONFIG=${2:-config/config.yaml}
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
BACKUP_DIR="models/checkpoints/backup_${TIMESTAMP}"

echo "========================================"
echo " Robot Vision — Retrain"
echo " Category  : $CATEGORY"
echo " Config    : $CONFIG"
echo " Timestamp : $TIMESTAMP"
echo "========================================"

# backup existing model before overwriting
EXISTING="models/checkpoints/${CATEGORY}_patchcore.pt"
if [ -f "$EXISTING" ]; then
    mkdir -p "$BACKUP_DIR"
    cp "$EXISTING" "$BACKUP_DIR/"
    echo "[INFO] Backed up existing model → $BACKUP_DIR"
fi

# check GPU
echo "[INFO] Checking GPU..."
python -c "from src.gpu.gpu_utils import get_device; get_device()"

# train
echo ""
echo "[INFO] Building memory bank..."
python -c "
from src.utils.config_loader import load_config
from src.training.train import train
train('$CONFIG')
"

# evaluate after training
echo ""
echo "[INFO] Evaluating new model..."
python -c "
from src.evaluation.evaluate import evaluate
auroc, f1 = evaluate('$CATEGORY', 'models/checkpoints/${CATEGORY}_patchcore.pt')
print(f'AUROC: {auroc:.4f} | F1: {f1:.4f}')
# fail the script if model is bad
import sys
if auroc < 0.80:
    print('[WARN] AUROC below 0.80 — consider reverting to backup')
    sys.exit(1)
"

echo ""
echo "[DONE] Retrain complete → models/checkpoints/${CATEGORY}_patchcore.pt"
