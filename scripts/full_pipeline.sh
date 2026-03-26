#!/bin/bash
# Usage: ./scripts/full_pipeline.sh [category] [image_dir]
# Runs the complete feedback loop: train → infer → collect → retrain if needed

set -e

CATEGORY=${1:-screw}
IMAGE_DIR=${2:-"data/raw/${CATEGORY}/test"}
CONFIG="config/config.yaml"
LOG_FILE="experiments/pipeline_$(date +%Y%m%d_%H%M%S).log"

mkdir -p experiments

echo "========================================"        | tee -a "$LOG_FILE"
echo " Robot Vision — Full Feedback Pipeline"          | tee -a "$LOG_FILE"
echo " Category  : $CATEGORY"                          | tee -a "$LOG_FILE"
echo " Image Dir : $IMAGE_DIR"                         | tee -a "$LOG_FILE"
echo " Log       : $LOG_FILE"                          | tee -a "$LOG_FILE"
echo "========================================"        | tee -a "$LOG_FILE"

# Step 1: Train if no model exists
WEIGHTS="models/checkpoints/${CATEGORY}_patchcore.pt"
if [ ! -f "$WEIGHTS" ]; then
    echo ""                                            | tee -a "$LOG_FILE"
    echo "[STEP 1/3] No model found — training..."     | tee -a "$LOG_FILE"
    bash scripts/retrain.sh "$CATEGORY" "$CONFIG"      | tee -a "$LOG_FILE"
else
    echo "[STEP 1/3] Model found — skipping train"     | tee -a "$LOG_FILE"
fi

# Step 2: Run inference + collect edge cases
echo ""                                                | tee -a "$LOG_FILE"
echo "[STEP 2/3] Running inference..."                 | tee -a "$LOG_FILE"
bash scripts/run_inference.sh "$IMAGE_DIR" "$CATEGORY" | tee -a "$LOG_FILE"

# Step 3: Run feedback loop (retrain if edge cases >= threshold)
echo ""                                                | tee -a "$LOG_FILE"
echo "[STEP 3/3] Checking feedback loop..."            | tee -a "$LOG_FILE"
python -c "
from src.pipeline.feedback_loop import run_feedback_loop
run_feedback_loop('$IMAGE_DIR', '$CONFIG')
"                                                      | tee -a "$LOG_FILE"

echo ""                                                | tee -a "$LOG_FILE"
echo "[DONE] Full pipeline complete. Log → $LOG_FILE"  | tee -a "$LOG_FILE"
