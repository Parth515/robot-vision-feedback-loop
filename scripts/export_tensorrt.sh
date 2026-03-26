#!/bin/bash
# Usage: ./scripts/export_tensorrt.sh [category]

set -e

CATEGORY=${1:-screw}
ONNX_DIR="models/exported/onnx"
TRT_DIR="models/exported/tensorrt"

mkdir -p "$ONNX_DIR" "$TRT_DIR"

echo "========================================"
echo " Robot Vision — Export"
echo " Category : $CATEGORY"
echo "========================================"

# Step 1: Export PyTorch → ONNX
echo "[1/2] Exporting to ONNX..."
python -c "
import torch
from src.anomaly.feature_extractor import FeatureExtractor

extractor = FeatureExtractor()
dummy_input = torch.randn(1, 3, 224, 224).to(extractor.device)

torch.onnx.export(
    extractor.model,
    dummy_input,
    '$ONNX_DIR/${CATEGORY}_backbone.onnx',
    opset_version=17,
    input_names=['image'],
    output_names=['features'],
    dynamic_axes={'image': {0: 'batch_size'}}
)
print('[OK] ONNX saved → $ONNX_DIR/${CATEGORY}_backbone.onnx')
"

# Step 2: ONNX → TensorRT (requires trtexec on system)
echo "[2/2] Converting to TensorRT..."
if command -v trtexec &> /dev/null; then
    trtexec \
        --onnx="$ONNX_DIR/${CATEGORY}_backbone.onnx" \
        --saveEngine="$TRT_DIR/${CATEGORY}_backbone.engine" \
        --fp16 \
        --workspace=2048
    echo "[OK] TensorRT engine saved → $TRT_DIR/${CATEGORY}_backbone.engine"
else
    echo "[WARN] trtexec not found — skipping TensorRT conversion"
    echo "       Install TensorRT: https://developer.nvidia.com/tensorrt"
    echo "       ONNX model is still usable for faster inference"
fi

echo ""
echo "[DONE] Export complete."
