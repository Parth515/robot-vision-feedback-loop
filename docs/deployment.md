# Deployment & Export

This guide explains how to benchmark the model and export the backbone for faster deployment.

## 1. Benchmark GPU Inference

Run latency and throughput measurements:

```bash
bash scripts/benchmark_gpu.sh screw 100
```

This reports:

- mean latency
- median latency
- p95 latency
- throughput (FPS)
- GPU memory usage

## 2. Export to ONNX

Run:

```bash
bash scripts/export_tensorrt.sh screw
```

This creates an ONNX export under:

```bash
models/exported/onnx/
```

## 3. Optional TensorRT Export

If TensorRT is installed and `trtexec` is available, the same script will also create a TensorRT engine under:

```bash
models/exported/tensorrt/
```

## 4. Deployment Strategy

Recommended deployment flow:

```text
PyTorch model
   ↓
ONNX export
   ↓
TensorRT engine
   ↓
Low-latency inference on target GPU
```

## 5. Suggested Production Layout

```text
production/
├── weights/
│   ├── screw_patchcore.pt
│   ├── screw_backbone.onnx
│   └── screw_backbone.engine
├── logs/
├── incoming/
├── reviewed/
└── rejected/
```

## 6. Practical Notes

- Keep the training pipeline in PyTorch.
- Use ONNX or TensorRT only for deployment inference.
- Version exported models together with the source checkpoint.
- Log anomaly scores for traceability.
- Re-export after every accepted retrain.

## 7. Example Deployment Cycle

```bash
# benchmark current model
bash scripts/benchmark_gpu.sh screw 100

# export for deployment
bash scripts/export_tensorrt.sh screw

# run inference on production batch
bash scripts/run_inference.sh production/incoming screw
```