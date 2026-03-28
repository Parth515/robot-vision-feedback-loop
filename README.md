# Robot Vision Feedback Loop — Industrial Anomaly Detection

Unsupervised anomaly detection system for industrial inspection using **PatchCore + ResNet50**.

Learns **normal patterns only** and detects defects as deviations, making it suitable for real-world factory environments with unknown or rare failures.

---

## Features
- No defect labels required  
- Detects unseen anomalies  
- PatchCore memory bank inference  
- GPU support (CUDA, FP16)  
- ONNX / TensorRT export  
- Continuous feedback loop for improvement  

---

## How It Works
Train on normal images → build memory bank → score new samples → flag anomalies → collect edge cases → retrain.

---

## Quick Start
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
bash scripts/full_pipeline.sh screw data/raw/screw/test
```

## Documentation
- [Setup & Installation](docs/setup.md)
- [Running Cycle & Usage](docs/usage.md)
- [Architecture](docs/architecture.md)
- [Deployment & Export](docs/deployment.md)

## Tech Stack
- PatchCore (anomaly detection)
- PyTorch + CUDA (GPU acceleration)
- MVTec AD dataset
- ONNX / TensorRT (deployment)

### Core Idea

The system learns what normal looks like and flags anything that deviates as anomalous.

### Use Cases
Industrial inspection  
Surface defect detection  
PCB / metal part quality control  

### Summary

A practical anomaly detection pipeline designed for industrial environments with unknown and evolving defects.
