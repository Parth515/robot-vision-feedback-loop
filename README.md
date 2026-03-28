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

### Train
```bash
./scripts/retrain.sh screw
```

### Inference
```bash
./scripts/run_inference.sh data/raw/screw/test screw
```

### Full Pipeline
```bash
./scripts/full_pipeline.sh screw
```

### Dataset

Built for MVTec AD

Recommended categories:  
screw  
metal_nut  
bottle  

### Evaluation
AUROC  
F1 Score  


### Project Structure
```bash
src/
  anomaly/
  training/
  inference/
  evaluation/
  pipeline/
```

### Core Idea

The system learns what normal looks like and flags anything that deviates as anomalous.

### Use Cases
Industrial inspection  
Surface defect detection  
PCB / metal part quality control  

### Summary

A practical anomaly detection pipeline designed for industrial environments with unknown and evolving defects.
