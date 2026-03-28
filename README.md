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

### Dataset

Built for MVTec AD

## Official Download

The dataset should be downloaded from the official MVTec AD download page.
Official source:
```text
https://www.mvtec.com/company/research/datasets/mvtec-ad/downloads
```
## Download Script

Use the provided script:

```bash
chmod +x scripts/download_mvtec.sh
./scripts/download_mvtec.sh /path/to/mvtec_anomaly_detection.tar.xz
```

If your official download is a ZIP archive, the same script also supports that format:

```bash
./scripts/download_mvtec.sh /path/to/mvtec_ad.zip
```
The script extracts only the selected categories into `data/raw/`:
- `screw`
- `metal_nut`
- `bottle`

This keeps the project lightweight and avoids copying the full dataset into the working directory when you only need a few categories.

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

### Evaluation
AUROC  
F1 Score  

Evaluation uses:
- `test/good/` as normal samples
- defect folders under `test/` as anomalous samples

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
