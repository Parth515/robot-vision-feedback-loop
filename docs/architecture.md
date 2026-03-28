# Architecture

This project implements an anomaly detection feedback loop for industrial parts such as screws, metal nuts, and bottles.

## Goal

The system learns only from normal images and marks anything that deviates from normal appearance as anomalous.

This is useful when:

- defect labels are limited
- defect types are unknown in advance
- new defect categories may appear in production

## High-Level Flow

```text
Normal training images
        ↓
Feature extraction
        ↓
Memory bank creation
        ↓
Threshold estimation
        ↓
Inference on new image
        ↓
Anomaly score
        ↓
NORMAL / DEFECT decision
        ↓
Save edge cases for review
        ↓
Retrain with updated data
```

## Folder Overview

```text
robot-vision-feedback-loop/
│
├── README.md
├── requirements.txt
├── config/
│   └── config.yaml
│
├── data/
│   ├── raw/
│   ├── labeled/
│   ├── processed/
│   └── splits/
│
├── models/
│   ├── checkpoints/
│   └── exported/
│
├── src/
│   ├── anomaly/
│   │   ├── patchcore.py
│   │   ├── feature_extractor.py
│   │   └── threshold.py
│   ├── inference/
│   │   └── detect.py
│   ├── data_collection/
│   │   └── collect.py
│   ├── training/
│   │   └── train.py
│   ├── evaluation/
│   │   └── evaluate.py
│   ├── gpu/
│   │   └── gpu_utils.py
│   ├── utils/
│   │   ├── logger.py
│   │   └── config_loader.py
│   └── pipeline/
│       └── feedback_loop.py
│
├── scripts/
│   ├── run_inference.sh
│   ├── retrain.sh
│   ├── full_pipeline.sh
│   ├── benchmark_gpu.sh
│   └── export_tensorrt.sh
│
└── docs/
    ├── setup.md
    ├── usage.md
    ├── architecture.md
    └── deployment.md
```

## Core Modules

### `src/anomaly/patchcore.py`
Builds the memory bank from normal images and computes anomaly scores.

### `src/anomaly/feature_extractor.py`
Handles backbone feature extraction.

### `src/anomaly/threshold.py`
Finds the decision threshold from known-good images.

### `src/inference/detect.py`
Runs scoring on new images and returns `NORMAL` or `DEFECT`.

### `src/data_collection/collect.py`
Stores edge cases and metadata for later review.

### `src/evaluation/evaluate.py`
Measures AUROC and F1 on the test set.

### `src/pipeline/feedback_loop.py`
Connects inference, collection, retraining, and evaluation.

## Configuration

Main runtime settings are stored in:

```bash
config/config.yaml
```

Typical options include:

- selected category
- device (`cuda` or `cpu`)
- precision (`fp16` or `fp32`)
- threshold percentile
- retrain trigger count
- minimum accepted AUROC

## Feedback Loop Logic

```text
1. Train on normal images
2. Score incoming images
3. Save suspicious samples
4. Review or accumulate edge cases
5. Retrain when threshold is reached
6. Evaluate the updated model
7. Keep the new model only if quality is acceptable
```