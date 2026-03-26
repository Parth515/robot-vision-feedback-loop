# Robot Vision Feedback Loop — Industrial Anomaly Detection

An end-to-end unsupervised anomaly detection system for factory part inspection using PatchCore with a ResNet50 backbone. The project is designed for industrial parts such as screws, metal nuts, bottles, PCBs, and similar components, and uses only normal images for training.

---

## Overview

In real industrial environments, collecting labeled defect data for every possible failure mode is difficult, expensive, and often unrealistic. New defect types can appear at any time, and supervised detectors usually fail when they encounter defects they were never trained on.

This project solves that problem with anomaly detection. Instead of learning defect classes, the system learns what **normal** parts look like and flags anything that deviates from that normal pattern. This makes it practical for factory inspection pipelines where defect types are unknown or constantly changing.

---

## Problem Statement

Traditional defect detection systems require:
- Large labeled datasets
- Defect annotations for each category
- Retraining whenever a new defect type appears

For industrial quality inspection, this creates major limitations:
- Labeling is slow and costly
- Rare defects may have almost no samples
- New failure modes are not covered
- Manual maintenance becomes difficult

This project uses an anomaly detection approach:
- Train only on good samples
- Detect unseen defects
- Avoid defect-type labeling
- Build a feedback loop for continuous improvement

---

## How It Works

The system follows this pipeline:

1. Collect only normal images from factory parts.
2. Build a memory bank of patch-level features from those normal samples.
3. At inference time, compare a new image to the memory bank.
4. Compute an anomaly score based on distance from known normal patterns.
5. If the score exceeds a threshold, flag the sample as defective.
6. Save flagged edge cases for later review and model improvement.
7. Retrain the memory bank when enough new useful samples are collected.

---

## Project Structure

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
│   │   ├── screw/
│   │   ├── metal_nut/
│   │   ├── bottle/
│   │   └── mistakes/
│   ├── labeled/
│   ├── processed/
│   └── splits/
│
├── models/
│   ├── checkpoints/
│   └── exported/
│       ├── onnx/
│       └── tensorrt/
│
├── src/
│   ├── anomaly/
│   │   ├── __init__.py
│   │   ├── patchcore.py
│   │   ├── feature_extractor.py
│   │   └── threshold.py
│   │
│   ├── inference/
│   │   └── detect.py
│   │
│   ├── data_collection/
│   │   └── collect.py
│   │
│   ├── training/
│   │   └── train.py
│   │
│   ├── evaluation/
│   │   └── evaluate.py
│   │
│   ├── utils/
│   │   ├── logger.py
│   │   └── config_loader.py
│   │
│   ├── gpu/
│   │   └── gpu_utils.py
│   │
│   └── pipeline/
│       └── feedback_loop.py
│
├── scripts/
│   ├── run_inference.sh
│   ├── retrain.sh
│   ├── export_tensorrt.sh
│   ├── benchmark_gpu.sh
│   └── full_pipeline.sh
│
└── experiments/
    ├── exp_v1/
    ├── exp_v2/
    └── notes.md

Dataset
This project is intended to use the MVTec Anomaly Detection (MVTec AD) dataset.

MVTec AD is one of the most widely used industrial anomaly detection benchmarks and includes categories such as:

Screw

Metal nut

Bottle

Capsule

Hazelnut

Cable

PCB

Leather

Grid

Wood

Tile

For this project, a good starting point is:

screw

metal_nut

bottle

These are simple, industrially relevant, and easy to explain in demos and documentation.

Why Anomaly Detection Instead of Object Detection
Object detection is useful when:

You know all defect classes in advance

You have labels for each defect type

You want bounding boxes for specific known defects

Anomaly detection is better here because:

You do not need labels for each defect type

You can use all normal training data

The system can detect unseen defect types

The pipeline is closer to real factory quality control problems

This is especially useful in industry because many production defects are rare, unstructured, and difficult to label consistently.

Model Choice
This project uses PatchCore as the anomaly detection method.

Why PatchCore
Strong performance for industrial anomaly detection

No heavy gradient-based training loop

Works well with small sets of normal images

Simple deployment logic

Suitable for GPU acceleration

Good fit for MVTec AD

Backbone
The default backbone is:

ResNet50

The backbone extracts patch-level features from normal images, and PatchCore stores those features in a memory bank. At inference time, abnormal regions produce larger feature distances from that bank.

Configuration
All configuration should be managed through:

text
config/config.yaml
Example:

text
device: cuda
precision: fp16
category: screw
batch_size: 1
num_workers: 4
img_size: 224

model:
  name: patchcore
  backbone: resnet50

threshold:
  percentile: 95

retrain_threshold: 20
min_auroc: 0.80
Main Settings
device: choose cuda or cpu

precision: use fp16 for faster GPU inference

category: active MVTec category

img_size: resized input dimension

threshold.percentile: threshold sensitivity

retrain_threshold: number of saved edge cases before retraining

min_auroc: minimum evaluation score to accept updated model

Installation
1. Clone the repository
bash
git clone https://github.com/yourname/robot-vision-feedback-loop.git
cd robot-vision-feedback-loop
2. Create and activate a virtual environment
bash
python -m venv venv
source venv/bin/activate
For Windows:

bash
venv\Scripts\activate
3. Install PyTorch with GPU support
Example for CUDA 12.1:

bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
4. Install project requirements
bash
pip install -r requirements.txt
Requirements
Example requirements.txt:

text
torch>=2.2.0
torchvision>=0.17.0
anomalib>=1.1.0
opencv-python>=4.9.0
Pillow>=10.2.0
numpy>=1.26.0
scikit-learn>=1.4.0
pyyaml>=6.0.1
onnx>=1.16.0
onnxruntime-gpu>=1.17.0
tqdm>=4.66.0
Data Layout
Place dataset folders inside data/raw/.

Example:

text
data/raw/screw/
├── train/
│   └── good/
└── test/
    ├── good/
    ├── scratch_head/
    ├── scratch_neck/
    └── thread_top/
Training Strategy
Use only train/good/ to build the memory bank

Use test/good/ and defect folders for evaluation

Save suspicious samples to data/raw/mistakes/

Training
Training in this project means building the PatchCore memory bank from good images.

Example:

bash
./scripts/retrain.sh screw
This script should:

Check device availability

Load configuration

Build the memory bank from normal images

Compute threshold from good validation data

Save the checkpoint

Evaluate the updated model

Inference
Inference computes an anomaly score for a given image or folder.

Single image
bash
./scripts/run_inference.sh data/raw/screw/test/good/000.png screw
Directory
bash
./scripts/run_inference.sh data/raw/screw/test screw
Expected output:

anomaly score

threshold

predicted status: NORMAL or DEFECT

Evaluation
The evaluation stage should measure how well the system separates good and defective images.

Typical metrics:

AUROC

F1 Score

Example:

bash
python src/evaluation/evaluate.py
The evaluation script should:

Load the saved PatchCore checkpoint

Score good and defective test images

Compute classification metrics

Report whether the model is acceptable for use

Feedback Loop
The feedback loop is what makes this project more than just a simple anomaly detector.

Loop Logic
Run inference on incoming factory images

Flag high-score anomalies

Save suspicious images and metadata

Collect enough edge cases

Retrain the memory bank

Re-evaluate the updated model

Accept or reject the updated checkpoint

This creates a repeatable industrial workflow where the system improves over time without requiring full manual re-labeling.

Edge Case Collection
Flagged samples should be stored in:

text
data/raw/mistakes/<category>/
Each image can have a matching metadata file such as:

json
{
  "original_path": "data/raw/screw/test/scratch_head/000.png",
  "anomaly_score": 0.8421,
  "threshold": 0.6230,
  "delta": 0.2191,
  "timestamp": "20260326_102345",
  "category": "screw",
  "reviewed": false
}
This helps track:

why the sample was saved

how abnormal it looked

when it was collected

whether it has been reviewed

GPU Usage
The project supports GPU acceleration through PyTorch.

GPU-related features
Automatic device selection

Mixed precision inference with FP16

Faster feature extraction

ONNX export support

TensorRT export option for deployment

The GPU utility file should handle:

device detection

GPU name display

VRAM reporting

memory usage stats

Export and Deployment
The project includes export support for deployment optimization.

ONNX export
Used to convert the backbone or model components into a portable format.

TensorRT export
Used to accelerate deployment on NVIDIA GPUs.

Example:

bash
./scripts/export_tensorrt.sh screw
Benchmarking
To measure performance:

bash
./scripts/benchmark_gpu.sh screw 100
This should report:

mean latency

median latency

p95 latency

min/max latency

approximate throughput

This is useful when you want to showcase GPU acceleration in your project.

Full Pipeline
To run the full process end-to-end:

bash
./scripts/full_pipeline.sh screw
This should:

Train if no checkpoint exists

Run inference

Collect anomalies

Check whether retraining is needed

Re-evaluate the updated model

Shell Scripts
run_inference.sh
Runs anomaly detection on a single image or folder.

retrain.sh
Builds a new memory bank and evaluates the result.

export_tensorrt.sh
Exports the backbone for optimized deployment.

benchmark_gpu.sh
Measures inference speed and memory usage.

full_pipeline.sh
Runs the full feedback loop.

Main Python Modules
src/anomaly/patchcore.py
Core PatchCore implementation:

memory bank creation

anomaly score computation

checkpoint save/load

src/anomaly/feature_extractor.py
Backbone feature extraction logic:

load pretrained ResNet50

extract patch embeddings

src/anomaly/threshold.py
Threshold estimation:

score good validation images

compute percentile-based threshold

src/training/train.py
Training entry point:

load config

fit PatchCore

compute threshold

save checkpoint

src/inference/detect.py
Inference entry point:

load checkpoint

score image

classify as normal or defect

src/data_collection/collect.py
Save edge cases and metadata.

src/evaluation/evaluate.py
Compute AUROC and F1 score.

src/pipeline/feedback_loop.py
End-to-end automation of the system.

Recommended First Demo
For a clean demo:

Use the screw category

Train PatchCore on train/good/

Run inference on:

one good sample

one defective sample

Show anomaly score difference

Run benchmark script

Explain how the feedback loop saves suspicious cases

This gives a very industry-relevant project demonstration without needing a huge setup.

Future Improvements
Possible next steps:

Add anomaly heatmap visualization

Add web dashboard for monitoring

Support multi-category inference in one run

Containerize with Docker

Add camera-based real-time live inspection

Integrate with ROS2 for robotic inspection workflows

Add review interface for human validation

Add experiment tracking with MLflow or Weights & Biases

Use Cases
This project can be adapted for:

Screw inspection

PCB inspection

Surface scratch detection

Metal part quality control

Packaging inspection

Conveyor-belt visual quality assurance

Robot-based inspection cells

Resume / Portfolio Value
This project is strong for robotics, automation, and computer vision roles because it demonstrates:

industrial vision understanding

anomaly detection pipeline design

GPU acceleration

model deployment awareness

automated retraining logic

real-world production thinking

It is more realistic than a simple object detector because it addresses a problem that actually appears in factories: unknown and rare defects.

License
MIT License

Acknowledgment
This project is inspired by industrial anomaly detection workflows and benchmark datasets such as MVTec AD, with PatchCore-style memory bank anomaly scoring for practical factory inspection systems.



# First time — train + run everything
./scripts/full_pipeline.sh screw

# Daily use — just inference on new images
./scripts/run_inference.sh data/raw/screw/test screw

# Check how fast your GPU is
./scripts/benchmark_gpu.sh screw 100

# After collecting enough edge cases — retrain manually
./scripts/retrain.sh screw

# Before deployment — export optimized model
./scripts/export_tensorrt.sh screw
