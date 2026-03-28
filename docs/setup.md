# Setup & Installation

This guide covers environment setup, dependency installation, GPU verification, and dataset download.

## 1. Clone the Project

```bash
git clone <your-repo-url>
cd robot-vision-feedback-loop
```

## 2. Create a Virtual Environment

```bash
python -m venv .venv
source .venv/bin/activate
```

For Windows:

```bash
.venv\Scripts\activate
```

## 3. Install PyTorch with CUDA

Install the CUDA-enabled PyTorch build first:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

Then install the remaining project dependencies:

```bash
pip install -r requirements.txt
```

## 4. Make Shell Scripts Executable

```bash
chmod +x scripts/*.sh
```

## 5. Verify GPU Access

```bash
python -c "from src.gpu.gpu_utils import get_device; get_device()"
```

Expected output should show your NVIDIA GPU name and available VRAM.

## 6. Download MVTec AD Data

Example for the `screw` category:

```bash
wget https://www.mvtec.com/fileadmin/Redaktion/mvtec.com/company/research/datasets/mvtec_anomaly_detection/screw.tar.xz -P data/raw/
tar -xf data/raw/screw.tar.xz -C data/raw/
```

You can repeat the same pattern for other categories such as `bottle` or `metal_nut`.

## 7. Configure the Project

Open:

```bash
config/config.yaml
```

Update fields such as:

- `category`
- `device`
- `precision`
- `threshold.percentile`

Example:

```yaml
device: cuda
precision: fp16
category: screw

model:
  name: patchcore
  backbone: resnet50

threshold:
  percentile: 95

retrain_threshold: 20
min_auroc: 0.85
```

## 8. Project Check

Run this to confirm imports work:

```bash
python -c "from src.training.train import train; print('setup ok')"
```

If that runs without errors, the environment is ready.