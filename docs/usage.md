# Running Cycle & Usage

This document shows the full day-to-day workflow: train, infer, evaluate, collect edge cases, and retrain.

## 1. Build the Memory Bank

Train PatchCore on normal images only:

```bash
python src/training/train.py
```

What this does:

- Loads normal training images from `data/raw/<category>/train/good`
- Extracts patch features with the backbone
- Builds the memory bank
- Computes a threshold from good validation/test images
- Saves the model checkpoint to `models/checkpoints/`

## 2. Run Inference on One Defective Image

```bash
python -c "from src.inference.detect import run_inference; run_inference('data/raw/screw/test/scratch_head/000.png')"
```

Expected result: the image should be flagged as `DEFECT`.

## 3. Run Inference on One Good Image

```bash
python -c "from src.inference.detect import run_inference; run_inference('data/raw/screw/test/good/000.png')"
```

Expected result: the image should be flagged as `NORMAL`.

## 4. Run Inference on a Folder

```bash
bash scripts/run_inference.sh data/raw/screw/test screw
```

This runs anomaly scoring on all images in the folder and stores flagged edge cases in:

```bash
data/raw/mistakes/screw/
```

## 5. Evaluate the Model

```bash
python -c "from src.evaluation.evaluate import evaluate; evaluate('screw', 'models/checkpoints/screw_patchcore.pt')"
```

This reports image-level metrics such as AUROC and F1.

## 6. Inspect Collected Edge Cases

```bash
ls data/raw/mistakes/screw/
```

Each flagged image can have a matching JSON metadata file with anomaly score and timestamp.

## 7. Retrain the Model

```bash
bash scripts/retrain.sh screw
```

This script:

- Backs up the previous checkpoint
- Rebuilds the memory bank
- Recomputes the threshold
- Runs evaluation after training

## 8. Run the Full Pipeline

```bash
bash scripts/full_pipeline.sh screw data/raw/screw/test
```

This is the main automation entry point. It will:

- Train if no checkpoint exists
- Run inference
- Save edge cases
- Check whether retraining should be triggered

## 9. Daily Workflow

For normal usage, this is enough:

```bash
# score new images
bash scripts/run_inference.sh <new_image_dir> screw

# retrain when enough new samples exist
bash scripts/retrain.sh screw

# or automate the whole process
bash scripts/full_pipeline.sh screw <new_image_dir>
```