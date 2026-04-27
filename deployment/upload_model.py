import os
from pathlib import Path
from huggingface_hub import HfApi

HF_TOKEN = os.environ["HF_TOKEN"]
HF_MODEL_REPO = os.environ["HF_MODEL_REPO"]
CHECKPOINT = os.environ.get("HF_CHECKPOINT_FILENAME", "screw_patchcore.pt")

api = HfApi(token=HF_TOKEN)
api.create_repo(repo_id=HF_MODEL_REPO, repo_type="model", exist_ok=True)

checkpoint_path = Path("models/checkpoints") / CHECKPOINT
config_path = Path("config/config.yaml")

if not checkpoint_path.exists():
    raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

model_card = f"""---
license: mit
tags:
- anomaly-detection
- patchcore
- industrial-inspection
- mvtec-ad
- computer-vision
---

# Robot Vision Anomaly Detection Model

This model is a PatchCore-based anomaly detection system for industrial parts.

## Files
- `{CHECKPOINT}`: trained checkpoint
- `config.yaml`: runtime configuration

## Use case
Industrial visual inspection for factory parts such as screws and metal components.
"""

tmp_card = Path("deployment/MODEL_CARD.md")
tmp_card.write_text(model_card, encoding="utf-8")

api.upload_file(
    path_or_fileobj=str(checkpoint_path),
    path_in_repo=CHECKPOINT,
    repo_id=HF_MODEL_REPO,
    repo_type="model",
)

api.upload_file(
    path_or_fileobj=str(config_path),
    path_in_repo="config.yaml",
    repo_id=HF_MODEL_REPO,
    repo_type="model",
)

api.upload_file(
    path_or_fileobj=str(tmp_card),
    path_in_repo="README.md",
    repo_id=HF_MODEL_REPO,
    repo_type="model",
)

print(f"Uploaded model artifacts to {HF_MODEL_REPO}")