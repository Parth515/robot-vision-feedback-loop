import os
import shutil
import tempfile
from pathlib import Path
from huggingface_hub import HfApi

HF_TOKEN = os.environ["HF_TOKEN"]
HF_SPACE_REPO = os.environ["HF_SPACE_REPO"]

api = HfApi(token=HF_TOKEN)
api.create_repo(repo_id=HF_SPACE_REPO, repo_type="space", space_sdk="docker", exist_ok=True)

with tempfile.TemporaryDirectory() as tmpdir:
    tmp = Path(tmpdir)

    # Copy Docker Space app files
    shutil.copytree("deployment/hf_space", tmp, dirs_exist_ok=True)

    # Copy project modules required by the app
    shutil.copytree("src", tmp / "src", dirs_exist_ok=True)
    shutil.copytree("config", tmp / "config", dirs_exist_ok=True)

    api.upload_folder(
        folder_path=str(tmp),
        repo_id=HF_SPACE_REPO,
        repo_type="space",
    )

print(f"Uploaded Space files to {HF_SPACE_REPO}")