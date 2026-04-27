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

    # app files from hf_space/
    shutil.copy("deployment/hf_space/app.py", tmp / "app.py")
    shutil.copy("deployment/hf_space/requirements.txt", tmp / "requirements.txt")
    shutil.copy("deployment/hf_space/README.md", tmp / "README.md")
    shutil.copy("deployment/hf_space/Dockerfile", tmp / "Dockerfile")

    # src and config from project root
    shutil.copytree("src", tmp / "src", dirs_exist_ok=True)
    (tmp / "config").mkdir(exist_ok=True)
    shutil.copy("config/config.yaml", tmp / "config" / "config.yaml")

    api.upload_folder(
        folder_path=str(tmp),
        repo_id=HF_SPACE_REPO,
        repo_type="space",
    )

print(f"Uploaded Space files to {HF_SPACE_REPO}")