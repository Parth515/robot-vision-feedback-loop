import yaml
from pathlib import Path

def load_config(path="config/config.yaml"):
    with open(Path(path), "r") as f:
        return yaml.safe_load(f)
