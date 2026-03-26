from src.anomaly.patchcore import PatchCore
from src.anomaly.threshold import compute_threshold
from src.utils.config_loader import load_config

def train(config_path="config/config.yaml"):
    """
    Train the PatchCore model using the configuration provided.

    Args:
        config_path (str): Path to the YAML config file.

    The config file must contain:
        category (str): The category name used for data and model checkpoint paths.
    """
    cfg = load_config(config_path)
    model = PatchCore()

    # fit on normal images only — no labels needed
    model.fit(f"data/raw/{cfg['category']}/train/good")

    # auto-compute threshold from val good images
    model.threshold = compute_threshold(model, f"data/raw/{cfg['category']}/test/good")
    print(f"Auto threshold set: {model.threshold:.4f}")

    model.save(f"models/checkpoints/{cfg['category']}_patchcore.pt")

if __name__ == "__main__":
    train()
