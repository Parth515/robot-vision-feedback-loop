import torch
from pathlib import Path
from PIL import Image

def compute_threshold(model, good_test_dir, percentile=95):
    """
    Score all known-good test images.
    Set threshold at 95th percentile → anything above = anomaly.
    """
    scores = []
    for p in Path(good_test_dir).glob("*.png"):
        img = model.transform(Image.open(p).convert("RGB")).unsqueeze(0)
        scores.append(model.score(img))

    threshold = torch.tensor(scores).quantile(percentile / 100).item()
    print(f"Scores on good images: min={min(scores):.3f}, max={max(scores):.3f}")
    return threshold
