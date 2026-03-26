from sklearn.metrics import roc_auc_score, f1_score
from src.anomaly.patchcore import PatchCore
from pathlib import Path
from PIL import Image
import numpy as np

def evaluate(category, weights_path):
    model = PatchCore()
    model.load(weights_path)

    scores, labels = [], []

    # good test images → label 0
    for p in Path(f"data/raw/{category}/test/good").glob("*.png"):
        img = model.transform(Image.open(p).convert("RGB")).unsqueeze(0)
        scores.append(model.score(img))
        labels.append(0)

    # defective test images → label 1 (all defect subdirs)
    test_dir = Path(f"data/raw/{category}/test")
    for defect_dir in test_dir.iterdir():
        if defect_dir.name == "good": continue
        for p in defect_dir.glob("*.png"):
            img = model.transform(Image.open(p).convert("RGB")).unsqueeze(0)
            scores.append(model.score(img))
            labels.append(1)

    auroc = roc_auc_score(labels, scores)
    preds = [1 if s > model.threshold else 0 for s in scores]
    f1 = f1_score(labels, preds)

    print(f"[{category}] AUROC: {auroc:.4f} | F1: {f1:.4f}")
    return auroc, f1

if __name__ == "__main__":
    evaluate("screw", "models/checkpoints/screw_patchcore.pt")
