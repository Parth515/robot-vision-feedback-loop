from src.anomaly.patchcore import PatchCore
from PIL import Image
from pathlib import Path

def run_inference(image_path, weights="models/checkpoints/screw_patchcore.pt"):
    model = PatchCore()
    model.load(weights)

    img = model.transform(Image.open(image_path).convert("RGB")).unsqueeze(0)
    score = model.score(img)

    status = "DEFECT" if score > model.threshold else "NORMAL"
    print(f"[{Path(image_path).name}] Score: {score:.4f} | Threshold: {model.threshold:.4f} | {status}")

    # save to data collection if edge case
    if score > model.threshold:
        Path("data/raw/mistakes").mkdir(parents=True, exist_ok=True)
        Image.open(image_path).save(f"data/raw/mistakes/{Path(image_path).name}")

    return score, status
