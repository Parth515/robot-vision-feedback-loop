import json, shutil
from pathlib import Path
from datetime import datetime

def save_edge_case(image_path, score, threshold, category):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = Path(f"data/raw/mistakes/{category}")
    save_dir.mkdir(parents=True, exist_ok=True)

    # copy image
    dest = save_dir / f"{timestamp}_{Path(image_path).name}"
    shutil.copy(image_path, dest)

    # save metadata as json alongside image
    meta = {
        "original_path": str(image_path),
        "anomaly_score": round(score, 4),
        "threshold": round(threshold, 4),
        "delta": round(score - threshold, 4),
        "timestamp": timestamp,
        "category": category,
        "reviewed": False       # flag for human review later
    }
    with open(dest.with_suffix(".json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"[COLLECT] Saved edge case → {dest}")
