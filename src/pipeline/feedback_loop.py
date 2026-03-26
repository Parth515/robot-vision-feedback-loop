from src.inference.detect import run_inference
from src.data_collection.collect import save_edge_case
from src.training.train import train
from src.evaluation.evaluate import evaluate
from src.utils.config_loader import load_config
from pathlib import Path

def run_feedback_loop(image_dir, config_path="config/config.yaml"):
    cfg = load_config(config_path)
    category = cfg["category"]
    weights = f"models/checkpoints/{category}_patchcore.pt"
    mistakes_dir = Path(f"data/raw/mistakes/{category}")

    print(f"\n{'='*50}")
    print(f"Starting feedback loop for: {category}")
    print(f"{'='*50}\n")

    # Step 1: Run inference on all images
    for img_path in Path(image_dir).glob("*.png"):
        score, status = run_inference(str(img_path), weights)

        # Step 2: Collect edge cases
        if "DEFECT" in status:
            save_edge_case(str(img_path), score,
                           threshold=None, category=category)

    # Step 3: Retrain if enough new samples collected
    edge_cases = list(mistakes_dir.glob("*.png")) if mistakes_dir.exists() else []
    print(f"\n[LOOP] Edge cases collected: {len(edge_cases)}")

    if len(edge_cases) >= cfg.get("retrain_threshold", 20):
        print("[LOOP] Threshold reached → retraining memory bank...")
        train(config_path)

        # Step 4: Evaluate new model
        auroc, f1 = evaluate(category, weights)

        # Step 5: Only keep new model if it's better
        if auroc > cfg.get("min_auroc", 0.85):
            print(f"[LOOP] New model accepted (AUROC: {auroc:.4f})")
        else:
            print(f"[LOOP] New model rejected (AUROC: {auroc:.4f} < threshold)")
    else:
        print(f"[LOOP] Not enough edge cases yet ({len(edge_cases)}/20)")
