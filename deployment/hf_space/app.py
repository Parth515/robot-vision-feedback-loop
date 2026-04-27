import os
import sys

sys.path.insert(0, "/app") 


import gradio as gr
from PIL import Image
from huggingface_hub import hf_hub_download

from src.anomaly.patchcore import PatchCore

HF_TOKEN = os.getenv("HF_TOKEN")
HF_MODEL_REPO = os.getenv("HF_MODEL_REPO", "parth515/robot-vision-anomaly-model")
HF_CHECKPOINT_FILENAME = os.getenv("HF_CHECKPOINT_FILENAME", "screw_patchcore.pt")

model = PatchCore(device="cpu")
checkpoint_path = hf_hub_download(
    repo_id=HF_MODEL_REPO,
    filename=HF_CHECKPOINT_FILENAME,
    repo_type="model",
    token=HF_TOKEN,
)
model.load(checkpoint_path)

def predict(image):
    if image is None:
        return None, "No image provided."

    if not isinstance(image, Image.Image):
        image = Image.fromarray(image)

    img_tensor = model.transform(image.convert("RGB")).unsqueeze(0)
    score = model.score(img_tensor)
    threshold = model.threshold if model.threshold is not None else 0.5
    status = "DEFECT" if score > threshold else "NORMAL"

    result = (
        f"### Result\n"
        f"- **Status:** {status}\n"
        f"- **Anomaly score:** {score:.4f}\n"
        f"- **Threshold:** {threshold:.4f}\n"
        f"- **Model repo:** `{HF_MODEL_REPO}`"
    )
    return image, result

demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="Upload part image"),
    outputs=[
        gr.Image(type="pil", label="Input image"),
        gr.Markdown(label="Prediction"),
    ],
    title="Robot Vision Anomaly Detection",
    description="PatchCore-based industrial anomaly detection demo deployed with Docker on Hugging Face Spaces.",
    flagging_mode="never", 
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)