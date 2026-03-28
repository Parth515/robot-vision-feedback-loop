import torch
import torch.nn.functional as F
from torchvision import models, transforms
from pathlib import Path
from PIL import Image
from src.gpu.gpu_utils import get_device

class PatchCore:
    def __init__(self, backbone="resnet50", device=None):
        self.device = device or get_device()
        self.memory_bank = None
        self.threshold = None

        # pretrained backbone, remove final layers
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        self.feature_extractor = torch.nn.Sequential(*list(resnet.children())[:-3])
        self.feature_extractor.eval().to(self.device)

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

    def extract_features(self, img_tensor):
        with torch.no_grad(), torch.amp.autocast('cuda'):
            feats = self.feature_extractor(img_tensor.to(self.device))
        # flatten spatial patches → (N_patches, C)
        B, C, H, W = feats.shape
        return feats.permute(0, 2, 3, 1).reshape(-1, C)

    def fit(self, good_image_dir):
        """Build memory bank from normal images only."""
        all_features = []
        paths = list(Path(good_image_dir).glob("*.png"))
        print(f"Building memory bank from {len(paths)} normal images...")

        
        for p in paths:
            img = self.transform(Image.open(p).convert("RGB")).unsqueeze(0)
            feats = self.extract_features(img)
            all_features.append(feats.cpu())

        self.memory_bank = torch.cat(all_features, dim=0)
        print(f"Memory bank size: {self.memory_bank.shape}")

    def score(self, img_tensor):
        """Return anomaly score — higher = more anomalous."""
        query_feats = self.extract_features(img_tensor).cpu()
        # nearest neighbor distance to memory bank
        dists = torch.cdist(query_feats, self.memory_bank)
        min_dists, _ = dists.min(dim=1)
        return min_dists.max().item()  # image-level score

    def save(self, path="models/checkpoints/patchcore_memory.pt"):
        torch.save({"memory_bank": self.memory_bank,
                    "threshold": self.threshold}, path)

    def load(self, path="models/checkpoints/patchcore_memory.pt"):
        ckpt = torch.load(path, map_location=self.device)
        self.memory_bank = ckpt["memory_bank"]
        self.threshold = ckpt["threshold"]
