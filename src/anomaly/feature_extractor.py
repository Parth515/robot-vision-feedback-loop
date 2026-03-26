import torch
from torchvision import models, transforms
from src.gpu.gpu_utils import get_device

class FeatureExtractor:
    def __init__(self, backbone="resnet50", device=None):
        self.device = device or get_device()
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        self.model = torch.nn.Sequential(*list(resnet.children())[:-3])
        self.model.eval().to(self.device)

    def extract(self, img_tensor):
        with torch.no_grad(), torch.amp.autocast('cuda'):
            feats = self.model(img_tensor.to(self.device))
        B, C, H, W = feats.shape
        return feats.permute(0, 2, 3, 1).reshape(-1, C)
