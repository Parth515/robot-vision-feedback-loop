import torch
from torchvision import models, transforms

class FeatureExtractor:
    def __init__(self, backbone="resnet50", device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        self.model = torch.nn.Sequential(*list(resnet.children())[:-3])
        self.model.eval().to(self.device)

    def extract(self, img_tensor):
        with torch.no_grad(), torch.cuda.amp.autocast():
            feats = self.model(img_tensor.to(self.device))
        B, C, H, W = feats.shape
        return feats.permute(0, 2, 3, 1).reshape(-1, C)