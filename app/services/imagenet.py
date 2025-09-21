# app/services/imagenet.py
import torch
from torch import Tensor
from torchvision import models, transforms
from typing import List, Tuple

class ImageNetService:
    def __init__(self, device: str = "cpu"):
        weights = models.AlexNet_Weights.DEFAULT
        self.model = models.alexnet(weights=weights)
        self.model.eval().to(device)

        meta = weights.transforms()
        self.input_size = meta.crop_size[0]   # 224
        self.mean = meta.mean
        self.std = meta.std
        self.device = torch.device(device)

        self.normalize = transforms.Normalize(self.mean, self.std)

    def preprocess01_to_norm(self, x01: Tensor) -> Tensor:
        return self.normalize(x01.to(self.device))

    @torch.inference_mode()
    def predict_topk(
        self, x01_batched: Tensor, k: int = 5
    ) -> Tuple[List[List[int]], List[List[float]]]:
        x_norm = self.preprocess01_to_norm(x01_batched)
        logits = self.model(x_norm)
        probs = logits.softmax(dim=1)
        top_probs, top_idx = probs.topk(k, dim=1)
        return top_idx.cpu().tolist(), top_probs.cpu().tolist()
