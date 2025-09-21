# app/services/attacks.py
from dataclasses import dataclass
import torch
import torch.nn.functional as F
from torch import Tensor

@dataclass
class FGSMConfig:
    epsilon: float = 0.1

class FGSM:
    def __init__(self, model, preprocess):
        self.model = model
        self.preprocess = preprocess

    @torch.inference_mode(False)
    def __call__(self, x01: Tensor, y_true: Tensor, cfg: FGSMConfig) -> Tensor:
        assert x01.dim() == 4, "Expected 4D tensor [N,C,H,W]"

        x = x01.clone().detach().requires_grad_(True)
        x_norm = self.preprocess(x)
        logits = self.model(x_norm)
        loss = F.cross_entropy(logits, y_true)

        grad = torch.autograd.grad(loss, x, retain_graph=False, create_graph=False)[0]
        x_adv = x + cfg.epsilon * grad.sign()
        return torch.clamp(x_adv.detach(), 0.0, 1.0)
