# app/utils/labels.py
from typing import List, Tuple
from torchvision.models import AlexNet_Weights

def imagenet_labels() -> List[str]:
    """
    Return list of 1000 ImageNet class labels for AlexNet.
    """
    return AlexNet_Weights.DEFAULT.meta["categories"]

def format_topk(
    indices: list[int],
    probs: list[float],
    labels: List[str],
    k: int = 5
) -> list[Tuple[str, float]]:
    """
    Map top-k indices & probs → (label, prob).
    """
    out: list[Tuple[str, float]] = []
    for i in range(min(k, len(indices))):
        out.append((labels[indices[i]], float(probs[i])))
    return out
