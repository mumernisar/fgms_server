# app/utils/image_io.py
import io, base64
from PIL import Image
import torch
from torchvision import transforms
from typing import cast

def load_image_to_01(image_bytes: bytes, resize_to: int) -> torch.Tensor:
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    t = transforms.Compose([
        transforms.Resize(resize_to, antialias=True),
        transforms.CenterCrop(resize_to),
        transforms.ToTensor(),  # [0,1]
    ])
    tensor = cast(torch.Tensor, t(img))
    return tensor.unsqueeze(0)

def tensor01_to_png_base64(x01_batched: torch.Tensor) -> str:
    x01 = x01_batched.squeeze(0).clamp(0,1)
    pil = transforms.ToPILImage()(x01.cpu())
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")
