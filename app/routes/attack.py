from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Request
import io, time, logging
import torch
from PIL import Image
from torchvision import transforms

from app.services.imagenet import ImageNetService
from app.services.attacks import FGSM, FGSMConfig
from app.utils.image_io import load_image_to_01, tensor01_to_png_base64
from app.utils.labels import imagenet_labels, format_topk
from app.schemas.attack import AttackResponse, Prediction

log = logging.getLogger("attack")
router = APIRouter()

_svc = ImageNetService(device="cpu")
_attacker = FGSM(_svc.model, _svc.preprocess01_to_norm)
_LABELS = imagenet_labels()

@router.post("/attack", response_model=AttackResponse)
async def attack_endpoint(
    request: Request,
    file: UploadFile = File(...),
    epsilon: str = Form("0.1"),
):
    t = time.perf_counter  # local alias
    t0 = t()

    if file.content_type not in ("image/png", "image/jpeg"):
        raise HTTPException(415, "Only PNG or JPEG supported")

    eps_str = request.query_params.get("epsilon", epsilon)
    try:
        eps_val = float(eps_str)
    except Exception:
        eps_val = 0.1
    if not (0.0 <= eps_val <= 1.0):
        raise HTTPException(400, "epsilon must be between 0.0 and 1.0")

    img_bytes = await file.read()
    t1 = t(); log.info("read_upload_ms=%.1f", (t1 - t0) * 1e3)

    # decode + basic preprocess to [0,1]
    x01 = load_image_to_01(img_bytes, _svc.input_size)
    t2 = t(); log.info("decode_pre_ms=%.1f", (t2 - t1) * 1e3)

    # clean prediction (no grads)
    with torch.no_grad():
        clean_idx, clean_probs = _svc.predict_topk(x01, k=5)
    t3 = t(); log.info("clean_pred_ms=%.1f", (t3 - t2) * 1e3)

    clean_topk = [Prediction(label=l, prob=p)
                  for l, p in format_topk(clean_idx[0], clean_probs[0], _LABELS)]

    # run FGSM (grads only where needed inside FGSM)
    y_true = torch.tensor([clean_idx[0][0]], dtype=torch.long)
    x_adv = _attacker(x01, y_true, FGSMConfig(epsilon=eps_val))
    t4 = t(); log.info("attack_ms=%.1f", (t4 - t3) * 1e3)

    # adversarial prediction (no grads)
    with torch.no_grad():
        adv_idx, adv_probs = _svc.predict_topk(x_adv, k=5)
    t5 = t(); log.info("adv_pred_ms=%.1f", (t5 - t4) * 1e3)

    adversarial_topk = [Prediction(label=l, prob=p)
                        for l, p in format_topk(adv_idx[0], adv_probs[0], _LABELS)]
    success = adversarial_topk[0].label != clean_topk[0].label

    # encode result image
    adv_png_b64 = tensor01_to_png_base64(x_adv)
    t6 = t(); log.info("encode_png_ms=%.1f total_ms=%.1f", (t6 - t5) * 1e3, (t6 - t0) * 1e3)

    return AttackResponse(
        clean_topk=clean_topk,
        adversarial_topk=adversarial_topk,
        epsilon=eps_val,
        attack_success=success,
        # adversarial_image_base64_png=adv_png_b64,
    )
