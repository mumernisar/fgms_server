from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Request
from fastapi.responses import StreamingResponse
import time, logging, base64
from io import BytesIO
import torch
from torchvision.transforms.functional import to_pil_image

# Optional fast JSON: use orjson if available; otherwise stdlib json (bytes)
try:
    import orjson as _json
    def _dumps_bytes(obj) -> bytes:
        return _json.dumps(obj)
except Exception:
    import json as _json
    def _dumps_bytes(obj) -> bytes:
        # ensure we return compact bytes with no spaces
        return _json.dumps(obj, separators=(",", ":")).encode("utf-8")

from app.services.imagenet import ImageNetService
from app.services.attacks import FGSM, FGSMConfig
from app.utils.image_io import load_image_to_01
from app.utils.labels import imagenet_labels, format_topk

log = logging.getLogger("attack")
router = APIRouter()

# Services (CPU for now; flip to "cuda" when ready)
_svc = ImageNetService(device="cpu")
_attacker = FGSM(_svc.model, _svc.preprocess01_to_norm)
_LABELS = imagenet_labels()

# ---------- helpers ----------

def tensor01_to_jpeg_bytes(x01: torch.Tensor, quality: int = 85):
    """
    x01: torch.Tensor in [0,1], shape (1,3,H,W) or (3,H,W)
    returns: (jpeg_bytes, width, height)
    """
    x = x01.squeeze(0) if x01.dim() == 4 else x01
    img = to_pil_image(x.clamp(0, 1).cpu())  # RGB PIL Image
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=quality, optimize=True)
    data = buf.getvalue()
    return data, img.width, img.height

def stream_b64_json(meta: dict, jpeg_bytes: bytes, chunk_size: int = 3 * 8192):
    """
    Stream a compact JSON object that includes all of `meta` plus:
      "img":"<base64 of jpeg_bytes>"
    Base64 chunks are 3-byte aligned so concatenation remains valid.
    """
    head = _dumps_bytes(meta)  # e.g. b'{"epsilon":0.1,"..."}'
    if not head.endswith(b"}"):
        # Extremely defensive; shouldn't happen given _dumps_bytes
        raise RuntimeError("meta serialization did not end with '}'")
    # Emit everything except the closing brace, then start the "img" field
    yield head[:-1] + b',"img":"'

    mv = memoryview(jpeg_bytes)
    step = max(3, (chunk_size // 3) * 3)  # enforce multiple of 3
    for i in range(0, len(mv), step):
        block = mv[i : i + step]
        yield base64.b64encode(block)  # contiguous base64 is valid

    yield b'"}'  # close JSON

# ---------- route ----------

@router.post("/attack")
async def attack_endpoint(
    request: Request,
    file: UploadFile = File(...),
    epsilon: str = Form("0.1"),
):
    t = time.perf_counter
    t0 = t()

    if file.content_type not in ("image/png", "image/jpeg"):
        raise HTTPException(415, "Only PNG or JPEG supported")

    # epsilon from form or query
    eps_str = request.query_params.get("epsilon", epsilon)
    try:
        eps_val = float(eps_str)
    except Exception:
        eps_val = 0.1
    if not (0.0 <= eps_val <= 1.0):
        raise HTTPException(400, "epsilon must be between 0.0 and 1.0")

    # read upload
    img_bytes = await file.read()
    t1 = t(); log.info("attack: read_upload_ms=%.1f", (t1 - t0) * 1e3)

    # decode + preprocess to [0,1]
    x01 = load_image_to_01(img_bytes, _svc.input_size)
    t2 = t(); log.info("attack: decode_pre_ms=%.1f", (t2 - t1) * 1e3)

    # clean prediction
    with torch.no_grad():
        clean_idx, clean_probs = _svc.predict_topk(x01, k=5)
    t3 = t(); log.info("attack: clean_pred_ms=%.1f", (t3 - t2) * 1e3)

    clean_pairs = format_topk(clean_idx[0], clean_probs[0], _LABELS)
    clean_topk = [{"label": l, "prob": p} for l, p in clean_pairs]

    # FGSM attack
    y_true = torch.tensor([clean_idx[0][0]], dtype=torch.long)
    x_adv = _attacker(x01, y_true, FGSMConfig(epsilon=eps_val))
    t4 = t(); log.info("attack: attack_ms=%.1f", (t4 - t3) * 1e3)

    # adversarial prediction
    with torch.no_grad():
        adv_idx, adv_probs = _svc.predict_topk(x_adv, k=5)
    t5 = t(); log.info("attack: adv_pred_ms=%.1f", (t5 - t4) * 1e3)

    adv_pairs = format_topk(adv_idx[0], adv_probs[0], _LABELS)
    adversarial_topk = [{"label": l, "prob": p} for l, p in adv_pairs]
    success = adversarial_topk[0]["label"] != clean_topk[0]["label"]

    # JPEG bytes (smaller over network than PNG)
    jpeg_bytes, w, h = tensor01_to_jpeg_bytes(x_adv, quality=85)
    t6 = t(); total_ms = (t6 - t0) * 1e3
    log.info("attack: encode_jpeg_ms=%.1f total_ms=%.1f", (t6 - t5) * 1e3, total_ms)

    # minimal, compact meta (client builds data URI: data:image/{fmt};base64,...)
    meta = {
        "epsilon": eps_val,
        "attack_success": success,
        "clean_topk": clean_topk,
        "adversarial_topk": adversarial_topk,
        "fmt": "jpeg",
        "w": w,
        "h": h,
    }

    # stream compact JSON with base64 image
    return StreamingResponse(
        stream_b64_json(meta, jpeg_bytes),
        media_type="application/json",
        headers={"X-Server-Time-ms": f"{total_ms:.1f}", "Cache-Control": "no-store"},
    )
