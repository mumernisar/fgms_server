from fastapi.responses import StreamingResponse, Response
import base64, orjson
from io import BytesIO
from torchvision.transforms.functional import to_pil_image

def tensor01_to_jpeg_bytes(x01, quality: int = 85):
    """
    x01: torch.Tensor in [0,1], shape (1,3,H,W) or (3,H,W)
    returns: (jpeg_bytes, width, height)
    """
    if x01.dim() == 4:
        x = x01.squeeze(0)
    else:
        x = x01
    img = to_pil_image(x.clamp(0, 1).cpu())  # RGB PIL
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=quality, optimize=True)
    data = buf.getvalue()
    return data, img.width, img.height

def stream_b64_json(meta: dict, jpeg_bytes: bytes, chunk_size: int = 3 * 8192):
    """
    Stream a compact JSON object with all of `meta` plus "img": "<base64...>".
    Chunks are 3-byte aligned for base64 correctness.
    """
    head = orjson.dumps(meta)  # b'{"a":1,...}'
    # replace trailing '}' with ',"img":"'
    if not head.endswith(b"}"):  # safety
        raise RuntimeError("meta serialization did not end with '}'")
    yield head[:-1] + b',"img":"'

    mv = memoryview(jpeg_bytes)
    for i in range(0, len(mv), chunk_size):
        block = mv[i : i + chunk_size]
        # b64-encode each binary block; concatenation stays valid base64
        yield base64.b64encode(block)

    yield b'"}'  # close the JSON object


def tensor01_to_jpeg_bytes(x01, quality: int = 85):
    """
    x01: torch.Tensor in [0,1], shape (1,3,H,W) or (3,H,W)
    returns: (jpeg_bytes, width, height)
    """
    if x01.dim() == 4:
        x = x01.squeeze(0)
    else:
        x = x01
    img = to_pil_image(x.clamp(0, 1).cpu())  # RGB PIL
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=quality, optimize=True)
    data = buf.getvalue()
    return data, img.width, img.height

def stream_b64_json(meta: dict, jpeg_bytes: bytes, chunk_size: int = 3 * 8192):
    """
    Stream a compact JSON object with all of `meta` plus "img": "<base64...>".
    Chunks are 3-byte aligned for base64 correctness.
    """
    head = orjson.dumps(meta)  # b'{"a":1,...}'
    # replace trailing '}' with ',"img":"'
    if not head.endswith(b"}"):  # safety
        raise RuntimeError("meta serialization did not end with '}'")
    yield head[:-1] + b',"img":"'

    mv = memoryview(jpeg_bytes)
    for i in range(0, len(mv), chunk_size):
        block = mv[i : i + chunk_size]
        # b64-encode each binary block; concatenation stays valid base64
        yield base64.b64encode(block)

    yield b'"}'  # close the JSON object
