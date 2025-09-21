# FGSM Server

A minimal FastAPI service that demonstrates the Fast Gradient Sign Method (FGSM) adversarial attack using an ImageNet‑pretrained AlexNet model.

## Features

- `GET /health` – simple health check
- `POST /attack` – upload an image (PNG/JPEG) and optional `epsilon` to generate an adversarial example
- Returns:
  - `clean_topk` and `adversarial_topk` (top‑5 predictions)
  - `epsilon`, `attack_success`
  - `adversarial_image_base64_png` (base64‑encoded PNG)

## Requirements

- Python 3.12 (recommended; other recent 3.x versions may work)
- See `requirements.txt` for Python packages

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Run (development or simple production)

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
# For a bit more throughput (optional):
# uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 2
```

- CORS is open to all origins by default in `app/main.py`.
- The app includes a generic exception handler that returns `{ "detail": "Internal Server Error" }` for unhandled errors.

## API

### Health

- Request:
  - `GET /health`
- Response example:

```json
{
  "status": "ok",
  "model": "alexnet(ImageNet-1K)"
}
```

### Attack

- Request:

  - `POST /attack`
  - Content‑Type: `multipart/form-data`
  - Fields:
    - `file`: required image (`image/png` or `image/jpeg`)
    - `epsilon`: optional string/float in `[0.0, 1.0]` (default `0.1`)
  - You can also pass `epsilon` as a query param, e.g. `POST /attack?epsilon=0.05`

- Response (shape):

```json
{
  "clean_topk": [{ "label": "...", "prob": 0.123 }],
  "adversarial_topk": [{ "label": "...", "prob": 0.456 }],
  "epsilon": 0.1,
  "attack_success": true,
  "adversarial_image_base64_png": "iVBORw0KGgo..."
}
```

## API

- Interactive API docs (Swagger UI):
  - http://localhost:8000/docs

## Examples

Use the Swagger UI at `/docs` to try the endpoints in your browser. You can upload an image file and set `epsilon` directly in the form.
