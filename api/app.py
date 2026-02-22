"""
FastAPI inference service for Cats vs Dogs classification.

Endpoints:
  GET  /health   — liveness/readiness probe
  POST /predict  — image upload → label + probabilities
  GET  /metrics  — request count and latency stats
"""

import json
import logging
import os
import time
from contextlib import asynccontextmanager
from io import BytesIO
from pathlib import Path

import torch
import torch.nn.functional as F
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import Response

# ─── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("cats-dogs-api")

# ─── Prometheus metrics ────────────────────────────────────────────────────────
REQUEST_COUNT = Counter("prediction_requests_total", "Total prediction requests")
REQUEST_LATENCY = Histogram("prediction_latency_seconds", "Prediction latency in seconds")
ERROR_COUNT = Counter("prediction_errors_total", "Total prediction errors")

# ─── Config ───────────────────────────────────────────────────────────────────
MODEL_PATH = os.getenv("MODEL_PATH", "models/model.pt")
MODEL_META_PATH = os.getenv("MODEL_META_PATH", "models/model_meta.json")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Global model holder
_model = None
_class_names = ["Cat", "Dog"]
_model_type = "simple_cnn"


def load_app_model():
    global _model, _class_names, _model_type
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
    from catsml.model import load_model

    if Path(MODEL_META_PATH).exists():
        with open(MODEL_META_PATH) as f:
            meta = json.load(f)
        _model_type = meta.get("model_type", "simple_cnn")
        _class_names = meta.get("class_names", ["Cat", "Dog"])

    logger.info(f"Loading model from {MODEL_PATH} (type={_model_type}) on {DEVICE}")
    _model = load_model(MODEL_PATH, model_type=_model_type, device=DEVICE)
    logger.info("Model loaded successfully.")


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_app_model()
    yield


app = FastAPI(
    title="Cats vs Dogs Classifier",
    description="MLOps Assignment 2 — Binary image classification service",
    version="1.0.0",
    lifespan=lifespan,
)


def preprocess(image_bytes: bytes) -> torch.Tensor:
    """Preprocess raw image bytes into a model-ready tensor."""
    from torchvision import transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    img = Image.open(BytesIO(image_bytes)).convert("RGB")
    return transform(img).unsqueeze(0).to(DEVICE)


@app.get("/health", tags=["Ops"])
async def health():
    """Liveness and readiness probe."""
    return {"status": "ok", "model_loaded": _model is not None, "device": DEVICE}


@app.post("/predict", tags=["Inference"])
async def predict(file: UploadFile = File(...)):
    """
    Classify an uploaded image as Cat or Dog.
    Returns: label, confidence, and per-class probabilities.
    """
    REQUEST_COUNT.inc()
    start = time.time()

    try:
        contents = await file.read()
        tensor = preprocess(contents)

        with torch.no_grad():
            logits = _model(tensor)
            probs = F.softmax(logits, dim=1).squeeze().cpu().tolist()

        label_idx = int(torch.argmax(torch.tensor(probs)))
        label = _class_names[label_idx]
        confidence = probs[label_idx]

        latency = time.time() - start
        REQUEST_LATENCY.observe(latency)

        logger.info(
            f"Prediction | filename={file.filename} label={label} "
            f"confidence={confidence:.4f} latency={latency:.4f}s"
        )

        return {
            "label": label,
            "confidence": round(confidence, 6),
            "probabilities": {cls: round(p, 6) for cls, p in zip(_class_names, probs)},
            "latency_seconds": round(latency, 6),
        }

    except Exception as e:
        ERROR_COUNT.inc()
        logger.error(f"Prediction error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/metrics", tags=["Ops"])
async def metrics():
    """Expose Prometheus metrics."""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/", tags=["Ops"])
async def root():
    return {"message": "Cats vs Dogs Classifier API — visit /docs for Swagger UI"}
