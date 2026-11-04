import io
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from PIL import Image, UnidentifiedImageError

from backend.model import (
    DEFAULT_THRESHOLD,
    get_active_model_path,
    is_model_available,
    load_trained_model,
    predict_image,
)

logger = logging.getLogger("mediscan.api")

_cached_model = None
_cached_device = None

MAX_UPLOAD_SIZE = 12 * 1024 * 1024
UPLOAD_CHUNK_SIZE = 1024 * 1024

# ---------------------------------------------------------------------------
# CORS — read from environment variable (comma-separated list of origins).
# Set ALLOWED_ORIGINS in Render dashboard to your Vercel production domain.
# Falls back to localhost for local development only.
# ---------------------------------------------------------------------------
_raw_origins = os.environ.get(
    "ALLOWED_ORIGINS",
    "http://localhost:5173,http://localhost:3000,http://127.0.0.1:5173"
)
ALLOWED_ORIGINS: list = [o.strip() for o in _raw_origins.split(",") if o.strip()]


def reset_cached_model():
    """Reset cached model instance for test isolation."""
    global _cached_model, _cached_device
    _cached_model = None
    _cached_device = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model weights at startup if a checkpoint exists."""
    global _cached_model, _cached_device

    if is_model_available():
        try:
            _cached_model, _cached_device = load_trained_model()
            logger.info("DenseNet121 checkpoint loaded on %s", _cached_device)
        except Exception as exc:
            logger.warning("Could not load checkpoint at startup: %s", exc)
    else:
        logger.info("API started without an active model checkpoint.")

    yield


app = FastAPI(
    title="MediScan Guardian API",
    version="2.0.0",
    description="AI-assisted chest X-ray screening research prototype.",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Accept", "Authorization"],
)


@app.get("/api/health")
def health():
    trained = is_model_available()
    active_path = get_active_model_path()

    return {
        "status": "running",
        "model_trained": trained,
        "model": "DenseNet121",
        "target": "Active Pulmonary Tuberculosis",
        "device": str(_cached_device) if _cached_device else "uninitialized",
        "checkpoint_path": str(active_path),
        "message": (
            "DenseNet121 active-TB model is loaded and ready for screening."
            if trained
            else "API is running, but no trained model checkpoint is installed yet."
        ),
        "disclaimer": (
            "MediScan Guardian is an educational research prototype, "
            "not a medical device or diagnostic tool. Does not diagnose disease."
        ),
    }


@app.post("/api/predict")
async def predict(
    file: UploadFile = File(...),
    threshold: Optional[float] = Query(
        default=DEFAULT_THRESHOLD,
        ge=0.0,
        le=1.0,
        description="Decision threshold for active TB signal",
    ),
):
    global _cached_model, _cached_device

    # 1. Validate MIME type
    allowed_types = {
        "image/jpeg",
        "image/png",
        "image/jpg",
        "image/webp",
    }

    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=415,
            detail="Upload a JPG, PNG, or WEBP image.",
        )

    # 2. Check model availability — return 503 (never a fake prediction)
    if not is_model_available():
        raise HTTPException(
            status_code=503,
            detail=(
                "The AI model checkpoint is not installed. "
                "Train the model using backend/train.py or install a valid checkpoint."
            ),
        )

    # 3. Read file in chunks and enforce the upload limit
    file_size = getattr(file, "size", None)

    if file_size is not None and file_size > MAX_UPLOAD_SIZE:
        raise HTTPException(
            status_code=413,
            detail="Image must be smaller than 12 MB.",
        )

    raw = bytearray()

    while True:
        chunk = await file.read(UPLOAD_CHUNK_SIZE)

        if not chunk:
            break

        raw.extend(chunk)

        if len(raw) > MAX_UPLOAD_SIZE:
            raise HTTPException(
                status_code=413,
                detail="Image must be smaller than 12 MB.",
            )

    if not raw:
        raise HTTPException(
            status_code=400,
            detail="Uploaded file is empty.",
        )

    # 4. Validate image integrity
    try:
        image = Image.open(io.BytesIO(raw))
        image.verify()
        image = Image.open(io.BytesIO(raw))

        if image.width <= 0 or image.height <= 0:
            raise HTTPException(
                status_code=400,
                detail="This file is not a valid image.",
            )

    except (UnidentifiedImageError, OSError):
        raise HTTPException(
            status_code=400,
            detail="This file is not a valid image.",
        )

    # 5. Load model if not already cached
    if _cached_model is None:
        try:
            _cached_model, _cached_device = load_trained_model()
        except Exception:
            logger.exception("Model initialization failed")
            raise HTTPException(
                status_code=503,
                detail="The trained AI model is currently unavailable.",
            )

    # 6. Run inference
    try:
        prediction = predict_image(
            image,
            _cached_model,
            _cached_device,
            threshold=threshold,
        )
    except Exception:
        logger.exception("Inference execution failed")
        raise HTTPException(
            status_code=500,
            detail="Inference execution failed due to an internal server error.",
        )

    # 7. Preserve Phase 5 frontend response compatibility
    signals = (
        prediction.get("results", [])
        if isinstance(prediction, dict)
        else prediction
    )

    return {
        "filename": file.filename,
        "results": signals,
        "prediction_details": prediction if isinstance(prediction, dict) else {},
        "notice": "Research screening output only. Not a medical diagnosis.",
    }


# NOTE: Static file serving is disabled in production (Vercel serves the frontend).
# Uncomment the two lines below only for local all-in-one development:
# FRONTEND = Path(__file__).resolve().parents[1] / "frontend"
# app.mount("/", StaticFiles(directory=FRONTEND, html=True), name="frontend")
