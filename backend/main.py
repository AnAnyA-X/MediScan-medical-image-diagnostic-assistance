import io
import logging
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, UnidentifiedImageError

from backend.model import load_trained_model, predict_image, MODEL_PATH, get_active_model_path

logger = logging.getLogger("mediscan.api")

app = FastAPI(title="MediScan Guardian API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict to your deployed frontend domain before production.
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_cached_model = None
_cached_device = None

MAX_UPLOAD_SIZE = 12 * 1024 * 1024  # 12 MB
UPLOAD_CHUNK_SIZE = 1024 * 1024     # 1 MB chunk


def reset_cached_model():
    """Helper to reset in-memory cached model instance (used for test isolation)."""
    global _cached_model, _cached_device
    _cached_model = None
    _cached_device = None


@app.get("/api/health")
def health():
    active_path = get_active_model_path()
    model_exists = active_path.exists()
    return {
        "status": "running",
        "model_trained": model_exists,
        "message": "Model is ready." if model_exists
                   else "API is running, but no trained model checkpoint is installed yet."
    }


@app.post("/api/predict")
async def predict(file: UploadFile = File(...)):
    global _cached_model, _cached_device
    if file.content_type not in ("image/jpeg", "image/png", "image/jpg", "image/webp"):
        raise HTTPException(status_code=415, detail="Upload a JPG, PNG, or WEBP image.")

    active_path = get_active_model_path()
    if not active_path.exists():
        raise HTTPException(
            status_code=503,
            detail="The AI model has not been trained yet. Add the NIH dataset and run the training command in README.md."
        )

    # Early size check if file.size is known by the server
    file_size = getattr(file, "size", None)
    if file_size is not None and file_size > MAX_UPLOAD_SIZE:
        raise HTTPException(status_code=413, detail="Image must be smaller than 12 MB.")

    # Read in chunks (max 1 MB per read) without buffering excessive data in memory
    raw = bytearray()
    while True:
        chunk = await file.read(UPLOAD_CHUNK_SIZE)
        if not chunk:
            break
        raw.extend(chunk)
        if len(raw) > MAX_UPLOAD_SIZE:
            raise HTTPException(status_code=413, detail="Image must be smaller than 12 MB.")

    if len(raw) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    try:
        image = Image.open(io.BytesIO(raw))
        image.verify()
        image = Image.open(io.BytesIO(raw))
        if image.width <= 0 or image.height <= 0:
            raise HTTPException(status_code=400, detail="This file is not a valid image.")
    except (UnidentifiedImageError, OSError):
        raise HTTPException(status_code=400, detail="This file is not a valid image.")

    try:
        if _cached_model is None:
            _cached_model, _cached_device = load_trained_model()
        prediction = predict_image(image, _cached_model, _cached_device)
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=503,
            detail="The trained AI model is currently unavailable. Please try again later."
        )
    except Exception as exc:
        logger.exception("Unexpected inference execution error: %s", exc)
        raise HTTPException(
            status_code=500,
            detail="Inference execution failed due to an internal server error. Please try again."
        )

    # Ensure frontend compatibility: results list formatted for array slicing
    signals = prediction.get("results", []) if isinstance(prediction, dict) else prediction
    return {
        "filename": file.filename,
        "results": signals,
        "prediction_details": prediction if isinstance(prediction, dict) else {},
        "notice": "Research screening output only. Not a medical diagnosis."
    }


FRONTEND = Path(__file__).resolve().parents[1] / "frontend"
app.mount("/", StaticFiles(directory=FRONTEND, html=True), name="frontend")
