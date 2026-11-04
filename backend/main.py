from pathlib import Path
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, UnidentifiedImageError
import io

from backend.model import load_trained_model, predict_image, MODEL_PATH

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

@app.get("/api/health")
def health():
    return {
        "status": "running",
        "model_trained": MODEL_PATH.exists(),
        "message": "Model is ready." if MODEL_PATH.exists()
                   else "API is running, but no trained model checkpoint is installed yet."
    }

@app.post("/api/predict")
async def predict(file: UploadFile = File(...)):
    global _cached_model, _cached_device
    if file.content_type not in ("image/jpeg", "image/png", "image/jpg", "image/webp"):
        raise HTTPException(status_code=415, detail="Upload a JPG, PNG, or WEBP image.")
    if not MODEL_PATH.exists():
        raise HTTPException(
            status_code=503,
            detail="The AI model has not been trained yet. Add the NIH dataset and run the training command in README.md."
        )
    raw = await file.read()
    if len(raw) > 12 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="Image must be smaller than 12 MB.")
    try:
        image = Image.open(io.BytesIO(raw))
        image.verify()
        image = Image.open(io.BytesIO(raw))
    except (UnidentifiedImageError, OSError):
        raise HTTPException(status_code=400, detail="This file is not a valid image.")
    if _cached_model is None:
        _cached_model, _cached_device = load_trained_model()
    results = predict_image(image, _cached_model, _cached_device)
    return {
        "filename": file.filename,
        "results": results,
        "notice": "Research screening output only. Not a medical diagnosis."
    }

FRONTEND = Path(__file__).resolve().parents[1] / "frontend"
app.mount("/", StaticFiles(directory=FRONTEND, html=True), name="frontend")
