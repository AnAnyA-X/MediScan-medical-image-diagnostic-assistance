from pathlib import Path
from typing import Dict, Union
import torch
from torch import nn
from torchvision import models
from PIL import Image

from backend.preprocessing import preprocess_cxr, DEFAULT_TARGET_SIZE

# Binary Active TB Classification
CLASS_NAMES = ["Negative_Control", "Active_TB"]
TARGET_NAME = "Active_TB"
MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "densenet121_active_tb.pth"
# Backwards-compatibility fallback path
LEGACY_MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "densenet121_chestxray.pth"


def build_model(pretrained: bool = False):
    """
    Builds DenseNet121 with a single logit output for binary Active TB detection.
    """
    weights = models.DenseNet121_Weights.DEFAULT if pretrained else None
    model = models.densenet121(weights=weights)
    model.classifier = nn.Linear(model.classifier.in_features, 1)
    return model


def get_active_model_path() -> Path:
    """Return existing trained model path, prioritizing the active TB model."""
    if MODEL_PATH.exists():
        return MODEL_PATH
    if LEGACY_MODEL_PATH.exists():
        return LEGACY_MODEL_PATH
    return MODEL_PATH


def load_trained_model(device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    active_path = get_active_model_path()
    if not active_path.exists():
        raise FileNotFoundError(
            f"Trained model not found at {active_path}. Train it using: "
            "python backend/train.py --metadata-dir data/metadata --data-dir data"
        )
    model = build_model(pretrained=False)
    checkpoint = torch.load(active_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()
    return model, device


def predict_image(image_input: Union[Image.Image, Path, str, bytes], model, device) -> Dict[str, any]:
    """
    Inference function applying the exact Phase 7 clinical preprocessing pipeline.
    """
    tensor = preprocess_cxr(image_input, target_size=DEFAULT_TARGET_SIZE).unsqueeze(0).to(device)
    with torch.no_grad():
        logit = model(tensor).item()
        prob_active_tb = float(torch.sigmoid(torch.tensor(logit)).item())

    prob_control = 1.0 - prob_active_tb
    is_positive = prob_active_tb >= 0.5

    if prob_active_tb >= 0.70:
        risk_level = "High Risk"
    elif prob_active_tb >= 0.40:
        risk_level = "Moderate Risk"
    else:
        risk_level = "Low Risk"

    return {
        "target": TARGET_NAME,
        "probability_active_tb": round(prob_active_tb, 4),
        "probability_control": round(prob_control, 4),
        "predicted_class": "Active_TB" if is_positive else "Negative_Control",
        "risk_level": risk_level,
        "results": [
            {"label": "Active Pulmonary TB", "probability": round(prob_active_tb, 4)},
            {"label": "Normal / Negative Control", "probability": round(prob_control, 4)}
        ]
    }
