"""
MediScan Guardian — Model Architecture & Inference Module
Architecture: DenseNet121 Binary Classifier (Active Pulmonary TB vs Control)
Preprocessing: Phase 7 Standard Letterbox (512x512) + ImageNet Normalization
Execution: torch.inference_mode() with CPU and CUDA support
"""

import os
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import torch
from torch import nn
from torchvision import models
from PIL import Image

from backend.preprocessing import DEFAULT_TARGET_SIZE, preprocess_cxr

# Class mappings & target specification
CLASS_NAMES = ["Negative_Control", "Active_TB"]
TARGET_NAME = "Active_TB"
DEFAULT_THRESHOLD = float(os.environ.get("MEDISCAN_THRESHOLD", "0.50"))

# Checkpoint paths (configurable via environment variable)
ENV_MODEL_PATH = os.environ.get("MEDISCAN_MODEL_PATH")
DEFAULT_MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "densenet121_active_tb.pth"
LEGACY_MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "densenet121_chestxray.pth"


def get_active_model_path() -> Path:
    """Return the configured or discovered model checkpoint path."""
    env_path = os.environ.get("MEDISCAN_MODEL_PATH")
    if env_path:
        return Path(env_path)
    if DEFAULT_MODEL_PATH.exists():
        return DEFAULT_MODEL_PATH
    if LEGACY_MODEL_PATH.exists():
        return LEGACY_MODEL_PATH
    return DEFAULT_MODEL_PATH


def is_model_available() -> bool:
    """Check whether a trained model checkpoint file exists."""
    p = get_active_model_path()
    return p.is_file() and p.stat().st_size > 0


def build_model(pretrained: bool = False) -> nn.Module:
    """
    Constructs DenseNet121 with a single logit output neuron for binary active TB screening.
    """
    weights = models.DenseNet121_Weights.DEFAULT if pretrained else None
    model = models.densenet121(weights=weights)
    model.classifier = nn.Linear(model.classifier.in_features, 1)
    return model


def load_trained_model(device: Optional[str] = None) -> Tuple[nn.Module, str]:
    """
    Loads model weights from the active checkpoint once during application startup,
    sets evaluation mode, and moves weights to target device.
    """
    chosen_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model_path = get_active_model_path()

    if not model_path.exists():
        raise FileNotFoundError(
            f"Trained model checkpoint not found at {model_path}. "
            "Run backend/train.py or set MEDISCAN_MODEL_PATH."
        )

    model = build_model(pretrained=False)
    checkpoint = torch.load(model_path, map_location=chosen_device)

    state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    model.to(chosen_device)
    model.eval()

    return model, chosen_device


def predict_image(
    image_input: Union[Image.Image, Path, str, bytes],
    model: nn.Module,
    device: str,
    threshold: float = DEFAULT_THRESHOLD
) -> Dict[str, any]:
    """
    Executes binary active-TB inference using torch.inference_mode().
    Returns standard Phase 4 response structure.
    """
    tensor = preprocess_cxr(image_input, target_size=DEFAULT_TARGET_SIZE).unsqueeze(0).to(device)

    with torch.inference_mode():
        logits = model(tensor)
        prob_active_tb = float(torch.sigmoid(logits).item())

    prob_control = 1.0 - prob_active_tb
    is_positive = prob_active_tb >= threshold

    # Determine risk category & prediction description
    if prob_active_tb >= 0.70:
        risk_level = "High Risk"
    elif prob_active_tb >= threshold:
        risk_level = "Moderate Risk"
    else:
        risk_level = "Low Risk"

    prediction_str = (
        "TB screening signal detected"
        if is_positive
        else "No TB screening signal detected"
    )

    return {
        "status": "success",
        "prediction": prediction_str,
        "probability": round(prob_active_tb, 4),
        "probability_control": round(prob_control, 4),
        "predicted_class": "Active_TB" if is_positive else "Negative_Control",
        "risk_level": risk_level,
        "threshold_used": threshold,
        "model": "DenseNet121",
        "notice": (
            "Research screening support only. Educational research prototype, "
            "not a medical device or diagnostic tool."
        ),
        # Backward-compatibility signal list for existing UI
        "results": [
            {"label": "Active Pulmonary TB", "probability": round(prob_active_tb, 4)},
            {"label": "Normal / Negative Control", "probability": round(prob_control, 4)},
        ],
    }
