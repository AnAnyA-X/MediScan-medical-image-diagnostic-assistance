from pathlib import Path
import torch
from torch import nn
from torchvision import models, transforms
from PIL import Image

LABELS = [
    "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration",
    "Mass", "Nodule", "Pneumonia", "Pneumothorax",
    "Consolidation", "Edema", "Emphysema", "Fibrosis",
    "Pleural_Thickening", "Hernia"
]
MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "densenet121_chestxray.pth"

def build_model():
    # Weights are trained by our training script, not downloaded implicitly.
    model = models.densenet121(weights=None)
    model.classifier = nn.Linear(model.classifier.in_features, len(LABELS))
    return model

def image_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

def load_trained_model(device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Trained model not found at {MODEL_PATH}. Train it using: "
            "python backend/train.py --data-dir data/nih --csv data/nih/Data_Entry_2017.csv"
        )
    model = build_model()
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()
    return model, device

def predict_image(image: Image.Image, model, device):
    tensor = image_transform()(image.convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        probabilities = torch.sigmoid(model(tensor))[0].cpu().tolist()
    return [
        {"label": label, "probability": round(float(prob), 4)}
        for label, prob in sorted(zip(LABELS, probabilities),
                                  key=lambda x: x[1], reverse=True)
    ]
