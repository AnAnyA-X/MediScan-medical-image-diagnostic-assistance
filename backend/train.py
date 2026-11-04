"""
Baseline NIH ChestX-ray14 multi-label training script.
Expected:
  data/nih/Data_Entry_2017.csv
  data/nih/images/   (all extracted image files)

Run from project root:
  python backend/train.py --data-dir data/nih --csv data/nih/Data_Entry_2017.csv --epochs 3

This is a starter baseline. Use a GPU/Colab for the full dataset.
"""
import argparse, random
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, UnidentifiedImageError
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import models, transforms
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

LABELS = [
    "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration",
    "Mass", "Nodule", "Pneumonia", "Pneumothorax",
    "Consolidation", "Edema", "Emphysema", "Fibrosis",
    "Pleural_Thickening", "Hernia"
]

class NIHChestDataset(Dataset):
    def __init__(self, frame, image_dir, transform):
        self.frame = frame.reset_index(drop=True)
        self.image_dir = Path(image_dir)
        self.transform = transform

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, idx):
        row = self.frame.iloc[idx]
        path = self.image_dir / row["Image Index"]
        image = Image.open(path).convert("RGB")
        image = self.transform(image)
        target = torch.tensor(row[LABELS].to_numpy(dtype=np.float32))
        return image, target

def make_targets(csv_path, image_dir, limit=0):
    df = pd.read_csv(csv_path)
    required = {"Image Index", "Finding Labels"}
    if not required.issubset(df.columns):
        raise ValueError(f"CSV must contain columns: {required}")
    # NIH labels are pipe-separated. 'No Finding' is represented by all-zero disease labels.
    for label in LABELS:
        df[label] = df["Finding Labels"].fillna("").apply(
            lambda s: 1.0 if label in str(s).split("|") else 0.0
        )
    image_dir = Path(image_dir)
    df = df[df["Image Index"].apply(lambda name: (image_dir / name).is_file())].copy()
    if limit and limit > 0:
        df = df.sample(n=min(limit, len(df)), random_state=42)
    if df.empty:
        raise ValueError(f"No images found in {image_dir}. Check the image folder and CSV.")
    return df

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/nih")
    parser.add_argument("--csv", default="data/nih/Data_Entry_2017.csv")
    parser.add_argument("--image-dir", default=None)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--limit", type=int, default=0, help="Optional small subset for a quick smoke test")
    parser.add_argument("--lr", type=float, default=1e-4)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    image_dir = Path(args.image_dir) if args.image_dir else data_dir / "images"
    print("Loading NIH annotations and checking image paths...")
    df = make_targets(args.csv, image_dir, args.limit)
    print(f"Usable images: {len(df)}")
    print("Positive examples by label:")
    print(df[LABELS].sum().sort_values(ascending=False).to_string())

    train_df, val_df = train_test_split(df, test_size=0.15, random_state=42)
    tfm = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    train_ds = NIHChestDataset(train_df, image_dir, tfm)
    val_ds = NIHChestDataset(val_df, image_dir, tfm)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=0, pin_memory=torch.cuda.is_available())
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training on: {device}")
    model = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
    model.classifier = nn.Linear(model.classifier.in_features, len(LABELS))
    model.to(device)

    # Calculate per-label positive weights from the training split.
    positives = train_df[LABELS].sum().to_numpy(dtype=np.float32)
    negatives = len(train_df) - positives
    pos_weight = torch.tensor(negatives / np.maximum(positives, 1), dtype=torch.float32).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        model.train()
        running = 0.0
        for images, targets in train_loader:
            images, targets = images.to(device), targets.to(device)
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()
            running += loss.item() * images.size(0)
        print(f"Epoch {epoch+1}/{args.epochs} | train loss: {running/len(train_ds):.4f}")

    model.eval()
    all_probs, all_targets = [], []
    with torch.no_grad():
        for images, targets in val_loader:
            probs = torch.sigmoid(model(images.to(device))).cpu().numpy()
            all_probs.append(probs)
            all_targets.append(targets.numpy())
    probs = np.concatenate(all_probs)
    targets = np.concatenate(all_targets)
    aucs = {}
    for i, label in enumerate(LABELS):
        try:
            aucs[label] = float(roc_auc_score(targets[:, i], probs[:, i]))
        except ValueError:
            aucs[label] = None
    valid_aucs = [v for v in aucs.values() if v is not None]
    print("Validation ROC-AUC by label (actual validation results):")
    for label, value in aucs.items():
        print(f"  {label}: {value:.4f}" if value is not None else f"  {label}: unavailable (one class in validation split)")
    if valid_aucs:
        print(f"Macro ROC-AUC: {np.mean(valid_aucs):.4f}")

    output = Path("models")
    output.mkdir(exist_ok=True)
    torch.save({
        "model_state_dict": model.cpu().state_dict(),
        "labels": LABELS,
        "validation_roc_auc": aucs,
        "epochs": args.epochs,
        "training_images": len(train_ds),
        "validation_images": len(val_ds)
    }, output / "densenet121_chestxray.pth")
    print(f"Saved model to {output / 'densenet121_chestxray.pth'}")

if __name__ == "__main__":
    main()
