"""
MediScan Guardian — Phase 3 Binary Active-TB Training & Evaluation Pipeline
Backbone: DenseNet121 (ImageNet Pretrained)
Target: Binary Active Pulmonary Tuberculosis (Class 1) vs Negative Control (Class 0)
Dataset: Strict Phase 7 Supervised Cohort (540 Train / 116 Val / 117 Held-out Test)
Evaluation: Pooled Multi-Center + Subgroup Analysis (Shenzhen, Montgomery)
"""

import argparse
import json
import os
import random
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

# Ensure utf-8 terminal output on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from backend.dataset import (
    ActiveTBCXRDataset,
    create_tb_dataloaders,
    load_split_metadata,
    verify_patient_leakage,
)
from backend.model import (
    CLASS_NAMES,
    DEFAULT_MODEL_PATH,
    LEGACY_MODEL_PATH,
    TARGET_NAME,
    build_model,
)
from backend.preprocessing import DEFAULT_TARGET_SIZE


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def compute_binary_metrics(y_true: np.ndarray, y_probs: np.ndarray, threshold: float = 0.5) -> dict:
    y_pred = (y_probs >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    # Sensitivity / Recall
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    # Specificity
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    # Precision
    precision = precision_score(y_true, y_pred, zero_division=0)
    # F1
    f1 = f1_score(y_true, y_pred, zero_division=0)
    # Accuracy
    acc = accuracy_score(y_true, y_pred)
    # ROC-AUC
    try:
        auc = float(roc_auc_score(y_true, y_probs))
    except ValueError:
        auc = None

    return {
        "accuracy": round(float(acc), 4),
        "precision": round(float(precision), 4),
        "sensitivity_recall": round(float(sensitivity), 4),
        "specificity": round(float(specificity), 4),
        "f1_score": round(float(f1), 4),
        "roc_auc": round(float(auc), 4) if auc is not None else None,
        "confusion_matrix": {
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_positives": int(tp),
        },
        "sample_count": len(y_true),
        "positive_count": int(np.sum(y_true == 1)),
        "negative_count": int(np.sum(y_true == 0)),
    }


def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    total_batches = len(dataloader)
    for batch_idx, (images, targets) in enumerate(dataloader, 1):
        images = images.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        if batch_idx % 25 == 0 or batch_idx == total_batches:
            print(f"  Step [{batch_idx:3d}/{total_batches:3d}] | Batch Loss: {loss.item():.4f}", flush=True)
    return running_loss / len(dataloader.dataset)


def evaluate_split(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_probs = []
    all_targets = []
    with torch.no_grad():
        for images, targets in dataloader:
            images = images.to(device)
            targets = targets.to(device)
            logits = model(images)
            loss = criterion(logits, targets)
            running_loss += loss.item() * images.size(0)

            probs = torch.sigmoid(logits).cpu().numpy().flatten()
            all_probs.extend(probs)
            all_targets.extend(targets.cpu().numpy().flatten())

    val_loss = running_loss / len(dataloader.dataset)
    y_true = np.array(all_targets, dtype=int)
    y_probs = np.array(all_probs, dtype=float)
    metrics = compute_binary_metrics(y_true, y_probs)
    return val_loss, y_probs, y_true, metrics


def plot_confusion_matrix(cm_dict: dict, output_path: Path, title: str = "Test Confusion Matrix"):
    tn = cm_dict["true_negatives"]
    fp = cm_dict["false_positives"]
    fn = cm_dict["false_negatives"]
    tp = cm_dict["true_positives"]

    cm = np.array([[tn, fp], [fn, tp]])
    fig, ax = plt.subplots(figsize=(5, 4.5))
    cax = ax.matshow(cm, cmap="Blues", alpha=0.85)
    fig.colorbar(cax)

    for (i, j), z in np.ndenumerate(cm):
        ax.text(j, i, f"{z}", ha="center", va="center", fontsize=16, fontweight="bold",
                color="white" if z > (cm.max() / 2) else "black")

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Control (0)", "Active TB (1)"])
    ax.set_yticklabels(["Control (0)", "Active TB (1)"])
    ax.set_xlabel("Predicted Label", fontweight="bold", labelpad=8)
    ax.set_ylabel("True Label", fontweight="bold", labelpad=8)
    ax.set_title(title, fontweight="bold", pad=12)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_roc_curve(y_true: np.ndarray, y_probs: np.ndarray, auc_val: float, output_path: Path):
    fpr, tpr, _ = roc_curve(y_true, y_probs)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color="#087f82", lw=2.5, label=f"DenseNet121 (AUC = {auc_val:.4f})")
    ax.plot([0, 1], [0, 1], color="#999999", lw=1.5, linestyle="--", label="Chance (AUC = 0.50)")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate (1 - Specificity)", fontweight="bold")
    ax.set_ylabel("True Positive Rate (Sensitivity)", fontweight="bold")
    ax.set_title("ROC Curve — Active Pulmonary TB Screening", fontweight="bold")
    ax.legend(loc="lower right", frameon=True)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Train DenseNet121 for Active TB Detection")
    parser.add_argument("--metadata-dir", default="data/metadata", help="Path to folder containing metadata CSVs")
    parser.add_argument("--data-dir", default="data", help="Root directory where images are located")
    parser.add_argument("--output-dir", default="outputs", help="Directory for output metrics and plots")
    parser.add_argument("--model-dir", default="models", help="Directory to save model checkpoints")
    parser.add_argument("--epochs", type=int, default=4, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size for training and evaluation")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--patience", type=int, default=4, help="Early stopping patience")
    parser.add_argument("--seed", type=int, default=42, help="Deterministic random seed")
    parser.add_argument("--num-workers", type=int, default=0, help="Number of DataLoader workers")
    parser.add_argument(
        "--freeze-early-layers",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Freeze early blocks (1-3) and fine-tune denseblock4 + classifier for fast CPU training (use --no-freeze-early-layers for full model fine-tuning)",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint (.pth) to resume training from",
    )
    args = parser.parse_args()

    set_seed(args.seed)

    output_dir = PROJECT_ROOT / args.output_dir
    model_dir = PROJECT_ROOT / args.model_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("  MEDISCAN GUARDIAN — PHASE 3 ACTIVE TB MODEL TRAINING")
    print("=" * 70)

    # 1. Load DataLoaders and Verify Patient Leakage
    meta_dir = PROJECT_ROOT / args.metadata_dir
    data_dir = PROJECT_ROOT / args.data_dir
    print(f"Loading Phase 7 splits from: {meta_dir}")
    print(f"Resolving images from root:  {data_dir}")

    train_loader, val_loader, test_loader, stats = create_tb_dataloaders(
        metadata_dir=meta_dir,
        dataset_root=data_dir,
        batch_size=args.batch_size,
        target_size=DEFAULT_TARGET_SIZE,
        num_workers=args.num_workers,
    )

    print(f"\nCohort Overview:")
    print(f"  - Training Set:   {stats['train_count']} images ({stats['train_pos']} Active TB, {stats['train_neg']} Controls)")
    print(f"  - Validation Set: {stats['val_count']} images")
    print(f"  - Held-out Test:  {stats['test_count']} images")
    print(f"  - Patient Exclusivity: ZERO leakage verified across all partitions ({stats['leakage_verification']['zero_leakage_verified']})")

    # 2. Setup Device, Model, Criterion, Optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nTraining Device: {device}")
    if device.type == "cpu":
        torch.set_num_threads(min(8, os.cpu_count() or 4))
        print(f"Using {torch.get_num_threads()} CPU threads for acceleration.")

    print("\nInitializing DenseNet121 with ImageNet pretrained weights...")
    model = build_model(pretrained=True)
    model.to(device)

    # Pos weight = Negatives / Positives = 292 / 248 = 1.1774
    pos_weight_val = float(stats["train_neg"]) / max(float(stats["train_pos"]), 1.0)
    pos_weight = torch.tensor([pos_weight_val], dtype=torch.float32).to(device)
    print(f"Class imbalance weighting: pos_weight = {pos_weight_val:.4f}")

    if args.freeze_early_layers:
        print("Freezing early layers (blocks 1-3); fine-tuning denseblock4 and classifier for high-efficiency CPU convergence...")
        for name, param in model.features.named_parameters():
            if not ("denseblock4" in name or "norm5" in name):
                param.requires_grad = False

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    print(f"Trainable parameter tensors: {len(trainable_params)}")

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=2
    )

    # Save Configurations
    training_config = {
        "model_architecture": "DenseNet121",
        "pretrained_weights": "ImageNet",
        "classifier_head": "Linear(in_features=1024, out_features=1)",
        "input_target_size": list(DEFAULT_TARGET_SIZE),
        "target_channels": 3,
        "normalization": {
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
            "scaling": "[0.0, 1.0] min-max",
            "aspect_ratio": "letterbox padded with black borders"
        },
        "target_class": TARGET_NAME,
        "class_mapping": {"0": "Negative_Control", "1": "Active_TB"},
        "pos_weight": round(pos_weight_val, 4),
        "learning_rate": args.lr,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "seed": args.seed,
        "patience": args.patience,
        "freeze_early_layers": args.freeze_early_layers,
        "device": str(device)
    }

    with open(output_dir / "config.json", "w") as fp:
        json.dump(training_config, fp, indent=2)

    with open(output_dir / "class_mapping.json", "w") as fp:
        json.dump({"0": "Negative_Control", "1": "Active_TB"}, fp, indent=2)

    # 3. Training Loop with Validation Model Selection & Incremental Checkpointing
    print("\n--- Starting Training & Validation Loop ---")
    start_epoch = 1
    best_val_auc = -1.0
    best_epoch = 0
    best_model_state = None
    best_val_metrics = None
    epochs_no_improve = 0

    history = {
        "train_loss": [],
        "val_loss": [],
        "val_roc_auc": [],
        "val_accuracy": [],
        "val_sensitivity": [],
        "val_specificity": [],
    }

    # Resume capability from existing checkpoint
    if args.resume:
        resume_path = Path(args.resume)
        if not resume_path.is_absolute():
            resume_path = PROJECT_ROOT / resume_path
        if resume_path.exists():
            print(f"\nResuming training from checkpoint: {resume_path}")
            checkpoint = torch.load(resume_path, map_location=device)
            state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
            model.load_state_dict(state_dict)
            print("  - Restored model weights.")

            if "optimizer_state_dict" in checkpoint:
                try:
                    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                    print("  - Restored optimizer state.")
                except Exception as e:
                    print(f"  - Warning: Could not restore optimizer state ({e}). Using fresh optimizer.")

            if "scheduler_state_dict" in checkpoint:
                try:
                    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
                    print("  - Restored learning rate scheduler state.")
                except Exception as e:
                    print(f"  - Warning: Could not restore scheduler state ({e}).")

            saved_epoch = checkpoint.get("epoch", 0)
            start_epoch = saved_epoch + 1
            best_val_auc = float(checkpoint.get("best_val_roc_auc", -1.0))
            best_epoch = int(checkpoint.get("best_epoch", 0))
            best_val_metrics = checkpoint.get("best_val_metrics", None)

            if "history" in checkpoint and isinstance(checkpoint["history"], dict):
                history = checkpoint["history"]
                print(f"  - Restored training history for {len(history.get('train_loss', []))} completed epochs.")
            elif (output_dir / "history.json").exists():
                try:
                    with open(output_dir / "history.json", "r") as fp:
                        history = json.load(fp)
                    print(f"  - Restored training history from {output_dir / 'history.json'}.")
                except Exception:
                    pass

            best_chk_path = model_dir / "densenet121_active_tb.pth"
            if best_chk_path.exists():
                try:
                    best_chk = torch.load(best_chk_path, map_location="cpu")
                    best_model_state = best_chk.get("model_state_dict")
                except Exception:
                    pass

            print(f"  - Resuming starting at Epoch {start_epoch} of {args.epochs} (Best Val AUC: {best_val_auc:.4f} at Epoch {best_epoch})\n")
        else:
            print(f"Warning: Checkpoint to resume not found at {resume_path}. Starting training from scratch.\n")

    if start_epoch > args.epochs:
        print(f"\nAll {args.epochs} requested epochs already completed (checkpoint was at epoch {start_epoch - 1}).")
        print("Skipping training loop and proceeding directly to held-out test evaluation.\n")
    else:
        for epoch in range(start_epoch, args.epochs + 1):
            train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
            val_loss, _, _, val_metrics = evaluate_split(model, val_loader, criterion, device)
            val_auc = val_metrics["roc_auc"] if val_metrics["roc_auc"] is not None else 0.0

            history["train_loss"].append(round(train_loss, 4))
            history["val_loss"].append(round(val_loss, 4))
            history["val_roc_auc"].append(round(val_auc, 4))
            history["val_accuracy"].append(val_metrics["accuracy"])
            history["val_sensitivity"].append(val_metrics["sensitivity_recall"])
            history["val_specificity"].append(val_metrics["specificity"])

            scheduler.step(val_auc)

            is_best = val_auc > best_val_auc
            if is_best:
                best_val_auc = val_auc
                best_epoch = epoch
                best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                best_val_metrics = val_metrics
                epochs_no_improve = 0
                best_flag = " * [Best Checkpoint]"
            else:
                epochs_no_improve += 1
                best_flag = ""

            print(
                f"Epoch {epoch:2d}/{args.epochs:2d} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"Val AUC: {val_auc:.4f} | "
                f"Val Acc: {val_metrics['accuracy']:.4f} | "
                f"Val Sens: {val_metrics['sensitivity_recall']:.4f} | "
                f"Val Spec: {val_metrics['specificity']:.4f}{best_flag}",
                flush=True
            )

            # A. Persist training history after every completed epoch
            with open(output_dir / "history.json", "w") as fp:
                json.dump(history, fp, indent=2)

            # B. Save last checkpoint after every completed epoch for resume support
            last_checkpoint_path = model_dir / "last_checkpoint.pth"
            last_checkpoint_data = {
                "epoch": epoch,
                "model_state_dict": {k: v.cpu().clone() for k, v in model.state_dict().items()},
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "model_architecture": "densenet121",
                "labels": CLASS_NAMES,
                "target_name": TARGET_NAME,
                "class_mapping": {"0": "Negative_Control", "1": "Active_TB"},
                "best_epoch": best_epoch,
                "best_val_roc_auc": best_val_auc,
                "best_val_metrics": best_val_metrics,
                "history": history,
                "preprocessing_config": training_config["normalization"],
            }
            torch.save(last_checkpoint_data, last_checkpoint_path)

            # C. Save best checkpoint immediately if this epoch improved validation AUC
            if is_best:
                best_checkpoint_data = {
                    "epoch": epoch,
                    "model_state_dict": best_model_state,
                    "model_architecture": "densenet121",
                    "labels": CLASS_NAMES,
                    "target_name": TARGET_NAME,
                    "class_mapping": {"0": "Negative_Control", "1": "Active_TB"},
                    "best_epoch": best_epoch,
                    "best_val_roc_auc": best_val_auc,
                    "best_val_metrics": best_val_metrics,
                    "history": history,
                    "preprocessing_config": training_config["normalization"],
                }
                torch.save(best_checkpoint_data, model_dir / "densenet121_active_tb.pth")
                torch.save(best_checkpoint_data, model_dir / "densenet121_chestxray.pth")
                torch.save(best_checkpoint_data, model_dir / "best_checkpoint.pth")
                print(f"  -> Persisted best checkpoint to {model_dir / 'densenet121_active_tb.pth'} (Val AUC: {best_val_auc:.4f})", flush=True)

            if epochs_no_improve >= args.patience:
                print(f"\nEarly stopping triggered after {epoch} epochs (no improvement for {args.patience} epochs).")
                break

    # 4. Resolve Best Model Checkpoint for Test Evaluation
    checkpoint_path = model_dir / "densenet121_active_tb.pth"
    legacy_path = model_dir / "densenet121_chestxray.pth"

    if best_model_state is None and checkpoint_path.exists():
        print(f"Loading best weights from existing checkpoint: {checkpoint_path}")
        chk = torch.load(checkpoint_path, map_location="cpu")
        best_model_state = chk.get("model_state_dict")
        best_epoch = chk.get("best_epoch", best_epoch)
        best_val_auc = chk.get("best_val_roc_auc", best_val_auc)
    elif best_model_state is None:
        best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        best_epoch = args.epochs
        best_val_auc = history["val_roc_auc"][-1] if history["val_roc_auc"] else 0.0

    print(f"\nFinal Best Model Selected: Epoch {best_epoch} with Val AUC: {best_val_auc:.4f}")
    print(f"Saved Checkpoints:")
    print(f"  - Best Checkpoint: {checkpoint_path}")
    print(f"  - Legacy Mirror:   {legacy_path}")
    print(f"  - Last Checkpoint: {model_dir / 'last_checkpoint.pth'}")

    # 5. Held-Out Test Evaluation (Evaluated ONLY after model selection)
    print("\n" + "=" * 70)
    print("  HELD-OUT TEST SET EVALUATION (Unseen Patients Only)")
    print("=" * 70)

    # Load best model for evaluation
    model.load_state_dict(best_model_state)
    model.to(device).eval()

    # Load test split with metadata passthrough for subgroup evaluations
    test_df = load_split_metadata(meta_dir / "Test_Metadata.csv", dataset_root=data_dir)
    test_ds = ActiveTBCXRDataset(test_df, dataset_root=data_dir, target_size=DEFAULT_TARGET_SIZE, return_meta=True)
    test_meta_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    test_probs = []
    test_targets = []
    test_sources = []
    test_img_ids = []

    with torch.no_grad():
        for images, targets, meta in test_meta_loader:
            images = images.to(device)
            logits = model(images)
            probs = torch.sigmoid(logits).cpu().numpy().flatten()

            test_probs.extend(probs)
            test_targets.extend(targets.numpy().flatten())
            test_sources.extend(meta["source_dataset"])
            test_img_ids.extend(meta["image_id"])

    y_test_true = np.array(test_targets, dtype=int)
    y_test_probs = np.array(test_probs, dtype=float)
    sources = np.array(test_sources)

    # A. Pooled Multi-Center Metrics
    pooled_metrics = compute_binary_metrics(y_test_true, y_test_probs)

    # B. Shenzhen-Only Metrics
    shz_mask = (sources == "TB_Shenzhen")
    shz_metrics = compute_binary_metrics(y_test_true[shz_mask], y_test_probs[shz_mask])

    # C. Montgomery-Only Metrics
    mc_mask = (sources == "TB_Montgomery")
    mc_metrics = compute_binary_metrics(y_test_true[mc_mask], y_test_probs[mc_mask])

    # Generate and Save Evaluation Plots
    plot_confusion_matrix(pooled_metrics["confusion_matrix"], output_dir / "confusion_matrix.png", title="Test Confusion Matrix (Pooled Multi-Center)")
    plot_roc_curve(y_test_true, y_test_probs, pooled_metrics["roc_auc"], output_dir / "roc_curve.png")

    full_evaluation_report = {
        "pooled_multicenter_metrics": pooled_metrics,
        "subgroup_metrics": {
            "shenzhen_only": shz_metrics,
            "montgomery_only": mc_metrics,
            "montgomery_warning": (
                "Warning: The Montgomery test subgroup contains only 15 test images (3 positive, 12 negative); "
                "small-sample statistics may exhibit wide confidence intervals."
            )
        },
        "checkpoint_path": str(checkpoint_path),
        "best_epoch": best_epoch,
        "best_val_roc_auc": best_val_auc,
    }

    with open(output_dir / "test_metrics.json", "w") as fp:
        json.dump(full_evaluation_report, fp, indent=2)

    # Print Final Audit Summary
    print(f"\n[A] POOLED MULTI-CENTER TEST METRICS (N = {pooled_metrics['sample_count']}):")
    print(f"  - Accuracy:         {pooled_metrics['accuracy'] * 100:.2f}%")
    print(f"  - Precision:        {pooled_metrics['precision'] * 100:.2f}%")
    print(f"  - Sensitivity/Recall: {pooled_metrics['sensitivity_recall'] * 100:.2f}%")
    print(f"  - Specificity:      {pooled_metrics['specificity'] * 100:.2f}%")
    print(f"  - F1-Score:         {pooled_metrics['f1_score']:.4f}")
    print(f"  - ROC-AUC:          {pooled_metrics['roc_auc']:.4f}")
    print(f"  - Confusion Matrix: TP={pooled_metrics['confusion_matrix']['true_positives']}, FP={pooled_metrics['confusion_matrix']['false_positives']}, TN={pooled_metrics['confusion_matrix']['true_negatives']}, FN={pooled_metrics['confusion_matrix']['false_negatives']}")

    print(f"\n[B] SHENZHEN-ONLY SUBGROUP METRICS (N = {shz_metrics['sample_count']}):")
    print(f"  - Accuracy:         {shz_metrics['accuracy'] * 100:.2f}%")
    print(f"  - Precision:        {shz_metrics['precision'] * 100:.2f}%")
    print(f"  - Sensitivity:      {shz_metrics['sensitivity_recall'] * 100:.2f}%")
    print(f"  - Specificity:      {shz_metrics['specificity'] * 100:.2f}%")
    print(f"  - F1-Score:         {shz_metrics['f1_score']:.4f}")
    print(f"  - ROC-AUC:          {shz_metrics['roc_auc']:.4f}")

    print(f"\n[C] MONTGOMERY-ONLY SUBGROUP METRICS (N = {mc_metrics['sample_count']}):")
    print(f"  ! WARNING: Subgroup has only {mc_metrics['sample_count']} images (3 Positives, 12 Negatives); metrics subject to small-sample variance.")
    print(f"  - Accuracy:         {mc_metrics['accuracy'] * 100:.2f}%")
    print(f"  - Sensitivity:      {mc_metrics['sensitivity_recall'] * 100:.2f}%")
    print(f"  - Specificity:      {mc_metrics['specificity'] * 100:.2f}%")
    print(f"  - ROC-AUC:          {mc_metrics['roc_auc']}")

    print("\n" + "=" * 70)
    print("  ALL ARTIFACTS SAVED SUCCESSFULLY:")
    print(f"  - Checkpoint:       {checkpoint_path}")
    print(f"  - Test Metrics:     {output_dir / 'test_metrics.json'}")
    print(f"  - Training History: {output_dir / 'history.json'}")
    print(f"  - Confusion Matrix: {output_dir / 'confusion_matrix.png'}")
    print(f"  - ROC Curve:        {output_dir / 'roc_curve.png'}")
    print(f"  - Training Config:  {output_dir / 'config.json'}")
    print("=" * 70)


if __name__ == "__main__":
    main()
