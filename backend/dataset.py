"""
MediScan Guardian — Phase 7 Tuberculosis Dataset Loader
Strict patient-stratified data loading for binary active-TB classification:
- Loads Train_Metadata.csv, Validation_Metadata.csv, Test_Metadata.csv
- Enforces strict patient-level exclusivity (zero patient leakage)
- Dynamic, configurable path resolution without hardcoding specific environments
- Excludes non-binary labels (strictly Final_TB_Label in {0, 1})
- Integrates standardized Phase 7 aspect-ratio preserving 512x512 preprocessing
- Supports subgroup analysis (Shenzhen, Montgomery, Hospital-grade controls)
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from backend.preprocessing import DEFAULT_TARGET_SIZE, preprocess_cxr


def resolve_image_path(
    row: Union[pd.Series, dict],
    dataset_root: Optional[Union[str, Path]] = None
) -> Tuple[Path, bool]:
    """
    Dynamically resolve the file path of a radiograph.
    Searches multiple standard directory conventions without hardcoding
    host-specific absolute paths.

    Returns:
    - (resolved_path, exists_boolean)
    """
    image_id = str(row.get("Image_ID", "")).strip()
    orig_path_str = str(row.get("Original_Image_Path", "")).strip()
    source_dataset = str(row.get("Source_Dataset", "")).strip()
    final_path_str = str(row.get("Final_Image_Path", "")).strip()

    candidate_paths: List[Path] = []

    if dataset_root is not None:
        root = Path(dataset_root)

        # Candidate 1: Direct join with Original_Image_Path (e.g. root/medical-analysis/datasets/...)
        if orig_path_str:
            candidate_paths.append(root / orig_path_str)

        # Candidate 2: Relative to 'datasets/' if root already points inside
        if orig_path_str and "datasets" in orig_path_str:
            sub = orig_path_str.split("datasets", 1)[-1].lstrip("\\/")
            candidate_paths.append(root / sub)

        # Candidate 3: Subfolder by Source_Dataset (e.g. root/TB_Shenzhen/images/CHNCXR_0001_0.png)
        if source_dataset and image_id:
            candidate_paths.append(root / source_dataset / "images" / image_id)
            candidate_paths.append(root / source_dataset / image_id)

        # Candidate 4: Standard 'images/' subfolder (e.g. root/images/CHNCXR_0001_0.png)
        if image_id:
            candidate_paths.append(root / "images" / image_id)

        # Candidate 5: Direct flat directory (e.g. root/CHNCXR_0001_0.png)
        if image_id:
            candidate_paths.append(root / image_id)

    # Candidate 6: Check Original_Image_Path as a relative path from current working directory
    if orig_path_str:
        candidate_paths.append(Path(orig_path_str))

    # Candidate 7: Check Final_Image_Path if accessible on this machine
    if final_path_str:
        candidate_paths.append(Path(final_path_str))

    # Search candidates in priority order
    for p in candidate_paths:
        if p.is_file():
            return p, True

    # If not found, return the most descriptive candidate path with exists=False
    primary_candidate = candidate_paths[0] if candidate_paths else Path(orig_path_str or image_id)
    return primary_candidate, False


def load_split_metadata(
    csv_path: Union[str, Path],
    dataset_root: Optional[Union[str, Path]] = None,
    filter_valid_labels: bool = True
) -> pd.DataFrame:
    """
    Load a Phase 7 split CSV and validate essential columns and labels.
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {csv_path}")

    df = pd.read_csv(csv_path)

    # Validate essential columns discovered in Phase 1 audit
    required_cols = {"Image_ID", "Patient_Group_ID", "Final_TB_Label"}
    if not required_cols.issubset(df.columns):
        raise ValueError(
            f"CSV {csv_path.name} is missing mandatory columns: {required_cols - set(df.columns)}"
        )

    # Filter strictly to binary labels 0 and 1
    if filter_valid_labels:
        df = df[df["Final_TB_Label"].isin([0, 1])].copy()

    # Pre-resolve and annotate image existence status
    resolved_paths = []
    exists_flags = []
    for _, row in df.iterrows():
        p, exists = resolve_image_path(row, dataset_root=dataset_root)
        resolved_paths.append(str(p))
        exists_flags.append(exists)

    df["resolved_path"] = resolved_paths
    df["file_exists"] = exists_flags

    return df.reset_index(drop=True)


def verify_patient_leakage(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame
) -> Dict[str, Union[int, bool]]:
    """
    Mathematically verify zero patient data leakage across all 3 partitions.
    """
    train_patients = set(train_df["Patient_Group_ID"])
    val_patients = set(val_df["Patient_Group_ID"])
    test_patients = set(test_df["Patient_Group_ID"])

    train_val_overlap = len(train_patients & val_patients)
    train_test_overlap = len(train_patients & test_patients)
    val_test_overlap = len(val_patients & test_patients)

    has_leakage = (train_val_overlap > 0) or (train_test_overlap > 0) or (val_test_overlap > 0)

    results = {
        "train_patients": len(train_patients),
        "val_patients": len(val_patients),
        "test_patients": len(test_patients),
        "train_val_overlap": train_val_overlap,
        "train_test_overlap": train_test_overlap,
        "val_test_overlap": val_test_overlap,
        "zero_leakage_verified": not has_leakage,
    }

    if has_leakage:
        raise ValueError(f"CRITICAL: Patient data leakage detected across splits! {results}")

    return results


class ActiveTBCXRDataset(Dataset):
    """
    PyTorch Dataset for Phase 7 Active Pulmonary Tuberculosis Screening.
    Returns:
    - image_tensor: FloatTensor of shape (3, 512, 512)
    - target: FloatTensor of shape (1,) with value 1.0 (Active TB) or 0.0 (Control)
    """

    def __init__(
        self,
        metadata_df: pd.DataFrame,
        dataset_root: Optional[Union[str, Path]] = None,
        target_size: Tuple[int, int] = DEFAULT_TARGET_SIZE,
        return_meta: bool = False
    ):
        self.frame = metadata_df.reset_index(drop=True)
        self.dataset_root = Path(dataset_root) if dataset_root else None
        self.target_size = target_size
        self.return_meta = return_meta

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, idx: int):
        row = self.frame.iloc[idx]

        # Use pre-resolved path or resolve dynamically
        if "resolved_path" in row and row.get("file_exists", False):
            img_path = Path(row["resolved_path"])
        else:
            img_path, exists = resolve_image_path(row, self.dataset_root)
            if not exists:
                raise FileNotFoundError(
                    f"Radiograph file not found for Image_ID '{row.get('Image_ID')}'. "
                    f"Attempted path: {img_path}. "
                    "Ensure the dataset root directory is correctly configured."
                )

        try:
            image_tensor = preprocess_cxr(img_path, target_size=self.target_size)
        except Exception as e:
            raise RuntimeError(
                f"Failed to process radiograph {img_path} ({row.get('Image_ID')}): {e}"
            ) from e

        label_val = float(row["Final_TB_Label"])
        target = torch.tensor([label_val], dtype=torch.float32)

        if self.return_meta:
            meta = {
                "image_id": str(row.get("Image_ID") or ""),
                "patient_group_id": str(row.get("Patient_Group_ID") or ""),
                "source_dataset": str(row.get("Source_Dataset") or ""),
                "clinical_status": str(row.get("Clinical_Status") or ""),
                "diagnostic_evidence": str(row.get("Diagnostic_Evidence_Category") or ""),
                "file_path": str(img_path),
            }
            return image_tensor, target, meta

        return image_tensor, target


def create_tb_dataloaders(
    metadata_dir: Union[str, Path],
    dataset_root: Optional[Union[str, Path]] = None,
    batch_size: int = 16,
    target_size: Tuple[int, int] = DEFAULT_TARGET_SIZE,
    num_workers: int = 0
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict[str, any]]:
    """
    Constructs PyTorch DataLoaders for Train, Validation, and Test splits.
    Preserves exact Phase 7 patient stratification.
    """
    meta_path = Path(metadata_dir)
    train_df = load_split_metadata(meta_path / "Train_Metadata.csv", dataset_root=dataset_root)
    val_df = load_split_metadata(meta_path / "Validation_Metadata.csv", dataset_root=dataset_root)
    test_df = load_split_metadata(meta_path / "Test_Metadata.csv", dataset_root=dataset_root)

    # Verify zero patient leakage
    leakage_stats = verify_patient_leakage(train_df, val_df, test_df)

    train_ds = ActiveTBCXRDataset(train_df, dataset_root=dataset_root, target_size=target_size)
    val_ds = ActiveTBCXRDataset(val_df, dataset_root=dataset_root, target_size=target_size)
    test_ds = ActiveTBCXRDataset(test_df, dataset_root=dataset_root, target_size=target_size)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=torch.cuda.is_available()
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers
    )

    stats = {
        "train_count": len(train_df),
        "val_count": len(val_df),
        "test_count": len(test_df),
        "train_pos": int((train_df["Final_TB_Label"] == 1).sum()),
        "train_neg": int((train_df["Final_TB_Label"] == 0).sum()),
        "leakage_verification": leakage_stats,
    }

    return train_loader, val_loader, test_loader, stats
