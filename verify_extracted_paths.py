import sys
from pathlib import Path
sys.path.insert(0, ".")
from backend.dataset import load_split_metadata, verify_patient_leakage

meta_dir = Path("data/metadata")
data_root = Path("data")

train_df = load_split_metadata(meta_dir / "Train_Metadata.csv", dataset_root=data_root)
val_df = load_split_metadata(meta_dir / "Validation_Metadata.csv", dataset_root=data_root)
test_df = load_split_metadata(meta_dir / "Test_Metadata.csv", dataset_root=data_root)

train_found = int(train_df["file_exists"].sum())
val_found = int(val_df["file_exists"].sum())
test_found = int(test_df["file_exists"].sum())

print(f"Train images resolved: {train_found}/{len(train_df)}")
print(f"Val images resolved: {val_found}/{len(val_df)}")
print(f"Test images resolved: {test_found}/{len(test_df)}")

total_found = train_found + val_found + test_found
total_expected = len(train_df) + len(val_df) + len(test_df)
print(f"Total resolved: {total_found}/{total_expected}")

leakage = verify_patient_leakage(train_df, val_df, test_df)
print("Patient leakage check:", leakage)
