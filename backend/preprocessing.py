"""
MediScan Guardian — Chest Radiograph Preprocessing Pipeline
Strict implementation of PREPROCESSING_GUIDE.md and PHASE7_README.md:
- Supports PNG, JPG, and native DICOM (.dcm, .dicom)
- Full Modality LUT and VOI LUT handling
- MONOCHROME1 polarity inversion mandate (inverts bone/air polarity)
- MONOCHROME2 standard preservation (no inversion)
- Min-Max dynamic range scaling to [0.0, 1.0]
- Aspect-ratio preserving letterbox resizing to 512x512 with zero (black) padding
- 3-channel grayscale replication
- ImageNet channel normalization
- Unified for both offline training and online API inference
"""

import io
import os
from pathlib import Path
from typing import Union

import numpy as np
import torch
from PIL import Image

try:
    import pydicom
    from pydicom.pixel_data_handlers.util import apply_modality_lut, apply_voi_lut
    HAS_PYDICOM = True
except ImportError:
    HAS_PYDICOM = False

# ImageNet normalization statistics
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)
DEFAULT_TARGET_SIZE = (512, 512)


def is_dicom_file(file_input: Union[str, Path, bytes, io.BytesIO]) -> bool:
    """Check if the input represents a DICOM file."""
    if isinstance(file_input, (str, Path)):
        ext = os.path.splitext(str(file_input))[1].lower()
        if ext in [".dcm", ".dicom"]:
            return True
        try:
            with open(file_input, "rb") as fp:
                header = fp.read(132)
                return len(header) >= 132 and header[128:132] == b"DICM"
        except Exception:
            return False
    elif isinstance(file_input, (bytes, io.BytesIO)):
        data = file_input if isinstance(file_input, bytes) else file_input.getvalue()
        return len(data) >= 132 and data[128:132] == b"DICM"
    return False


def load_dicom_pixels(dicom_input: Union[str, Path, bytes, io.BytesIO]) -> np.ndarray:
    """
    Load DICOM pixel array following Phase 7 clinical imaging protocols:
    1. Apply Modality LUT (RescaleSlope / RescaleIntercept)
    2. Apply VOI LUT (WindowCenter / WindowWidth)
    3. Invert MONOCHROME1 images (mandated so bone is bright and lungs dark)
    4. Preserve MONOCHROME2 without inversion
    """
    if not HAS_PYDICOM:
        raise ImportError(
            "pydicom is required to process DICOM radiographs. "
            "Install it via: pip install pydicom"
        )

    if isinstance(dicom_input, (str, Path)):
        ds = pydicom.dcmread(str(dicom_input), force=True)
    else:
        stream = io.BytesIO(dicom_input) if isinstance(dicom_input, bytes) else dicom_input
        stream.seek(0)
        ds = pydicom.dcmread(stream, force=True)

    raw_arr = ds.pixel_array
    photometric = getattr(ds, "PhotometricInterpretation", "MONOCHROME2")
    if photometric is not None:
        photometric = str(photometric).strip().upper()
    else:
        photometric = "MONOCHROME2"

    # Step 1: Modality LUT
    try:
        mod_arr = apply_modality_lut(raw_arr, ds).astype(np.float32)
    except Exception:
        mod_arr = raw_arr.astype(np.float32)

    # Step 2: VOI LUT (WindowCenter / WindowWidth / VOI LUT)
    try:
        voi_arr = apply_voi_lut(mod_arr, ds).astype(np.float32)
    except Exception:
        voi_arr = mod_arr

    # Step 3: MONOCHROME1 Inversion Mandate
    if photometric == "MONOCHROME1":
        pixel_array = np.max(voi_arr) - voi_arr
    else:
        pixel_array = voi_arr

    return pixel_array.astype(np.float32)


def preprocess_cxr(
    image_input: Union[str, Path, bytes, io.BytesIO, Image.Image],
    target_size=(512, 512)
) -> torch.Tensor:
    """
    Standardized Phase 7 preprocessing pipeline for Chest X-Rays.

    Parameters:
    - image_input: File path, bytes, BytesIO stream, or PIL.Image.Image.
    - target_size: (height, width) tuple, defaults to (512, 512).

    Returns:
    - PyTorch FloatTensor of shape (3, target_height, target_width)
      normalized with ImageNet statistics.
    """
    # 1. Pixel Extraction
    if isinstance(image_input, Image.Image):
        # Already a PIL image (e.g. from FastAPI file upload or unit test)
        img_gray = image_input.convert("L")
        pixel_array = np.array(img_gray, dtype=np.float32)
    elif is_dicom_file(image_input):
        # Native DICOM image
        pixel_array = load_dicom_pixels(image_input)
    else:
        # Standard image (PNG, JPG, WEBP)
        if isinstance(image_input, (bytes, io.BytesIO)):
            stream = io.BytesIO(image_input) if isinstance(image_input, bytes) else image_input
            stream.seek(0)
            img = Image.open(stream).convert("L")
        else:
            img = Image.open(str(image_input)).convert("L")
        pixel_array = np.array(img, dtype=np.float32)

    # Validate image dimensions
    if pixel_array.size == 0 or len(pixel_array.shape) < 2 or pixel_array.shape[0] == 0 or pixel_array.shape[1] == 0:
        raise ValueError("Invalid image dimensions: image must have non-zero height and width.")

    # 2. Min-Max Dynamic Range Scaling to [0.0, 1.0]
    p_min = float(np.min(pixel_array))
    p_max = float(np.max(pixel_array))
    pixel_norm = (pixel_array - p_min) / (p_max - p_min + 1e-7)

    # 3. Aspect-Ratio Preserving Letterbox Resizing
    h, w = pixel_norm.shape
    th, tw = target_size
    scale = min(th / h, tw / w)
    nh, nw = int(round(h * scale)), int(round(w * scale))

    pil_img = Image.fromarray((pixel_norm * 255.0).astype(np.uint8))
    resized = pil_img.resize((nw, nh), resample=Image.Resampling.BILINEAR)
    resized_arr = np.array(resized, dtype=np.float32) / 255.0

    padded = np.zeros(target_size, dtype=np.float32)
    top = (th - nh) // 2
    left = (tw - nw) // 2
    padded[top:top + nh, left:left + nw] = resized_arr

    # 4. Replicate 3 Channels and Apply ImageNet Normalization
    # Shape: (3, th, tw)
    tensor = np.stack([padded, padded, padded], axis=0)
    tensor_norm = (tensor - IMAGENET_MEAN) / IMAGENET_STD

    return torch.from_numpy(tensor_norm.astype(np.float32))
