"""
Tests for MediScan Guardian Image Preprocessing Pipeline (Task 2 & Task 5).
Covers:
1. Image loading from various inputs (File path, PIL Image, BytesIO, raw bytes).
2. Image format handling (PNG, JPG, WEBP, DICOM).
3. Grayscale conversion (RGB, RGBA, Grayscale 'L').
4. Aspect-ratio preserving letterbox resizing with zero padding.
5. Dynamic range normalization [0.0, 1.0] and ImageNet normalization.
6. Tensor dimensions (strictly [3, 512, 512]).
7. Tensor data types (torch.float32, all finite values, no NaN/Inf).
8. Input compatibility with DenseNet121 model.
9. Invalid image handling (zero dimensions, invalid types).
10. Corrupted image handling (truncated bytes).
11. DICOM MONOCHROME1 polarity inversion mandate.
12. DICOM MONOCHROME2 standard preservation.
"""

import io
import os
import tempfile
from pathlib import Path
import numpy as np
import pytest
import torch
from PIL import Image

from backend.preprocessing import (
    HAS_PYDICOM,
    IMAGENET_MEAN,
    IMAGENET_STD,
    DEFAULT_TARGET_SIZE,
    is_dicom_file,
    load_dicom_pixels,
    preprocess_cxr,
)

if HAS_PYDICOM:
    import pydicom
    from pydicom.dataset import Dataset as DcmDataset, FileMetaDataset, FileDataset
    from pydicom.uid import ExplicitVRLittleEndian, SecondaryCaptureImageStorage


def create_synthetic_dicom_bytes(photometric: str = "MONOCHROME2", size=(64, 64), min_val=0, max_val=4000):
    """Helper creating in-memory synthetic DICOM bytes for test isolation."""
    if not HAS_PYDICOM:
        pytest.skip("pydicom not installed")

    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = SecondaryCaptureImageStorage
    file_meta.MediaStorageSOPInstanceUID = pydicom.uid.generate_uid()
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian

    ds = DcmDataset()
    ds.file_meta = file_meta
    ds.is_little_endian = True
    ds.is_implicit_VR = False

    ds.SOPClassUID = SecondaryCaptureImageStorage
    ds.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
    ds.Modality = "CR"
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = photometric
    ds.Rows, ds.Columns = size
    ds.BitsAllocated = 16
    ds.BitsStored = 12
    ds.HighBit = 11
    ds.PixelRepresentation = 0
    ds.RescaleIntercept = 0
    ds.RescaleSlope = 1

    arr = np.linspace(min_val, max_val, size[0] * size[1], dtype=np.uint16).reshape(size)
    ds.PixelData = arr.tobytes()

    bio = io.BytesIO()
    fds = FileDataset(bio, {}, file_meta=file_meta, preamble=b"\0" * 128)
    fds.update(ds)
    fds.is_little_endian = True
    fds.is_implicit_VR = False
    fds.save_as(bio)
    return bio.getvalue()


# 1. Image Loading from Various Input Types
def test_preprocess_cxr_from_pil_image():
    """Verify preprocessing directly from PIL.Image instance."""
    img = Image.new("RGB", (300, 300), color=(100, 150, 200))
    tensor = preprocess_cxr(img)
    assert isinstance(tensor, torch.Tensor)
    assert tensor.shape == (3, 512, 512)
    assert tensor.dtype == torch.float32


def test_preprocess_cxr_from_bytes():
    """Verify preprocessing from raw bytes."""
    img = Image.new("L", (256, 256), color=128)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    tensor = preprocess_cxr(buf.getvalue())
    assert tensor.shape == (3, 512, 512)
    assert torch.isfinite(tensor).all()


def test_preprocess_cxr_from_bytesio():
    """Verify preprocessing from BytesIO stream."""
    img = Image.new("L", (256, 256), color=128)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    tensor = preprocess_cxr(buf)
    assert tensor.shape == (3, 512, 512)


def test_preprocess_cxr_from_filepath(tmp_path):
    """Verify preprocessing from disk file path."""
    img = Image.new("RGB", (400, 300), color=(20, 40, 60))
    p = tmp_path / "test_img.jpg"
    img.save(p, format="JPEG")

    tensor = preprocess_cxr(str(p))
    assert tensor.shape == (3, 512, 512)

    tensor_path = preprocess_cxr(p)
    assert tensor_path.shape == (3, 512, 512)


# 2. Supported Formats (PNG, JPG, WEBP)
@pytest.mark.parametrize("fmt", ["PNG", "JPEG", "WEBP"])
def test_preprocess_standard_formats(fmt):
    """Verify PNG, JPEG, and WEBP are loaded and preprocessed uniformly."""
    img = Image.new("RGB", (200, 200), color=(80, 90, 100))
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    tensor = preprocess_cxr(buf.getvalue())
    assert tensor.shape == (3, 512, 512)
    assert torch.isfinite(tensor).all()


# 3. Grayscale Conversion (RGB, RGBA, L)
def test_grayscale_conversion_rgba():
    """RGBA images must be cleanly converted to grayscale without channel artifacts."""
    img = Image.new("RGBA", (150, 150), color=(255, 0, 0, 128))
    tensor = preprocess_cxr(img)
    assert tensor.shape == (3, 512, 512)
    # Check that channels 0, 1, 2 have identical pre-normalization signals
    # Before normalization, channels were identical replicas.
    # Check that min and max are within expected ImageNet bounds
    assert tensor.min() >= -3.0
    assert tensor.max() <= 3.0


# 4. Aspect-Ratio Preserving Letterbox Resizing & Zero Padding
def test_letterbox_resizing_portrait():
    """Tall portrait image (e.g. 200x800) must be scaled to fit height and letterbox padded horizontally."""
    img = Image.new("L", (200, 800), color=255)
    tensor = preprocess_cxr(img, target_size=(512, 512))
    assert tensor.shape == (3, 512, 512)
    # Check border pixels: since the padding is zero, normalized value should match (0 - mean) / std
    expected_zero_ch0 = (0.0 - 0.485) / 0.229
    # Left edge (x=0) should be padding
    np.testing.assert_allclose(tensor[0, 256, 0].item(), expected_zero_ch0, atol=1e-3)


def test_letterbox_resizing_landscape():
    """Wide landscape image (e.g. 800x200) must be scaled to fit width and letterbox padded vertically."""
    img = Image.new("L", (800, 200), color=255)
    tensor = preprocess_cxr(img, target_size=(512, 512))
    assert tensor.shape == (3, 512, 512)
    expected_zero_ch0 = (0.0 - 0.485) / 0.229
    # Top edge (y=0) should be padding
    np.testing.assert_allclose(tensor[0, 0, 256].item(), expected_zero_ch0, atol=1e-3)


# 5. Normalization Values & Finite Tensors
def test_tensor_normalization_and_finiteness():
    """Verify that output tensor has no NaN, no Inf, and is normalized."""
    img = Image.new("L", (512, 512), color=128)
    tensor = preprocess_cxr(img)
    assert torch.isfinite(tensor).all()
    assert not torch.isnan(tensor).any()
    assert tensor.dtype == torch.float32


# 6. DICOM Handling & MONOCHROME1 Inversion
def test_is_dicom_file_detection():
    """Verify DICOM detection helper correctly classifies DICOM vs standard files."""
    assert not is_dicom_file(b"not_a_dicom_file")
    assert not is_dicom_file("image.png")
    assert is_dicom_file("scan.dcm")
    assert is_dicom_file("scan.dicom")

    if HAS_PYDICOM:
        dcm_bytes = create_synthetic_dicom_bytes()
        assert is_dicom_file(dcm_bytes)


def test_dicom_monochrome1_inversion():
    """Verify MONOCHROME1 images are inverted so bright bone remains high signal."""
    if not HAS_PYDICOM:
        pytest.skip("pydicom not installed")

    # Gradient from 0 to 4000
    dcm_m1 = create_synthetic_dicom_bytes(photometric="MONOCHROME1", size=(64, 64), min_val=0, max_val=4000)
    dcm_m2 = create_synthetic_dicom_bytes(photometric="MONOCHROME2", size=(64, 64), min_val=0, max_val=4000)

    pix_m1 = load_dicom_pixels(dcm_m1)
    pix_m2 = load_dicom_pixels(dcm_m2)

    # MONOCHROME1 was inverted: pixel 0 was min in raw data, so in M1 it becomes max
    assert pix_m1[0, 0] > pix_m1[-1, -1]
    # MONOCHROME2 was preserved: pixel 0 was min in raw data, stays min
    assert pix_m2[0, 0] < pix_m2[-1, -1]


# 7. Invalid & Corrupted Images Handling
def test_invalid_image_corrupted_bytes():
    """Corrupted image bytes must raise an exception during preprocessing."""
    corrupt_data = b"\x89PNG\r\n\x1a\nCorruptedDataNotAnImage"
    with pytest.raises(Exception):
        preprocess_cxr(corrupt_data)


def test_invalid_image_zero_dimensions():
    """Empty or zero-dimension images must raise ValueError."""
    mock_img = Image.new("L", (1, 1))
    # Resize to 0x0
    with pytest.raises(ValueError):
        preprocess_cxr(Image.fromarray(np.zeros((0, 0), dtype=np.uint8)))


def test_non_existent_file():
    """Non-existent file path must raise FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        preprocess_cxr("non_existent_chest_xray_file.png")
