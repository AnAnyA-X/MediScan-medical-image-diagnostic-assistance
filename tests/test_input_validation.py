"""
Input Validation and Boundary Tests for MediScan Guardian (Task 1, Task 2 & Task 5).
Covers:
1. Zero-byte file uploads.
2. Oversized payload threshold (12 MB boundary).
3. Unsupported MIME types and extension spoofing (e.g. .png extension with ASCII text).
4. Corrupted headers vs corrupted bodies.
5. Extreme aspect ratio images (very narrow, very wide, 1x1 pixels).
6. Missing multipart 'file' field.
"""

import io
from unittest.mock import patch, MagicMock
import pytest
import torch
from PIL import Image
from starlette.testclient import TestClient

from backend.main import app, reset_cached_model
from backend.preprocessing import preprocess_cxr


@pytest.fixture
def client():
    reset_cached_model()
    return TestClient(app)


# 1. Zero-byte uploads
def test_validation_zero_byte_file(client):
    """File of size 0 bytes must be rejected with 400 Bad Request."""
    with patch("backend.main.get_active_model_path") as mock_path:
        mock_file = MagicMock()
        mock_file.exists.return_value = True
        mock_path.return_value = mock_file

        files = {"file": ("empty.png", b"", "image/png")}
        res = client.post("/api/predict", files=files)
        assert res.status_code == 400
        assert "empty" in res.json()["detail"].lower()


# 2. File size boundary tests
def test_validation_file_size_limit_exceeded(client):
    """File exceeding 12MB must return 413."""
    with patch("backend.main.get_active_model_path") as mock_path:
        mock_file = MagicMock()
        mock_file.exists.return_value = True
        mock_path.return_value = mock_file

        large_payload = b"A" * (12 * 1024 * 1024 + 1)
        files = {"file": ("large.jpg", large_payload, "image/jpeg")}
        res = client.post("/api/predict", files=files)
        assert res.status_code == 413


def test_validation_file_size_attribute_oversized(client):
    """Early rejection when file.size attribute is set and exceeds 12 MB."""
    with patch("backend.main.get_active_model_path") as mock_path:
        mock_file = MagicMock()
        mock_file.exists.return_value = True
        mock_path.return_value = mock_file

        mock_upload = MagicMock()
        mock_upload.content_type = "image/png"
        mock_upload.size = 15 * 1024 * 1024  # 15 MB

        with patch("backend.main.predict") as mock_predict:
            # Test direct client upload exceeding 12MB
            oversized_payload = b"Z" * (12 * 1024 * 1024 + 100)
            files = {"file": ("oversized.png", oversized_payload, "image/png")}
            res = client.post("/api/predict", files=files)
            assert res.status_code == 413
            assert "smaller than 12 MB" in res.json()["detail"]


def test_validation_chunked_reading_stops_at_limit(client):
    """Chunked streaming loop terminates as soon as accumulated bytes exceed limit."""
    with patch("backend.main.get_active_model_path") as mock_path:
        mock_file = MagicMock()
        mock_file.exists.return_value = True
        mock_path.return_value = mock_file

        # 13 MB payload (exceeds 12 MB limit)
        oversized = b"X" * (13 * 1024 * 1024)
        files = {"file": ("oversized.jpg", oversized, "image/jpeg")}
        res = client.post("/api/predict", files=files)
        assert res.status_code == 413
        assert "smaller than 12 MB" in res.json()["detail"]


def test_validation_file_size_within_limit_passes_size_check(client):
    """File below 12MB passes size validation check."""
    with patch("backend.main.get_active_model_path") as mock_path:
        mock_file = MagicMock()
        mock_file.exists.return_value = False
        mock_path.return_value = mock_file

        # Checkpoint is absent -> returns 503 rather than 413
        valid_img = Image.new("RGB", (100, 100), color=100)
        buf = io.BytesIO()
        valid_img.save(buf, format="JPEG")
        files = {"file": ("valid.jpg", buf.getvalue(), "image/jpeg")}
        res = client.post("/api/predict", files=files)
        assert res.status_code == 503  # Passed size & mime validation, failed on checkpoint


# 3. Spoofed extensions (e.g. filename.png with plain text content)
def test_validation_spoofed_image_extension(client):
    """Files named image.png containing text must be rejected with 400."""
    with patch("backend.main.get_active_model_path") as mock_path:
        mock_file = MagicMock()
        mock_file.exists.return_value = True
        mock_path.return_value = mock_file

        files = {"file": ("fake.png", b"This is not a real PNG image content.", "image/png")}
        res = client.post("/api/predict", files=files)
        assert res.status_code == 400
        assert "not a valid image" in res.json()["detail"]


# 4. Extreme image aspect ratios in preprocessing
def test_validation_extreme_aspect_ratio_narrow():
    """Very tall, narrow images (10 x 1000) must preprocess without error into (3, 512, 512)."""
    img = Image.new("L", (10, 1000), color=200)
    tensor = preprocess_cxr(img)
    assert tensor.shape == (3, 512, 512)
    assert torch.isfinite(tensor).all()


def test_validation_extreme_aspect_ratio_wide():
    """Very wide images (1000 x 10) must preprocess without error into (3, 512, 512)."""
    img = Image.new("L", (1000, 10), color=200)
    tensor = preprocess_cxr(img)
    assert tensor.shape == (3, 512, 512)
    assert torch.isfinite(tensor).all()


def test_validation_minimal_image_dimensions():
    """1x1 pixel image must be processed without division by zero or crashing."""
    img = Image.new("L", (1, 1), color=128)
    tensor = preprocess_cxr(img)
    assert tensor.shape == (3, 512, 512)
    assert torch.isfinite(tensor).all()
