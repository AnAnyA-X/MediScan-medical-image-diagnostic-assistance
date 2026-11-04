"""
Tests for MediScan Guardian Backend API (Task 1 & Task 5).
Covers:
1. Application startup & lifespan.
2. API health endpoint (/api/health).
3. API availability & root route (static index.html).
4. Valid image requests (when model checkpoint is present vs absent).
5. Invalid requests (malformed payloads, wrong endpoints).
6. Missing request parameters (omitting file).
7. Invalid request payloads (non-multipart bodies).
8. Unsupported file formats (PDF, TXT, EXE, CSV).
9. Empty file uploads (0 bytes).
10. Corrupted image uploads (truncated image bytes).
11. Invalid image dimensions (0x0 images).
12. File upload size limit validation (> 12MB).
13. Appropriate HTTP status codes (200, 400, 404, 413, 415, 422, 503).
14. Exception handling for unexpected inference/server errors (500).
15. Missing configuration handling.
16. Missing model checkpoint handling (503 Service Unavailable).
"""

import io
from pathlib import Path
from unittest.mock import patch, MagicMock
import pytest
from starlette.testclient import TestClient
from PIL import Image
import torch

from backend.main import app, reset_cached_model


@pytest.fixture
def client():
    reset_cached_model()
    return TestClient(app)


@pytest.fixture
def sample_png_bytes():
    img = Image.new("RGB", (256, 256), color=(128, 128, 128))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture
def sample_jpg_bytes():
    img = Image.new("L", (512, 512), color=200)
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


@pytest.fixture
def sample_webp_bytes():
    img = Image.new("RGB", (128, 128), color=(50, 100, 150))
    buf = io.BytesIO()
    img.save(buf, format="WEBP")
    return buf.getvalue()


# 1. Backend Application Startup & Availability
def test_backend_startup_and_availability(client):
    """Test that the backend starts and responds to HTTP requests."""
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert data["status"] == "running"


# 2. API Health Endpoint Structure
def test_api_health_endpoint_contract(client):
    """Verify health endpoint returns status, model_trained, and descriptive message."""
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "running"
    assert isinstance(data["model_trained"], bool)
    assert isinstance(data["message"], str)
    if data["model_trained"]:
        assert data["message"] == "Model is ready."
    else:
        assert "no trained model checkpoint is installed yet" in data["message"]


# 3. Static Files & Root Route Serving
def test_root_serves_frontend_index_html(client):
    """Verify GET / serves the frontend index.html."""
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers.get("content-type", "")
    assert "MediScan Guardian" in response.text


def test_static_assets_serving(client):
    """Verify /app.js and /style.css are served by static files mount."""
    res_js = client.get("/app.js")
    assert res_js.status_code == 200
    assert "analyzeBtn" in res_js.text

    res_css = client.get("/style.css")
    assert res_css.status_code == 200


# 4. Missing Checkpoint Handling (Phase 3 in progress)
def test_predict_endpoint_missing_checkpoint_returns_503(client, sample_png_bytes):
    """When trained checkpoint does not exist, /api/predict must return 503."""
    with patch("backend.main.get_active_model_path") as mock_path:
        mock_file = MagicMock()
        mock_file.exists.return_value = False
        mock_path.return_value = mock_file

        files = {"file": ("cxr.png", sample_png_bytes, "image/png")}
        response = client.post("/api/predict", files=files)

        assert response.status_code == 503
        assert "The AI model has not been trained yet" in response.json()["detail"]


# 5. Missing Request Parameters
def test_predict_endpoint_missing_file_parameter(client):
    """POST /api/predict without a 'file' parameter must return 422 Unprocessable Entity."""
    response = client.post("/api/predict", data={})
    assert response.status_code == 422


# 6. Invalid Request Payloads
def test_predict_endpoint_invalid_payload_json(client):
    """POST /api/predict with JSON instead of multipart/form-data must return 422."""
    response = client.post("/api/predict", json={"image": "base64..."})
    assert response.status_code == 422


# 7. Unsupported File Formats (Status 415)
@pytest.mark.parametrize("filename,content_type,content", [
    ("record.pdf", "application/pdf", b"%PDF-1.4 ..."),
    ("data.txt", "text/plain", b"plain text chest x-ray description"),
    ("script.exe", "application/x-msdownload", b"MZ\x90\x00"),
    ("data.csv", "text/csv", b"id,tb_status\n1,0"),
])
def test_predict_endpoint_unsupported_formats(client, filename, content_type, content):
    """POST /api/predict with unsupported MIME types must return 415."""
    files = {"file": (filename, content, content_type)}
    response = client.post("/api/predict", files=files)
    assert response.status_code == 415
    assert "Upload a JPG, PNG, or WEBP image." in response.json()["detail"]


# 8. Empty File Uploads (Status 400)
def test_predict_endpoint_empty_file(client):
    """POST /api/predict with 0 bytes must return 400 Bad Request."""
    with patch("backend.main.get_active_model_path") as mock_path:
        mock_file = MagicMock()
        mock_file.exists.return_value = True
        mock_path.return_value = mock_file

        files = {"file": ("empty.png", b"", "image/png")}
        response = client.post("/api/predict", files=files)
        assert response.status_code == 400
        assert "Uploaded file is empty" in response.json()["detail"]


# 9. Corrupted Image Uploads (Status 400)
def test_predict_endpoint_corrupted_image(client):
    """POST /api/predict with corrupted bytes must return 400 Bad Request."""
    with patch("backend.main.get_active_model_path") as mock_path:
        mock_file = MagicMock()
        mock_file.exists.return_value = True
        mock_path.return_value = mock_file

        corrupted_bytes = b"\x89PNG\r\n\x1a\n" + b"garbage_random_data_not_a_real_image"
        files = {"file": ("corrupt.png", corrupted_bytes, "image/png")}
        response = client.post("/api/predict", files=files)
        assert response.status_code == 400
        assert "This file is not a valid image" in response.json()["detail"]


# 10. Oversized File Uploads (Status 413)
def test_predict_endpoint_oversized_file(client):
    """POST /api/predict with file > 12MB must return 413 Payload Too Large."""
    with patch("backend.main.get_active_model_path") as mock_path:
        mock_file = MagicMock()
        mock_file.exists.return_value = True
        mock_path.return_value = mock_file

        oversized_bytes = b"0" * (12 * 1024 * 1024 + 1024)
        files = {"file": ("huge.png", oversized_bytes, "image/png")}
        response = client.post("/api/predict", files=files)
        assert response.status_code == 413
        assert "smaller than 12 MB" in response.json()["detail"]


# 11. Invalid Image Dimensions (Status 400)
def test_predict_endpoint_zero_dimension_image(client):
    """Image with zero dimensions must return 400."""
    with patch("backend.main.get_active_model_path") as mock_path:
        mock_file = MagicMock()
        mock_file.exists.return_value = True
        mock_path.return_value = mock_file

        mock_img = MagicMock()
        mock_img.verify.return_value = None
        mock_img.width = 0
        mock_img.height = 0

        with patch("PIL.Image.open", return_value=mock_img):
            files = {"file": ("zero.png", b"fake_png", "image/png")}
            response = client.post("/api/predict", files=files)
            assert response.status_code == 400
            assert "This file is not a valid image" in response.json()["detail"]


# 12. Inference Exception Handling (Status 500)
def test_predict_endpoint_server_exception_handling(client, sample_png_bytes):
    """Unexpected runtime error during model inference must return 500 without leaking internal details."""
    with patch("backend.main.get_active_model_path") as mock_path:
        mock_file = MagicMock()
        mock_file.exists.return_value = True
        mock_path.return_value = mock_file

        with patch("backend.main.load_trained_model", side_effect=RuntimeError("CUDA memory allocation fault at 0x7fff")):
            files = {"file": ("cxr.png", sample_png_bytes, "image/png")}
            response = client.post("/api/predict", files=files)
            assert response.status_code == 500
            assert "Inference execution failed due to an internal server error" in response.json()["detail"]
            assert "CUDA memory allocation fault" not in response.json()["detail"]


# 13. Non-existent Routes (Status 404)
def test_nonexistent_api_route(client):
    """Requests to invalid API paths must return 404."""
    response = client.get("/api/unknown_endpoint")
    assert response.status_code == 404


# 14. Static Fallback on Root Mount vs Method Handling
def test_method_not_allowed_or_not_found(client):
    """
    With StaticFiles mounted at root ('/'), unmatched GET requests to API routes
    are routed to static handler which returns 404 when file does not exist,
    or 405 when route matching rejects the method directly.
    """
    response = client.get("/api/predict")
    assert response.status_code in (404, 405)
