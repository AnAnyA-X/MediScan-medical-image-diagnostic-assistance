"""
Frontend and Backend Integration Tests (Task 4 & Task 5).
Covers:
1. API base URL configuration and route consistency.
2. Request methods (GET /api/health, POST /api/predict).
3. Image upload payload structure (FormData key 'file' matches UploadFile).
4. Response schema contract compatibility (data.results, data.notice, data.detail).
5. Frontend UI element bindings and HTML structure consistency.
6. CORS configuration and header validation.
7. Error state propagation and response parsing.
8. Status display logic (online vs model pending vs unavailable).
"""

import re
from pathlib import Path
from unittest.mock import patch, MagicMock
import pytest
from starlette.testclient import TestClient
from PIL import Image
import io

from backend.main import app, reset_cached_model


@pytest.fixture
def client():
    reset_cached_model()
    return TestClient(app)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FRONTEND_DIR = PROJECT_ROOT / "frontend"


# 1. HTML Element ID Bindings with app.js
def test_frontend_dom_elements_consistency():
    """Verify that all DOM element IDs referenced in app.js exist in index.html."""
    html_content = (FRONTEND_DIR / "index.html").read_text(encoding="utf-8")
    js_content = (FRONTEND_DIR / "app.js").read_text(encoding="utf-8")

    # Extract all document.getElementById('...') from app.js
    id_matches = set(re.findall(r"document\.getElementById\(['\"]([^'\"]+)['\"]\)", js_content))
    assert len(id_matches) > 0

    for elem_id in id_matches:
        pattern = f'id="{elem_id}"'
        assert pattern in html_content, f"Element ID '{elem_id}' referenced in app.js not found in index.html"


# 2. API Routes Contract in app.js
def test_frontend_api_routes_match_backend():
    """Verify endpoints referenced in frontend app.js match backend defined routes."""
    js_content = (FRONTEND_DIR / "app.js").read_text(encoding="utf-8")

    # Check /api/health fetch
    assert "fetch('/api/health')" in js_content or 'fetch("/api/health")' in js_content

    # Check /api/predict fetch
    assert "fetch('/api/predict'" in js_content or 'fetch("/api/predict"' in js_content

    # Check FormData file key
    assert "body.append('file'" in js_content or 'body.append("file"' in js_content


# 3. Response Schema Parsing Contract
def test_frontend_response_schema_compatibility(client):
    """
    Simulate a prediction response and verify all fields expected by frontend
    (data.results array, x.label, x.probability, data.notice) exist and conform.
    """
    mock_prediction = {
        "target": "Active_TB",
        "probability_active_tb": 0.825,
        "probability_control": 0.175,
        "predicted_class": "Active_TB",
        "risk_level": "High Risk",
        "results": [
            {"label": "Active Pulmonary TB", "probability": 0.825},
            {"label": "Normal / Negative Control", "probability": 0.175}
        ]
    }

    with patch("backend.main.get_active_model_path") as mock_path, \
         patch("backend.main.load_trained_model", return_value=(MagicMock(), "cpu")), \
         patch("backend.main.predict_image", return_value=mock_prediction):

        mock_file = MagicMock()
        mock_file.exists.return_value = True
        mock_path.return_value = mock_file

        img = Image.new("RGB", (100, 100), color=(120, 120, 120))
        buf = io.BytesIO()
        img.save(buf, format="JPEG")

        files = {"file": ("cxr.jpg", buf.getvalue(), "image/jpeg")}
        response = client.post("/api/predict", files=files)

        assert response.status_code == 200
        data = response.json()

        # Check frontend compatibility
        assert "results" in data
        assert isinstance(data["results"], list)
        assert len(data["results"]) == 2
        for item in data["results"]:
            assert "label" in item
            assert "probability" in item
            assert isinstance(item["label"], str)
            assert isinstance(item["probability"], (int, float))

        assert "notice" in data
        assert isinstance(data["notice"], str)


# 4. Error Detail Contract
def test_frontend_error_detail_contract(client):
    """Frontend expects `data.detail` on non-200 responses (e.g. 400 or 503) to display in UI."""
    response = client.post("/api/predict", files={"file": ("test.png", b"", "image/png")})
    assert response.status_code in (400, 503)
    data = response.json()
    assert "detail" in data
    assert len(data["detail"]) > 0


# 5. CORS Configuration Verification
def test_cors_headers_present(client):
    """
    Verify that CORS middleware headers are properly returned.
    When allow_credentials=True, Starlette CORS middleware correctly reflects the request Origin.
    """
    headers = {
        "Origin": "http://localhost:3000",
        "Access-Control-Request-Method": "POST",
        "Access-Control-Request-Headers": "content-type",
    }
    response = client.options("/api/predict", headers=headers)
    assert response.status_code == 200
    assert response.headers.get("access-control-allow-origin") in ("*", "http://localhost:3000")
    assert response.headers.get("access-control-allow-credentials") == "true"
    assert "POST" in response.headers.get("access-control-allow-methods", "")
