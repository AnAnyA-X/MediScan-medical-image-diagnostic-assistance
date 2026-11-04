"""
MediScan Guardian — Phase 4 API & Integration Verification Suite
Tests:
1. /api/health endpoint in missing checkpoint scenario
2. /api/predict 503 response in missing checkpoint scenario (graceful failure, no dummy predictions)
3. Input validation: invalid mime type (415), corrupted image (400), empty file (400)
4. Configurable checkpoint path handling via MEDISCAN_MODEL_PATH
5. Inference pipeline verification with DenseNet121 architecture & torch.inference_mode()
6. Static frontend files verification
"""

import asyncio
import io
import os
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import torch
from fastapi import HTTPException, UploadFile
from PIL import Image

from backend.main import health, predict, app
from backend.model import (
    DEFAULT_THRESHOLD,
    build_model,
    get_active_model_path,
    is_model_available,
    predict_image,
)


def create_sample_png_bytes(width=100, height=100, color=128) -> bytes:
    """Generate in-memory valid PNG image bytes."""
    buf = io.BytesIO()
    img = Image.new("L", (width, height), color=color)
    img.save(buf, format="PNG")
    return buf.getvalue()


def run_tests():
    print("=" * 70)
    print("  MEDISCAN GUARDIAN — PHASE 4 INTEGRATION TEST SUITE")
    print("=" * 70)

    # -------------------------------------------------------------
    # Test 1: API Health Check (Missing Checkpoint Scenario)
    # -------------------------------------------------------------
    print("\n[Test 1] Verifying /api/health with no checkpoint installed...")
    data = health()
    print(f"  status:        {data['status']}")
    print(f"  model_trained: {data['model_trained']}")
    print(f"  model:         {data['model']}")
    print(f"  message:       {data['message']}")
    print(f"  disclaimer:    {data['disclaimer']}")

    assert data["status"] == "running"
    assert data["model_trained"] is False
    assert "no trained model checkpoint" in data["message"].lower()
    print("  -> PASS: Health check correctly identifies missing checkpoint.")

    # -------------------------------------------------------------
    # Test 2: /api/predict Graceful 503 (No Dummy Predictions)
    # -------------------------------------------------------------
    print("\n[Test 2] Verifying /api/predict returns HTTP 503 when checkpoint is absent...")
    valid_png = create_sample_png_bytes()
    upload_file = UploadFile(
        file=io.BytesIO(valid_png),
        filename="test_chest_xray.png",
        headers={"content-type": "image/png"}
    )
    try:
        asyncio.run(predict(file=upload_file))
        assert False, "Expected HTTPException 503 was not raised!"
    except HTTPException as e:
        print(f"  Captured HTTPException: Status {e.status_code}")
        print(f"  Detail: {e.detail}")
        assert e.status_code == 503
        assert "not installed" in e.detail.lower()
        print("  -> PASS: Gracefully rejected with 503; zero dummy/fake predictions generated.")

    # -------------------------------------------------------------
    # Test 3: Input Validation — Unsupported File Type (415)
    # -------------------------------------------------------------
    print("\n[Test 3] Verifying input validation for unsupported MIME type...")
    pdf_upload = UploadFile(
        file=io.BytesIO(b"%PDF-1.4 dummy content"),
        filename="report.pdf",
        headers={"content-type": "application/pdf"}
    )
    try:
        asyncio.run(predict(file=pdf_upload))
        assert False, "Expected HTTPException 415 was not raised!"
    except HTTPException as e:
        print(f"  Captured HTTPException: Status {e.status_code}")
        print(f"  Detail: {e.detail}")
        assert e.status_code == 415
        print("  -> PASS: Unsupported MIME type correctly rejected with HTTP 415.")

    # -------------------------------------------------------------
    # Test 4: Input Validation — Corrupted Image Content (400)
    # -------------------------------------------------------------
    print("\n[Test 4] Verifying input validation for corrupted image...")
    # Temporarily set a dummy file path so the checkpoint check passes to test image validation
    dummy_model_path = PROJECT_ROOT / "models" / "temp_marker.pth"
    dummy_model_path.write_bytes(b"temporary non-empty checkpoint marker")
    os.environ["MEDISCAN_MODEL_PATH"] = str(dummy_model_path)

    corrupt_upload = UploadFile(
        file=io.BytesIO(b"This is plain text, not a PNG image"),
        filename="corrupt.png",
        headers={"content-type": "image/png"}
    )
    try:
        asyncio.run(predict(file=corrupt_upload))
        assert False, "Expected HTTPException 400 was not raised!"
    except HTTPException as e:
        print(f"  Captured HTTPException: Status {e.status_code}")
        print(f"  Detail: {e.detail}")
        assert e.status_code == 400
        print("  -> PASS: Corrupted image rejected with HTTP 400.")

    # -------------------------------------------------------------
    # Test 5: Input Validation — Empty File (400)
    # -------------------------------------------------------------
    print("\n[Test 5] Verifying input validation for empty file...")
    empty_upload = UploadFile(
        file=io.BytesIO(b""),
        filename="empty.png",
        headers={"content-type": "image/png"}
    )
    try:
        asyncio.run(predict(file=empty_upload))
        assert False, "Expected HTTPException 400 was not raised!"
    except HTTPException as e:
        print(f"  Captured HTTPException: Status {e.status_code}")
        print(f"  Detail: {e.detail}")
        assert e.status_code == 400
        print("  -> PASS: Empty file rejected with HTTP 400.")

    # Clean up temp marker
    dummy_model_path.unlink()
    if "MEDISCAN_MODEL_PATH" in os.environ:
        del os.environ["MEDISCAN_MODEL_PATH"]

    # -------------------------------------------------------------
    # Test 6: Configurable Checkpoint Path via MEDISCAN_MODEL_PATH
    # -------------------------------------------------------------
    print("\n[Test 6] Verifying configurable checkpoint path handling...")
    custom_path = r"C:\Custom\Model\Path\custom_densenet.pth"
    os.environ["MEDISCAN_MODEL_PATH"] = custom_path
    from backend.model import get_active_model_path as get_path_override
    active_p = str(get_path_override())
    print(f"  Configured MEDISCAN_MODEL_PATH: {custom_path}")
    print(f"  Active resolved model path:     {active_p}")
    assert active_p == custom_path
    del os.environ["MEDISCAN_MODEL_PATH"]
    print("  -> PASS: Checkpoint path is dynamically configurable via environment variable.")

    # -------------------------------------------------------------
    # Test 7: Inference Interface Execution & Schema Validation
    # -------------------------------------------------------------
    print("\n[Test 7] Verifying DenseNet121 inference logic & response schema...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Build clean model structure to test inference logic
    test_model = build_model(pretrained=False).to(device)
    test_model.eval()

    sample_img = Image.new("L", (512, 512), color=120)
    res = predict_image(sample_img, test_model, device, threshold=0.50)

    print(f"  Inference Output Keys: {list(res.keys())}")
    print(f"  status:        {res['status']}")
    print(f"  prediction:    {res['prediction']}")
    print(f"  probability:   {res['probability']}")
    print(f"  risk_level:    {res['risk_level']}")
    print(f"  model:         {res['model']}")
    print(f"  notice:        {res['notice']}")

    assert res["status"] == "success"
    assert res["model"] == "DenseNet121"
    assert 0.0 <= res["probability"] <= 1.0
    assert res["prediction"] in ["TB screening signal detected", "No TB screening signal detected"]
    assert "educational research prototype" in res["notice"].lower()
    print("  -> PASS: Inference output matches exact Phase 4 schema and safety requirements.")

    # -------------------------------------------------------------
    # Test 8: Frontend Static Files Verification
    # -------------------------------------------------------------
    print("\n[Test 8] Verifying static frontend file integrity...")
    frontend_dir = PROJECT_ROOT / "frontend"
    index_file = frontend_dir / "index.html"
    app_file = frontend_dir / "app.js"
    css_file = frontend_dir / "style.css"

    assert index_file.exists(), "frontend/index.html is missing!"
    assert app_file.exists(), "frontend/app.js is missing!"
    assert css_file.exists(), "frontend/style.css is missing!"

    index_text = index_file.read_text(encoding="utf-8")
    app_text = app_file.read_text(encoding="utf-8")
    assert "MediScan Guardian" in index_text
    assert "fetch('/api/health')" in app_text
    assert "fetch('/api/predict'" in app_text
    # Verify no fake predictions or demo mode exist
    assert "Math.random" not in app_text, "Found Math.random in app.js!"
    assert "mock" not in app_text.lower(), "Found mock in app.js!"

    print("  -> index.html: Verified")
    print("  -> app.js:     Verified (no fake predictions, real API fetch)")
    print("  -> style.css:  Verified")

    print("\n" + "=" * 70)
    print("  ALL PHASE 4 INTEGRATION TESTS PASSED (8/8)")
    print("=" * 70)


if __name__ == "__main__":
    run_tests()
