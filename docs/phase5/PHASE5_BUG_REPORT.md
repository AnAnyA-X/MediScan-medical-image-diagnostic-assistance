# MediScan Guardian — Phase 5 Bug Report

**Execution Date:** 2026-11-04  
**Target Branch:** `phase5`  
**Quality Assurance Lead:** Phase 5 Testing Engineer  

---

## 1. Summary of Identified & Resolved Issues

| Bug ID | Title | Component | Severity | Status |
| :--- | :--- | :--- | :--- | :--- |
| `BUG-P5-01` | Frontend TypeError on API Response Parsing | `frontend/app.js` & `backend/main.py` | **High** | **FIXED** |
| `BUG-P5-02` | Inconsistent Checkpoint Resolution Path in API | `backend/main.py` | **Medium** | **FIXED** |
| `BUG-P5-03` | Division by Zero in Preprocessing on Zero-Dimension Images | `backend/preprocessing.py` | **Medium** | **FIXED** |
| `BUG-P5-04` | Unhandled Inference Exceptions Yielding Generic 500 Responses | `backend/main.py` | **Medium** | **FIXED** |
| `BUG-P5-05` | Empty (0-byte) Uploads Not Explicitly Handled | `backend/main.py` | **Low** | **FIXED** |

---

## 2. Granular Bug Details

### BUG-P5-01: Frontend TypeError on API Response Parsing
- **Component:** `frontend/app.js` (line 28) & `backend/main.py` (line 53)
- **Severity:** High
- **Description:** 
  In `backend/model.py`, `predict_image()` returns a structured dictionary containing metadata (`target`, `risk_level`, `predicted_class`, `probability_active_tb`, and a nested `results` list). In `backend/main.py`, `results` was directly set to this dictionary. When `app.js` executed `data.results.slice(0, 6).map(...)`, it called `.slice()` on a plain Object, throwing `TypeError: data.results.slice is not a function` in the browser console and crashing the user interface display.
- **Root Cause:** Incompatible schema evolution between Phase 2 and Phase 3 where the dictionary structure replaced a flat list without updating the response packaging in `main.py` or defensive array parsing in `app.js`.
- **Fix Implemented:**
  1. Updated `backend/main.py` to extract `signals = prediction.get("results", [])` so `data.results` is always a serializable list of signal objects, while keeping full prediction details under `prediction_details`.
  2. Updated `frontend/app.js` to defensively inspect `data.results`:
     ```javascript
     const signalList = Array.isArray(data.results) 
       ? data.results 
       : (data.results && Array.isArray(data.results.results) ? data.results.results : []);
     ```
- **Verification:** Verified via `tests/test_frontend_integration.py::test_frontend_response_schema_compatibility` (PASSED).

---

### BUG-P5-02: Inconsistent Checkpoint Resolution Path in API
- **Component:** `backend/main.py` (lines 27, 37)
- **Severity:** Medium
- **Description:**
  `backend/model.py` defines `get_active_model_path()` which prioritizes `MODEL_PATH` (`models/densenet121_active_tb.pth`) while falling back to `LEGACY_MODEL_PATH` (`models/densenet121_chestxray.pth`). However, `backend/main.py` called `MODEL_PATH.exists()` directly in both `/api/health` and `/api/predict`. If a legacy or renamed checkpoint existed, the API reported `model_trained: false` and returned 503, bypassing the fallback loader in `model.py`.
- **Root Cause:** Hardcoded reference to `MODEL_PATH` instead of utilizing the helper `get_active_model_path()`.
- **Fix Implemented:** Imported and used `get_active_model_path()` in `backend/main.py` for health checks and inference route gating.
- **Verification:** Verified via `tests/test_model_pipeline.py::test_checkpoint_path_resolution` and `tests/test_backend_api.py::test_predict_endpoint_missing_checkpoint_returns_503` (PASSED).

---

### BUG-P5-03: Division by Zero in Preprocessing on Zero-Dimension Images
- **Component:** `backend/preprocessing.py` (line 145)
- **Severity:** Medium
- **Description:**
  If an image with zero height or zero width (e.g. 0x0 array from corrupted image headers or malformed arrays) was passed to `preprocess_cxr()`, calculating the aspect scaling factor `min(th / h, tw / w)` resulted in an unhandled `ZeroDivisionError: division by zero`.
- **Root Cause:** Missing dimension validation before calculating the aspect scaling ratio.
- **Fix Implemented:** Added an explicit dimension guard before scaling:
  ```python
  if pixel_array.size == 0 or len(pixel_array.shape) < 2 or pixel_array.shape[0] == 0 or pixel_array.shape[1] == 0:
      raise ValueError("Invalid image dimensions: image must have non-zero height and width.")
  ```
- **Verification:** Verified via `tests/test_preprocessing.py::test_invalid_image_zero_dimensions` and `tests/test_backend_api.py::test_predict_endpoint_zero_dimension_image` (PASSED).

---

### BUG-P5-04: Unhandled Inference Exceptions Yielding Generic Responses
- **Component:** `backend/main.py` (line 52)
- **Severity:** Medium
- **Description:**
  During `/api/predict`, any runtime exception from PyTorch (such as CUDA memory allocation errors, device tensor mismatch, or corrupt checkpoint deserialization) would produce an unhandled internal exception, returning an opaque Starlette 500 without a descriptive detail message for the client.
- **Root Cause:** Missing `try...except` wrapper around the model loading and inference execution block.
- **Fix Implemented:** Wrapped the inference block in a try-except statement:
  ```python
  try:
      if _cached_model is None:
          _cached_model, _cached_device = load_trained_model()
      prediction = predict_image(image, _cached_model, _cached_device)
  except FileNotFoundError as exc:
      raise HTTPException(status_code=503, detail=str(exc))
  except Exception as exc:
      raise HTTPException(status_code=500, detail=f"Inference execution failed: {str(exc)}")
  ```
- **Verification:** Verified via `tests/test_backend_api.py::test_predict_endpoint_server_exception_handling` (PASSED).

---

### BUG-P5-05: Empty (0-byte) Uploads Not Explicitly Handled
- **Component:** `backend/main.py` (line 42)
- **Severity:** Low
- **Description:**
  Empty file uploads (0 bytes) fell through to PIL `Image.open()`, raising a generic `UnidentifiedImageError` with message `"This file is not a valid image."` rather than an explicit 400 validation error informing the user that the uploaded file was empty.
- **Root Cause:** Missing check for `len(raw) == 0`.
- **Fix Implemented:** Added an explicit check:
  ```python
  raw = await file.read()
  if len(raw) == 0:
      raise HTTPException(status_code=400, detail="Uploaded file is empty.")
  ```
- **Verification:** Verified via `tests/test_backend_api.py::test_predict_endpoint_empty_file` and `tests/test_input_validation.py::test_validation_zero_byte_file` (PASSED).
