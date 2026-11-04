# MediScan Guardian — Phase 5 Test Plan

**Document Version:** 1.0.0  
**Execution Date:** 2026-11-04  
**Target Branch:** `phase5`  
**System Status:** Complete Quality Assurance & Test Implementation  

---

## 1. Testing Objectives

The primary objective of Phase 5 is to deliver a comprehensive, reproducible, and automated testing suite for MediScan Guardian covering:
1. **Backend API Stability:** Validate application startup, static file serving, endpoint routing, HTTP status code contracts, and error resilience.
2. **Clinical Preprocessing Verification:** Guarantee that Chest X-Ray (CXR) radiograph processing strictly adheres to Phase 7 specifications (DICOM LUT, MONOCHROME1 inversion, MONOCHROME2 preservation, aspect-ratio preserving letterboxing to 512x512, and ImageNet normalization).
3. **Model Pipeline Architecture & Loading:** Confirm DenseNet121 architecture configuration (1024 to 1 linear classifier), checkpoint path resolution, device selection, evaluation mode semantics, and structured output formatting.
4. **Input Validation & Security Boundaries:** Verify request bounds, payload size thresholds (12 MB ceiling), format whitelisting (PNG, JPG, WEBP), and rejection of empty or corrupted payloads.
5. **Frontend / Backend Integration:** Ensure UI element IDs in `index.html` seamlessly connect with `app.js` and that response data structures match UI rendering requirements.
6. **Zero Falsification Mandate:** Real inference validation is kept strictly as `BLOCKED/PENDING` until Phase 3 training completes and the genuine `densenet121_active_tb.pth` checkpoint is produced.

---

## 2. Testing Scope

| Subsystem | Scope Description |
| :--- | :--- |
| **Backend API** | FastAPI startup, health checks (`/api/health`), `/api/predict`, static file serving (`/`, `/app.js`, `/style.css`), HTTP response codes (200, 400, 404, 405, 413, 415, 422, 500, 503). |
| **Image Preprocessing** | Loading PNG, JPG, WEBP, and DICOM from file paths, byte streams, and PIL images; aspect-ratio letterboxing to 512x512; zero padding; MONOCHROME1 inversion; ImageNet channel normalization. |
| **Input Validation** | Zero-byte payloads, files > 12 MB, corrupted headers, spoofed MIME extensions, invalid/extreme aspect ratios, missing request parameters. |
| **Model Pipeline** | DenseNet121 architecture initialization, logit output dimensions `[B, 1]`, checkpoint priority resolution, `FileNotFoundError` handling, risk categorization rules. |
| **Frontend Integration** | DOM selector integrity, API endpoint path agreement, CORS middleware validation, response schema JSON parsing. |
| **Data Integrity** | Patient leakage check, split verification across Train (540), Validation (116), Test (117) datasets. |

---

## 3. Testing Methodology

- **Automated Framework:** Pytest 9.1.1 execution utilizing `starlette.testclient.TestClient`.
- **Unit & Boundary Testing:** Granular function-level tests for mathematical operations (aspect scaling, padding, normalization, sigmoid conversion).
- **Mocking & Isolation:** Synthetic, in-memory DICOM headers and mock file uploads generated dynamically during tests without touching training data or altering repository assets.
- **Negative Testing:** Injection of corrupted byte streams, oversized payloads, non-image files, and missing checkpoint scenarios.
- **Strict Compliance:** No fake weights, no dummy predictions, no demo mode. Real model weights testing is tagged with `pytest.skip` and marked `BLOCKED`.

---

## 4. Test Environment

- **Operating System:** Windows 11 (win32)
- **Python Version:** 3.13.7
- **Core Packages:**
  - `pytest`: 9.1.1
  - `torch`: 2.15.0+cpu
  - `torchvision`: 0.29.2+cpu
  - `fastapi`: 0.162.0
  - `starlette`: 1.10.0
  - `pydicom`: 3.0.2
  - `pillow`: 13.0.0
  - `numpy`: 2.5.4
  - `pandas`: 3.1.0

---

## 5. Test Suite Architecture

```text
tests/
├── test_backend_api.py          # 18 test cases (API startup, endpoints, status codes, errors)
├── test_preprocessing.py        # 16 test cases (DICOM LUT, letterbox, normalization, channels)
├── test_model_pipeline.py       # 11 test cases (DenseNet121, checkpoint resolution, logits, risk)
├── test_input_validation.py     # 7 test cases (0-byte, 12MB limit, spoofed MIME, extreme aspect)
├── test_frontend_integration.py # 5 test cases (DOM bindings, routes, CORS, schema contracts)
└── test_dataset_pipeline.py     # 1 comprehensive end-to-end dataset & leakage validation suite
```

---

## 6. Test Cases Specification

### 6.1 Backend API Tests (`test_backend_api.py`)
- `TC-API-01`: Verify backend startup and availability (`GET /api/health`).
- `TC-API-02`: Validate health response structure (`status`, `model_trained`, `message`).
- `TC-API-03`: Confirm root route serves `index.html` with `text/html` header.
- `TC-API-04`: Confirm static asset delivery (`/app.js`, `/style.css`).
- `TC-API-05`: Validate missing model checkpoint returns 503 with descriptive message.
- `TC-API-06`: Validate missing multipart `file` parameter returns 422.
- `TC-API-07`: Validate JSON payload to `/api/predict` returns 422.
- `TC-API-08` to `TC-API-11`: Validate unsupported file types (PDF, TXT, EXE, CSV) return 415.
- `TC-API-12`: Validate 0-byte file upload returns 400.
- `TC-API-13`: Validate corrupted image byte stream returns 400.
- `TC-API-14`: Validate oversized image (> 12MB) returns 413.
- `TC-API-15`: Validate 0x0 image dimensions return 400.
- `TC-API-16`: Validate internal server exception handling returns 500.
- `TC-API-17`: Validate non-existent API routes return 404.
- `TC-API-18`: Validate method handling on `/api/predict`.

### 6.2 Preprocessing Tests (`test_preprocessing.py`)
- `TC-PRE-01`: Process image from `PIL.Image` instance.
- `TC-PRE-02`: Process image from raw `bytes`.
- `TC-PRE-03`: Process image from `io.BytesIO` stream.
- `TC-PRE-04`: Process image from filesystem path (`str` and `Path`).
- `TC-PRE-05` to `TC-PRE-07`: Format tests for PNG, JPEG, and WEBP.
- `TC-PRE-08`: Verify RGBA transparency conversion to grayscale.
- `TC-PRE-09`: Validate aspect-ratio letterboxing for portrait orientation (200x800).
- `TC-PRE-10`: Validate aspect-ratio letterboxing for landscape orientation (800x200).
- `TC-PRE-11`: Validate output tensor shape `(3, 512, 512)`, dtype `torch.float32`, and finite values.
- `TC-PRE-12`: Validate DICOM magic byte and extension identification.
- `TC-PRE-13`: Validate DICOM MONOCHROME1 polarity inversion.
- `TC-PRE-14`: Validate corrupted bytes exception raising.
- `TC-PRE-15`: Validate zero-dimension array exception raising.
- `TC-PRE-16`: Validate missing file path raises `FileNotFoundError`.

### 6.3 Model Pipeline Tests (`test_model_pipeline.py`)
- `TC-MOD-01`: Validate DenseNet121 architecture (1024 input features, 1 output logit).
- `TC-MOD-02`: Verify checkpoint priority resolution (`densenet121_active_tb.pth` > legacy).
- `TC-MOD-03`: Verify genuine checkpoint presence check.
- `TC-MOD-04`: Verify `load_trained_model` raises `FileNotFoundError` when weights missing.
- `TC-MOD-05`: Verify checkpoint with missing keys raises `KeyError`.
- `TC-MOD-06`: Verify CPU/GPU device selection resolution.
- `TC-MOD-07`: Verify forward pass dimensions with batch `(2, 3, 512, 512)` -> `(2, 1)`.
- `TC-MOD-08`: Verify evaluation mode (`model.eval()`).
- `TC-MOD-09`: Verify probability calculation, class prediction, and risk thresholds (High >= 0.70, Moderate >= 0.40, Low < 0.40).
- `TC-MOD-10`: Verify genuine inference status is marked as BLOCKED pending Phase 3 completion.

### 6.4 Input Validation Tests (`test_input_validation.py`)
- `TC-VAL-01`: 0-byte file upload rejection.
- `TC-VAL-02`: File size limit boundary (12 MB + 1 byte) rejection (413).
- `TC-VAL-03`: File size within limit boundary passes size check.
- `TC-VAL-04`: Spoofed extension (`.png` containing ASCII text) rejection (400).
- `TC-VAL-05`: Extreme tall aspect ratio (10 x 1000) letterboxing.
- `TC-VAL-06`: Extreme wide aspect ratio (1000 x 10) letterboxing.
- `TC-VAL-07`: Minimal image dimension (1 x 1) preprocessing resilience.

### 6.5 Frontend Integration Tests (`test_frontend_integration.py`)
- `TC-INT-01`: HTML DOM IDs (`#fileInput`, `#fileName`, `#analyzeBtn`, `#message`, `#results`, `#apiStatus`) match JavaScript event listeners.
- `TC-INT-02`: Endpoints `/api/health` and `/api/predict` in `app.js` match FastAPI routes.
- `TC-INT-03`: Prediction response JSON schema contract (`results` array, `label`, `probability`, `notice`).
- `TC-INT-04`: Non-200 responses return `detail` string displayed in UI message.
- `TC-INT-05`: CORS headers verification with credentials support.

### 6.6 Dataset & Leakage Tests (`test_dataset_pipeline.py`)
- `TC-DAT-01`: Metadata split presence (Train: 540, Val: 116, Test: 117).
- `TC-DAT-02`: Zero patient-level data leakage across splits.
- `TC-DAT-03`: All 773 local CXR images resolved.
- `TC-DAT-04`: Synthetic cohort batch loading via PyTorch `DataLoader`.
