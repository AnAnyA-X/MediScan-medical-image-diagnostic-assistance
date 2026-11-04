# MediScan Guardian — Phase 5 Test Results

**Execution Date:** 2026-11-04  
**Environment:** Python 3.13.7 (Win32), PyTorch 2.15.0+cpu, Pytest 9.1.1  
**Target Branch:** `phase5`  
**Summary:** 58 PASSED, 0 FAILED, 1 BLOCKED/SKIPPED (Total: 59 Test Cases)  

---

## 1. Test Execution Summary

```text
============================= test session starts =============================
platform win32 -- Python 3.13.7, pytest-9.1.1, pluggy-1.6.0
rootdir: C:\Users\santh\Desktop\MediScan-medical-image-diagnostic-assistance
collected 59 items

tests/test_backend_api.py ...................                           [ 30%]
tests/test_dataset_pipeline.py .                                        [ 32%]
tests/test_frontend_integration.py .....                                [ 40%]
tests/test_input_validation.py .........                                [ 55%]
tests/test_model_pipeline.py ..........s                                [ 72%]
tests/test_preprocessing.py ................                            [100%]

================= 58 passed, 1 skipped, 23 warnings in 33.28s =================
```

---

## 2. Granular Test Results Table

| Test ID | Test Name | Category | Expected Result | Actual Result | Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `TC-API-01` | `test_backend_startup_and_availability` | Backend | HTTP 200, `status: running` | HTTP 200, `{"status": "running", ...}` | **PASS** |
| `TC-API-02` | `test_api_health_endpoint_contract` | Backend | Health JSON with model status | HTTP 200, schema fully conformant | **PASS** |
| `TC-API-03` | `test_root_serves_frontend_index_html` | Backend | HTTP 200, `text/html` Content-Type | HTTP 200, HTML contains brand mark | **PASS** |
| `TC-API-04` | `test_static_assets_serving` | Backend | HTTP 200 for `/app.js` and `/style.css` | HTTP 200, assets served accurately | **PASS** |
| `TC-API-05` | `test_predict_endpoint_missing_checkpoint_returns_503` | Backend | HTTP 503 when weights missing | HTTP 503, message mentions train.py | **PASS** |
| `TC-API-06` | `test_predict_endpoint_missing_file_parameter` | Backend | HTTP 422 Unprocessable Entity | HTTP 422 returned | **PASS** |
| `TC-API-07` | `test_predict_endpoint_invalid_payload_json` | Backend | HTTP 422 Unprocessable Entity | HTTP 422 returned | **PASS** |
| `TC-API-08` | `test_predict_endpoint_unsupported_formats[PDF]` | Backend | HTTP 415 Unsupported Media Type | HTTP 415 returned | **PASS** |
| `TC-API-09` | `test_predict_endpoint_unsupported_formats[TXT]` | Backend | HTTP 415 Unsupported Media Type | HTTP 415 returned | **PASS** |
| `TC-API-10` | `test_predict_endpoint_unsupported_formats[EXE]` | Backend | HTTP 415 Unsupported Media Type | HTTP 415 returned | **PASS** |
| `TC-API-11` | `test_predict_endpoint_unsupported_formats[CSV]` | Backend | HTTP 415 Unsupported Media Type | HTTP 415 returned | **PASS** |
| `TC-API-12` | `test_predict_endpoint_empty_file` | Backend | HTTP 400 Bad Request | HTTP 400, "Uploaded file is empty" | **PASS** |
| `TC-API-13` | `test_predict_endpoint_corrupted_image` | Backend | HTTP 400 Bad Request | HTTP 400, "not a valid image" | **PASS** |
| `TC-API-14` | `test_predict_endpoint_oversized_file` | Backend | HTTP 413 Payload Too Large | HTTP 413, "smaller than 12 MB" | **PASS** |
| `TC-API-15` | `test_predict_endpoint_zero_dimension_image` | Backend | HTTP 400 Bad Request | HTTP 400 returned | **PASS** |
| `TC-API-16` | `test_predict_endpoint_server_exception_handling` | Backend | HTTP 500 without leaking internals | HTTP 500, sanitized generic message | **PASS** |
| `TC-API-17` | `test_nonexistent_api_route` | Backend | HTTP 404 Not Found | HTTP 404 returned | **PASS** |
| `TC-API-18` | `test_method_not_allowed_or_not_found` | Backend | HTTP 404 or 405 on invalid method | HTTP 404/405 handled appropriately | **PASS** |
| `TC-DAT-01` | `test_dataset_pipeline` | Dataset | End-to-end dataset integrity check | All splits resolved, 0 leakage verified | **PASS** |
| `TC-INT-01` | `test_frontend_dom_elements_consistency` | Integration | DOM element IDs in app.js exist in HTML | All 6 IDs match 100% | **PASS** |
| `TC-INT-02` | `test_frontend_api_routes_match_backend` | Integration | JS fetch URLs match API endpoints | `/api/health` and `/api/predict` match | **PASS** |
| `TC-INT-03` | `test_frontend_response_schema_compatibility` | Integration | API response matches UI array slicing | `data.results` is array of signals | **PASS** |
| `TC-INT-04` | `test_frontend_error_detail_contract` | Integration | `detail` field present in non-200 | `detail` string extracted and verified | **PASS** |
| `TC-INT-05` | `test_cors_headers_present` | Integration | CORS middleware headers present | Origin reflected with credentials | **PASS** |
| `TC-VAL-01` | `test_validation_zero_byte_file` | Validation | Rejection of 0-byte file (400) | HTTP 400 returned | **PASS** |
| `TC-VAL-02` | `test_validation_file_size_limit_exceeded` | Validation | 12MB limit enforcement (413) | HTTP 413 returned | **PASS** |
| `TC-VAL-03` | `test_validation_file_size_attribute_oversized` | Validation | Fast rejection on oversized size attribute | HTTP 413 returned | **PASS** |
| `TC-VAL-04` | `test_validation_chunked_reading_stops_at_limit` | Validation | Streaming stops immediately at 12MB limit | HTTP 413 returned, zero excess buffering | **PASS** |
| `TC-VAL-05` | `test_validation_file_size_within_limit_passes` | Validation | Normal size files pass size check | Size check passed, proceeded to route | **PASS** |
| `TC-VAL-06` | `test_validation_spoofed_image_extension` | Validation | Rejection of fake `.png` text file | HTTP 400 returned | **PASS** |
| `TC-VAL-07` | `test_validation_extreme_aspect_ratio_narrow` | Validation | Preprocess 10x1000 image to (3,512,512) | Shape (3, 512, 512), all finite | **PASS** |
| `TC-VAL-08` | `test_validation_extreme_aspect_ratio_wide` | Validation | Preprocess 1000x10 image to (3,512,512) | Shape (3, 512, 512), all finite | **PASS** |
| `TC-VAL-09` | `test_validation_minimal_image_dimensions` | Validation | Preprocess 1x1 image without crashing | Shape (3, 512, 512), all finite | **PASS** |
| `TC-MOD-01` | `test_build_model_architecture` | Model | DenseNet121 with (1024 -> 1) head | Linear head confirmed, in=1024, out=1 | **PASS** |
| `TC-MOD-02` | `test_checkpoint_path_resolution` | Model | Priority resolution for active model | Prioritizes `densenet121_active_tb.pth` | **PASS** |
| `TC-MOD-03` | `test_trained_checkpoint_current_status` | Model | Returns boolean existence | Confirms checkpoint absence (Phase 3) | **PASS** |
| `TC-MOD-04` | `test_load_trained_model_missing_checkpoint` | Model | FileNotFoundError with training hint | FileNotFoundError raised as expected | **PASS** |
| `TC-MOD-05` | `test_load_trained_model_corrupt_checkpoint` | Model | KeyError on corrupted dictionary | KeyError raised as expected | **PASS** |
| `TC-MOD-06` | `test_device_selection_defaults` | Model | CUDA if available else CPU | Resolves cleanly without errors | **PASS** |
| `TC-MOD-07` | `test_model_forward_pass_dimensions` | Model | Forward pass `(2, 3, 512, 512)` -> `(2, 1)` | Shape `torch.Size([2, 1])`, finite | **PASS** |
| `TC-MOD-08` | `test_model_eval_mode` | Model | `model.eval()`, `training == False` | Eval mode verified across all submodules | **PASS** |
| `TC-MOD-09` | `test_predict_image_output_contract` | Model | Sigmoid, risk levels, labels | Risk levels (High/Mod/Low) verified | **PASS** |
| `TC-MOD-10` | `test_genuine_model_inference_status` | Model | Real inference with genuine weights | **BLOCKED (SKIPPED)** — Awaiting Phase 3 | **BLOCKED** |
| `TC-PRE-01` | `test_preprocess_cxr_from_pil_image` | Preprocessing | Preprocessing from PIL.Image | Shape (3, 512, 512), dtype float32 | **PASS** |
| `TC-PRE-02` | `test_preprocess_cxr_from_bytes` | Preprocessing | Preprocessing from raw bytes | Shape (3, 512, 512), all finite | **PASS** |
| `TC-PRE-03` | `test_preprocess_cxr_from_bytesio` | Preprocessing | Preprocessing from BytesIO stream | Shape (3, 512, 512), all finite | **PASS** |
| `TC-PRE-04` | `test_preprocess_cxr_from_filepath` | Preprocessing | Preprocessing from file path | Shape (3, 512, 512), all finite | **PASS** |
| `TC-PRE-05` | `test_preprocess_standard_formats[PNG]` | Preprocessing | PNG loading and normalization | Shape (3, 512, 512), all finite | **PASS** |
| `TC-PRE-06` | `test_preprocess_standard_formats[JPEG]` | Preprocessing | JPEG loading and normalization | Shape (3, 512, 512), all finite | **PASS** |
| `TC-PRE-07` | `test_preprocess_standard_formats[WEBP]` | Preprocessing | WEBP loading and normalization | Shape (3, 512, 512), all finite | **PASS** |
| `TC-PRE-08` | `test_grayscale_conversion_rgba` | Preprocessing | RGBA converted to grayscale | Normalized channels finite, no artifacts | **PASS** |
| `TC-PRE-09` | `test_letterbox_resizing_portrait` | Preprocessing | Aspect ratio letterboxing (portrait) | Horizontal zero padding verified | **PASS** |
| `TC-PRE-10` | `test_letterbox_resizing_landscape` | Preprocessing | Aspect ratio letterboxing (landscape) | Vertical zero padding verified | **PASS** |
| `TC-PRE-11` | `test_tensor_normalization_and_finiteness` | Preprocessing | Normalization within ImageNet range | Zero NaN/Inf, dtype float32 | **PASS** |
| `TC-PRE-12` | `test_is_dicom_file_detection` | Preprocessing | DICOM header and extension detection | Accurately identifies DICOM vs PNG/text | **PASS** |
| `TC-PRE-13` | `test_dicom_monochrome1_inversion` | Preprocessing | MONOCHROME1 inverted, M2 preserved | Polarity inversion validated | **PASS** |
| `TC-PRE-14` | `test_invalid_image_corrupted_bytes` | Preprocessing | Exception raised on corrupt bytes | Exception caught and handled | **PASS** |
| `TC-PRE-15` | `test_invalid_image_zero_dimensions` | Preprocessing | ValueError raised on 0-dim image | ValueError raised as expected | **PASS** |
| `TC-PRE-16` | `test_non_existent_file` | Preprocessing | FileNotFoundError on missing file | FileNotFoundError raised as expected | **PASS** |

---

## 3. Blocked / Skipped Test Analysis

- **Test ID:** `TC-MOD-10` (`test_genuine_model_inference_status`)
- **Reason:** Model checkpoint `models/densenet121_active_tb.pth` has not yet been produced because Phase 3 model training is actively executing.
- **Strict Compliance:** In accordance with project instructions, no synthetic weights were generated, no random classifier was injected, and no fake inference was recorded. The test gracefully skips and records `BLOCKED` status until genuine weights are deposited.
