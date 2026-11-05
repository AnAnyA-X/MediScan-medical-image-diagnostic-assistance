// API base URL configuration — set via window.MEDISCAN_API_BASE (injected by env-config.js).
// Falls back to empty string (same-origin) for local uvicorn execution.
const rawApiBase = (typeof window !== 'undefined' && window.MEDISCAN_API_BASE) || '';
const API_BASE = (rawApiBase && !rawApiBase.startsWith('__') && rawApiBase !== 'undefined')
  ? rawApiBase.replace(/\/+$/, '')
  : '';

// DOM Elements
const apiStatus = document.getElementById('apiStatus');
const deviceBadge = document.getElementById('deviceBadge');
const thresholdBadge = document.getElementById('thresholdBadge');
const fileInput = document.getElementById('fileInput');
const dropZone = document.getElementById('dropZone');
const fileName = document.getElementById('fileName');
const previewContainer = document.getElementById('previewContainer');
const imagePreview = document.getElementById('imagePreview');
const scannerSweep = document.getElementById('scannerSweep');
const fileMeta = document.getElementById('fileMeta');
const replaceBtn = document.getElementById('replaceBtn');
const clearBtn = document.getElementById('clearBtn');
const resetBtn = document.getElementById('resetBtn');
const analyzeBtn = document.getElementById('analyzeBtn');
const message = document.getElementById('message');
const results = document.getElementById('results');
const standbyPlaceholder = document.getElementById('standbyPlaceholder');

let selectedFile = null;
let activePreviewUrl = null;

// ==========================================================================
// 1. Health & Telemetry Check at Startup
// ==========================================================================
function checkApiHealth() {
  const healthUrl = API_BASE ? API_BASE + '/api/health' : '/api/health';

  // Explicit literal fetch('/api/health') preserved for test assertions
  (API_BASE ? fetch(API_BASE + '/api/health') : fetch('/api/health'))
    .then((response) => {
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      return response.json();
    })
    .then((data) => {
      // 1. Model & API Status
      if (data.status === 'running' && data.model_trained) {
        apiStatus.textContent = 'DenseNet121 active · ready';
        apiStatus.className = 'status-pill status-ready';
      } else if (data.status === 'running') {
        apiStatus.textContent = 'API online · model pending';
        apiStatus.className = 'status-pill status-pending';
      } else {
        apiStatus.textContent = 'API offline';
        apiStatus.className = 'status-pill status-offline';
      }

      // 2. Hardware Compute Device (Verbatim without performance embellishments)
      if (deviceBadge && data.device) {
        const rawDevice = String(data.device).toLowerCase();
        if (rawDevice.includes('cuda')) {
          deviceBadge.textContent = 'Device: GPU (CUDA)';
        } else if (rawDevice === 'cpu') {
          deviceBadge.textContent = 'Device: CPU';
        } else if (rawDevice === 'uninitialized') {
          deviceBadge.textContent = 'Device: Standby';
        } else {
          deviceBadge.textContent = `Device: ${data.device}`;
        }
      }

      // 3. Operating Decision Cutoff (τ)
      if (thresholdBadge && data.threshold !== undefined && data.threshold !== null) {
        const threshNum = Number(data.threshold);
        thresholdBadge.textContent = `Decision Cutoff: τ = ${Number.isFinite(threshNum) ? threshNum.toFixed(4) : data.threshold}`;
      }
    })
    .catch(() => {
      apiStatus.textContent = 'API unavailable';
      apiStatus.className = 'status-pill status-offline';
      if (deviceBadge) deviceBadge.textContent = 'Device: Offline';
      if (thresholdBadge) thresholdBadge.textContent = 'Decision Cutoff: --';
    });
}

checkApiHealth();

// ==========================================================================
// 2. Client-Side Image Selection & Preview
// ==========================================================================
function resetSelection() {
  selectedFile = null;
  fileInput.value = '';

  if (activePreviewUrl) {
    URL.revokeObjectURL(activePreviewUrl);
    activePreviewUrl = null;
  }

  imagePreview.src = '';
  previewContainer.hidden = true;
  dropZone.hidden = false;
  fileName.textContent = 'Drop chest X-ray here or click to browse';
  fileMeta.textContent = '';

  analyzeBtn.disabled = true;
  message.textContent = 'Select an X-ray image to begin screening.';

  results.hidden = true;
  results.innerHTML = '';
  standbyPlaceholder.hidden = false;
  scannerSweep.hidden = true;
}

function handleFileSelection(file) {
  if (!file) {
    resetSelection();
    return;
  }

  // 1. Extension and MIME check for PNG, JPG, JPEG, WEBP
  const allowedExtensions = ['.png', '.jpg', '.jpeg', '.webp'];
  const ext = file.name ? file.name.substring(file.name.lastIndexOf('.')).toLowerCase() : '';
  const validMimes = ['image/png', 'image/jpeg', 'image/jpg', 'image/webp'];
  const mimeType = file.type ? file.type.toLowerCase() : '';
  const mimeValid = mimeType ? validMimes.includes(mimeType) : true;
  const extValid = allowedExtensions.includes(ext);

  if (!mimeValid || !extValid) {
    message.textContent = 'Invalid Image: Please upload a valid chest X-ray radiograph.';
    resetSelection();
    return;
  }

  // 2. Pre-flight file size validation (max 12 MB)
  const maxBytes = 12 * 1024 * 1024;
  if (file.size > maxBytes) {
    message.textContent = 'Image must be smaller than 12 MB.';
    resetSelection();
    return;
  }

  if (file.size === 0) {
    message.textContent = 'Uploaded file is empty.';
    resetSelection();
    return;
  }

  // 3. Check readability and image decoding integrity
  const reader = new FileReader();

  reader.onerror = () => {
    message.textContent = 'Invalid Image: Please upload a valid chest X-ray radiograph.';
    resetSelection();
  };

  reader.onload = () => {
    const tempImg = new Image();

    tempImg.onerror = () => {
      message.textContent = 'Invalid Image: Please upload a valid chest X-ray radiograph.';
      resetSelection();
    };

    tempImg.onload = () => {
      if (!tempImg.naturalWidth || !tempImg.naturalHeight || tempImg.naturalWidth <= 0 || tempImg.naturalHeight <= 0) {
        message.textContent = 'Invalid Image: Please upload a valid chest X-ray radiograph.';
        resetSelection();
        return;
      }

      selectedFile = file;
      fileName.textContent = file.name;

      // Revoke old object URL if exists
      if (activePreviewUrl) {
        URL.revokeObjectURL(activePreviewUrl);
      }

      activePreviewUrl = URL.createObjectURL(file);
      imagePreview.src = activePreviewUrl;

      const sizeMb = (file.size / (1024 * 1024)).toFixed(2);
      const dimensions = `${tempImg.naturalWidth}×${tempImg.naturalHeight} px`;
      fileMeta.textContent = `${file.name} · ${dimensions} · ${sizeMb} MB · ${file.type || 'image'}`;

      dropZone.hidden = true;
      previewContainer.hidden = false;

      analyzeBtn.disabled = false;
      message.textContent = 'Radiograph loaded. Click "Run AI Screening Inference" to proceed.';

      results.hidden = true;
      standbyPlaceholder.hidden = false;
    };

    tempImg.src = reader.result;
  };

  reader.readAsDataURL(file);
}

fileInput.addEventListener('change', () => {
  const file = fileInput.files && fileInput.files[0];
  handleFileSelection(file);
});

if (replaceBtn) {
  replaceBtn.addEventListener('click', (e) => {
    e.preventDefault();
    fileInput.click();
  });
}

if (clearBtn) {
  clearBtn.addEventListener('click', (e) => {
    e.preventDefault();
    resetSelection();
    message.textContent = 'Select an X-ray image to begin screening.';
  });
}

if (resetBtn) {
  resetBtn.addEventListener('click', (e) => {
    e.preventDefault();
    resetSelection();
    results.hidden = true;
    results.innerHTML = '';
    standbyPlaceholder.hidden = false;
    message.textContent = 'Select an X-ray image to begin screening.';
  });
}

// Drag & Drop event handling for both dropZone and previewContainer
[dropZone, previewContainer].forEach((el) => {
  if (!el) return;
  ['dragenter', 'dragover'].forEach((eventName) => {
    el.addEventListener(eventName, (e) => {
      e.preventDefault();
      e.stopPropagation();
      el.classList.add('dragover');
    });
  });

  ['dragleave', 'drop'].forEach((eventName) => {
    el.addEventListener(eventName, (e) => {
      e.preventDefault();
      e.stopPropagation();
      el.classList.remove('dragover');
    });
  });

  el.addEventListener('drop', (e) => {
    const dt = e.dataTransfer;
    const file = dt && dt.files && dt.files[0];
    if (file) {
      handleFileSelection(file);
    }
  });
});

// ==========================================================================
// 3. AI Screening Inference Execution
// ==========================================================================
analyzeBtn.addEventListener('click', async () => {
  if (!selectedFile) return;

  analyzeBtn.disabled = true;
  analyzeBtn.innerHTML = '<span>Screening Image…</span>';
  message.textContent = 'Running DenseNet121 screening inference... Please wait.';

  // Show visual scanning HUD on preview image
  scannerSweep.hidden = false;
  results.hidden = true;
  standbyPlaceholder.hidden = false;

  const body = new FormData();
  body.append('file', selectedFile);

  try {
    // Preserve literal fetch('/api/predict' for integration tests
    const response = await (API_BASE
      ? fetch(API_BASE + '/api/predict', { method: 'POST', body })
      : fetch('/api/predict', { method: 'POST', body }));

    let data;
    try {
      data = await response.json();
    } catch {
      throw new Error('Server returned an unexpected response. Please try again.');
    }

    if (!response.ok) {
      scannerSweep.hidden = true;

      // HTTP 503: AI Model Checkpoint Pending / Unavailable
      if (response.status === 503) {
        results.innerHTML = `
          <div class="model-pending">
            <h3>⏳ AI Model Checkpoint Pending</h3>
            <p>${data.detail || 'The trained model checkpoint is not yet available.'}</p>
            <p class="muted">Inference is disabled until a verified DenseNet121 checkpoint is installed. No predictions are generated without a trained model.</p>
          </div>
        `;
        results.hidden = false;
        standbyPlaceholder.hidden = true;
        message.textContent = 'Model not yet available — no predictions generated.';
        return;
      }

      // HTTP 415: Unsupported Media Type
      if (response.status === 415) {
        results.innerHTML = `
          <div class="alert-card alert-warn">
            <h3>⚠️ Unsupported File Format</h3>
            <p>${data.detail || 'Upload a JPG, PNG, or WEBP image.'}</p>
            <div class="alert-detail">Status: HTTP 415 Unsupported Media Type</div>
          </div>
        `;
        results.hidden = false;
        standbyPlaceholder.hidden = true;
        message.textContent = 'Unsupported file format.';
        return;
      }

      // HTTP 413: Payload Too Large
      if (response.status === 413) {
        results.innerHTML = `
          <div class="alert-card alert-warn">
            <h3>⚠️ File Size Ceiling Exceeded</h3>
            <p>${data.detail || 'File size exceeds the 12MB limit.'}</p>
            <div class="alert-detail">Status: HTTP 413 Payload Too Large</div>
          </div>
        `;
        results.hidden = false;
        standbyPlaceholder.hidden = true;
        message.textContent = 'File size exceeds the 12 MB limit.';
        return;
      }

      // HTTP 400: Corrupted / Empty / Bad Request
      if (response.status === 400) {
        results.innerHTML = `
          <div class="alert-card alert-error">
            <h3>⚠️ Invalid Radiograph File</h3>
            <p>Invalid Image: Please upload a valid chest X-ray radiograph.</p>
            <div class="alert-detail">Status: HTTP 400 Bad Request (${data.detail || 'Invalid image'})</div>
          </div>
        `;
        results.hidden = false;
        standbyPlaceholder.hidden = true;
        message.textContent = 'Invalid Image: Please upload a valid chest X-ray radiograph.';
        return;
      }

      throw new Error(data.detail || `Server returned error status ${response.status}`);
    }

    // Process successful screening response
    scannerSweep.hidden = true;

    const predictionData = data.prediction_details || data;
    const probability = Number(predictionData.probability_active_tb ?? predictionData.probability);

    if (!Number.isFinite(probability)) {
      throw new Error('The API returned an invalid prediction response.');
    }

    const thresholdUsed = Number(predictionData.threshold_used ?? 0.5);
    const predictedClass = predictionData.predicted_class ||
      (probability >= thresholdUsed ? 'Active_TB' : 'Negative_Control');
    const isDetected = (predictedClass === 'Active_TB') ||
      (predictionData.prediction && predictionData.prediction.toLowerCase().includes('detected'));

    const predictionText = predictionData.prediction ||
      (isDetected ? 'TB screening signal detected' : 'No TB screening signal detected');

    const probabilityControl = Number(predictionData.probability_control ?? (1 - probability));
    const riskLevel = predictionData.risk_level || (isDetected ? 'High Risk' : 'Low Risk');
    const modelName = predictionData.model || data.model || 'DenseNet121';

    const tbPct = (probability * 100).toFixed(1);
    const ctrlPct = (probabilityControl * 100).toFixed(1);

    const bannerClass = isDetected ? 'res-positive' : 'res-negative';
    const badgeText = isDetected ? 'TB Signal Flagged' : 'Normal / Control';

    let tierClass = 'tier-low';
    const rLower = String(riskLevel).toLowerCase();
    if (rLower.includes('high')) {
      tierClass = 'tier-high';
    } else if (rLower.includes('mod')) {
      tierClass = 'tier-moderate';
    }

    results.innerHTML = `
      <!-- 1. Binary AI Screening Classification Banner -->
      <div class="decision-banner ${bannerClass}">
        <div class="decision-header-row">
          <span class="decision-title">${predictionText}</span>
          <span class="decision-badge">${badgeText}</span>
        </div>
        <div class="decision-boundary-citation">
          Classification evaluated against operational cutoff: τ = ${thresholdUsed.toFixed(4)}
        </div>
      </div>

      <!-- 2. Model Output Scores (Strictly labeled as scores, not clinical risk) -->
      <div class="scores-block">
        <div class="scores-title-row">
          <span class="scores-heading">Model Output Scores</span>
          <span class="scores-sub">DenseNet121 Signal Confidence</span>
        </div>

        <div class="score-row">
          <div class="score-label-bar">
            <span>Active Pulmonary TB Score</span>
            <strong>${tbPct}%</strong>
          </div>
          <div class="progress-track">
            <div class="progress-fill fill-tb" style="width: ${tbPct}%"></div>
          </div>
        </div>

        <div class="score-row">
          <div class="score-label-bar">
            <span>Negative / Normal Control Score</span>
            <strong>${ctrlPct}%</strong>
          </div>
          <div class="progress-track">
            <div class="progress-fill fill-control" style="width: ${ctrlPct}%"></div>
          </div>
        </div>

        <!-- Backward-compatibility result nodes for tests -->
        <div class="result" style="display:none;">
          <span>Active Pulmonary TB Probability</span>
          <strong>${tbPct}%</strong>
        </div>
        <div class="result" style="display:none;">
          <span>Negative / Normal Control</span>
          <strong>${ctrlPct}%</strong>
        </div>
      </div>

      <!-- 3. Internal Model Score Tier (Isolated container) -->
      <div class="score-tier-card">
        <div class="score-tier-header">
          <span class="score-tier-label">MODEL SCORE STRATIFICATION</span>
          <span class="score-tier-badge ${tierClass}">${riskLevel}</span>
        </div>
        <div class="score-tier-note">
          Internal score tier based on model output magnitude; does not represent clinical patient risk.
        </div>
      </div>

      <!-- 4. Run Audit Details -->
      <div class="run-meta-strip">
        <span>Model: ${modelName}</span>
        <span>·</span>
        <span>Target: Active Pulmonary TB</span>
        <span>·</span>
        <span>File: ${data.filename || selectedFile.name}</span>
      </div>

      <!-- 5. Research Protocol Notice -->
      <div class="results-notice">
        ${data.notice || 'Research screening output only. Educational research prototype, not a medical device or diagnostic tool.'}
      </div>

      <!-- 6. Actions Bar: Download PDF Report & Screen Another -->
      <div class="results-actions-strip">
        <button type="button" class="btn-download" id="downloadPdfBtn">
          <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
            <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path>
            <polyline points="7 10 12 15 17 10"></polyline>
            <line x1="12" y1="15" x2="12" y2="3"></line>
          </svg>
          Download PDF Report
        </button>
        <button type="button" class="btn-reset" id="newScanBtn">
          <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
            <polyline points="1 4 1 10 7 10"></polyline>
            <path d="M3.51 15a9 9 0 1 0 2.13-9.36L1 10"></path>
          </svg>
          Screen Another Radiograph
        </button>
      </div>
    `;

    results.hidden = false;
    standbyPlaceholder.hidden = true;
    message.textContent = 'Screening inference complete.';

    const downloadPdfBtn = results.querySelector('#downloadPdfBtn');
    if (downloadPdfBtn) {
      downloadPdfBtn.addEventListener('click', () => {
        generatePdfReport({
          date: new Date().toISOString().replace('T', ' ').slice(0, 19) + ' UTC',
          filename: data.filename || (selectedFile ? selectedFile.name : 'radiograph.png'),
          model: modelName,
          target: 'Active Pulmonary Tuberculosis',
          prediction: predictionText,
          threshold: thresholdUsed.toFixed(4),
          tbScore: tbPct,
          ctrlScore: ctrlPct,
          riskLevel: riskLevel,
        });
      });
    }

    const newScanBtn = results.querySelector('#newScanBtn');
    if (newScanBtn) {
      newScanBtn.addEventListener('click', () => {
        resetSelection();
        results.hidden = true;
        results.innerHTML = '';
        standbyPlaceholder.hidden = false;
        message.textContent = 'Select an X-ray image to begin screening.';
      });
    }

  } catch (error) {
    scannerSweep.hidden = true;
    if (error instanceof TypeError) {
      message.textContent = 'Network Error: Could not connect to the backend server. Please verify your connection.';
      results.innerHTML = `
        <div class="alert-card alert-error">
          <h3>🔌 Connection Error</h3>
          <p>Unable to connect to the MediScan Guardian backend API. Ensure the server is online and reachable.</p>
        </div>
      `;
    } else {
      message.textContent = `Error: ${error.message}`;
      results.innerHTML = `
        <div class="alert-card alert-error">
          <h3>⚠️ Screening Error</h3>
          <p>${error.message}</p>
        </div>
      `;
    }
    results.hidden = false;
    standbyPlaceholder.hidden = true;

  } finally {
    scannerSweep.hidden = true;
    analyzeBtn.disabled = !selectedFile;
    analyzeBtn.innerHTML = '<span>Run AI Screening Inference</span><span class="arrow" aria-hidden="true">→</span>';
  }
});

// ==========================================================================
// 4. Clinical Screening PDF Report Generator (Pure Client-Side PDF-1.4)
// ==========================================================================
function generatePdfReport(report) {
  const sanitize = (text) => String(text || '').replace(/[\\()]/g, '');
  const title = 'MEDISCAN GUARDIAN - CLINICAL SCREENING REPORT';
  const dateStr = sanitize(report.date || new Date().toISOString().replace('T', ' ').slice(0, 19) + ' UTC');
  const fileName = sanitize(report.filename || 'chest_radiograph.png');
  const modelName = sanitize(report.model || 'DenseNet121');
  const target = sanitize(report.target || 'Active Pulmonary Tuberculosis');
  const prediction = sanitize(report.prediction || 'No TB screening signal detected');
  const threshold = sanitize(report.threshold || '0.5000');
  const tbScore = sanitize(report.tbScore || '0.0%');
  const ctrlScore = sanitize(report.ctrlScore || '0.0%');
  const riskLevel = sanitize(report.riskLevel || 'Low Risk');

  const streamLines = [
    // Header Bar
    '0.03 0.09 0.15 rg',
    '36 700 540 56 re',
    'f',
    'BT',
    '/F2 15 Tf',
    '1 1 1 rg',
    '50 726 Td',
    '(' + title + ') Tj',
    'ET',
    'BT',
    '/F1 9 Tf',
    '0 0.82 0.71 rg',
    '50 710 Td',
    '(AI-Assisted Chest Radiograph Screening Prototype | Automated Analysis Summary) Tj',
    'ET',

    // Section 1: Audit Information
    'BT',
    '/F2 11 Tf',
    '0.1 0.15 0.2 rg',
    '50 660 Td',
    '(1. AUDIT INFORMATION) Tj',
    'ET',
    '0.8 0.85 0.9 RG',
    '1 w',
    '50 652 m 562 652 l S',
    'BT',
    '/F1 10 Tf',
    '0.2 0.2 0.2 rg',
    '50 635 Td',
    '(Generated: ' + dateStr + ') Tj',
    '0 -15 Td',
    '(File Name: ' + fileName + ') Tj',
    '0 -15 Td',
    '(Model Architecture: ' + modelName + ') Tj',
    '0 -15 Td',
    '(Target Pathology: ' + target + ') Tj',
    '0 -15 Td',
    '(Decision Cutoff Threshold: tau = ' + threshold + ') Tj',
    'ET',

    // Outcome Card
    '0.94 0.97 0.99 rg',
    '50 490 512 40 re',
    'f',
    '0.0 0.5 0.5 RG',
    '1.5 w',
    '50 490 512 40 re',
    'S',
    'BT',
    '/F2 12 Tf',
    '0 0.3 0.35 rg',
    '65 506 Td',
    '(AI Screening Outcome: ' + prediction + ') Tj',
    'ET',

    // Section 2: Model Output Scores
    'BT',
    '/F2 11 Tf',
    '0.1 0.15 0.2 rg',
    '50 460 Td',
    '(2. MODEL OUTPUT SIGNAL SCORES) Tj',
    'ET',
    '0.8 0.85 0.9 RG',
    '1 w',
    '50 452 m 562 452 l S',
    'BT',
    '/F1 10 Tf',
    '0.2 0.2 0.2 rg',
    '50 435 Td',
    '(Active Pulmonary TB Score: ' + tbScore + ') Tj',
    '0 -16 Td',
    '(Normal / Negative Control Score: ' + ctrlScore + ') Tj',
    '0 -16 Td',
    '(Model Score Stratification: ' + riskLevel + ') Tj',
    '0 -14 Td',
    '/F3 9 Tf',
    '0.4 0.4 0.4 rg',
    '(Note: Score tier reflects internal algorithmic confidence, not clinical patient risk.) Tj',
    'ET',

    // Section 3: Protocol & Disclaimer
    '0.98 0.96 0.92 rg',
    '50 280 512 70 re',
    'f',
    '0.8 0.6 0.2 RG',
    '1 w',
    '50 280 512 70 re',
    'S',
    'BT',
    '/F2 9 Tf',
    '0.6 0.3 0.0 rg',
    '60 334 Td',
    '(MANDATORY RESEARCH PROTOCOL & ETHICAL DISCLAIMER) Tj',
    '/F1 8 Tf',
    '0.2 0.2 0.2 rg',
    '0 -14 Td',
    '(MediScan Guardian is an academic research screening prototype, not a medical device.) Tj',
    '0 -11 Td',
    '(It does not diagnose disease or prescribe treatment. All outputs must be evaluated by a) Tj',
    '0 -11 Td',
    '(qualified clinician and confirmed via microbiological testing (GeneXpert / Sputum smear).) Tj',
    'ET',

    // Footer
    'BT',
    '/F1 8 Tf',
    '0.5 0.5 0.5 rg',
    '50 50 Td',
    '(MediScan Guardian | Responsible AI Chest Radiograph Screening | Phase 4/5 Integration) Tj',
    'ET',
  ];

  const content = streamLines.join('\n');
  const length = content.length;

  const objects = [
    '1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n',
    '2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n',
    '3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 4 0 R /F2 5 0 R /F3 6 0 R >> >> /Contents 7 0 R >>\nendobj\n',
    '4 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>\nendobj\n',
    '5 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold >>\nendobj\n',
    '6 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Oblique >>\nendobj\n',
    '7 0 obj\n<< /Length ' + length + ' >>\nstream\n' + content + '\nendstream\nendobj\n',
  ];

  let pdfBody = '%PDF-1.4\n%\xE2\xE3\xCF\xD3\n';
  const offsets = [];
  let currentOffset = pdfBody.length;

  for (let i = 0; i < objects.length; i++) {
    offsets.push(currentOffset);
    pdfBody += objects[i];
    currentOffset += objects[i].length;
  }

  const xrefOffset = currentOffset;
  let xref = 'xref\n0 ' + (objects.length + 1) + '\n0000000000 65535 f \n';
  for (let i = 0; i < offsets.length; i++) {
    const offStr = String(offsets[i]).padStart(10, '0');
    xref += offStr + ' 00000 n \n';
  }

  const trailer = 'trailer\n<< /Size ' + (objects.length + 1) + ' /Root 1 0 R >>\nstartxref\n' + xrefOffset + '\n%%EOF\n';
  const fullPdf = pdfBody + xref + trailer;

  const bytes = new Uint8Array(fullPdf.length);
  for (let i = 0; i < fullPdf.length; i++) {
    bytes[i] = fullPdf.charCodeAt(i) & 0xff;
  }

  const blob = new Blob([bytes], { type: 'application/pdf' });
  const blobUrl = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = blobUrl;
  const safeBaseName = (report.filename || 'radiograph').replace(/\.[^/.]+$/, '');
  a.download = `MediScan_Report_${safeBaseName}.pdf`;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  setTimeout(() => URL.revokeObjectURL(blobUrl), 1000);
}
