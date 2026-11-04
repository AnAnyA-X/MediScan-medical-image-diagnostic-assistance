// API base URL — set via window.MEDISCAN_API_BASE (injected by env-config.js).
// Falls back to empty string (same-origin) for local dev.
const API_BASE = (typeof window !== 'undefined' && window.MEDISCAN_API_BASE) || '';

const fileInput = document.getElementById('fileInput');
const fileName = document.getElementById('fileName');
const analyzeBtn = document.getElementById('analyzeBtn');
const message = document.getElementById('message');
const results = document.getElementById('results');
const apiStatus = document.getElementById('apiStatus');

let selectedFile = null;

// --- Health check at startup ---
fetch(API_BASE + '/api/health')
  .then((response) => response.json())
  .then((data) => {
    if (data.model_trained) {
      apiStatus.textContent = 'DenseNet121 active · ready';
      apiStatus.style.color = '#087f82';
      apiStatus.style.background = '#eaf5f3';
    } else {
      apiStatus.textContent = 'API online · model pending';
      apiStatus.style.color = '#a06a12';
      apiStatus.style.background = '#fef6e7';
    }
  })
  .catch(() => {
    apiStatus.textContent = 'API unavailable';
    apiStatus.style.color = '#b3261e';
    apiStatus.style.background = '#fdeded';
  });

// --- File selection ---
fileInput.addEventListener('change', () => {
  selectedFile = fileInput.files[0] || null;
  fileName.textContent = selectedFile
    ? selectedFile.name
    : 'Choose an X-ray image';

  analyzeBtn.disabled = !selectedFile;

  message.textContent = selectedFile
    ? 'Image selected. Ready to run screening inference.'
    : 'Select an image to begin.';

  results.hidden = true;
});

// --- Analyze ---
analyzeBtn.addEventListener('click', async () => {
  if (!selectedFile) return;

  analyzeBtn.disabled = true;
  analyzeBtn.textContent = 'Analyzing...';
  message.textContent = 'Running DenseNet121 screening inference...';
  results.hidden = true;

  const body = new FormData();
  body.append('file', selectedFile);

  try {
    const response = await fetch(API_BASE + '/api/predict', {
      method: 'POST',
      body,
    });

    let data;
    try {
      data = await response.json();
    } catch {
      throw new Error('Server returned an unexpected response. Please try again.');
    }

    if (!response.ok) {
      // 503 = model pending — show friendly model-pending block
      if (response.status === 503) {
        results.innerHTML = `
          <div class="model-pending">
            <h3>⏳ AI Model Pending</h3>
            <p>${data.detail || 'The trained model checkpoint is not yet available.'}</p>
            <p class="muted">Inference will be enabled once the verified checkpoint is installed. No predictions are generated without a trained model.</p>
          </div>`;
        results.hidden = false;
        message.textContent = 'Model not yet available — no predictions generated.';
        return;
      }
      throw new Error(data.detail || 'Prediction failed.');
    }

    // Support both Phase 5 and Phase 4 response formats
    const predictionData = data.prediction_details || data;

    const probability = Number(predictionData.probability_active_tb ?? predictionData.probability);

    if (!Number.isFinite(probability)) {
      throw new Error('The API returned an invalid prediction response.');
    }

    const thresholdUsed = Number(predictionData.threshold_used ?? 0.5);
    const isDetected = probability >= thresholdUsed;
    const prediction = predictionData.predicted_class ||
      predictionData.prediction ||
      (isDetected ? 'TB Signal Detected' : 'No TB Signal Detected');
    const riskLevel = predictionData.risk_level || (isDetected ? 'High Risk' : 'Low Risk');
    const probabilityControl = Number(predictionData.probability_control ?? (1 - probability));

    const badgeColor = isDetected ? '#b3261e' : '#1e7b4c';
    const badgeBg = isDetected ? '#fdeded' : '#edf7f1';

    results.innerHTML = `
      <div style="margin-bottom:14px;padding:12px 14px;border-radius:10px;background:${badgeBg};border:1px solid ${badgeColor}33;display:flex;justify-content:space-between;align-items:center;">
        <div>
          <strong style="color:${badgeColor};font-size:14px;display:block;">
            ${prediction.replaceAll('_', ' ')}
          </strong>
          <small style="color:#5c6b73;">
            Risk level: ${riskLevel} · Model: ${predictionData.model || 'DenseNet121'}
          </small>
        </div>
        <span style="font-size:18px;font-weight:800;color:${badgeColor};">
          ${(probability * 100).toFixed(1)}%
        </span>
      </div>

      <h3>Screening probabilities</h3>

      <div class="result">
        <span>Active Pulmonary TB Probability</span>
        <strong>${(probability * 100).toFixed(1)}%</strong>
      </div>

      <div class="result">
        <span>Negative / Normal Control</span>
        <strong>${(probabilityControl * 100).toFixed(1)}%</strong>
      </div>

      <p class="muted" style="margin-top:14px;font-size:11px;line-height:1.5;">
        ${data.notice || 'Research screening output only. Not a medical diagnosis.'}
      </p>
    `;

    results.hidden = false;
    message.textContent = 'Screening inference complete.';

  } catch (error) {
    if (error instanceof TypeError) {
      message.textContent = 'Network error: could not reach the backend. Please check your connection.';
    } else {
      message.textContent = `Error: ${error.message}`;
    }
    results.hidden = true;

  } finally {
    analyzeBtn.disabled = false;
    analyzeBtn.innerHTML = 'Analyze image <span>→</span>';
  }
});
