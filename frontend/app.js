// API base URL — set via window.MEDISCAN_API_BASE (injected by env-config.js) or
// fall back to same-origin (local dev mode where backend serves frontend).
const API_BASE = (typeof window !== 'undefined' && window.MEDISCAN_API_BASE) || '';

const fileInput = document.getElementById('fileInput');
const fileName = document.getElementById('fileName');
const analyzeBtn = document.getElementById('analyzeBtn');
const message = document.getElementById('message');
const results = document.getElementById('results');
const apiStatus = document.getElementById('apiStatus');

// --- Health check ---
fetch(API_BASE + '/api/health')
  .then(r => r.json())
  .then(data => {
    apiStatus.textContent = data.model_trained ? 'AI model ready' : 'API online · model pending';
  })
  .catch(() => { apiStatus.textContent = 'API unavailable'; });

// --- File selection ---
fileInput.addEventListener('change', () => {
  selectedFile = fileInput.files[0] || null;
  fileName.textContent = selectedFile ? selectedFile.name : 'Choose an X-ray image';
  analyzeBtn.disabled = !selectedFile;
  message.textContent = selectedFile ? 'Image selected. Ready to analyze.' : 'Select an image to begin.';
  results.hidden = true;
});

let selectedFile = null;

// --- Analyze ---
analyzeBtn.addEventListener('click', async () => {
  if (!selectedFile) return;
  analyzeBtn.disabled = true;
  analyzeBtn.textContent = 'Analyzing…';
  message.textContent = 'Running model inference…';
  results.hidden = true;

  const body = new FormData();
  body.append('file', selectedFile);

  try {
    const response = await fetch(API_BASE + '/api/predict', { method: 'POST', body });

    // Handle non-JSON or network-level errors
    let data;
    try {
      data = await response.json();
    } catch {
      throw new Error('Server returned an unexpected response. Please try again.');
    }

    if (!response.ok) {
      // 503 = model pending — show friendly message
      if (response.status === 503) {
        results.innerHTML = `
          <div class="model-pending">
            <h3>⏳ AI Model Pending</h3>
            <p>${data.detail || 'The trained model checkpoint is not yet available.'}</p>
            <p class="muted">Inference will be enabled once the verified checkpoint is installed.</p>
          </div>`;
        results.hidden = false;
        message.textContent = 'Model not yet available.';
        return;
      }
      throw new Error(data.detail || 'Prediction failed.');
    }

    const signalList = Array.isArray(data.results)
      ? data.results
      : (data.results && Array.isArray(data.results.results) ? data.results.results : []);

    results.innerHTML =
      '<h3>Model screening signals</h3>' +
      signalList.slice(0, 6).map(x =>
        `<div class="result"><span>${(x.label || '').replaceAll('_', ' ')}</span><strong>${((x.probability || 0) * 100).toFixed(1)}%</strong></div>`
      ).join('') +
      `<p class="muted">${data.notice || ''}</p>`;
    results.hidden = false;
    message.textContent = 'Inference complete.';

  } catch (error) {
    if (error instanceof TypeError) {
      // Network error (fetch failed entirely)
      message.textContent = 'Network error: could not reach the backend. Please check your connection.';
    } else {
      message.textContent = error.message;
    }
  } finally {
    analyzeBtn.disabled = false;
    analyzeBtn.innerHTML = 'Analyze image <span>→</span>';
  }
});
