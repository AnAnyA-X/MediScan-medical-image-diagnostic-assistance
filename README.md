# 🩺 MediScan Guardian

### AI-Assisted Tuberculosis Screening Using Chest X-Ray Images

MediScan Guardian is an AI-assisted medical imaging research prototype designed to analyze chest X-ray images and provide preliminary Tuberculosis (TB) screening results using a DenseNet121 deep learning model.

The application combines a modern web interface with a Python-based backend to simplify image uploading, AI inference and prediction result visualization.

> **Medical Disclaimer:** MediScan Guardian is an educational and research prototype. It is not a certified medical diagnostic system and must not replace professional medical advice, diagnosis or treatment.

---

## ✨ Key Features

- **Modern Dashboard:** Futuristic, user-friendly medical interface.
- **Image Upload:** Drag-and-drop and file-picker support.
- **Image Preview:** Displays uploaded image, filename, dimensions and file size.
- **Input Validation:** Checks image formats, file size and image readability.
- **AI-Powered Screening:** Uses a DenseNet121 model for TB prediction.
- **Prediction Results:** Displays TB probability, Normal probability and risk category using backend responses.
- **API Integration:** Connects the frontend with a FastAPI prediction endpoint.
- **Error Handling:** Handles invalid uploads, unsupported formats, oversized files and backend connectivity issues.
- **Responsive Interface:** Designed for accessible interaction across screen sizes.

## 🛠️ Technology Stack

| Component | Technologies |
|---|---|
| Frontend | HTML, CSS, JavaScript |
| Backend | Python, FastAPI |
| Deep Learning | PyTorch, DenseNet121 |
| Image Processing | Pillow, NumPy |
| API Communication | REST API, Fetch API |
| Version Control | Git, GitHub |
| Deployment | Vercel, Render (planned/configured as applicable) |

## 🏗️ System Architecture

```text
           USER
             |
             v
     MediScan Dashboard
             |
             v
       Image Upload
             |
             v
    Image Validation
             |
             v
      FastAPI Backend
             |
             v
    Image Preprocessing
             |
             v
       DenseNet121
       TB Screening
             |
             v
     Prediction Response
             |
             v
      Results Dashboard
```

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/AnAnyA-X/MediScan-medical-image-diagnostic-assistance.git
```

### 2. Navigate to the Project

```bash
cd MediScan-medical-image-diagnostic-assistance
```

### 3. Create a Virtual Environment

Windows PowerShell:

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.venv\Scripts\Activate.ps1
```

### 4. Install Dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Install a compatible PyTorch build from [PyTorch Official Installation Guide](https://pytorch.org/get-started/locally/) if required.

### 5. Start the Backend

From the project root:

```bash
uvicorn backend.main:app --reload
```

Open the application:

http://127.0.0.1:8000

API Health Check:

http://127.0.0.1:8000/api/health

API Documentation:

http://127.0.0.1:8000/docs

**Note:** A compatible trained model checkpoint must be available for actual inference. The API may report that the model is unavailable if the checkpoint is missing.

---

## 📂 Project Structure

```text
mediscan_guardian/
│
├── backend/
│   ├── main.py
│   ├── model.py
│   ├── preprocessing.py
│   └── train.py
│
├── frontend/
│   ├── index.html
│   ├── style.css
│   └── app.js
│
├── models/
├── tests/
├── data/
├── requirements.txt
├── render.yaml
└── vercel.json
```

*Refer to the repository for the complete and current project structure.*

## 🔬 Dataset and Model Information

- **Model Architecture:** DenseNet121.
- **Framework:** PyTorch.
- **Task:** AI-assisted chest X-ray TB screening.
- **Inference:** Uses the existing trained model checkpoint.

The repository also contains a baseline training pipeline for NIH ChestX-ray14. Its multi-label training baseline should not be confused with the current TB screening inference model.

Dataset availability, training configuration and evaluation results must be verified independently before reporting performance metrics.

## ⚠️ Current Limitations

- Current image validation checks file format, size and readability. It does not reliably distinguish genuine chest X-rays from ordinary photographs or screenshots.
- A valid image file is not necessarily a valid chest radiograph.
- Model predictions are experimental outputs and are not confirmed medical diagnoses.
- No accuracy, sensitivity, specificity or AUROC claims are made without independently verified evaluation results.

## 🔮 Future Scope

- Dedicated chest X-ray versus non-X-ray classification.
- Improved model evaluation and validation.
- Explainable AI using genuine Grad-CAM visualizations.
- Expanded dataset support.
- Enhanced screening reports and history.
- Further improvements to deployment and accessibility.

---

## 🔗 Project Links

| Resource | Link |
|---|---|
| GitHub Repository | [MediScan Guardian](https://github.com/AnAnyA-X/MediScan-medical-image-diagnostic-assistance) |
| Latest Implementation Branch | [feature/futuristic-frontend](https://github.com/AnAnyA-X/MediScan-medical-image-diagnostic-assistance/tree/feature/futuristic-frontend) |
| Live Demo | To be added after successful deployment |
| Demo Video | Optional, to be added |

**Important:** The latest implementation is maintained on `feature/futuristic-frontend`. Ensure the evaluator can access this branch. Do not assume the `main` branch contains the latest implementation until changes are merged.

---

## 👩‍💻 Developers

**Ananya Kulal**

**Santhiya Murugesh**

Developed as an academic AI-assisted medical imaging research project.

---

*Building technology to make preliminary medical image screening more accessible through AI.*
