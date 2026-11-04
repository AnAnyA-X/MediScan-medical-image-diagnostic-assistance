# MediScan Guardian

AI-assisted chest X-ray screening research prototype. This repository contains:
- A FastAPI backend and browser-based frontend.
- A DenseNet121 multi-label training baseline for NIH ChestX-ray14.
- An inference endpoint that only activates when a real trained checkpoint exists.



## 1. Open in VS Code
Extract the ZIP, then open the `mediscan_guardian` folder in VS Code.

## 2. Create Python environment (Windows PowerShell)
```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```
If PowerShell blocks activation, run:
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.venv\Scripts\Activate.ps1
```
Install the PyTorch build suitable for your machine from https://pytorch.org/get-started/locally/ if the default install fails.

## 3. Run the application
From the project root:
```powershell
uvicorn backend.main:app --reload
```
Open http://127.0.0.1:8000

API status: http://127.0.0.1:8000/api/health
API docs: http://127.0.0.1:8000/docs

## 4. Add NIH ChestX-ray14 data
Obtain the dataset through its official distribution source and follow its terms. Extract images so the layout is:
```text
data/
  nih/
    Data_Entry_2017.csv
    images/
      00000001_000.png
      ...
```
The CSV must contain `Image Index` and `Finding Labels`. The training script parses NIH's pipe-separated labels for the 14 disease classes.

## 5. Train a real baseline
From the project root:
```powershell
python backend/train.py --data-dir data/nih --csv data/nih/Data_Entry_2017.csv --epochs 3
```
Optional small subset for a smoke test:
```powershell
python backend/train.py --data-dir data/nih --csv data/nih/Data_Entry_2017.csv --limit 500 --epochs 1 --batch-size 8
```
A 500-image smoke test is only to verify the pipeline, not to claim meaningful clinical performance. The full NIH dataset is large; use a GPU/Colab for training if your laptop is not equipped for it. First training may download ImageNet pretrained DenseNet121 weights.

The trained checkpoint is saved to:
`models/densenet121_chestxray.pth`

Restart the API after training. `/api/health` should then report `model_trained: true`.

## Dataset integration status
- NIH ChestX-ray14: baseline training pipeline implemented.
- TB Montgomery / TB Shenzhen: not merged into the NIH model. Their labels and image formats must be audited and mapped in a separate step.
- BIMCV COVID-19: requires its metadata/label files to be inspected before integration.
- PadChest: do not use until the actual dataset is available and verified.

Do not report accuracy, AUC, or other metrics unless produced by an actual run on a held-out test set. The starter script prints validation ROC-AUC; for a formal result, keep a separate untouched test set and document the split.

## Demo checklist
1. Show the home screen and upload control.
2. Show `/api/health` explaining whether a trained model is installed.
3. If trained, upload a held-out X-ray and show actual model output.
4. Show training script, dataset source/label handling, and actual evaluation output.
5. Keep the medical disclaimer visible.
