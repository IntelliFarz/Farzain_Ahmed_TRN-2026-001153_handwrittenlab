# The Handwriting Lab — Farzain Ahmed

**Training ID:** TRN-2026-001153  
**Seed S:** 1153  
**Editor:** VS Code  
**Assignment:** ML + Deep Learning, Parts A–G

## Important status note
This is an executable working project pack based on the supplied assignment, starter ZIP, and handwriting photograph. It includes implementations and generated experiment artifacts. It is **not yet a submission-ready claim that every personal requirement is complete**: review the 50 digit crops, perform and photograph the required handwritten derivations, run Part F2 with the Hugging Face model, and write genuine dated experiment-log entries. Read `COMPLETION_STATUS.md` before submission.

## Windows / VS Code setup
1. Extract this ZIP completely.
2. Open the extracted `Farzain_Ahmed_TRN-2026-001153_handwrittenlab` folder in VS Code.
3. Open **Terminal → New Terminal**.
4. Create an environment (recommended):
   ```powershell
   py -m venv .venv
   .\.venv\Scripts\Activate.ps1
   python -m pip install --upgrade pip
   python -m pip install -r requirements.txt
   ```
5. Select `.venv` as the VS Code Python interpreter and install/enable the Python and Jupyter extensions.
6. Run the project parts individually:
   ```powershell
   python run_project.py --part A
   python run_project.py --part B
   python run_project.py --part C
   python run_project.py --part D
   python run_project.py --part E
   python run_project.py --part F1
   python run_project.py --part F2
   ```
   Or run `python run_project.py --part all`.

## Notebooks
Open `notebooks/` and run the six notebooks top-to-bottom. Each notebook calls the matching reproducible script section. The same code can be run from the terminal using the commands above.

## Outputs
Generated CSV tables, Markdown tables, plots, confusion matrices, and saved model weights are written to `outputs/` and `tables/`. Re-run the code to regenerate them. Numbers in a final report must be copied from your own latest run.

## Handwriting sheet
The provided sheet is a 5-row by 10-column table. The processing routine in `run_project.py` applies a perspective warp and creates 50 8×8 samples. **Inspect `outputs/handwriting_5x10_grid.png` carefully**: if any crop is incomplete or contains a grid line, adjust the crop coordinates in `load_handwriting()` before using handwriting accuracy in your report. The labels are assumed to be row-major: each row contains digits 0 through 9.

## Part F2
The sentiment pipeline requires an internet connection on first run to download its pretrained model. Install transformers (`python -m pip install transformers`) and run `python run_project.py --part F2`. The supplied review labels are a starting set; verify/edit them yourself before running the model. Mixed/neutral labels are not binary sentiment labels and are excluded from the simple binary accuracy calculation.

## Submission checklist
See `README_CHECKLIST.md` and `COMPLETION_STATUS.md`. Do not submit template answers or unverified results as if they were your own observations.
