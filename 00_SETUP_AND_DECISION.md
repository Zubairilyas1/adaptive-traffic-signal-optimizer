# 00 - Setup and Decision Guide

**Project:** AI2002 Artificial Intelligence - Semester Project (Fall 2026)
**Group size:** 3 members
**Path:** A (Software-Based)
**Purpose of this file:** the shared foundation for all three candidate ideas. Environment setup, tooling, dataset-access rules, and the idea comparison matrix. The per-idea roadmaps live in `01_`, `02_`, `03_`.

> Read this file FIRST. Do not start any idea roadmap until Phase 0 is complete and the dataset gate (Phase 1 of the chosen idea) has passed.

---

## 1. The rule that governs the whole project

From `AI_Project_Statement.pdf`:

- Dataset MUST be **identified** and documented (source, features, target, preprocessing, limitations).
- **External datasets may be used** where appropriate, but the group MUST understand, document, and justify their use.
- **Streamlit deployment is MANDATORY.**
- Directly copying an existing project, tutorial, repository, or complete solution is **strictly prohibited**.

**Interpretation:** you may download a public dataset. You may NOT clone a finished Streamlit app and call it yours. The **pipeline, the analysis layer, and the decision layer must be your own work.** Every idea file includes a "verify ownership" note in its Phase 1 gate.

---

## 2. Confirmed environment audit (checked on this machine)

| Item | Value | Status |
|---|---|---|
| OS | Windows (win32), PowerShell 5.1 | OK |
| Python | 3.12.10 | OK |
| CPU | Intel i7-11850H, 8 cores / 16 threads | OK |
| RAM | 31.7 GB | OK |
| GPU | NVIDIA T1200 Laptop, 4 GB VRAM, compute 7.5, driver 610.60 | OK but underused |
| Disk (D:) | 99.3 GB free | OK |
| git | 2.55.0.windows.1 | OK |
| torch (GLOBAL) | 2.13.0+**cpu** | Broken but no longer used |
| torch (.venv) | **2.6.0+cu124** | **FIXED - CUDA works** |
| streamlit (.venv) | 1.64.0 | installed |
| albumentations (.venv) | 2.0.8 | installed |
| kaggle (.venv) | 2.2.4 | installed |
| numpy (.venv) | 2.5.3 | installed |
| opencv (.venv) | opencv-python-headless 5.0.0.93 | single build, clean |
| timm | not installed | install if Idea 1 |
| mediapipe | not installed | install if Idea 2 |

### 2.1 The critical blocker - RESOLVED

The global Python had a CPU-only torch build. This was fixed on **02 Oct 2026** by creating an isolated venv with the CUDA build:

```
.venv\Scripts\python.exe
torch: 2.6.0+cu124
CUDA available: True
device: NVIDIA T1200 Laptop GPU
torchvision: 0.21.0+cu124
```

A real GPU matmul was run successfully (2000x2000). **Always activate/use `.venv` - never the global Python.**

> Note: the `cu124` Windows/cp312 index tops out at torch 2.6.0. That is the correct current version for this machine.

> Note: C: drive is full (0 GB free). pip's temp and cache were redirected to `D:\...\Project\.build\` for the install. Keep this in mind for future installs - see Section 3 for the `$env:TMP` pattern.

### 2.2 The OpenCV conflict - RESOLVED

The global Python had three conflicting OpenCV builds. The venv has exactly one: **opencv-python-headless 5.0.0.93**. `cv2 5.0.0` imports cleanly.

---

## 3. Phase 0 - Environment setup (do this once, before 11 Oct 2026)

All commands are PowerShell. Run them from `D:\5th Semester\2.AI\Project`.

### Step 1 - Create an isolated virtual environment

Do NOT train from global site-packages; it is already in a conflicted state.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

If activation is blocked by execution policy, run once:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

### Step 2 - Install PyTorch WITH CUDA (fixes the blocker)

Install the CUDA build explicitly. Do not use the default `pip install torch` (that is what produced the +cpu wheel).

```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

> If `cu124` is unavailable for your torch version, check https://pytorch.org/get-started/locally/ and use the CUDA build it recommends. The T1200 is compute capability 7.5 (Turing), which is supported by every current CUDA build.

**Verify immediately:**

```powershell
python -c "import torch; print(torch.__version__); print('CUDA:', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE')"
```

Expected: `torch ... +cu124`, `CUDA: True`, `NVIDIA T1200 Laptop GPU`. **Do not proceed until this prints `CUDA: True`.**

### Step 3 - Install the shared stack

```powershell
pip install numpy pandas matplotlib seaborn scikit-learn tqdm pillow
pip install opencv-python-headless albumentations
pip install streamlit
pip install kaggle
```

> `opencv-python-headless` is chosen deliberately (no GUI dependencies, best for Streamlit servers). If you need `cv2.imshow` locally, use `opencv-python` instead - but never both.

### Step 4 - Idea-specific dependencies (install only for the idea you pick)

| Idea | Extra packages |
|---|---|
| 01 DR Triage | `pip install timm scikit-image grad-cam` |
| 02 Drowsiness | `pip install mediapipe ultralytics` |
| 03 Traffic | `pip install ultralytics` |

> Dependency risk: mediapipe (Idea 2) sometimes pins `numpy<2`. If `pip` downgrades numpy, that is fine inside the venv; just confirm the install completes without resolver errors. Verify with `python -c "import numpy; print(numpy.__version__)"`.

### Step 5 - Freeze the environment

```powershell
pip freeze > requirements.txt
```

Commit `requirements.txt`. It is required evidence of a reproducible setup.

### Step 6 - Verify the full stack

```powershell
python -c "import torch, cv2, numpy, pandas, sklearn, streamlit; print('all imports OK'); print('cuda', torch.cuda.is_available()); print('cv2', cv2.__version__)"
```

### Step 7 - Kaggle API setup (for Ideas 1 and 2)

1. Create a free Kaggle account.
2. Go to Account -> API -> "Create New Token". Download `kaggle.json`.
3. Place it at `C:\Users\<you>\.kaggle\kaggle.json`.

```powershell
kaggle --version
```

4. For competition datasets (e.g. APTOS 2019) you must click **"Join Competition"** on the Kaggle page and accept the rules, otherwise the API download returns 403. This is a required, easy-to-miss step.

### Phase 0 exit criteria

- [ ] `.venv` created and activated
- [ ] `torch.cuda.is_available()` prints `True`
- [ ] `streamlit` imports
- [ ] `opencv` shows exactly one version
- [ ] `requirements.txt` written
- [ ] `kaggle --version` works

---

## 4. Phase 1 - Dataset access gate (applies to the chosen idea)

This is the single most important risk control. **Do not commit to an idea on 25 Oct until its dataset is downloaded and a first sample loads.**

Rules:

1. Download the dataset for real. Do not assume a link works because it was in a list.
2. Load at least 10 samples and print their shape/label.
3. Count class distribution. If a class has < 30 samples, flag it.
4. Read the license. Confirm academic use is allowed.
5. If the dataset fails, **pivot to the backup** - do not "make it work" with a tiny scraped set.

### Dataset verification status (from research on this machine)

| Dataset | Idea | Status | Notes |
|---|---|---|---|
| APTOS 2019 | 01 | Exists, gated | Kaggle competition page live. Requires account + Join Competition. Free. |
| Messidor-2 / EyePACS | 01 (backup) | Known public | Messidor: request form. EyePACS: Kaggle "Diabetic Retinopathy Detection". |
| MRL Eye Dataset | 02 | **UNVERIFIED** | Lab page (mrl.cs.vsb.cz) does not render the dataset section reliably. My guessed Kaggle mirror returned 404. Treat as unknown until confirmed. |
| State Farm Distracted Driver | 02 | Exists, gated | Kaggle competition page live. |
| Roboflow Universe vehicle sets | 03 (PRIMARY) | **CONFIRMED, instant** | Free account + instant YOLO-format export. Vehicle Detection (by Learning) ~22k imgs incl. Autorickshaw; other sets 9.6k-51k imgs. No request form. |
| UA-DETRAC | 03 (backup) | Exists, request process | Official page describes dataset (10 hrs, 24 locations, 960x540 @ 25fps, 140k+ frames, 8,250 vehicles, 1.21M boxes) but exposes **no direct download link** - treat as request-based backup. |

> Idea 03's dataset access is now solved via Roboflow Universe (instant, free). Idea 02 (MRL Eye) remains the only unverified primary; if you pick 02, run Phase 1 **today** and have State Farm plus self-captured webcam footage as fallback.

---

## 5. Mandatory hybrid dataset strategy (all ideas)

The statement requires documenting **data limitations**. The highest-leverage move available to you is:

1. **Public dataset as the training base** (~90%). Fast, reproducible, benchmarkable.
2. **Your own small test set** (30-50 self-collected samples you annotate). This gives:
   - a real generalisation result ("94% on benchmark, 78% on our local images"),
   - authentic content for the mandatory limitations section,
   - proof of ownership during the viva.

The gap between benchmark and local performance is exactly the honest analysis that separates a top grade from a passing one.

---

## 6. Marker-mapped timeline

| Date | Deadline | What must be true |
|---|---|---|
| **11 Oct 2026 (Sun)** | Finalize group + path | Phase 0 done, one idea chosen, idea entered on the shared sheet and cell locked |
| **25 Oct 2026 (Sun)** | Finalize dataset + models + use case | Phase 1 gate passed, dataset downloaded and documented, model approach fixed |
| **06 Dec 2026 (Sun)** | Final project submission | All deliverables zipped (see Section 8) |
| **07-09 Dec 2026 (Mon-Wed)** | Presentations / demos | Live Streamlit demo rehearsed and working |

All deadlines are FINAL per the statement.

---

## 7. Idea comparison matrix

Scoring: 3 = best, 1 = weakest.

| # | Idea | Unique | Feasible (4GB / 6wk) | Metric depth | Demo impact | Dataset risk (3=low) | **Total /15** |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|
| 01 | DR Triage | 2 | 3 | 3 | 2 | 3 | **13** |
| 02 | Drowsiness | 2 | 3 | 3 | 3 | 1 | **12** |
| 03 | Traffic Signal | 3 | 2 | 3 | 3 | 3 | **14** |

### Recommendation

- **Safest for a top grade:** Idea 03 (Traffic Signal) scores highest and has an instantly-available dataset (Roboflow Universe, no request form). It needs all 3 members engaged - most engineering hours.
- **Best demo with zero dataset risk:** Idea 01 (DR Triage). APTOS is instant, and the quadratic-weighted-kappa metric is genuinely impressive.
- **Highest wow-factor but dataset risk:** Idea 02 (Drowsiness). Verify MRL Eye today or plan to rely on State Farm + self-captured video.

**Do not pick based only on this table.** Pick the idea your group can explain confidently at the viva, because every member is personally responsible for understanding the work.

---

## 8. Mandatory deliverables checklist (all ideas)

From `AI_Project_Statement.pdf`, "Mandatory Deliverables":

- [ ] **Source Code** - complete, organized, runnable, commented, meaningful naming, clear structure
- [ ] **Project Report** - problem statement, real-world justification, selected path, objectives, methodology, algorithm description, system architecture/workflow (**proper pipeline diagram**), dataset description, implementation details, testing, results, limitations, conclusion
- [ ] **Demonstration** - live Streamlit app (mandatory live, video is not a substitute)
- [ ] **Presentation Slides** - problem, motivation, solution, AI/ML technique, implementation, testing, results (qualitative + quantitative), limitations, conclusion
- [ ] **Evidence of Evaluation** - ML evaluation results and metrics (Path A)
- [ ] Submission: **single zip per group on GCR**, one member submits

### Compliance quick-reference

| Statement says | Every idea file must show |
|---|---|
| "MUST implement >= 1 ML algorithm/NN" | Section 4 model layer |
| "MUST NOT be merely ... rule-based" | Section 4 analysis + decision layers |
| "algorithm MUST be justified against a specific use case" | Section 2 + 3 |
| "suitable dataset MUST be identified and documented" | Section 6 |
| "MUST be evaluated using appropriate performance metrics" | Section 11 |
| "Streamlit deployment is MANDATORY" | Section 12 |
| "complete usable flow: input -> model execution -> presentation" | Section 12 flow map |

---

## 9. Cross-idea conventions

Use these in every idea so the repo stays consistent.

### 9.1 Repository skeleton

```
project-root/
  .venv/                     (not committed)
  data/
    raw/                     downloaded, never edited
    processed/               cleaned/derived
    external/                self-collected test set
  notebooks/                 exploration only, not the deliverable
    01_eda.ipynb
    02_baseline.ipynb
    03_model_training.ipynb
    04_evaluation.ipynb
  src/
    config.py                paths, hyperparameters, constants
    data/                    loaders, preprocessing, augmentation
    models/                  architecture definitions
    training/                train loop, callbacks
    evaluation/              metrics, plots
    inference/               single-sample prediction API
  app/
    streamlit_app.py         entry point
    pages/                   multi-page Streamlit
    components/              reusable UI blocks
    assets/                  css, images
  models/                    saved weights (.pt/.pth) - small ones committed
  reports/
    figures/                 exported plots
    report.md / report.pdf
    slides.pdf
  requirements.txt
  README.md
```

### 9.2 Config discipline

All paths and hyperparameters live in `src/config.py`. No hardcoded paths anywhere else. This satisfies "meaningful naming" and "clear project structure" and makes the Streamlit app portable.

### 9.3 GPU discipline (T1200, 4 GB)

- Use **mixed precision** (`torch.cuda.amp` / `torch.amp.autocast`) to roughly halve memory.
- Prefer small backbones: EfficientNet-B0, ResNet18/34, YOLOv8n/s, MobileNetV3.
- Start at 224 px. Only go to 380/512 px if memory allows.
- If you hit OOM: reduce batch size first, then image size, then use gradient accumulation.
- Set `torch.backends.cudnn.benchmark = True` for fixed input sizes.
- Watch memory live: `nvidia-smi -l 2` in a second terminal.

### 9.4 Streamlit discipline

- Cache model loading with `@st.cache_resource`.
- Cache data loading with `@st.cache_data`.
- Never retrain in the app - load saved weights only.
- The app must demonstrate the full flow: **input -> execution -> understandable output**.
- Include at least one exportable artifact (CSV/PNG) so results are demonstrable evidence.

---

## 10. Ownership and academic integrity protocol

Since the statement strictly prohibits copying a complete solution:

1. **Never** clone a "end-to-end + Streamlit" repo. Reading a tutorial to learn a technique is fine; submitting its code is not.
2. Write your own training loop, even if it is simple. Reviewers value understanding over sophistication.
3. Keep a **decision log** (`reports/decision_log.md`): every choice, why, and what alternative you rejected. This is your viva armour.
4. Pretrained weights (ImageNet, COCO) are explicitly permitted - document them as such.
5. Cite every dataset, pretrained model, and external library in the report.

---

## 11. What to do right now (in order)

1. Complete **Phase 0** above. Confirm `CUDA: True`.
2. Read `01_`, `02_`, `03_` and choose one idea as a group.
3. Enter the idea on the shared Google Sheet and **lock the cell** before someone else takes it.
4. Run that idea's **Phase 1 dataset gate**.
5. Only then begin Phase 2.

---

*Shared setup document - companion to the per-idea roadmaps. Every `MUST` quoted here is taken verbatim from `AI_Project_Statement.pdf`. Where a dataset link is unverified, it is labelled as such and must be confirmed in Phase 1.*
