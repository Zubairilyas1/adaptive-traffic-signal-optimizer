# 01 - Diabetic Retinopathy Severity Grading + Referral Triage

**Path:** A (Software-Based)
**Primary technique:** Transfer learning with an ordinal 5-class CNN (EfficientNet-B0)
**Deployment:** Streamlit (mandatory)
**Group size:** 3

Companion file: `00_SETUP_AND_DECISION.md` (environment, timeline, conventions).

---

## 1. Verdict

A clinically meaningful, low-risk project with an impressive metric (quadratic weighted kappa). The model layer is a standard transfer-learning task, so the grade is won or lost in the **analysis layer (ordinal + attribution)** and the **decision layer (triage window)**. Dataset risk is the lowest of the three ideas because APTOS is instantly downloadable. The main intellectual risk is that this is a well-known Kaggle problem - the triage and explainability layers are what make it unambiguously yours.

| Dimension | Score /5 | Note |
|---|---|---|
| Uniqueness | 3 | Known Kaggle task; differentiated by triage + Grad-CAM |
| Feasibility on 4 GB / 6 weeks | 5 | EfficientNet-B0 @ 224-380 px fits easily |
| Metric depth | 5 | Quadratic Weighted Kappa is the clinical standard |
| Demo impact | 3 | Strong if you show the heatmap + triage card |
| Dataset risk | 5 | APTOS is instant; Messidor as backup |

---

## 2. Real-world problem and justification

**Problem.** Diabetic retinopathy (DR) is a leading cause of preventable blindness. It is graded on an ordinal scale from no disease to proliferative disease. Early stages are treatable; late stages are not. Grading requires an ophthalmologist or retina specialist reading a fundus photograph.

**Why it matters (Pakistan framing).** Diabetes prevalence in Pakistan is among the highest in the world, but the supply of retina specialists is concentrated in a few urban centres. In screening camps and rural clinics, fundus images are captured by a technician with no one qualified to grade them, so patients either wait weeks for a referral or are never screened again.

**Stakeholder and decision.** The primary user is a **screening-camp technician or a general practitioner**, not a specialist. They need one answer: *does this patient need to be seen urgently, soon, or routinely?* That is a triage decision, not a diagnosis. This framing is the project's justification.

**Why ML is appropriate here.** The mapping from pixel patterns (microaneurysms, haemorrhages, exudates, neovascularisation) to severity grade is not expressible as hand-written rules. Convolutional networks learn these lesion patterns from labelled examples. Because the grade order matters (grade 3 is closer to grade 4 than to grade 0), an **ordinal** treatment is more appropriate than plain categorical classification - this is the justification the statement demands.

**Cost of inaction.** Untreated proliferative DR causes irreversible vision loss. A triage tool that correctly routes even a modest fraction of urgent cases to timely care has direct clinical value.

---

## 3. Requirement-compliance matrix

Every mandatory requirement from `AI_Project_Statement.pdf` and the exact component satisfying it.

| Statement requirement (quoted) | Satisfied by |
|---|---|
| "MUST implement at least ONE Machine Learning algorithm/Neural Network" | EfficientNet-B0 transfer learning, 5-class ordinal output (Section 4.1) |
| "MUST NOT be merely a conventional software application ... or rule-based program with an AI-related label" | The core output is a learned CNN prediction; triage rules operate on the learned probability vector, never replace it |
| "algorithm MUST be justified against a specific real-world use case" | Section 2 - screening-camp triage, pixel patterns not hand-codable, ordinal scale |
| "explain why the selected algorithm is appropriate ... what input data ... what output ... how the output supports the decision" | Section 4 (input: fundus RGB; output: 5-class probability + triage tier) |
| "Generic statements such as 'AI is useful in this field' ... will NOT be accepted" | Justification is structural: ordinal loss, class imbalance handling, kappa metric |
| "A suitable dataset MUST be identified and documented ... source, features, target, preprocessing, limitations" | Section 6 (APTOS 2019) |
| "ML pipeline MUST be implemented and evaluated using appropriate performance metrics. Results MUST be reported and interpreted" | Section 11 (QWK, per-class recall, confusion matrix, calibration) |
| "Streamlit deployment is MANDATORY" | Section 12 (upload -> inference -> triage card + heatmap) |
| "complete usable flow: input/data entry, model execution, understandable presentation" | Section 12 flow map |
| "Project MUST be tested using representative inputs" | Section 11.4 (held-out split + self-collected local test set) |

---

## 4. Technical design - the three layers

### 4.1 Layer 1 - Model

- **Input:** single retinal fundus image, RGB, resized to 224x224 (or 380x380 if memory allows).
- **Backbone:** EfficientNet-B0 pretrained on ImageNet (transfer learning).
- **Head:** global average pool -> dropout -> linear layer with **5 outputs** (grades 0-4).
- **Loss - the key decision.** Two candidate approaches:

  **Option A (recommended): ordinal regression via cumulative-link / CORAL-style loss.**
  Encode the label as a set of 4 binary "is grade >= k" thresholds and predict them with sigmoid outputs, then derive class probabilities. This directly respects the ordinal structure and tends to improve kappa.

  **Option B (simpler fallback): soft cross-entropy with ordinal distance smoothing.** Standard softmax but move a fraction of label mass to neighbouring grades so the model is not over-penalised for off-by-one errors. Easier to implement, nearly as good.

  Start with Option B to get a working baseline, then upgrade to Option A and report the kappa difference. That comparison is itself a result.

- **Class imbalance:** grade 0 is by far the largest class; grades 3-4 are rare. Use a weighted sampler or class-weighted loss, and always report **per-class recall**, not just overall accuracy. A model that predicts grade 0 for everything scores deceptively high on accuracy but is clinically useless - say this explicitly in the report.

### 4.2 Layer 2 - Analysis (the grade-maker)

This layer is what separates the project from a Kaggle notebook.

1. **Ordinal-aware post-processing.** Convert raw outputs to a monotonic probability vector. Optionally use the **expected grade** (sum of k * p_k) as a continuous severity score.
2. **Attribution / explainability.** Use **Grad-CAM** to overlay which retinal regions drove the prediction. This is clinically interpretable: the overlay should land on haemorrhages/exudates. Include a caveat that Grad-CAM is a coarse saliency map, not a clinical diagnosis.
3. **Confidence calibration.** Report reliability (predicted confidence vs observed accuracy). Optionally apply temperature scaling on a validation split. Calibration matters because a triage system that is confidently wrong is dangerous.
4. **Uncertainty flag.** If the max class probability is below a threshold, or if the top-2 classes are close, output "uncertain - manual review" instead of a forced grade. This is a genuinely strong design point.

### 4.3 Layer 3 - Decision (triage)

Map the predicted grade (and uncertainty) to an actionable window:

| Predicted grade | Triage tier | Recommended action |
|---|---|---|
| 0 (No DR) | Routine | Rescreen in 12 months |
| 1 (Mild) | Monitor | Rescreen in 6-12 months |
| 2 (Moderate) | Soon | Ophthalmologist referral within 4-6 weeks |
| 3 (Severe) | Urgent | Referral within 1-2 weeks |
| 4 (Proliferative) | Emergency | Immediate referral |
| low confidence / ambiguous | Manual review | Flag for specialist reading |

The decision layer is deterministic given the model output - that is acceptable and expected. The **learning** is entirely in the model; the decision layer translates a learned prediction into a real-world action, which is exactly what the statement asks for.

---

## 5. Difficulty and risk assessment

### What is easy
- Dataset download and loading.
- Standard transfer-learning training loop.
- Basic Streamlit image upload + prediction.

### What will hurt
- **Class imbalance.** Grades 3 and 4 have few hundred images each; the model will drift toward grade 0. Budget real time for sampler/loss tuning.
- **Off-by-one errors.** Ordinal grading means grade 1 predicted as grade 2 is a mild error, but grade 0 predicted as grade 4 is catastrophic. QWK captures this; accuracy does not. Do not optimise accuracy.
- **Image quality variance.** Fundus images vary in illumination, field of view, and black border. Preprocessing (crop black borders, Ben Graham colour normalisation) matters more than people expect.
- **Grad-CAM on EfficientNet.** Requires hooking the correct target layer; getting a meaningful overlay takes iteration.

### Where groups fail
- Reporting accuracy only (hides the class-imbalance failure).
- No comparison to a baseline, so there is no evidence of learning.
- Training with unnormalised/uncropped images and blaming the model.
- Claiming clinical validity without the "not a diagnostic device" disclaimer.

### Effort estimate (3 people, ~6 weeks part-time)

| Phase | Person-days |
|---|---|
| Phase 1 dataset gate | 1 |
| Phase 2 EDA + preprocessing | 4 |
| Phase 3 baseline | 4 |
| Phase 4 main model + tuning | 10 |
| Phase 5 analysis + decision | 8 |
| Phase 6 Streamlit | 8 |
| Phase 7 report/slides/demo | 8 |

---

## 6. Dataset

### 6.1 Primary - APTOS 2019 Blindness Detection

- **Source:** Kaggle competition "APTOS 2019 Blindness Detection". Live competition page confirmed. Requires a Kaggle account and clicking **Join Competition** to accept the rules before the API download works.
- **Content:** colour fundus photographs with integer severity labels 0-4. Reported scale is a few thousand training images (3000-4000 range); confirm the exact count after download and report the true number.
- **Features:** raw RGB retinal images, varied resolution.
- **Target:** integer disease grade 0-4.
- **Rationale:** the labels used in this competition follow the standard 5-level diabetic retinopathy grading scale, which is exactly the clinical scale the project needs.

**Download (after joining the competition):**

```powershell
kaggle competitions download -c aptos2019-blindness-detection -p data/raw
Expand-Archive -Path data/raw/aptos2019-blindness-detection.zip -DestinationPath data/raw/aptos2019 -Force
```

**Verify the gate:**

```powershell
python -c "import pandas as pd; df=pd.read_csv('data/raw/aptos2019/train.csv'); print(df.shape); print(df['diagnosis'].value_counts().sort_index())"
```

If the value counts show any class under ~50, note it as a limitation and plan oversampling.

### 6.2 Backup datasets

- **Messidor-2** - public retinal dataset, usually via a request form; useful for external validation.
- **EyePACS** - Kaggle "Diabetic Retinopathy Detection" competition, much larger (tens of thousands of images). Use if APTOS access fails.

### 6.3 Self-collected local test set (mandatory for the limitations section)

Collect 30-50 fundus images available from public teaching sources or, if your group has any clinical access, real screening images with permission. Label them yourself where possible, or use them purely for qualitative assessment. Report benchmark vs local performance - the gap is the honest headline.

**If you cannot source local fundus images:** use a held-out slice of a *different* dataset (e.g. train on APTOS, test on Messidor) as the generalisation experiment. Document this choice.

### 6.4 Preprocessing pipeline

1. Crop black borders (threshold + largest contour).
2. Resize to model input size.
3. Ben Graham colour normalisation (optional but recommended): subtract local average colour to equalise illumination.
4. Augment: random horizontal/vertical flips, small rotations (fundus orientation is arbitrary), brightness/contrast jitter. **Do not** use aggressive crops that cut the optic disc/lesions.
5. Normalise with ImageNet mean/std (since the backbone is ImageNet-pretrained).

### 6.5 Documented limitations (required section)

- APTOS labels are cameraman/ophthalmologist graded and may carry inter-grader variability.
- Class imbalance biases the model toward no-DR.
- Single-source data - performance on local cameras is expected to be lower.
- Grad-CAM is a saliency aid, not a diagnosis.
- The system is a **triage aid, not a medical device**; a disclaimer is mandatory in the app and report.

---

## 7. Environment

Complete Phase 0 in `00_SETUP_AND_DECISION.md` first, then:

```powershell
pip install timm scikit-image grad-cam
```

Pins to watch: `timm` for EfficientNet weights; `grad-cam` for the pytorch-grad-cam package. If `grad-cam` internals conflict with your torch version, implement Grad-CAM manually (it is ~30 lines) - this is also better for the viva because you can explain it.

---

## 8. Roadmap: Phase 0 to Phase 7

### Phase 0 - Environment (before 11 Oct)
- Complete `00_SETUP_AND_DECISION.md` Phase 0. Confirm `CUDA: True`.
- **Exit:** imports work, GPU active, requirements frozen.

### Phase 1 - Dataset access gate (before 25 Oct) - BLOCKING
- Join the APTOS competition, download, extract.
- Run the verify command in Section 6.1. Print class distribution.
- Load 10 images, confirm they open and the labels align with filenames.
- Confirm the licence permits academic use.
- **Ownership check:** confirm you are not reusing a published solution. You will write your own training loop.
- **Exit:** dataset on disk, distribution known, limitations drafted. If APTOS fails, switch to EyePACS now - do not delay.

### Phase 2 - EDA and preprocessing (days 1-4)
- `notebooks/01_eda.ipynb`: class counts, sample grid, image size histogram, illumination check.
- Implement `src/data/preprocess.py`: black-border crop + resize.
- Implement `src/data/dataset.py`: a `torch.utils.data.Dataset` returning (image, grade).
- Implement augmentation with albumentations.
- **Exit:** can draw a batch and visualise it correctly augmented.

### Phase 3 - Baseline (days 5-8) - deliberately simple
- Train a small CNN from scratch (3-4 conv blocks) OR a linear head on frozen ImageNet features.
- Report the weak numbers. Expect modest accuracy and near-zero recall on grades 3-4.
- **Why this matters:** it establishes the "our simple baseline vs our final model" comparison the statement's "evidence of evaluation" expects.
- **Exit:** baseline metrics logged to `reports/figures/baseline_metrics.json`.

### Phase 4 - Main model + tuning (days 9-18)
- EfficientNet-B0 transfer learning, two-stage schedule:
  1. Freeze backbone, train head (few epochs).
  2. Unfreeze, fine-tune at low LR.
- Mixed precision, `cudnn.benchmark = True`.
- Track val QWK each epoch, save best checkpoint to `models/effnet_b0_best.pth`.
- Ablate: (a) CE vs ordinal loss, (b) with/without class weighting, (c) 224 vs 380 px.
- **Exit:** a trained model beating the baseline on QWK, with an ablation table.

### Phase 5 - Analysis + decision layers (days 19-26)
- Implement ordinal post-processing and expected-grade score.
- Implement Grad-CAM overlays; sanity-check that saliency lands on lesions.
- Implement temperature scaling; produce a reliability diagram.
- Implement the uncertainty flag and the triage mapping table (Section 4.3).
- **Exit:** every prediction returns grade + confidence + triage tier + heatmap.

### Phase 6 - Streamlit app (days 27-34) - MANDATORY
- Build per Section 12.
- `@st.cache_resource` for the model, `@st.cache_data` for reference data.
- **Exit:** a stranger can upload an image and understand the result without explanation.

### Phase 7 - Evidence, report, slides, demo (days 35-42, before 06 Dec)
- Run final evaluation on the held-out split and the external/local set. Freeze the numbers.
- Export every figure to `reports/figures/`.
- Write the report per Section 13. Build the pipeline diagram.
- Build slides. Rehearse the live demo twice.
- **Exit:** single zip assembled, demo runs offline from a fresh venv.

---

## 9. Repository structure

```
diabetic-retinopathy-triage/
  data/
    raw/aptos2019/            train.csv, train_images/
    processed/                cached resized tensors
    external/                 local/other-source test images
  notebooks/
    01_eda.ipynb
    02_baseline.ipynb
    03_training.ipynb
    04_evaluation.ipynb
  src/
    config.py
    data/
      preprocess.py           border crop, colour normalisation
      dataset.py              DRDataset
      augment.py              albumentations transforms
    models/
      backbone.py             efficientnet_b0 builder + ordinal head
      ordinal.py              CORAL / cumulative-link loss
    training/
      train.py
      losses.py
      sampler.py              weighted sampler
    evaluation/
      metrics.py              qwk, per-class recall, calibration
      gradcam.py              attribution
      plots.py                confusion matrix, reliability diagram
    inference/
      predict.py              single-image predict() API
  app/
    streamlit_app.py
    pages/
      1_Single_Image.py
      2_Batch_Screening.py
    components/
      triage_card.py
      heatmap.py
    assets/style.css
  models/
    effnet_b0_best.pth
  reports/
    figures/
    decision_log.md
    report.md
  requirements.txt
  README.md
```

---

## 10. Training configuration (tuned for T1200 4 GB)

Starting point - adjust after measuring actual memory.

```python
# src/config.py
SEED = 42
IMG_SIZE = 224              # raise to 380 only if memory allows
BATCH_SIZE = 16             # try 32 after enabling AMP
NUM_WORKERS = 4             # Windows: keep <= 4; if DataLoader hangs, set 0
EPOCHS_HEAD = 5             # frozen backbone
EPOCHS_FINETUNE = 20
LR_HEAD = 1e-3
LR_FINETUNE = 1e-4
WEIGHT_DECAY = 1e-4
AMP = True                  # mixed precision, essential on 4 GB
USE_CLASS_WEIGHTS = True
PRETRAINED = "efficientnet_b0"
NUM_CLASSES = 5
```

```python
# training essentials
import torch
torch.backends.cudnn.benchmark = True
scaler = torch.amp.GradScaler("cuda")
with torch.amp.autocast("cuda"):
    logits = model(batch)
    loss = criterion(logits, labels)
scaler.scale(loss).backward()
scaler.step(optimizer); scaler.update(); optimizer.zero_grad()
```

- If OOM: `BATCH_SIZE=8`, then `IMG_SIZE=192`, then gradient accumulation.
- Monitor: `nvidia-smi -l 2` in a second terminal.
- Expected wall-clock on the T1200: roughly 1-3 hours for the full schedule at 224 px, assuming the dataset is in the low thousands. Re-measure, do not assume.

---

## 11. Evaluation harness

### 11.1 Metrics and why

| Metric | Why it is the right choice here |
|---|---|
| **Quadratic Weighted Kappa (QWK)** | The standard for ordinal grading; penalises large mistakes more than off-by-one. This is the headline metric. |
| **Per-class recall** | The clinically critical metric. Recall on grades 3-4 is what stops blindness. |
| **Confusion matrix** | Shows exactly which grades get confused; drives error analysis. |
| **Macro F1** | Guards against the majority-class trap that accuracy hides. |
| **Calibration (reliability diagram)** | A triage system must not be confidently wrong. |

**Explicit anti-pattern to state in the report:** plain accuracy is misleading under class imbalance and will NOT be used as the primary metric.

### 11.2 Splits
- Train / validation / test from APTOS (stratified by grade).
- External test set (Messidor or local images) for generalisation.

### 11.3 Analysis to report
- QWK: baseline vs final vs ablations (table).
- Per-class recall before/after class weighting (table).
- Confusion matrix heatmaps (baseline vs final).
- Reliability diagram with and without temperature scaling.
- 5 qualitative examples: correct, off-by-one, catastrophic error, uncertain-flag case, Grad-CAM showing lesion focus.

### 11.4 Representative-input testing
- Feed blurred, dark, and incorrectly-oriented images to confirm graceful behaviour (uncertain flag rather than confident nonsense).

### 11.5 Target numbers (aspirational, state as such)
- QWK >= 0.80 on the APTOS test split.
- Recall >= 0.70 on grade 3 and grade 4 combined (urgent tiers).
- Any number below these is fine if interpreted honestly - the statement rewards interpretation, not perfection.

---

## 12. Streamlit application specification

### Page 1 - Single Image Triage (primary demo)

Flow (satisfies "input -> model execution -> understandable presentation"):

1. **Input:** `st.file_uploader` (jpg/png). Show original image.
2. **Preprocess preview:** display the cropped/normalised image so the user sees what the model sees.
3. **Run:** a "Grade image" button triggers inference (model loaded via `@st.cache_resource`).
4. **Outputs:**
   - Predicted grade label (0-4) with a human name.
   - Confidence bar and expected-grade score.
   - **Triage card**: colour-coded tier (Routine / Monitor / Soon / Urgent / Emergency) + recommended action + timeframe.
   - **Grad-CAM overlay** next to the original.
   - Uncertainty warning if triggered.
   - "Download result as CSV" button (evidence artifact).
5. **Disclaimer:** persistent banner - "Triage aid only. Not a medical diagnosis. Consult a specialist."

### Page 2 - Batch Screening (camp workflow)

1. Upload multiple images (or a zip).
2. Table: filename | grade | confidence | triage tier.
3. Sortable by urgency; count per tier.
4. Download the full triage list as CSV - this is the artefact a camp coordinator actually uses.

### Page 3 - Model & Metrics (transparency)

1. Display the evaluation results (QWK, confusion matrix, reliability diagram).
2. Short methodology summary.
3. Limitations list.

This page is worth real marks: it proves the evaluation is real and not decorative.

---

## 13. Report, slides, and viva

### 13.1 Report sections (map to statement deliverables)
1. Problem statement - DR, preventable blindness, screening gap.
2. Real-world justification - technicians/GPs, triage decision, cost of inaction.
3. Selected path - Path A, why software.
4. Objectives.
5. Methodology - transfer learning, ordinal classification, two-stage training.
6. Algorithm description - EfficientNet-B0, ordinal loss derivation, why appropriate.
7. System architecture - **pipeline diagram** (image -> preprocess -> CNN -> ordinal post-process -> Grad-CAM + calibration -> triage).
8. Dataset description - APTOS, source, distribution, preprocessing, limitations.
9. Implementation details - framework, hyperparameters, hardware (T1200), AMP.
10. Testing - splits, external set, representative inputs.
11. Results - QWK, per-class recall, confusion matrices, calibration, ablation table, qualitative examples.
12. Limitations - imbalance, single source, Grad-CAM caveat, not a medical device.
13. Conclusion and future work - larger datasets, multi-camera validation, clinician study.

### 13.2 Slides (15-18)
Problem -> motivation -> solution -> DR grading scale -> dataset -> architecture diagram -> ordinal loss -> training -> results (QWK table + confusion matrix) -> Grad-CAM examples -> triage design -> demo -> limitations -> conclusion.

### 13.3 Likely viva questions and prepared answers
- *Why kappa and not accuracy?* -> ordinal + imbalance; accuracy is dominated by grade 0.
- *Why ordinal loss?* -> grade order is meaningful; CE treats grade 1 as equally wrong as grade 4.
- *How do you handle rare grades?* -> weighted sampler + per-class recall reporting.
- *Why can't you use this in a clinic?* -> single-source data, no regulatory validation, needs a clinician study.
- *Is this just a Kaggle notebook?* -> no; and point to the analysis/decision layers and the ablation study.
- *What does Grad-CAM add?* -> interpretability and trust; shows the model attends to lesions rather than borders.
- *What would you do with more time?* -> external validation, ensembling, and a clinician-in-the-loop evaluation.

### 13.4 Work split (3 members)

| Member | Owns |
|---|---|
| A - ML lead | Preprocessing, dataset, baseline, main model, ablations |
| B - Integration/UI lead | Grad-CAM, calibration, Streamlit app, CSV export, demo rehearsal |
| C - Evaluation/report lead | Metric harness, external test set, all figures/tables, report, slides, decision log |

Cross-training requirement: each member must be able to run the full pipeline and explain any section, per "Every group member is responsible for understanding the submitted work equally."

---

*End of Idea 01. See `00_SETUP_AND_DECISION.md` for shared setup and `02_`/`03_` for alternatives.*
