# 03 - Traffic Signal Timing Optimizer from Intersection Video

**Path:** A (Software-Based)
**Primary techniques:** Object detection (YOLOv8) + per-approach queue analysis + Webster's optimal signal timing
**Deployment:** Streamlit (mandatory)
**Group size:** 3

Companion file: `00_SETUP_AND_DECISION.md` (environment, timeline, conventions).

> This idea has the **highest composite score AND an instantly-accessible dataset (Roboflow Universe)**. It also has the most engineering work. Pick it only if all three members can commit.
>
> **Dataset access is SOLVED.** Roboflow Universe hosts free, instant vehicle-detection datasets with exactly the classes this project needs (car, bus, truck, motorcycle, **autorickshaw**) - no request form, no waiting. UA-DETRAC is now a **backup**, not the primary. See Section 6.

---

## 1. Verdict

The best balance of uniqueness, dataset confidence, and depth. It is a natural instance of the pattern the recommendations praised: **detection + analysis**. Detection alone is not enough; the grade comes from converting detections into PCU-weighted queue lengths and then into an optimal signal plan via Webster's formula, benchmarked against the existing fixed-time plan. Roboflow Universe provides instant access to a vehicle-detection dataset with the right classes, which removes the biggest risk of Idea 02. The cost is that video handling, per-lane ROI logic, and de-duplication are non-trivial engineering.

| Dimension | Score /5 | Note |
|---|---|---|
| Uniqueness | 5 | Likely unique in the class; specific local framing |
| Feasibility on 4 GB / 6 weeks | 4 | YOLOv8n feasible; engineering-heavy |
| Metric depth | 5 | mAP + count MAE + simulated delay reduction |
| Demo impact | 5 | Video with ROI overlays + signal plan table is very compelling |
| Dataset risk | 5 | Roboflow Universe: instant, free, correct classes |

---

## 2. Real-world problem and justification

**Problem.** Many intersections in Islamabad/Rawalpindi run fixed-time signal cycles that are set once and never adapt. Green time is spent on empty approaches while a different approach queues for several cycles. This wastes fuel, increases delay and emissions, and worsens congestion.

**Why it matters (Pakistan framing).** Rapid growth in vehicle ownership (including motorcycles and rickshaws) has outpaced signal design. Adaptive signal control systems are expensive to install because they rely on buried inductive-loop sensors. A **camera-only** system needs no road works - it can run on existing CCTV or a phone on a pole. That is the practical justification.

**Stakeholder and decision.** The user is a **traffic engineer or city traffic police operator**. The decision is: *what green-time split should this intersection run for the next cycle to minimize total delay?*

**Why ML is appropriate here.** Vehicle density and class must be inferred from pixels under varying light, occlusion, rain, and dust. Detecting and classifying vehicles (car vs motorcycle vs bus/truck vs rickshaw) is a learned visual task, not a hand-coded rule. The detection output is then converted to **PCU (Passenger Car Units)** because a bus occupies far more road capacity than a motorcycle - a purely rule-based count would badly mis-estimate saturation.

**The key subtlety to emphasise in the report:** the city's problem is not only *how many* vehicles wait but *how much road capacity* they occupy. Counting vehicles naively (one vehicle = one vehicle) over-weights motorcycles and under-weights buses. A PCU-weighted queue is a more faithful representation - and no hand-coded rule can produce the per-class detections it requires.

**Cost of inaction.** Wasted fuel, lost commuter hours, and avoidable emissions - quantifiable in the evaluation.

---

## 3. Requirement-compliance matrix

| Statement requirement (quoted) | Satisfied by |
|---|---|
| "MUST implement at least ONE Machine Learning algorithm/Neural Network" | Fine-tuned YOLOv8 vehicle detector, multi-class (Section 4.1) |
| "MUST NOT be merely a conventional software application ... rule-based" | Per-class detection is learned; Webster optimisation sits on top of the learned queue estimates |
| "algorithm MUST be justified against a specific real-world use case" | Section 2 - camera-only adaptive control, PCU capacity argument |
| "explain ... what input data ... what output ... how the output supports the decision" | Section 4 (input: intersection video/frame; output: recommended green split) |
| "Generic statements ... will NOT be accepted" | Justification is structural: PCU weighting, Webster delay formula, fixed-time baseline comparison |
| "A suitable dataset MUST be identified and documented ... limitations" | Section 6 (Roboflow Universe vehicle detection set) |
| "evaluated using appropriate performance metrics. Results MUST be reported and interpreted" | Section 11 (mAP, count MAE, simulated delay reduction) |
| "Streamlit deployment is MANDATORY" | Section 12 |
| "complete usable flow: input -> execution -> understandable presentation" | Section 12 flow map |
| "tested using representative inputs" | Section 11.4 (day/night, rain, local footage) |

---

## 4. Technical design - the three layers

### 4.1 Layer 1 - Model

- **Architecture:** **YOLOv8n** fine-tuned (from COCO-pretrained weights) for vehicle detection. Optionally YOLOv8s if the 4 GB card tolerates it at 416-640 px; start with n and only scale up if there is memory headroom.
- **Classes:** car, motorcycle, bus, truck, **autorickshaw**. Roboflow Universe provides vehicle detection datasets with exactly these classes (and the Roboflow rickshaw class directly supports the PCU argument). Map any extra source classes (van, bicycle, etc.) to your target PCU classes and document the mapping.
- **Output:** bounding boxes + class + confidence per frame.
- **Why detection and not classification:** the project needs to *locate and count* vehicles per lane; classification of a whole image cannot localise or count. This is the same "object detection + analysis" sophistication the recommendations praised.
- **Tracking (recommended):** pair YOLO with ByteTrack (built into ultralytics) to assign stable IDs and prevent double-counting when a vehicle is detected across consecutive frames. If you skip tracking, justify it (e.g. counting per frame and averaging over a window).

### 4.2 Layer 2 - Analysis (the grade-maker)

1. **Lane ROI definition.** Define polygon regions of interest per approach/lane. **Important design decision:** do NOT train a model to assign lanes; let the operator draw ROIs in the Streamlit UI (or define them in config). This is honest - lane assignment is a geometric operation, not a learning problem, and the learning is in the detection.
2. **Per-lane, per-class counts.** Count detections per class within each ROI, using tracking IDs to avoid double-counting.
3. **PCU conversion.** Multiply each class count by its PCU factor:

   | Class | Typical PCU |
   |---|---|
   | Motorcycle | 0.5 |
   | Car / van | 1.0 |
   | Rickshaw | 1.5 (verify against a source) |
   | Bus | 3.0-4.0 |
   | Truck | 3.0-4.0 |

   Cite a source for the PCU factors (e.g. an established traffic-engineering guideline) and state that factors vary by standard - this is exactly the kind of documented assumption the report needs.

4. **Saturation flow / flow ratio per approach.** Convert PCU queue to an arrival flow rate (vehicles/hour) using the time window. Compute the flow ratio y = q / s, where s is the saturation flow (a documented constant, e.g. ~1900 PCU/hour/lane - cite it).
5. **Queue / congestion severity.** Report queue length per approach as a bar chart and a congestion index.

### 4.3 Layer 3 - Decision (Webster's optimal timing)

1. **Webster's formula** computes the optimal cycle length:

   ```
   Co = (1.5 * L + 5) / (1 - Y)
   where:
     L = total lost time per cycle (start-up + clearance, seconds)
     Y = sum of critical flow ratios across all phases
   ```

2. **Effective green per phase** is then allocated in proportion to each phase's flow ratio:

   ```
   g_i = (y_i / Y) * (Co - L)
   ```

3. **Constraints and guard rails.** Clamp green times to practical minimum (e.g. 7-10 s) and maximum (e.g. 60 s), and clamp cycle length to a practical range. Document these.
4. **Baseline comparison.** Compare the Webster plan against the intersection's current **fixed-time plan**. Compute average vehicle delay using Webster's delay formula:

   ```
   d = C(1 - g/C)^2 / (2(1 - min(1, x) * g/C)) + x^2 / (2q(1 - x))   [+ correction term optional]
   where x = degree of saturation = q / (s * g/C)
   ```

5. **Output:** a recommended signal plan table (phase | green time | justification) plus a **% delay reduction** vs the fixed baseline. The delay reduction is the headline result.

**Honesty note for the report:** Webster's formula is a well-established traffic-engineering method, not your invention. Your contribution is the camera-based PCU queue estimation that feeds it, and the end-to-end integration. State this plainly - examiners reward clear scoping.

### 4.4 Why this is not "just detection"

> Our contribution is not the detector. It is the pipeline that converts per-class, PCU-weighted, de-duplicated queue estimates from video into an optimal signal plan, and quantifies the delay saving against the existing fixed-time plan. The detector is a means; the optimisation and its evaluation are the contribution.

---

## 5. Difficulty and risk assessment

### What is easy
- Getting a COCO-trained YOLO to detect cars out of the box.
- Basic counting.
- **Dataset access** - Roboflow Universe is instant and free (no request form).

### What will hurt
- **Fine-tuning quality.** Getting the detector to handle motorcycles/rickshaws at distance and under occlusion takes iteration. Roboflow's export is already YOLO-format, so this is training quality, not conversion pain.
- **Lane ROI + double-counting.** Without tracking, a vehicle counted in successive frames inflates the queue. This is the single most likely source of wrong numbers.
- **PCU justification.** You must cite factors; arbitrary numbers will be challenged.
- **Webster delay simulation.** Implementing the delay formula correctly (with the saturation correction term) is fiddly; verify against a textbook example.
- **Domain gap.** Public vehicle datasets may not match local road layout and vehicle mix (rickshaws, overloaded bikes). Collect local clips and report the gap.
- **Video processing throughput.** Detection per frame is slow on 4 GB; sample frames and/or stride.

### Where groups fail
- Reporting detection mAP only and never showing the optimisation result.
- Counting vehicles instead of PCU, then claiming capacity optimised.
- Choosing a "baseline" that is not the real fixed-time plan, making the improvement meaningless.
- No de-duplication, producing inflated queue lengths.

### Effort estimate (3 people, ~6 weeks part-time) - highest of the three

| Phase | Person-days |
|---|---|
| Phase 1 dataset gate (now low-risk) | 1 |
| Phase 2 EDA + preprocessing | 4 |
| Phase 3 baseline (pretrained YOLO, no fine-tune) | 3 |
| Phase 4 fine-tune + tune | 10 |
| Phase 5 analysis + PCU + Webster + delay sim | 10 |
| Phase 6 Streamlit | 9 |
| Phase 7 report/slides/demo | 8 |

---

## 6. Dataset

### 6.1 PRIMARY - Roboflow Universe vehicle-detection sets (VERIFIED, instant)

Roboflow Universe hosts hundreds of free object-detection datasets that export directly in YOLO format. Verified available vehicle sets (class lists shown as published):

| Dataset (Roboflow Universe) | Images | Classes |
|---|---|---|
| Vehicle Detection (by Learning) | ~22,080 | Autorickshaw, Bicycle, Bus, Car, Motorcycle, Truck |
| Vehicle Detection (10k) | ~9,620 | car, bus, truck, motorcycle |
| Vehicle Detection (by Skripsi) | ~51,000 | car, motorbike, truck |
| pakistan traffic | ~3,000 | (traffic footage) |

- **Source:** https://universe.roboflow.com (search "vehicle detection" / "pakistan traffic").
- **Why primary:** instant download, free account, YOLO-format export, and the classes needed for PCU (including **autorickshaw**, which is the local vehicle that justifies PCU weighting).
- **Target:** vehicle class + bounding box per detection.
- **Rationale:** it is a genuine multi-object detection dataset; fine-tuning a detector and building queue analysis on top is exactly the task.

**Recommended composition:** start with **Vehicle Detection (by Learning)** (~22k images, has Autorickshaw) as the main training pool, and optionally merge one more set to increase diversity.

**Download (Roboflow export):**

1. Create a free Roboflow account.
2. Open a dataset page (e.g. "Vehicle Detection" by Learning), click **Download Dataset**.
3. Choose format **YOLOv8** (or YOLOv5 PyTorch / "YOLO" text format).
4. Choose "Show download code" and copy the snippet, or download the zip directly.
5. Export into `data/raw/roboflow/`.

The dataset ships with a `data.yaml` listing classes and train/valid/test paths.

**Verify the gate:**

```powershell
# after extracting into data/raw/roboflow
python -c "import glob,os; imgs=glob.glob('data/raw/roboflow/**/images/**/*.jpg', recursive=True); print('images:', len(imgs)); print('sample:', imgs[:3])"
```

Then confirm you can load one image and its matching `.txt` label and draw the boxes.

### 6.2 BACKUP - UA-DETRAC (verified to exist, but access is a request process)

- **Source:** Computer Vision and Machine Learning Lab, University at Albany. Official benchmark page confirmed.
- **Verified facts from the official page:** 10 hours of video captured with a Canon EOS 550D at **24 different locations in Beijing and Tianjin, China**; 25 fps; resolution **960x540**; **more than 140,000 frames**; **8,250 manually annotated vehicles**; a total of **1.21 million labelled bounding boxes**. It is a real-world multi-object detection and tracking benchmark.
- **Important caveat:** the official page exposes the dataset description but **no direct download link** in the rendered page - access appears to be a request/agreement process. **Do not depend on it as the primary.** Use it only if you want its track annotations for the tracking component and access resolves quickly.
- If used: convert its XML annotations to YOLO format, and split by **sequence**, not frame (frames within a clip are highly correlated; frame-level splitting leaks data). State this care point in the report.

### 6.3 Supplementary - local footage (mandatory for the limitations section)

- Record 3-5 short clips (2-5 minutes) at a real Islamabad/Rawalpindi intersection or main road. Use a phone; keep it safe and legal (do not film from traffic lanes; obtain permission where required).
- Use these for the domain-gap experiment: detect on local clips and report the performance difference vs the public dataset.
- This is the strongest single evidence of genuine work and it directly feeds the required limitations section.

### 6.4 Quick-start note
- A COCO-pretrained YOLO already detects car/bus/truck/motorcycle - usable as the Phase 3 baseline with **zero** fine-tuning. This is intentional: it gives you a real comparison.

### 6.5 Preprocessing pipeline

1. Use Roboflow's YOLO export as-is (images + `.txt` labels + `data.yaml`).
2. Verify class indices in `data.yaml` and align them with your PCU mapping.
3. Letterbox resize to the training size (YOLO default 640; use 416 if memory-constrained).
4. Augmentation: YOLO's built-in flips, mosaic, HSV jitter, scale. Avoid horizontal flip only if it would create impossible road geometry - for detection it is usually fine.
5. Normalise / rely on ultralytics defaults.

### 6.6 Documented limitations (required section)

- Public vehicle datasets may not match local road marking styles and vehicle mix (domain gap, to be measured on local clips).
- If UA-DETRAC is used, it is Chinese traffic - a further domain gap.
- PCU factors are borrowed from a traffic-engineering standard and vary by source; stated explicitly.
- Saturation flow is a constant assumption, not measured on-site.
- ROI/lane assignment is manual, not learned.
- Webster assumes undersaturated conditions; oversaturated behaviour is out of scope.
- Occlusions and night footage may reduce detection quality.

---

## 7. Environment

Complete Phase 0 in `00_SETUP_AND_DECISION.md` first, then:

```powershell
pip install ultralytics streamlit
```

Notes:
- ultralytics bundles ByteTrack - no separate install needed for tracking.
- On the 4 GB T1200, train YOLOv8n at `imgsz=416` first; scale to 640 only if memory allows.
- `nvidia-smi -l 2` while training to watch memory.

---

## 8. Roadmap: Phase 0 to Phase 7

### Phase 0 - Environment (before 11 Oct)
- Complete `00_SETUP_AND_DECISION.md` Phase 0. Confirm `CUDA: True` (essential - CPU YOLO training is impractical).

### Phase 1 - Dataset access gate (before 25 Oct) - LOW RISK
- Create a Roboflow account; export **Vehicle Detection (by Learning)** (~22k images, includes Autorickshaw) in YOLOv8 format into `data/raw/roboflow/`.
- Run the verify command in Section 6.1; open `data.yaml` and confirm classes.
- Confirm the licence permits academic use.
- Record 1 placeholder local clip if possible.
- **Exit:** Roboflow dataset on disk, `data.yaml` parsed, classes aligned with PCU mapping. UA-DETRAC is optional and only if access is quick.

### Phase 2 - EDA + preprocessing (days 1-4)
- `notebooks/01_eda.ipynb`: class distribution, sample frames with boxes, image size histogram.
- Verify the train/val/test split shipped by Roboflow; adjust if needed (avoid duplicate/near-duplicate frames across splits).
- Implement frame sampling (stride) and the ROI overlay utility.
- **Exit:** a YOLO-format dataset that trains successfully for a few iterations.

### Phase 3 - Baseline (days 5-7) - deliberately simple
- Run **COCO-pretrained YOLOv8n** with no fine-tuning on the test set. Report mAP and count errors.
- This baseline is your evidence that fine-tuning was worthwhile.
- **Exit:** baseline metrics logged.

### Phase 4 - Fine-tune + tune (days 8-17)
- Fine-tune YOLOv8n on the Roboflow data; then try YOLOv8s if memory allows.
- Add ByteTrack for de-duplicated counting.
- Ablate: n vs s; 416 vs 640; with/without tracking; with/without merged second dataset.
- **Exit:** a fine-tuned detector beating the COCO baseline, with an ablation table.

### Phase 5 - Analysis + Webster + delay simulation (days 18-27)
- Implement per-ROI, per-class counting with track de-duplication.
- Implement PCU conversion (cite factors).
- Implement flow ratio -> Webster cycle -> green split, with clamps.
- Implement the Webster delay formula; **validate against a textbook worked example** before trusting it.
- Implement the fixed-time baseline and produce the % delay reduction.
- **Exit:** given a clip, output a recommended plan and a delay comparison.

### Phase 6 - Streamlit app (days 28-36) - MANDATORY
- Build per Section 12, including the ROI drawing interaction.
- **Exit:** a reviewer can upload a clip, see per-lane counts and a signal plan, and compare to the baseline.

### Phase 7 - Evidence, report, slides, demo (days 37-42, before 06 Dec)
- Freeze metrics on the Roboflow test split and the local clips.
- Write report + the pipeline diagram.
- Rehearse with a pre-loaded clip (never rely on live video upload during the demo).
- **Exit:** single zip assembled; demo runs from a fresh venv.

---

## 9. Repository structure

```
traffic-signal-optimizer/
  data/
    raw/roboflow/            primary vehicle detection export (images/ labels/ data.yaml)
    raw/uadetrac/            optional backup
    processed/yolo/          merged + relabelled data.yaml
    external/local_clips/    self-recorded footage
  notebooks/
    01_eda.ipynb
    02_baseline_yolo.ipynb
    03_finetune.ipynb
    04_webster_eval.ipynb
  src/
    config.py                PCU factors, saturation flow, lost time, clamps
    data/
      roboflow_load.py       read Roboflow YOLO export, align classes
      frames.py              extraction, striding
      roi.py                 polygon ROIs, point-in-polygon
    models/
      detector.py            YOLO load + train + predict wrapper
      tracking.py            ByteTrack integration
    analysis/
      counting.py            per-ROI per-class counts + de-dup
      pcu.py                 PCU conversion
      flow.py                flow ratios, saturation
      queue.py               queue/congestion metrics
    decision/
      webster.py             cycle length, green split
      delay.py               Webster delay formula + baseline
      baseline.py            fixed-time plan
    evaluation/
      metrics.py             mAP, count MAE, delay reduction
      plots.py               ROI overlays, counts, delay comparison
    inference/
      analyze_video.py
  app/
    streamlit_app.py
    pages/
      1_Intersection_Analysis.py
      2_Signal_Plan.py
      3_Model_and_Metrics.py
    components/
      roi_editor.py          draw ROIs on a frame
      plan_table.py
      delay_chart.py
    assets/style.css
  models/
    yolov8n_traffic.pt
  reports/
    figures/
    decision_log.md
    report.md
  requirements.txt
  README.md
```

---

## 10. Training configuration (tuned for T1200 4 GB)

```python
# src/config.py
SEED = 42
MODEL = "yolov8n.pt"         # scale to yolov8s.pt only if memory allows
IMG_SIZE = 416               # 640 if memory allows
EPOCHS = 50
BATCH = 16                   # drop to 8 on OOM
PATIENCE = 15                # early stopping
DEVICE = 0                   # GPU

# ---- analysis / decision constants (document + cite) ----
PCU = {"motorcycle": 0.5, "car": 1.0, "van": 1.0, "autorickshaw": 1.5,
       "rickshaw": 1.5, "bus": 3.5, "truck": 3.5}
SATURATION_FLOW = 1900       # PCU/hour/lane - CITE SOURCE
LOST_TIME_PER_PHASE = 4      # seconds - start-up + clearance
MIN_GREEN = 7                # seconds
MAX_GREEN = 60               # seconds
MIN_CYCLE = 40
MAX_CYCLE = 120
FRAME_STRIDE = 3
```

Training command (ultralytics):

```python
from ultralytics import YOLO
model = YOLO("yolov8n.pt")
model.train(data="data/raw/roboflow/data.yaml", imgsz=416, batch=16,
            epochs=50, patience=15, device=0, seed=42)
```

- ultralytics manages AMP automatically.
- If OOM: `batch=8`, then `imgsz=320`.
- Expect several hours for 50 epochs on the T1200 at 416 px - measure; do not assume.
- The bottleneck is video inference, not training. Measure inference FPS on a test clip.

---

## 11. Evaluation harness

### 11.1 Metrics and why

| Metric | Why |
|---|---|
| **mAP@0.5 and mAP@0.5:0.95** | Standard detection quality; the statement's "appropriate metrics" for detection |
| **Per-class AP** | Motorcycles/rickshaws are harder; per-class exposes it |
| **Vehicle-count MAE vs hand count** | Detection mAP does not guarantee correct counts; validate counts directly |
| **% Average Delay Reduction vs fixed-time baseline** | The decision-layer payoff - the headline result |
| **Congestion index / queue length per approach** | Interpretable traffic metric |

**Do not report mAP alone.** The project's claim is about signal timing; you must show the delay improvement.

### 11.2 Splits
- Roboflow ships train/valid/test - verify no near-duplicate frames cross splits; adjust if needed.
- Local clips: an untouched external test set.

### 11.3 Analysis to report
- Baseline (COCO YOLOv8n) vs fine-tuned vs fine-tuned+tracking (mAP + count MAE table).
- Correlation scatter: predicted count vs hand-counted ground truth per lane.
- Delay comparison table and chart: fixed-time vs Webster, per approach and overall.
- Ablation: with/without PCU weighting (show that naive counting gives a worse plan).
- Domain-gap table: public dataset test vs local clips.

### 11.4 Representative-input testing
- Day vs dusk vs night; light vs heavy traffic; local footage with rickshaws.
- Report degradation honestly.

### 11.5 Target numbers (aspirational)
- Fine-tuned mAP@0.5 >= 0.80 on the Roboflow test split.
- Count MAE within ~1-2 vehicles per lane per window.
- Average delay reduction >= 15-25% vs the fixed-time baseline (state assumptions).

---

## 12. Streamlit application specification

### Page 1 - Intersection Analysis (primary demo)

1. **Input:** upload a video clip, OR select a bundled sample clip.
2. **ROI editor:** show the first frame; let the user draw lane/approach polygons (`streamlit-drawable-canvas` or click-based polygon). Persist ROIs in session state.
3. **Run:** "Analyse" processes sampled frames with YOLO + ByteTrack; progress bar.
4. **Outputs:**
   - Annotated frame with boxes + ROI overlays + lane labels.
   - Per-lane, per-class count table.
   - Per-lane PCU queue bar chart.
   - Congestion index per approach.
   - Download counts as CSV.

### Page 2 - Signal Plan (the decision demo)

1. Show the current fixed-time plan (input or default).
2. Show the recommended Webster plan: cycle length + green split per phase, with the clamps applied.
3. **Delay comparison chart:** fixed-time vs Webster, per approach and overall, with % reduction.
4. Explanation panel: flow ratios, lost time, saturation flow assumptions (transparency).
5. Download the recommended plan and comparison as CSV/PDF.

### Page 3 - Model & Metrics (transparency)
- mAP table, count-MAE scatter, domain-gap table.
- Ablation table.
- Limitations list.

---

## 13. Report, slides, and viva

### 13.1 Report sections
1. Problem statement - fixed-time signals, wasted green, congestion.
2. Real-world justification - camera-only (no loop sensors), local traffic growth.
3. Selected path - Path A.
4. Objectives.
5. Methodology - detection + PCU + Webster.
6. Algorithm description - YOLOv8 detection, ByteTrack, PCU conversion, Webster cycle/green and delay formulas (with the equations written out).
7. System architecture - **pipeline diagram** (video -> detector -> tracker -> per-ROI counts -> PCU -> flow ratios -> Webster -> plan + delay comparison -> UI).
8. Dataset description - Roboflow Universe vehicle detection set (composition, classes), subset, local clips, limitations.
9. Implementation details - ultralytics, T1200, imgsz/batch, frame stride.
10. Testing - train/val/test split, local clips, day/night.
11. Results - mAP table, count MAE, delay reduction, ablations, domain gap.
12. Limitations - domain gap, PCU assumptions, saturation constant, ROI manual, Webster undersaturated assumption.
13. Conclusion and future work - real-time deployment, multi-intersection coordination, reinforcement-learning adaptive control.

### 13.2 Slides (15-18)
Problem -> motivation -> camera-only advantage -> architecture -> detector + tracking -> PCU insight -> Webster optimisation -> dataset -> results (mAP + count MAE + delay reduction) -> ROI demo -> limitations -> conclusion.

### 13.3 Likely viva questions
- *Why count PCUs and not vehicles?* -> capacity is about road space; a bus is not a motorcycle; naive counts produce a wrong plan.
- *Did you invent Webster's formula?* -> no; cite it. Our contribution is the camera-based PCU queue estimation and the end-to-end pipeline.
- *How do you avoid double-counting?* -> tracking IDs; or per-frame counts averaged over a window.
- *How did you assign lanes?* -> operator-drawn ROIs; lane assignment is geometric, not learned.
- *Where did your dataset come from?* -> Roboflow Universe vehicle detection sets; documented class composition and licence; local clips for the domain-gap test.
- *What about night/rain/rickshaws?* -> reported degradation on local clips; domain gap quantified.
- *Is the baseline fair?* -> the real intersection's fixed-time plan is the baseline; the delay formula is validated against a textbook example.
- *Does it run live?* -> report measured inference FPS and the frame-stride trade-off.

### 13.4 Work split (3 members)

| Member | Owns |
|---|---|
| A - Detection lead | Roboflow export + class alignment, YOLO fine-tuning, tracking, ablations |
| B - Optimization/UI lead | PCU, flow ratios, Webster + delay modules, Streamlit app, ROI editor, demo |
| C - Evaluation/report lead | Baseline, count ground-truth, delay validation, local clips, report, slides |

Cross-training requirement applies as in `00_SETUP_AND_DECISION.md`.

---

*End of Idea 03. See `00_SETUP_AND_DECISION.md` for shared setup and `01_`/`02_` for alternatives.*
