# 02 - Driver Drowsiness and Distraction Detection with Temporal Risk Scoring

**Path:** A (Software-Based)
**Primary techniques:** CNN eye/mouth-state classification + facial landmarks + temporal aggregation
**Deployment:** Streamlit (mandatory)
**Group size:** 3

Companion file: `00_SETUP_AND_DECISION.md` (environment, timeline, conventions).

> **Dataset-risk warning.** The MRL Eye Dataset is the natural primary dataset for this idea, but its public download could NOT be verified from this machine. Treat Phase 1 as a hard gate and plan for the State Farm fallback and a self-captured webcam set. This is the only idea of the three with a real dataset-access question mark.

---

## 1. Verdict

The most impressive **demo** of the three and the clearest argument that you are not "just doing classification", because the temporal risk score is provably a second layer on top of per-frame predictions. A single frame of closed eyes is meaningless; a rolling PERCLOS window is a real drowsiness signal. That argument alone satisfies the statement's "more substantial than simple image classification" bar. The cost is dataset uncertainty and a more involved pipeline (video, tracking, sliding windows).

| Dimension | Score /5 | Note |
|---|---|---|
| Uniqueness | 3 | Common topic; differentiated by temporal risk scoring + distraction fusion |
| Feasibility on 4 GB / 6 weeks | 5 | Small models, tiny crops; very GPU-friendly |
| Metric depth | 5 | Per-state F1 + false-alarms-per-hour is a genuine systems metric |
| Demo impact | 5 | Video playback with live risk graph is the best demo of the three |
| Dataset risk | 2 | MRL unverified; State Farm exists but is a different task; self-capture needed |

---

## 2. Real-world problem and justification

**Problem.** Driver fatigue and distraction are leading contributors to fatal road collisions. Fatigue degrades reaction time and attention progressively; the danger is that the driver does not perceive their own decline. Distraction (phone use, looking away) compounds this.

**Why it matters (Pakistan framing).** Long-haul routes such as the M-2 motorway involve hours of monotonous driving, and commercial fleets have no affordable fatigue monitoring. Commercial drowsiness detectors are expensive and not deployed in local fleets. A low-cost camera-based system running on commodity hardware is directly useful.

**Stakeholder and decision.** The user is a **fleet operator or an individual driver**. The decision is: *is it safe to continue driving right now, or should the driver pull over?* The output is a graded alert, not a raw label.

**Why ML is appropriate here.** Eye-closure and yawning must be inferred from images under varying illumination, glasses, head pose, and skin tone - not expressible as fixed pixel rules. A CNN learns robust appearance features. Critically, the *temporal* risk is a sequence problem: the meaning of a frame depends on the frames around it.

**Cost of inaction.** A microsleep at motorway speed covers hundreds of metres. Early warning has direct safety value.

---

## 3. Requirement-compliance matrix

| Statement requirement (quoted) | Satisfied by |
|---|---|
| "MUST implement at least ONE Machine Learning algorithm/Neural Network" | CNN eye-state classifier + (optional) YOLO phone/distraction detector (Section 4.1) |
| "MUST NOT be merely a conventional software application ... or rule-based program with an AI-related label" | Core signal is learned per-frame prediction; PERCLOS aggregation is the analysis layer |
| "algorithm MUST be justified against a specific real-world use case" | Section 2 - motorway fatigue, no affordable incumbent solution, sequence nature of drowsiness |
| "explain ... what input data ... what output ... how the output supports the decision" | Section 4 (input: video/webcam; output: risk tier per window) |
| "Generic statements ... will NOT be accepted" | Justification is structural: temporal aggregation, PERCLOS, false-alarm constraint |
| "A suitable dataset MUST be identified and documented ... limitations" | Section 6 |
| "evaluated using appropriate performance metrics. Results MUST be reported and interpreted" | Section 11 (F1, false alarms/hour, latency) |
| "Streamlit deployment is MANDATORY" | Section 12 |
| "complete usable flow: input -> execution -> understandable presentation" | Section 12 flow map |
| "tested using representative inputs" | Section 11.4 (varied lighting/glasses/webcam self-test) |

---

## 4. Technical design - the three layers

### 4.1 Layer 1 - Model(s)

Two detection components; you may implement one and document the other as future work, but both together is stronger.

**(a) Eye-state classifier (core).**
- **Input:** cropped eye region (from the MRL Eye Dataset this is already an eye crop; from video you extract it via landmarks).
- **Architecture:** small CNN (a 4-block CNN or MobileNetV3-small / a tiny pretrained backbone).
- **Output:** binary - open / closed.
- **Why a dedicated eye classifier rather than landmark math alone:** appearance-based eye closure is more robust under glasses and head rotation than geometric eye-aspect-ratio thresholds. You can compare both and report which wins - that is a result.

**(b) Mouth-state classifier or yawn detector (optional second model).**
- **Input:** mouth region.
- **Output:** yawning / not yawning.
- Yawning is a complementary fatigue indicator; combining eye + mouth is a stronger signal than either alone.

**(c) Face landmark extraction (not a model you train, a tool you use).**
- **MediaPipe Face Mesh** gives eye/mouth/head-pose landmarks per frame at the frame rate.
- **Important distinction for the viva:** MediaPipe is a pretrained tool, not your ML contribution. Your trained CNN is the ML contribution. State this explicitly to avoid the "you just used a library" critique.

**(d) Distraction detector (optional, strong bonus).**
- **YOLOv8n** fine-tuned to detect a phone in hand / hands off wheel.
- Adds the "distraction" half of the project and a second learned model.

### 4.2 Layer 2 - Analysis (the grade-maker)

This is where the project becomes genuinely more than classification.

1. **Per-frame feature extraction.** For each frame: eye state, mouth state, head-pose angles (from landmarks), phone present.
2. **PERCLOS.** The proportion of frames in a sliding window in which the eyes are closed. This is a standard, literature-backed drowsiness measure. Report the window length (e.g. 30-60 seconds or N frames) and justify it.
3. **Blink analysis.** Blink duration distribution. Abnormally long blinks are a fatigue marker distinct from PERCLOS.
4. **Yawn frequency.** Yawns per minute over the window.
5. **Head-pose deviation.** Time spent looking away from the forward direction (distraction + microsleep).
6. **Fusion into a cumulative fatigue score.** A weighted combination of the above, mapped to 0-100. Justify the weights (literature values or tuned on validation). Do not present the fusion as a trained model unless it is one - be honest that it is a defined scoring function over learned features.

### 4.3 Layer 3 - Decision (alerting)

| Risk score | Tier | Action |
|---|---|---|
| 0-39 | Normal | No alert |
| 40-69 | Caution | Soft chime + visual nudge |
| 70-100 | Critical | Loud alert: "Pull over now" |

Add **hysteresis / debouncing** so the alert does not flicker: require the tier to hold for K seconds before triggering, and require a longer recovery before clearing. This is a real engineering detail examiners appreciate.

### 4.4 The "not a classifier" argument (write this in the report)

> A single-frame classifier cannot detect drowsiness; it can only detect a closed eye. Drowsiness is a temporal property. Our system converts per-frame learned predictions into a PERCLOS-based risk trajectory over a sliding window, and thresholds that trajectory. This is a sequence-level inference problem layered on top of a per-frame learned model.

That paragraph is the single most valuable sentence in the report.

---

## 5. Difficulty and risk assessment

### What is easy
- Eye-crop classification once the dataset is loaded.
- MediaPipe landmark extraction.
- Basic Streamlit video upload.

### What will hurt
- **Dataset access (MRL).** Unverified. This is the top risk. Mitigate in Phase 1.
- **Face/eye detection in video.** Under sunglasses or extreme head pose, landmark extraction fails; decide what the system does when no face is found (should degrade to "unknown", never to "alert").
- **Video processing speed.** Frame-by-frame inference over a long clip is slow; sample frames (e.g. every 3rd-5th frame) and justify the rate.
- **False-alarm tuning.** A system that beeps constantly gets switched off. You must measure false alarms per hour, not just accuracy.
- **Webcam in Streamlit.** `streamlit-webrtc` is fiddly. Make video-upload the reliable demo path and webcam a stretch goal.
- **Annotating your own driving footage.** If using self-captured video, you must log ground-truth events; be systematic.

### Where groups fail
- Reporting only per-frame accuracy and never evaluating the temporal system.
- Ignoring the no-face / occlusion case.
- Building a demo that alerts on every blink.
- Not documenting that MediaPipe is a pretrained dependency.

### Effort estimate (3 people, ~6 weeks part-time)

| Phase | Person-days |
|---|---|
| Phase 1 dataset gate (highest uncertainty) | 2-3 |
| Phase 2 EDA + preprocessing | 3 |
| Phase 3 baseline | 3 |
| Phase 4 main models + tuning | 9 |
| Phase 5 temporal analysis + fusion + alerting | 9 |
| Phase 6 Streamlit (video + webcam) | 9 |
| Phase 7 report/slides/demo | 8 |

---

## 6. Dataset

### 6.1 Primary candidate - MRL Eye Dataset (UNVERIFIED)

- **Source:** Media Research Lab (MRL), VSB - Technical University of Ostrava. The lab site hosts a "Recognizing fatigue and gaze direction" project consistent with this dataset.
- **Status:** the dataset download section did not render reliably during research, and a guessed Kaggle mirror returned 404. **Do not assume it is downloadable.** Confirm first.
- **Expected content:** tens of thousands of near-infrared eye crops labelled open/closed, with subject and condition metadata.
- **Why it fits:** it is exactly eye-state classification, at the crop level, which is what Layer 1 needs.

**Phase 1 actions:**
1. Open the MRL lab site and locate the dataset link.
2. If downloadable: download, inspect the folder structure, confirm the open/closed labelling convention.
3. If NOT downloadable: pivot immediately to State Farm (below) plus self-captured eye crops.

### 6.2 Verified-to-exist alternatives

- **State Farm Distracted Driver Detection** - Kaggle competition page confirmed live. Ten classes of `c0`-`c9` including safe driving, texting, phone use, talking, reaching, etc. This is a **distraction** dataset, not a drowsiness one - it maps cleanly to the optional distraction model (Section 4.1d) and to the "looking away / hands off wheel" features. It does NOT directly give eye-open/closed labels.
- **NTHU-DDD** (NTHU Drowsy Driver Detection) - a video dataset with drowsiness states. Usually requires a request/agreement. Worth requesting early because approval takes time.

### 6.3 Self-captured data (strongly recommended, and probably necessary)

Because the primary dataset is uncertain, plan to capture your own:

1. **Eye crops for training/augmentation.** Record yourself and volunteers (with consent) opening/closing eyes, with and without glasses, in varied light. Even 500-1000 crops meaningfully augments a training set.
2. **Driving/drowsy simulation video for testing.** Simulate drowsy behaviour (long eye closures, yawning, looking away) in a parked car or at a desk with a laptop webcam. Label the timelines (when drowsy behaviour starts/ends) so you can evaluate the temporal system.
3. **Consent and privacy.** Get written consent from anyone recorded; do not publish footage. Document this in the report - it demonstrates responsible handling of personal data.

### 6.4 Composition plan

| Purpose | Source |
|---|---|
| Eye-state training | MRL if available, else self-captured + public eye datasets |
| Distraction model | State Farm (+ self-captured) |
| Temporal evaluation | Self-captured labelled drowsy video |
| Robustness testing | Self-captured glasses/low-light/head-pose cases |

### 6.5 Preprocessing pipeline

1. Face detection (MediaPipe) -> eye/mouth landmark extraction.
2. Crop and resize eye/mouth regions to the classifier input size (e.g. 24x24 or 48x48 - these are tiny, which is why this idea is GPU-cheap).
3. Normalise; grayscale for eyes is often sufficient and faster.
4. Frame sampling for video (e.g. every 3rd frame) - document the rate and why.
5. Augmentation for training: small brightness/contrast jitter, tiny rotations, horizontal flip (**note:** flipping eyes is physically fine, but verify it does not create unrealistic samples).

### 6.6 Documented limitations (required section)

- If MRL is unavailable, eye-state data is self-collected and small - generalisation is limited.
- MediaPipe requires a detectable face; sunglasses/occlusion degrade it.
- The fusion weights are hand-tuned, not learned (state honestly) unless you replace them with a learned model.
- IR-trained models may not transfer to visible-light webcam footage - test and report the gap.
- Simulation of drowsiness in self-captured video is not real fatigue; behavioural realism is limited.

---

## 7. Environment

Complete Phase 0 in `00_SETUP_AND_DECISION.md` first, then:

```powershell
pip install mediapipe ultralytics streamlit-webrtc
```

Watch: mediapipe may pin `numpy<2`. If pip downgrades numpy inside the venv, that is acceptable; verify imports after.

If `streamlit-webrtc` is unstable on Windows, drop the webcam feature and rely on video upload - the mandatory flow is satisfied by upload.

---

## 8. Roadmap: Phase 0 to Phase 7

### Phase 0 - Environment (before 11 Oct)
- Complete `00_SETUP_AND_DECISION.md` Phase 0. Confirm `CUDA: True`.

### Phase 1 - Dataset access gate (before 25 Oct) - BLOCKING, HIGH RISK
- Attempt MRL Eye download. Record outcome.
- If unavailable: immediately pull State Farm and start self-capture in parallel.
- Request NTHU-DDD access now in case it takes time.
- Capture an initial self-recorded set (eyes + a labelled drowsy clip).
- **Exit:** a loadable eye-state dataset exists (MRL or self-captured) and a labelled temporal clip exists. If neither is possible, flag a pivot to Idea 01 or 03 - do not fake it.

### Phase 2 - EDA + preprocessing (days 1-4)
- Inspect eye crops; check class balance (often near 50/50 for eye state - good).
- Build the landmark + crop pipeline; test it on a still image and a short video.
- Build the frame sampler and the sliding-window generator.
- **Exit:** given a video, you can output a sequence of (frame, eye crop, mouth crop, head pose).

### Phase 3 - Baseline (days 5-7)
- Geometric baseline: eye-aspect-ratio (EAR) threshold from landmarks, no learning.
- Report its F1 and where it fails (glasses, head pose).
- **This baseline is gold** - it lets you argue "a simple hand-crafted rule reaches F1 0.6, our learned CNN reaches 0.9". That is compelling evidence that ML was necessary.

### Phase 4 - Main models + tuning (days 8-16)
- Train the eye-state CNN (small architecture). Report F1.
- Train the mouth/yawn classifier or the YOLO distraction model.
- Compare: EAR rule vs CNN; grayscale vs colour; IR vs visible-light if applicable.
- **Exit:** trained eye-state model + distraction model with metrics, plus a documented train/val/test split.

### Phase 5 - Temporal analysis + fusion + alerting (days 17-25)
- Implement PERCLOS over a sliding window.
- Implement blink-duration and yawn-frequency features.
- Implement head-pose deviation timing.
- Implement the fusion score, tiers, and hysteresis.
- Evaluate on the labelled self-captured clip: event detection timing, false alarms/hour, missed events.
- **Exit:** the system outputs a risk trajectory and correctly tiered alerts on the test clip.

### Phase 6 - Streamlit app (days 26-34) - MANDATORY
- Build per Section 12; ship video upload first, webcam as stretch.
- **Exit:** a clean end-to-end demo with live risk graph and alert log.

### Phase 7 - Evidence, report, slides, demo (days 35-42, before 06 Dec)
- Freeze metrics; export figures.
- Write report + pipeline diagram.
- Rehearse the demo (pre-load a clip so it never fails live).
- **Exit:** single zip assembled; demo runs from a fresh venv.

---

## 9. Repository structure

```
driver-drowsiness/
  data/
    raw/
      mrl_eye/               if available
      state_farm/            distraction
    external/
      self_captured/         eye crops + labelled video
    processed/
  notebooks/
    01_eda_eye.ipynb
    02_baseline_ear.ipynb
    03_eye_model.ipynb
    04_temporal_eval.ipynb
  src/
    config.py
    data/
      face_extract.py        MediaPipe landmarks + crops
      eye_dataset.py
      video_loader.py        frame sampling
      window.py              sliding-window generator
    models/
      eye_cnn.py
      mouth_cnn.py
      distraction_yolo.py
    analysis/
      perclos.py
      blink.py
      headpose.py
      fusion.py              risk score
      alerting.py            tiers + hysteresis
    training/
      train_eye.py
      train_distraction.py
    evaluation/
      metrics.py             F1, false alarms/hour, event latency
      plots.py               risk timeline
    inference/
      predict_frame.py
      predict_video.py
  app/
    streamlit_app.py
    pages/
      1_Video_Analysis.py
      2_Live_Webcam.py       (stretch)
      3_Model_and_Metrics.py
    components/
      risk_gauge.py
      timeline_chart.py
    assets/style.css
  models/
    eye_cnn.pth
    distraction_yolo.pt
  reports/
    figures/
    decision_log.md
    report.md
  requirements.txt
  README.md
```

---

## 10. Training configuration (tuned for T1200 4 GB)

Eye crops are tiny, so this is GPU-light and fast.

```python
# src/config.py
SEED = 42
EYE_IMG_SIZE = 48           # 24x24 is also common; 48 is a good default
BATCH_SIZE = 128            # tiny images => large batches fit easily
NUM_WORKERS = 4
EPOCHS = 20
LR = 1e-3
WEIGHT_DECAY = 1e-4
AMP = True
GRAYSCALE = True            # documents a grayscale-vs-colour ablation
FRAME_STRIDE = 3            # process every 3rd frame
WINDOW_SECONDS = 30         # PERCLOS window
PERCLOS_HIGH = 0.4          # threshold justification required
ALERT_HOLD_SECONDS = 5      # hysteresis
```

```python
# distraction YOLO (if building it)
# ultralytics handles AMP + memory automatically
# img size 416-640, batch 16, ~30-50 epochs on State Farm
```

- If OOM on YOLO: `imgsz=416`, `batch=8`.
- The eye CNN will train in minutes per epoch; the bottleneck is video inference, not training.
- Measure true inference FPS on a 30-second clip before claiming real-time capability.

---

## 11. Evaluation harness

### 11.1 Metrics and why

| Metric | Why |
|---|---|
| **Per-state F1 (open/closed, distracted/safe)** | Accuracy hides imbalance; F1 per state is honest |
| **Temporal event detection (precision/recall on drowsy episodes)** | The real question: did it catch the episode? |
| **False alarms per hour** | The metric that decides whether a driver keeps it on |
| **Detection latency** | How long after fatigue onset before the alert fires - safety-critical |
| **Inference FPS on the T1200** | Feasibility evidence for real-time use |

**Comparison table to include:** EAR rule vs CNN (per-frame); and per-frame classifier vs full temporal system (event-level). The second comparison is the proof that the temporal layer adds value.

### 11.2 Splits
- Eye-state: train/val/test split by **subject**, not by frame. Splitting by frame leaks the same face into train and test and inflates results - call this out as a methodological care point.
- Temporal: held-out self-captured clips.

### 11.3 Analysis to report
- F1 comparison table (EAR vs CNN vs CNN+temporal).
- Risk timeline plots overlaying ground-truth drowsy episodes and detected alerts.
- False alarms/hour across lighting/glasses conditions.
- Ablation: with/without yawn feature; with/without head-pose feature.
- Frame-sampling stride vs detection latency trade-off.

### 11.4 Representative-input testing
- Test under low light, with glasses, with head turned away, and with no face visible (must output "unknown", not alert).

### 11.5 Target numbers (aspirational)
- Eye-state F1 >= 0.90.
- Drowsy-event recall >= 0.85 with false alarms <= 1-2 per hour on the test clip.
- Detection latency < 5-10 seconds after the window fills.

---

## 12. Streamlit application specification

### Page 1 - Video Analysis (primary demo)

1. **Input:** `st.file_uploader` for a video clip (mp4/avi). Optional: a "use sample clip" button so the demo never depends on the examiner having a file.
2. **Run:** "Analyse" button processes frames with sampled stride; show a progress bar.
3. **Outputs:**
   - Annotated video or key-frame grid with eye/mouth bounding boxes and state labels.
   - **Live risk timeline chart** (risk score vs time) with tier bands shaded.
   - Alert log: timestamp | tier | reason (PERCLOS high, long blink, yawn cluster, looking away).
   - Summary card: peak risk, time in each tier, number of alerts.
   - Download buttons: risk timeline CSV, alert log CSV.
4. **Disclaimer:** "Assistive prototype; not a certified safety device. Does not replace a rested driver."

### Page 2 - Live Webcam (stretch)
- `streamlit-webrtc` real-time feed with overlaid state and live risk gauge.
- If unstable, hide this page and note it as attempted/future work in the report.

### Page 3 - Model & Metrics (transparency)
- Per-state F1, event-level results, false alarms/hour.
- EAR-vs-CNN comparison chart.
- Limitations list.

---

## 13. Report, slides, and viva

### 13.1 Report sections
1. Problem statement - fatigue/distraction, motorway risk.
2. Real-world justification - fleet operators, absent affordable incumbents.
3. Selected path - Path A.
4. Objectives.
5. Methodology - per-frame CNN + temporal aggregation.
6. Algorithm description - eye CNN architecture, PERCLOS, fusion scoring, why temporal.
7. System architecture - **pipeline diagram** (video -> face detect -> crops -> CNN -> per-frame states -> sliding-window features -> fusion -> risk tier -> alert).
8. Dataset description - MRL/State Farm/self-captured, distribution, limitations.
9. Implementation details - MediaPipe (pretrained dependency, not your model), torch, ultralytics, T1200.
10. Testing - subject-wise split, self-captured clips, robustness cases.
11. Results - F1 tables, event detection, false alarms/hour, latency, risk timelines.
12. Limitations - dataset size, simulated drowsiness, MediaPipe dependency, not certified.
13. Conclusion and future work - real fatigue data, driver-specific calibration, edge deployment.

### 13.2 Slides (15-18)
Problem -> motivation -> why per-frame classification is insufficient -> architecture -> eye model -> temporal layer (PERCLOS) -> fusion + tiers -> dataset -> results (F1 + event detection + false alarms) -> risk timeline -> demo -> limitations -> conclusion.

### 13.3 Likely viva questions
- *Isn't this just image classification?* -> no; PERCLOS and event-level detection are sequence-level, demonstrated by the per-frame vs temporal comparison.
- *MediaPipe already detects eyes - did you even train anything?* -> yes, a CNN trained on eye crops, compared against the geometric baseline; MediaPipe only locates the face.
- *How do you avoid alerting on every blink?* -> PERCLOS window + tier hysteresis; measured false alarms/hour.
- *What happens with sunglasses?* -> reported degradation; system outputs unknown rather than a false alert.
- *Is this real-time?* -> report measured FPS and the frame-sampling trade-off.
- *Is this safe to ship?* -> no; assistive prototype, needs real fatigue data and certification.

### 13.4 Work split (3 members)

| Member | Owns |
|---|---|
| A - ML lead | Face/eye extraction, eye CNN, distraction model, training + tuning |
| B - Temporal/UI lead | PERCLOS, fusion, alerting, Streamlit app, webcam, demo |
| C - Evaluation/report lead | Baseline (EAR), metrics harness, self-captured test clips, report, slides |

Cross-training requirement applies as in `00_SETUP_AND_DECISION.md`.

---

*End of Idea 02. See `00_SETUP_AND_DECISION.md` for shared setup and `01_`/`03_` for alternatives.*
