# Live Demonstration Script

## Before the demo (do this once)

```powershell
.\.venv\Scripts\streamlit.exe run app\streamlit_app.py
```

Open http://localhost:8501 in a browser. Leave the terminal open.

> If the laptop is warm, the demo still works: single-image/short-clip inference
> is light. The synthetic demo needs no video and no GPU.

## Click path (5 minutes)

### 1. Home
- **Play the demo video**: the home page has a **Watch the demo** section that
  plays `reports/demo_video.mp4` (a ~36 s results walkthrough). Start the meeting
  with this if you want a quick overview before the live run.
- Show the title, the three-layer description, and the disclaimer.
- Sidebar: detector = **Fine-tuned (7 classes)**.

### 2. 1 - Intersection Analysis
- Intersection config: `default.json` (2 approaches, 2 phases).
- Lane ROIs: show the editable polygon table (preset `example.json`).
- Source options (pick one):
  - **Use synthetic demo (no video needed)** - always works, richest numbers.
  - **Use a bundled sample clip** - pick `street_clip.mp4`; it plays in the page,
    and use ROI preset `clip_768x432.json`.
  - **Upload a video clip** - your own phone footage; the uploaded clip plays too.
- Click **Run analysis**.
- Point out: annotated keyframes, vehicles counted, PCU total, counts chart,
  PCU chart, per-approach table, CSV download.

> If a real clip is available: choose **Upload a video clip** and adjust the ROIs.

### Bundled demo clips (already downloaded)

| File | Size | Source (public) |
|---|---|---|
| `data/samples/street_clip.mp4` | 5.7 MB | `raw.githubusercontent.com/intel-iot-devkit/sample-videos/master/person-bicycle-car-detection.mp4` |
| `data/samples/traffic_clip.mp4` | 2.7 MB | `raw.githubusercontent.com/intel-iot-devkit/sample-videos/master/car-detection.mp4` |

When uploading these, select the ROI preset **`clip_768x432.json`** (matches their 768x432 frame size).

> These public clips are **sparse** (few vehicles, distant view), so the counts are low.
> For a richer real demo, record a phone video of a busy intersection and draw ROIs on it.
> The **synthetic demo** always gives the fuller numbers (183 vehicles, 32% reduction).


### 3. 2 - Signal Plan
- Show parameters (window, fixed-time cycle).
- Point out: critical flow ratio Y, optimized cycle, green allocation table.
- **Headline:** delay reduction vs fixed-time (expected ~32% on the demo).
- Show the delay comparison chart and the congestion table.
- Open **Assumptions & constants** and explain the transparency.
- Download the signal plan CSV.

### 4. 3 - Model & Metrics
- mAP table: COCO baseline 0.727 vs fine-tuned 0.944.
- Per-class AP table (autorickshaw 0.976, bus 0.974, van 0.957).
- Training curves + confusion matrix.
- Read the limitations list aloud - examiners reward honesty.

## Expected numbers on the demo

| Metric | Value |
|---|---|
| Vehicles counted (synthetic) | ~183 |
| Total PCU | ~187 |
| Fixed-time avg delay | 20.64 s/veh |
| Webster avg delay | 14.01 s/veh |
| Delay reduction | 32.1 % |

## Likely questions (prepared answers)

- **Why kappa / why not accuracy for the detector?** -> we report mAP plus
  per-class AP and a separate *count* MAE, because counting is what the pipeline
  consumes.
- **Isn't this just detection?** -> no; detection is the means. The contribution
  is PCU-weighted demand -> Webster plan -> measured delay reduction (see ablation).
- **Did you invent Webster?** -> no, it is cited; our contribution is the
  camera-based PCU estimation and the end-to-end integration.
- **How do you avoid double counting?** -> ByteTrack IDs; each (track, lane)
  counted once.
- **How are lanes assigned?** -> operator-drawn geometric ROIs; explicit not learned.
- **What about night/rain/local roads?** -> domain-gap run shown; labels unusable
  so it is qualitative; quantified generalisation is future work.
- **Is the 0.944 reliable?** -> likely optimistic due to video-frame leakage;
  stated in the limitations.
- **Why train on Kaggle?** -> the local GPU thermally throttles and powers the
  laptop off; documented.

## Fallback if something breaks

- Synthetic demo needs neither video nor GPU -> always works.
- If the app fails entirely, run the CLI:
  ```powershell
  .\.venv\Scripts\python.exe scripts\analyze_video.py --synthetic
  ```
  and show the printed plan + `reports/figures/delay_synthetic.png`.
