# Dataset

Primary training dataset for the Traffic Signal Timing Optimizer.
Required documentation: source, features, target, preprocessing and limitations.

---

## Source

| Field | Value |
|---|---|
| Platform | Roboflow Universe |
| URL | https://universe.roboflow.com/thesis-swpxf/traffic-surveillance-system/dataset/8 |
| Workspace | `thesis-swpxf` |
| Project | `traffic-surveillance-system` |
| Version | `8` |
| License | **Public Domain** |
| Format | YOLOv8 (images + YOLO `.txt` labels + `data.yaml`) |

## Composition

| Split | Images |
|---|---|
| train | 1,595 |
| valid | 247 |
| test | 104 |
| **total** | **1,946** |

Instance counts per class:

| id | class | train | valid | test | total |
|---|---|---|---|---|---|
| 0 | bus | 123 | 23 | 8 | **154** |
| 1 | car | 4,435 | 438 | 223 | **5,096** |
| 2 | motorbike | 5,305 | 550 | 237 | **6,092** |
| 3 | truck | 538 | 66 | 40 | **644** |

The publisher reports its own model at mAP 91.19, precision 85.29, recall 88.60
(not our numbers - our fine-tuned results are reported separately).

## Features and target

- **Input:** RGB road/traffic images, resized (see preprocessing).
- **Target:** per-object bounding box + class (4 vehicle classes).
- **Use in the pipeline:** detections feed the lane/PCU analysis layer; the
  class is what justifies PCU weighting (a bus is not a motorcycle).

## Preprocessing (as published by the datacard)

- auto-orient enabled
- resize: stretch to 640 x 640

## How to re-download

```powershell
.\.venv\Scripts\python.exe scripts\download_dataset.py `
    --workspace thesis-swpxf --project traffic-surveillance-system --version 8
.\.venv\Scripts\python.exe scripts\verify_dataset.py
```

Credentials live in `.env` (git-ignored).

## Limitations (documented for the report)

1. **Class imbalance.** `bus` has only 154 instances versus 6,092 for
   `motorbike`. Per-class recall must be reported, not just overall accuracy.
   Mitigations: class-weighted loss, oversampling, or reporting macro metrics.
2. **No rickshaw / three-wheeler class.** The PCU argument for auto-rickshaws
   (a key local vehicle type) cannot be demonstrated from this dataset alone.
   Option: merge a small rickshaw set or document as future work.
3. **Domain gap.** Footage is not from Pakistani roads, so vehicle mix, road
   markings and camera angles differ. A self-recorded local clip is required
   to quantify the gap (limitations section of the report).
4. **Single source.** All images come from one publisher/version.
5. **Bounding-box quality is taken on trust** from the publisher; no hand
   verification of labels was performed.

## Candidate additions (not yet downloaded)

If extra data or the rickshaw class is wanted:
- `lynkeus/vehicle-detection` (v3) - 9,559 images, classes bus/car/motorbike/truck/microbus/pickup-van
- `amit-3ttxd/indian-traffic-objects` (v1) - 2,766 images, includes Three Wheeler
- `image-detection-kwo9b/surveillance-system-rnklm` - 1,670 images, includes auto/threewheel

Merging datasets requires remapping class indices in `data.yaml`.
