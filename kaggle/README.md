# Kaggle Training Package

Train YOLOv8n on the merged 7-class dataset on Kaggle's free GPU (your local
T1200 thermally throttles and shuts the laptop down, so training moves here).

Everything else (inference, Streamlit, PCU/Webster logic, report) runs locally.

---

## What's in this folder

| File | Purpose |
|---|---|
| `traffic_yolo.zip` | The merged dataset to upload as a Kaggle Dataset (529 MB) |
| `train_yolov8n_kaggle.ipynb` | The training notebook to upload/import into Kaggle |

---

## Complete steps

### Part 1 - Upload the dataset (one time)

1. Go to **https://www.kaggle.com/datasets** and click **New Dataset**.
2. Drag in **`kaggle/traffic_yolo.zip`** (from this project folder).
3. Title it `traffic-yolo` (URL slug becomes `traffic-yolo`).
4. Leave it **Private**, click **Create**.
5. Wait for it to finish processing (a few minutes). It will show ~8.5k files.

> The zip contains a top-level `traffic/` folder, so inside Kaggle the files land
> at `/kaggle/input/traffic-yolo/traffic/`.

### Part 2 - Create the training notebook

1. Go to **https://www.kaggle.com/code** and click **New Notebook**.
2. Menu: **File -> Import Notebook** -> upload **`train_yolov8n_kaggle.ipynb`**.
3. **Right panel settings:**
   - **Accelerator:** `GPU T4 x2` (or `GPU P100`)
   - **Internet:** `On` (needed for the one-time `pip install ultralytics`)
4. **Add Input:** click **Add Input** -> search `traffic-yolo` (your dataset) -> **Add**.
5. Click **Run All**. First run takes ~1-2 min to install, then training.

### Part 3 - Training

- ~100 epochs on a T4 at 640 px, batch 32. Expect roughly **40-90 minutes**.
- Kaggle sessions allow up to ~9-12 h GPU, so there is plenty of headroom.
- Watch the live loss/metrics as each epoch prints in the notebook.

### Part 4 - Download the results

In the notebook **Output** panel (right side), download:

- **`yolov8n_traffic_best.pt`** <- the trained weights (what we need)
- `results.png`, `confusion_matrix.png`, `PR_curve.png`, `F1_curve.png` (report figures)

Put `yolov8n_traffic_best.pt` into this project at:

```
D:\5th Semester\2.AI\Project\models\yolov8n_traffic_best.pt
```

Then tell me, and I will wire it into local inference + Streamlit and continue.

---

## Local fallback (no cloud)

If you ever must train locally, use the CPU so the GPU does not overheat:

```powershell
.\.venv\Scripts\python.exe scripts\train_detector.py --device cpu --imgsz 416 --batch 8 --epochs 50
```

Slower (several hours) but zero shutdown risk.

---

## Notes / troubleshooting

- **"No data.yaml found"** -> you forgot **Add Input** the dataset.
- **OOM on GPU** -> lower `batch=16` in the training cell.
- **Internet install fails** -> enable internet in the notebook settings, or skip
  the `pip install` line if `ultralytics` is already present (it usually is).
- The notebook copies the dataset to `/kaggle/working` first because `/kaggle/input`
  is read-only and Ultralytics writes label caches next to the data.
- Hyperparameters here match `scripts/train_detector.py`, so local and cloud runs
  are directly comparable.
