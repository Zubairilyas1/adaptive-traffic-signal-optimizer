r"""Smoke test: verify YOLOv8n loads and runs on the T1200 GPU.

Run:  .\.venv\Scripts\python.exe scripts\test_yolo_gpu.py
"""

from pathlib import Path
import time

import torch
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "yolov8n.pt"
IMAGE_PATH = ROOT / "data" / "samples" / "bus.jpg"
OUT_DIR = ROOT / "reports" / "figures"

COCO_TRAFFIC = {"car", "motorcycle", "bus", "truck", "bicycle", "person"}


def main() -> None:
    print("=" * 60)
    print("ENVIRONMENT")
    print("=" * 60)
    print(f"torch          : {torch.__version__}")
    print(f"cuda available : {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"device         : {torch.cuda.get_device_name(0)}")
        print(f"capability     : {torch.cuda.get_device_capability(0)}")

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Missing weights: {MODEL_PATH}")
    if not IMAGE_PATH.exists():
        raise FileNotFoundError(f"Missing test image: {IMAGE_PATH}")

    device = 0 if torch.cuda.is_available() else "cpu"
    print(f"\nLoading {MODEL_PATH.name} on device={device} ...")
    model = YOLO(str(MODEL_PATH))

    # Warm-up + timed run
    _ = model.predict(str(IMAGE_PATH), device=device, verbose=False)
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    start = time.perf_counter()
    results = model.predict(str(IMAGE_PATH), device=device, conf=0.25, verbose=False)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - start) * 1000

    result = results[0]
    names = result.names

    print("\n" + "=" * 60)
    print(f"INFERENCE  ({elapsed_ms:.1f} ms,  {1000 / elapsed_ms:.1f} FPS)")
    print("=" * 60)

    counts: dict[str, int] = {}
    for box in result.boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        name = names[cls_id]
        counts[name] = counts.get(name, 0) + 1
        print(f"  {name:<12} conf={conf:.2f}")

    print(f"\nTotal detections: {len(result.boxes)}")
    print(f"By class        : {counts}")

    traffic_hits = {k: v for k, v in counts.items() if k in COCO_TRAFFIC}
    print(f"Traffic-relevant: {traffic_hits}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "yolo_smoke_test.jpg"
    result.save(filename=str(out_path))
    print(f"\nAnnotated image saved -> {out_path}")

    assert torch.cuda.is_available(), "CUDA not available!"
    assert len(result.boxes) > 0, "No detections - model may not have loaded correctly."
    print("\nSMOKE TEST PASSED")


if __name__ == "__main__":
    main()
