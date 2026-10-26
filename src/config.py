"""Central configuration for the Traffic Signal Timing Optimizer project.

All paths, model settings, class mappings and traffic-engineering constants
live here so the rest of the codebase never hardcodes magic values.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
SAMPLES_DIR = DATA_DIR / "samples"
PROCESSED_DIR = DATA_DIR / "processed"
VIOLATIONS_DIR = DATA_DIR / "violations"
TELEMETRY_DB = DATA_DIR / "traffic_telemetry.db"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

COCO_WEIGHTS = MODELS_DIR / "yolov8n.pt"
BEST_WEIGHTS = MODELS_DIR / "yolov8n_traffic_best.pt"
PPO_MODEL = MODELS_DIR / "ppo_traffic"  # stable-baselines3 saves .zip

# Use the fine-tuned 7-class model when available, else the COCO baseline.
DETECTOR_WEIGHTS = BEST_WEIGHTS if BEST_WEIGHTS.exists() else COCO_WEIGHTS

DATASET_DIR = RAW_DIR / "roboflow"
DATASET_YAML = DATASET_DIR / "data.yaml"

# Merged 7-class dataset produced by scripts/merge_datasets.py
MERGED_DIR = PROCESSED_DIR / "traffic"
MERGED_YAML = MERGED_DIR / "data.yaml"

DEVICE = 0
IMGSZ = 640
CONF_THRESHOLD = 0.10
IOU_THRESHOLD = 0.45
MAX_DET = 300

# Canonical names the pipeline reasons about. These match the merged dataset
# class order produced by scripts/merge_datasets.py.
TARGET_CLASS_NAMES = (
    "car",
    "motorcycle",
    "autorickshaw",
    "van",
    "tractor",
    "bus",
    "truck",
)

# Maps alternate dataset spellings onto the canonical names above.
CLASS_ALIASES = {
    "motorbike": "motorcycle",
    "two wheeler": "motorcycle",
    "two-wheeler": "motorcycle",
    "rickshaw": "autorickshaw",
    "auto": "autorickshaw",
    "auto-rickshaw": "autorickshaw",
    "three wheeler": "autorickshaw",
    "threewheel": "autorickshaw",
    "vikram": "autorickshaw",
    "pickup-van": "van",
    "pickup van": "van",
    "microbus": "van",
    "minibus": "van",
    "lorry": "truck",
    "cars": "car",
}

# Passenger Car Unit factors (NOT VERIFIED against physical IRC:106-1990 document text).
# Values follow common literature estimates; verify against official standard before production deployment.
PCU_FACTORS = {
    "motorcycle": 0.5,     # NOT VERIFIED against IRC document text
    "car": 1.0,            # NOT VERIFIED (baseline unit)
    "autorickshaw": 1.5,   # NOT VERIFIED against IRC document text
    "van": 1.5,            # NOT VERIFIED against IRC document text
    "bicycle": 0.3,        # NOT VERIFIED against IRC document text
    "tractor": 3.0,        # NOT VERIFIED against IRC document text
    "bus": 3.5,            # NOT VERIFIED against IRC document text
    "truck": 3.5,          # NOT VERIFIED against IRC document text
}
DEFAULT_PCU = 1.0

# Webster signal-timing parameters. Theoretical textbook value is 1900 PCU/h/lane.
# In SUMO simulation under Krauss car-following (tau=1.0s, minGap=2.5m), measured straight/turning capacity is ~868-953 veh/h/lane.
SATURATION_FLOW = 1900.0      # Theoretical PCU per hour per lane (textbook baseline)
LOST_TIME_PER_PHASE = 4.0     # start-up + clearance, seconds
MIN_GREEN = 7.0               # seconds
MAX_GREEN = 90.0              # seconds (raised from 60s to allow full Webster allocation on long cycles)
MIN_CYCLE = 40.0              # seconds
MAX_CYCLE = 120.0             # seconds

# Video analysis defaults.
FRAME_STRIDE = 1              # process every Nth frame (1 = every frame for stable tracking)
COUNT_WINDOW_SECONDS = 60.0   # rolling aggregation window


def canonical_class(name: str) -> str:
    """Map a raw detector class name to its canonical pipeline name."""
    key = name.strip().lower()
    return CLASS_ALIASES.get(key, key)


def pcu_for(name: str) -> float:
    """Return the PCU factor for a (raw) class name."""
    return PCU_FACTORS.get(canonical_class(name), DEFAULT_PCU)
