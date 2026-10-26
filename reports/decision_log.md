# Decision Log

Every significant technical decision, why it was made, and what was rejected.
Kept as viva armour: for each choice we can explain the reasoning and the
alternative.

| # | Decision | Rationale | Alternative rejected |
|---|---|---|---|
| 1 | Object detection (not classification) | Need to localise and count per lane; classification cannot count | Whole-image classification (cannot localise) |
| 2 | YOLOv8n | Fits the 4 GB GPU; fast (~34 FPS); strong accuracy/size trade-off | YOLOv8s/l larger but slower; the mark is in the decision layer, not detector size |
| 3 | Fine-tune from COCO weights | 3 of 7 classes (autorickshaw, van, tractor) do not exist in COCO | Zero-shot COCO only (baseline showed it cannot detect those classes) |
| 4 | Combine 4 public datasets | Gives a South-Asian vehicle mix including autorickshaw/tractor | Single small dataset (misses PCU-relevant classes) |
| 5 | PCU weighting | A bus occupies far more capacity than a motorcycle; naive counts mislead | Naive vehicle counts (shown in ablation to change the plan by 3.1 s) |
| 6 | ByteTrack | Persistent IDs prevent counting the same vehicle each frame | Per-frame counting (noisy, double counts) |
| 7 | Geometric lane ROIs (not learned) | Lane assignment is geometry; honest scoping of the ML contribution | Training a lane classifier (over-engineering) |
| 8 | Webster's method | Established, defensible, matches a 2-phase model | Custom RL controller (out of scope, needs simulation) |
| 9 | Flag oversaturation | Webster is invalid when Y >= 1; returning a "plan" would be misleading | Silently clamping (hides a real failure mode) |
| 10 | Train on Kaggle, infer locally | Local T1200 thermally throttles and shuts the laptop down; inference is light | Local GPU training (grabs the machine) |
| 11 | CPU for batch evaluation | Sustained GPU load triggers thermal shutdown; CPU is slower but safe | GPU batch eval (crashed the laptop) |
| 12 | Streamlit (mandatory) | Required by the statement; fastest path to a full input->output flow | FastAPI + frontend (more work, not required) |
| 13 | Synthetic demo mode | Guarantees a working live demo even with no video file | Upload-only (risky in a live viva) |
| 14 | Config in JSON | ROIs and intersection specs editable without code changes | Hardcoded constants |

## Notable findings

- **Near-duplicate leakage.** The base dataset is video-derived, so train/test
  splits likely share visually similar frames; the 0.944 mAP is optimistic.
  This is the single most important honest caveat in the report.
- **PCU actually matters.** The ablation shows a 3.1 s green-time shift on P1,
  so PCU weighting is not cosmetic.
- **Oversaturation.** With Y >= 1 the Webster optimum does not exist; the code
  reports this rather than returning a bad plan.
- **Thermal reality.** The laptop's GPU cannot train; moving training to Kaggle
  was necessary, not a preference.
