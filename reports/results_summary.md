# Results Summary

## Detection accuracy (test split)

| Model | Classes | mAP@0.5 | mAP@0.5:0.95 |
|---|---|---|---|
| COCO zero-shot baseline | 4 | 0.727 | 0.549 |
| Fine-tuned (ours) | 7 | 0.944 | 0.782 |

### Per-class AP (fine-tuned)

| id | class | AP@0.5 | AP@0.5:0.95 |
|---|---|---|---|
| 0 | car | 0.932 | 0.777 |
| 1 | motorcycle | 0.902 | 0.579 |
| 2 | autorickshaw | 0.976 | 0.830 |
| 3 | van | 0.957 | 0.911 |
| 4 | tractor | 0.954 | 0.748 |
| 5 | bus | 0.974 | 0.862 |
| 6 | truck | 0.915 | 0.764 |

## Count accuracy (counting is what the analysis layer uses)

- Images: 805, ground-truth boxes: 1429, predicted: 1663
- **MAE**: 0.395 vehicles/image
- RMSE: 0.992
- Bias: +0.291
- Exact-count matches: 74.8%

## Ablation - does PCU weighting matter?

Window: 274s

| Quantity | Naive count | PCU-weighted |
|---|---|---|
| flow P1 (PCU/h) | 1639 | 1410 |
| flow P2 (PCU/h) | 761 | 1043 |
| cycle (s) | 46.1 | 47.9 |
| green P1 (s) | 26.1 | 23.0 |
| green P2 (s) | 12.1 | 17.0 |
| avg delay (s/veh) | 12.35 | 14.01 |

**Green-time difference on P1: 3.1 s** - PCU weighting changes the recommended plan.

## Signal optimization (synthetic 2-phase demo)

- Critical flow ratio Y: 0.645
- Optimized cycle: 47.9 s
- Fixed-time avg delay: 20.64 s/veh
- Webster avg delay: 14.01 s/veh
- **Delay reduction: 32.1%**

## Domain gap (external clip)

- External clip: external: subek-yolo/pakistan traffic (test split)
- Detection quality vs benchmark: Qualitative only: external labels have garbled class names, so no mAP. Confirms the model still detects vehicles on local footage; magnitude of the domain gap must be quantified with hand-labelled local data.

## Adaptive control: Fixed-Time vs Rule-Based vs PPO (SUMO)

**Normal demand**

| Controller | Avg queue | Avg delay (s/veh) | Throughput |
|---|---|---|---|
| Fixed-Time | 43.92 | 658.47 | 197.0 |
| Rule-Based | 49.65 | 335.57 | 166.0 |
| PPO | 47.09 | 279.00 | 147.5 |

**Stress demand**

| Controller | Avg queue | Avg delay (s/veh) | Throughput |
|---|---|---|---|
| Fixed-Time | 22.24 | 472.83 | 136.0 |
| Rule-Based | 14.60 | 107.76 | 147.5 |
| PPO | 13.80 | 97.61 | 113.5 |

- PPO delay reduction vs Fixed-Time: normal 57.6%, stress 79.4%
