---
name: analyze-results
description: >-
  Phân tích kết quả dự đoán của model: per-class mAP, so sánh baseline,
  visualize prediction timeline, và tìm các class có hiệu suất kém.
  Kích hoạt khi: xem kết quả eval, so sánh giữa các checkpoint, phân tích lỗi model.
---

# Skill: Analyze Results

## 1. Đọc Log Kết quả

```bash
# Xem kết quả eval mới nhất
cat models/ASmodels/MediaFusion/$(ls -t models/ASmodels/MediaFusion/*.log | head -1 | xargs basename)

# Grep chỉ metrics chính
grep -E "(loose_aMAP|tight_aMAP|a_mAP visibility)" models/ASmodels/MediaFusion/*.log | tail -20
```

## 2. Kết quả Baseline (MediaFusion, test split)

| Class | Loose aMAP | Tight aMAP |
|-------|-----------|-----------|
| Ball out of play | 97.79% | 96.45% |
| Throw-in | 84.29% | 69.32% |
| Foul | 89.96% | 85.18% |
| Indirect free-kick | 80.43% | 56.54% |
| Clearance | 66.49% | 58.32% |
| Shot on target | 64.81% | 61.68% |
| Shot off target | 71.08% | 68.49% |
| Tackle | 80.00% | 68.65% |
| Save attempt | 89.23% | 82.94% |
| Substitution | 89.44% | 81.60% |
| Yellow card | 86.60% | 80.51% |
| Red card | 74.35% | 57.47% |
| Yellow→Red card | 78.47% | 75.27% |
| Kick-off | 91.29% | 87.70% |
| Goal | 78.72% | 66.99% |
| Offside | **29.45%** | **26.73%** |
| Corner | **48.56%** | **35.21%** |
| **Tổng (all)** | **76.53%** | **68.18%** |

**Classes khó nhất**: Offside (29%), Corner (48%) → cần cải thiện.

## 3. So sánh Checkpoint

```bash
# Chạy eval trên 2 checkpoint khác nhau
./eval_checkpoint.sh MediaFusion test models/ASmodels/MediaFusion/model_epoch10.pth.tar
./eval_checkpoint.sh MediaFusion test models/ASmodels/MediaFusion/model.pth.tar

# Xem và so sánh kết quả
grep "loose_aMAP\|tight_aMAP" models/ASmodels/MediaFusion_eval_*/*.log
```

## 4. Per-class Analysis Script

```python
import json, os
import numpy as np

# Đọc prediction JSON
pred_dir = "models/MediaFusion/post_SNMS_window_8"
results = {}
for game_dir in os.listdir(pred_dir):
    pred_file = os.path.join(pred_dir, game_dir, "results_spotting.json")
    if os.path.exists(pred_file):
        preds = json.load(open(pred_file))
        for p in preds['predictions']:
            cls = p['label']
            if cls not in results:
                results[cls] = 0
            results[cls] += 1

# Số dự đoán per class
for cls, count in sorted(results.items(), key=lambda x: x[1]):
    print(f"{cls:25s}: {count}")
```

## 5. Visualize Predictions

```bash
# Notebook visualizer có sẵn
conda run -n soccernet jupyter notebook visualizer.ipynb
```

## 6. Phân tích Theo Split

```python
# Classes khó với "unshown" (không thấy trực tiếp trong video)
# Từ kết quả baseline:
unshown_hard = {
    'Ball out of play': 0.0,   # Không detect được khi không hiện camera
    'Foul': 0.0,                
    'Yellow->Red card': 0.0,
    'Offside': 0.0,
    'Corner': 0.0,
}
# → Cần cải thiện: temporal context dài hơn, hoặc thêm thông tin text/context
```

## 7. Tracking Epochs

```python
import re, glob

logs = sorted(glob.glob("models/ASmodels/MediaFusion/*.log"))
for log in logs:
    content = open(log).read()
    epoch_matches = re.findall(r'Epoch (\d+)/.*loss: ([\d.]+)', content)
    map_matches = re.findall(r'val.*aMAP.*: ([\d.]+)', content)
    print(f"\n{os.path.basename(log)}:")
    for epoch, loss in epoch_matches[-5:]:
        print(f"  Epoch {epoch}: loss={loss}")
    if map_matches:
        print(f"  Best val mAP: {max(map_matches)}")
```
