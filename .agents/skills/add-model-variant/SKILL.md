---
name: add-model-variant
description: >-
  Hướng dẫn step-by-step để thêm một model variant mới vào hệ thống.
  Kích hoạt khi muốn: thêm model mới, fork từ model hiện có, thêm module mới
  vào kiến trúc (attention, loss, head), hoặc thêm ablation study.
---

# Skill: Thêm Model Variant Mới

## 1. Tổng quan Hệ thống Model

Model được đăng ký trong `hadh_main.py` theo pattern:

```python
model_dict = {
    'MediaFusion': MediaFusion,
    'MediaFusion': MediaFusion,
    'BaseModel': BaseModel,
    # ...
}
```

Mỗi model cần có:
1. **File model**: `mediafusion_<tên>_model.py`
2. **Config YAML**: `configs/<TênModel>.yaml`
3. **Đăng ký**: thêm vào `model_dict` trong `hadh_main.py`

## 2. Step-by-step Thêm Model Mới

### Step 1: Tạo file model

```bash
cp mediafusion_model.py mediafusion_mymodel.py
```

Model cần implement `forward()` với signature:

```python
def forward(self, featsB=None, featsA=None, labels=None, labelsD=None, inference=False):
    # featsB: (B, T, sum(Bfeat_dim)) = (B, T, 9344)
    # featsA: (B, T*100, 128) hoặc None
    # labels: (B, T_out, num_classes+1) hoặc None
    # labelsD: (B, T_out, num_classes+1) hoặc None
    # return: dict với keys 'preds', 'predsD', 'labels', 'labelsD'
    ...
    return output  # dict
```

### Step 2: Tạo config YAML

```bash
cp configs/MediaFusion.yaml configs/MyModel.yaml
```

Chỉnh các params cần thiết trong YAML:
```yaml
model:
  name: MyModel   # Phải khớp với key trong model_dict
  dim: 512
  # ...
```

### Step 3: Đăng ký model

Trong `hadh_main.py`:

```python
from mediafusion_mymodel import MyModel

model_dict = {
    'MediaFusion': MediaFusion,
    'MyModel': MyModel,       # ← Thêm dòng này
    # ...
}
```

### Step 4: Chạy thử

```bash
conda run -n soccernet python hadh_main.py --model_name MyModel
```

## 3. Kiến trúc MediaFusion (để fork)

```
Input:
  featsB (B, T, 9344) → 6 Bfeat_modules → 6 × (B, T, 512)
  featsA (B, T*100, 128) → VGGish → (B, T, 512) [optional]

Grid: (B, T, M, D) với M=6 hoặc 7 (nếu có audio)

Encoder (8 lớp shared):
  Temporal Attention: (B×M, T, D) → layer → reshape
  Modality Attention: (B×T, M, D) → layer → reshape

Decoder:
  Queries (n_output=24, D) × Encoded features → decoder output

Head:
  clas_head (GroupedConvHead): → (B, 24, 18) classification
  displ_head (uncertainty_head): → (B, 24, 18, 2) mean+logvar
```

## 4. Thêm Module mới (ví dụ: Cross-Attention Audio-Video)

```python
# Trong __init__:
self.av_cross_attn = nn.MultiheadAttention(
    embed_dim=model_cfg['dim'],
    num_heads=8,
    batch_first=True
)

# Trong forward (sau khi encode visual):
if self.audio and featsA is not None:
    # Cross-attend: video queries, audio keys/values
    x_av, _ = self.av_cross_attn(x_video, featsA_encoded, featsA_encoded)
```

## 5. Tips khi Debug Model Mới

```python
# Luôn test model trước khi train
model = MyModel(chunk_size=50, n_output=24, baidu=True, audio=True, model_cfg=cfg).cuda()
model.eval()

B, T = 2, 50
featsB = torch.randn(B, T, sum(model.Bfeat_dim)).cuda()
featsA = torch.randn(B, T*100, 128).cuda()

with torch.no_grad():
    out = model(featsB=featsB, featsA=featsA, inference=True)
    print(out['preds'].shape)   # Should be (2, 24, 18)
    print(out['predsD'].shape)  # Should be (2, 24, 18, 2) if uncertainty
```

## 6. Ablation: Chạy bỏ Audio

Trong config YAML:
```yaml
audio: False   # Model sẽ chạy Visual-only
```

Không cần sửa code, model tự không init VGGish và không concat audio vào grid.
