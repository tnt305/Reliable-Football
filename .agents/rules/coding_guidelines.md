---
trigger: always_on
---

# Coding Guidelines cho SoccerNet Project

## Python Style

- Dùng type hints cho functions mới.
- Tên file model: `mediafusion_<tên>_model.py`.
- Tên class dataset: `SoccerNet<Tên>` (ví dụ: `SoccerNetFrames`, `SoccerNetFramesTesting`).
- Log với `logging.warning()`, `logging.info()` - không dùng `print()` trong dataset/model.
- Torch operations phải explicit `.cuda()` và `.float()` trước khi dùng.

## Quy tắc về Audio

- **KHÔNG BAO GIỜ** dùng `np.zeros()` thay thế cho audio bị lỗi khi training.
- **LUÔN** kiểm tra `featA is not None` trước khi dùng.
- Khi audio lỗi ở training/validation: dùng `continue` để skip game.
- Khi audio lỗi ở testing: trả về `None`, để inference layer xử lý.

## Model Forward Signature

```python
def forward(self, featsB=None, featsA=None, labels=None, labelsD=None, inference=False):
    # featsB: (B, T, D_baidu)
    # featsA: (B, T*100, 128) hoặc None
    # return: dict với 'preds', 'predsD', 'labels', 'labelsD'
```

## Config YAML

- Model name trong YAML phải match với key trong `model_dict` trong `hadh_main.py`.
- Sau khi thay đổi `chunk_size`, `outputrate`, `rC`, `rD` phải xóa cache `soccernet_dataloader/`.
- `test_only: True` để chỉ chạy eval (không train).

## Commands

- Mọi Python command phải chạy qua: `/root/miniconda3/envs/soccernet/bin/python` hoặc `conda run -n soccernet python`.
- Không dùng `python` trực tiếp (thiếu môi trường).
