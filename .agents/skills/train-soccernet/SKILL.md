---
name: train-soccernet
description: >-
  Hướng dẫn đầy đủ để train, eval và debug model MediaFusion trên dataset SoccerNet.
  Kích hoạt khi cần: chạy training, thay đổi config, theo dõi metrics, debug loss/OOM.
---

# Skill: Train & Eval SoccerNet

## 1. Khởi động Training

```bash
# Standard training (background, log ra file)
cd /workspace/ActionAware
conda run -n soccernet nohup python hadh_main.py --model_name MediaFusion > logs/train_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# Hoặc dùng tmux
tmux new -s train
conda activate soccernet
python hadh_main.py --model_name MediaFusion
```

## 2. Theo dõi Training

```bash
# Xem log gần nhất
ls -lt models/ASmodels/MediaFusion/*.log | head -1 | xargs tail -50

# Xem GPU usage
nvidia-smi -l 2
```

## 3. Eval Checkpoint (không cần dừng training)

```bash
# Eval checkpoint tốt nhất trên test split
./eval_checkpoint.sh MediaFusion test

# Eval trên valid split
./eval_checkpoint.sh MediaFusion valid

# Eval checkpoint tuỳ chỉnh
./eval_checkpoint.sh MediaFusion test /path/to/custom.pth.tar
```

## 4. Thay đổi Config

- File config: `configs/MediaFusion.yaml`
- Sau khi thay đổi config, cần **xóa cache dataloader** nếu thay đổi `chunk_size`, `outputrate`, `rC`, `rD`:
  ```bash
  rm -rf soccernet_dataloader/
  ```
- Nếu chỉ thay đổi hyperparams (LR, BS, dropout...) thì KHÔNG cần xóa cache.

## 5. Key Hyperparameters (MediaFusion.yaml)

| Param | Hiện tại | Ý nghĩa |
|-------|---------|---------|
| `chunk_size` | 50 | Độ dài clip (giây) |
| `outputrate` | 2 | Predictions/giây |
| `LR` | 5e-5 | Learning rate (auto-scale theo BS) |
| `BS` | 8 | Batch size |
| `dim` | 512 | Model dimension |
| `TE_layers` | 8 | Số lớp Transformer Encoder |
| `TD_layers` | 4 | Số lớp Transformer Decoder |
| `dropout` | 0.4 | Dropout rate |
| `audio` | True | Có dùng audio hay không |

## 6. Debug Thường Gặp

### OOM (Out of Memory)
```yaml
# Giảm BS trong config
BS: 4  # thay vì 8
# Hoặc giảm chunk_size
chunk_size: 32
```

### Loss không giảm
- Kiểm tra LR: `LR: 0.00005` → thử `0.0001`
- Kiểm tra `warmup_iter` có phù hợp không (nên là 2-5 epochs)
- Kiểm tra `mixup_alpha` và `mixup_beta`

### Dataset loading chậm
- Tăng `num_workers` trong config
- Đảm bảo `store: True` ở lần chạy đầu để cache dataloader

## 7. Kết quả Baseline (MediaFusion)

| Metric | Value |
|--------|-------|
| test/loose_aMAP | 76.53% |
| test/tight_aMAP | 68.18% |
| Thời gian inference (100 games test) | ~38 phút |
| Số params | 192,289,166 |
