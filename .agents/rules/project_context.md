---
trigger: always_on
---

# SoccerNet Training Project - Agent Rules

## Môi trường & Runtime

- **Python environment**: Luôn dùng `/root/miniconda3/envs/soccernet/bin/python` (KHÔNG dùng `python` thông thường).
- **Working directory**: `/workspace/ActionAware/`
- **GPU**: RTX A4000 (VRAM ~16GB). Mỗi lần chạy nặng hãy kiểm tra `nvidia-smi` trước.

## Cấu trúc file chính

| File | Vai trò |
|------|---------|
| `hadh_main.py` | Entry point chính (train + test) |
| `mediafusion_dataset.py` | Dataset loader (SoccerNetFrames + SoccerNetFramesTesting) |
| `mediafusion_model.py` | Model MediaFusion hiện tại |
| `mediafusion_train.py` | Training loop + testSpotting |
| `configs/MediaFusion.yaml` | Config model chính |
| `eval_checkpoint.sh` | Script eval nhanh không cần dừng training |
| `feature_extractor/mel.py` | Trích xuất Log-Mel audio features |

## Audio Feature Format

- File audio: `audio1.npy` (hiệp 1), `audio2.npy` (hiệp 2) trong thư mục game tương ứng.
- Shape chuẩn: `(T, 128)` - T frames Log-Mel spectrogram, 128 mel bands.
- Giá trị thực tế: khoảng `-108 dB` đến `-28 dB`. **Số 0.0 KHÔNG có nghĩa là khoảng lặng** (0 dB = cực đại).
- Có **56 file audio bị lỗi** (danh sách trong `mkv_audio_investigation.txt` và `audio_truncated_files.txt`).
- **Policy hiện tại**: Khi audio bị thiếu/lỗi → **skip game** (bỏ qua, không zero-pad).

## Baidu Feature Format

- File: `1_baidu_soccer_embeddings.npy`, `2_baidu_soccer_embeddings.npy`
- Shape: `(T, 9344)` - concat 6 visual streams: `[2048, 2048, 384, 2048, 2048, 768]`

## Model Architecture (MediaFusion)

- Input: 6 visual streams Baidu + 1 audio stream (tuỳ chọn)
- Backbone Audio: VGGish (frozen features → embeddings)
- Encoder: Shared Space-Time Attention Transformer (8 layers)
- Decoder: Transformer Decoder với learnable queries
- Output: (B, n_output=24, num_classes+1=18, 2) với uncertainty head

## Training

```bash
# Train chuẩn
conda run -n soccernet python hadh_main.py --model_name MediaFusion

# Eval riêng không cần dừng training  
./eval_checkpoint.sh MediaFusion test
```

## Evaluation Metrics

- **loose_aMAP**: Metric chính - hiện đạt ~76.5% trên test split
- **tight_aMAP**: Metric phụ - hiện đạt ~68.2% trên test split
- SoccerNet v2, 17 classes, 300 train / 100 valid / 100 test games

## Dataset Splits và Games bị lỗi Audio

| Split | Tổng games | Games có audio lỗi |
|-------|-----------|-------------------|
| train | 300 | 14 |
| valid | 100 | 9 |
| test  | 100 | 6 |

Games bị lỗi sẽ bị **skip** (bỏ qua) khi load dataset.
