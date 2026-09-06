---
name: audio-pipeline
description: >-
  Hướng dẫn chi tiết về audio pipeline: trích xuất Log-Mel spectrogram từ video MKV,
  kiểm tra chất lượng audio, xử lý file bị lỗi, và hiểu format dữ liệu âm thanh.
  Kích hoạt khi làm việc với audio features, file audio bị lỗi, hoặc feature extraction.
---

# Skill: Audio Pipeline

## Tổng quan Audio Pipeline

```
MKV video → FFmpeg (decode PCM) → Log-Mel Spectrogram → audio1.npy / audio2.npy
```

- Sample rate: 16000 Hz
- FFT size: 400 samples (25ms window)
- Hop length: 160 samples (10ms = 100 frames/giây)
- Mel bands: 128
- Output: `(T, 128)` float32 array, giá trị khoảng `-108 dB` đến `-28 dB`

## Trích xuất Audio Features từ Video

```bash
# Trích xuất audio cho 1 thư mục game cụ thể
conda run -n soccernet python feature_extractor/mel.py \
  --input_dir "downloads/dataset/england_epl/2014-2015/2015-02-21 - 18-00 Chelsea 1 - 1 Burnley"

# Trích xuất toàn bộ dataset (batch)
bash feature_extractor/run_mel.sh downloads/dataset/
```

## Kiểm tra chất lượng Audio File

```python
import numpy as np

a = np.load('path/to/audio1.npy')
print(f'shape: {a.shape}')        # Expect: (270000, 128) ~ 45min * 100fps
print(f'min/max: {a.min():.2f}, {a.max():.2f}')  # Expect: -108 to -28
print(f'std: {a.std():.2f}')      # Expect: ~14 dB - nếu std=0 là file lỗi!

# File bị lỗi: shape quá nhỏ VD (11, 128) hoặc (117, 128)
# File bình thường: T >= chunk_size * 100 = 5000 frames
```

## Danh sách File Audio Bị Lỗi

Xem danh sách đầy đủ trong:
- `mkv_audio_investigation.txt` - 56 file MKV gốc bị thiếu audio packet
- `audio_truncated_files.txt` - audio.npy đã trích xuất nhưng bị cụt (quá ngắn)

**Nguyên nhân**: File MKV gốc từ SoccerNet Server bị lỗi encoding. Cả bản 224p và 720p đều bị ảnh hưởng (không thể fix bằng cách tải lại bản 720p).

## Cách Dataset Xử Lý Audio Lỗi

Hiện tại (sau khi fix trong `mediafusion_dataset.py`):
- **Train/Valid**: Game bị lỗi audio → `continue` (bỏ qua hoàn toàn, không load vào tập)
- **Test**: Game bị lỗi audio → `featA1 = None`, `featA2 = None`
- **Inference**: Khi `featA_clips = None` → model chạy thuần Visual (không có audio token)

## So sánh Giá trị Audio Chuẩn vs. Lỗi

| Thuộc tính | File chuẩn | File bị lỗi |
|-----------|-----------|------------|
| Shape[0] | ~270,000 (45 phút) | 3 - 236 frames |
| Min dB | ~-108 | ~-108 |
| Std dB | ~14 | ~0 hoặc rất nhỏ |
| Nguyên nhân | OK | MKV thiếu audio packet |

## Không Dùng Zero Padding

**⚠️ Quan trọng**: KHÔNG dùng `np.zeros()` làm fallback cho audio lỗi vì:
- `0.0` trong log-Mel = `0 dB` = cực đại âm thanh (không phải silence)
- Silence thực tế ≈ `-80 dB` → dùng `np.full(-80.0)` nếu cần pad
- Feeding tensor 0 vào VGGish backbone tạo ra embedding giả gây nhiễu Transformer
- **Hướng xử lý đúng**: skip game khi train, trả về `None` khi inference

## Migration Audio Files

Nếu có file đặt tên `1_AST.npy` thay vì `audio1.npy`:
```bash
# Script đổi tên hàng loạt
conda run -n soccernet python migrate_audio_files.py downloads/dataset/
```
