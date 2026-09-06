---
name: dataset-debug
description: >-
  Hướng dẫn debug và kiểm tra dataset SoccerNet: kiểm tra tính toàn vẹn của features,
  tìm game bị thiếu data, phân tích phân phối nhãn, và sửa lỗi dataloader.
  Kích hoạt khi: dataloader bị lỗi, nghi ngờ thiếu dữ liệu, kiểm tra feature shape.
---

# Skill: Dataset Debug

## 1. Cấu trúc Thư mục Dataset

```
downloads/dataset/
  england_epl/
    2014-2015/
      2015-02-21 - 18-00 Chelsea 1 - 1 Burnley/
        1_224p.mkv                        # Video hiệp 1 (224p)
        2_224p.mkv                        # Video hiệp 2
        1_baidu_soccer_embeddings.npy     # Visual features hiệp 1 (T, 9344)
        2_baidu_soccer_embeddings.npy     # Visual features hiệp 2
        audio1.npy                        # Audio features hiệp 1 (T, 128)
        audio2.npy                        # Audio features hiệp 2
        Labels-v2.json                    # Ground truth annotations
```

## 2. Kiểm tra Nhanh Một Game

```python
import numpy as np, json, os

game = "england_epl/2014-2015/2015-02-21 - 18-00 Chelsea 1 - 1 Burnley"
base = f"downloads/dataset/{game}"

# Baidu features
b1 = np.load(f"{base}/1_baidu_soccer_embeddings.npy")
print(f"baidu1: {b1.shape}")  # Expect (2700, 9344)

# Audio
a1 = np.load(f"{base}/audio1.npy")
print(f"audio1: {a1.shape}, min={a1.min():.1f}, max={a1.max():.1f}")

# Labels
labels = json.load(open(f"{base}/Labels-v2.json"))
print(f"annotations: {len(labels['annotations'])}")
```

## 3. Quét Toàn bộ Dataset để Tìm File Thiếu

```python
from SoccerNet.Downloader import getListGames
import os, numpy as np

base = "downloads/dataset/"
issues = []
for split in ['train', 'valid', 'test']:
    for game in getListGames(split):
        game_dir = os.path.join(base, game)
        for half in ['1', '2']:
            for ftype in ['baidu_soccer_embeddings.npy', '../audio.npy']:
                fname = f"{half}_baidu_soccer_embeddings.npy"
                fpath = os.path.join(game_dir, fname)
                if not os.path.exists(fpath):
                    issues.append((game, fname))
                    
print(f"Missing files: {len(issues)}")
for g, f in issues[:5]:
    print(f"  {g}/{f}")
```

## 4. Kiểm tra Audio Bị Lỗi

```bash
# Chạy script check có sẵn
conda run -n soccernet python check_audio_shapes.py
```

File output: `audio_truncated_files.txt` - danh sách audio bị cụt.

## 5. Phân tích Phân phối Nhãn

```bash
conda run -n soccernet python count_classes.py
```

17 classes của SoccerNet v2:
```
0: Ball out of play
1: Throw-in
2: Foul
3: Indirect free-kick
4: Clearance
5: Shot on target
6: Shot off target
7: Tackle
8: Save attempt
9: Substitution
10: Yellow card
11: Red card
12: Yellow->Red card
13: Kick-off
14: Goal
15: Offside
16: Corner
```

## 6. Debug DataLoader

```python
# Test load 1 sample từ training set
from mediafusion_dataset import SoccerNetFrames
import logging
logging.basicConfig(level=logging.WARNING)

ds = SoccerNetFrames(
    path_labels='downloads/dataset/',
    path_store='soccernet_dataloader/',
    path_baidu='downloads/dataset/',
    path_audio='downloads/dataset/',
    split=['train'],
    chunk_size=50,
    outputrate=2,
    stride=25,
    rC=2, rD=3,
    store=True,
    max_games=5  # Chỉ load 5 games để test nhanh
)
print(f"Loaded {ds.n_clips} clips from {min(5, len(ds.listGames))} games")

# Test __getitem__
sample = ds[0]
print(f"Sample keys: {list(sample.keys())}")
print(f"featB shape: {sample['featB'].shape}")
print(f"featA shape: {sample['featA'].shape}")
```

## 7. Reset Cache Dataloader

```bash
# Xóa cache khi cần rebuild (thay đổi chunk_size, stride, etc.)
rm -rf soccernet_dataloader/

# Hoặc xóa chỉ 1 split
rm -rf "soccernet_dataloader/50_Splittrain1_Outputrate2_rC2_rD3/"
```

## 8. Kiểm tra Tương thích Audio ↔ Baidu

Số frames audio = 100× số giây game (~270,000 cho game 45 phút)
Số frames Baidu = số giây game (2,700 cho game 45 phút)

```python
# Tỷ lệ audio:baidu phải là 100:1
ratio = a1.shape[0] / b1.shape[0]
assert 95 < ratio < 105, f"Unexpected ratio: {ratio}"
```
