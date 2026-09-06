---
name: soccernet-download
description: >-
  Hướng dẫn tải dữ liệu SoccerNet: video MKV, Baidu features, audio features.
  Kích hoạt khi cần: tải thêm data, kiểm tra data có đủ không, xử lý lỗi download,
  hiểu cách dùng SoccerNet Downloader API.
---

# Skill: SoccerNet Download

## 1. Cài đặt & Xác thực

```python
from SoccerNet.Downloader import SoccerNetDownloader

mySoccerNetDownloader = SoccerNetDownloader(LocalDirectory="downloads/dataset")
mySoccerNetDownloader.password = "s0cc3rn3t"  # Password từ SoccerNet website
```

## 2. Tải Baidu Features (cần nhất)

```python
# Tải features cho training (train+valid+test)
mySoccerNetDownloader.downloadDataTask(
    task="action-spotting",
    split=["train", "valid", "test"],
    version=2,
    verbose=True,
    randomized=True
)
```

## 3. Tải Video (224p hoặc 720p)

```python
# 224p - nhỏ hơn (~170MB/game)
mySoccerNetDownloader.downloadGames(files=["1_224p.mkv", "2_224p.mkv"], split=["all"])

# 720p - nặng hơn (~900MB/game)  
mySoccerNetDownloader.downloadGames(files=["1_720p.mkv", "2_720p.mkv"], split=["all"])
```

⚠️ **LƯU Ý**: **56 game có audio bị lỗi** ở CẢ 2 phiên bản 224p VÀ 720p.
Đây là lỗi từ SoccerNet Server, không thể fix bằng tải lại.
Danh sách đầy đủ trong `mkv_audio_investigation.txt`.

## 4. Tải Labels

```python
mySoccerNetDownloader.downloadDataTask(
    task="action-spotting",
    split=["train", "valid", "test"],
    version=2
)
```

## 5. Kiểm tra Data Sau Khi Tải

```bash
# Kiểm tra số file trong dataset
find downloads/dataset -name "1_baidu_soccer_embeddings.npy" | wc -l
# Expect: 500 (train=300, valid=100, test=100)

find downloads/dataset -name "audio1.npy" | wc -l
# Expect: ~500 (trừ những game bị skip lỗi)

find downloads/dataset -name "Labels-v2.json" | wc -l
# Expect: 500
```

## 6. Tải 1 Game Test (Debug)

```python
from SoccerNet.Downloader import SoccerNetDownloader
import os

downloader = SoccerNetDownloader(LocalDirectory="downloads/test_720p")
downloader.password = "s0cc3rn3t"

# Tải 1 game cụ thể
target_game = "england_epl/2015-2016/2015-10-03 - 17-00 Manchester City 6 - 1 Newcastle Utd"
downloader.downloadSingleGame(
    game=target_game, 
    files=["1_720p.mkv"],
    verbose=True
)
```

## 7. Dung lượng Ổ Đĩa

| Loại data | ~GB cho full dataset (500 games) |
|-----------|--------------------------------|
| Labels (json) | ~0.2 GB |
| Baidu features (.npy) | ~90 GB |
| Audio features (.npy) | ~60 GB |
| Video 224p | ~80 GB |
| Video 720p | ~450 GB |

Hiện tại workspace có: Baidu + Audio features + Labels + 224p videos.

## 8. Cấu trúc Sau Khi Tải

```
downloads/dataset/
  england_epl/ (league)
  europe_uefa-champions-league/
  germany_bundesliga/
  spain_laliga/
  france_ligue-1/
  italy_serie-a/
```
