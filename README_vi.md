# ⚽ Reliable-Football (ActionAware)

[English](README.md) | **Tiếng Việt**

> **Space-Time Attention Transformer & KnowledgeBank for SoccerNet Action Spotting**  
> Hệ thống phát hiện sự kiện trận đấu bóng đá (Action Spotting) đa phương thức (Audio-Visual) đạt độ chính xác cao trên chuẩn dữ liệu **SoccerNet v2 (17 Classes)**.

---

## 📌 Mục Lục
- [1. Giới Thiệu & Điểm Nổi Bật](#1-giới-thiệu--điểm-nổi-bật)
- [2. Cấu Trúc Mã Nguồn](#2-cấu-trúc-mã-nguồn)
- [3. Cài Đặt Môi Trường](#3-cài-đặt-môi-trường)
- [4. Hướng Dẫn Chạy Full Luồng Qua Script `.sh`](#4-hướng-dẫn-chạy-full-luồng-qua-script-sh)
  - [4.1. Chạy 1-Click Toàn Bộ Pipeline](#41-chạy-1-click-toàn-bộ-pipeline)
  - [4.2. Chạy Từng Giai Đoạn Độc Lập](#42-chạy-từng-giai-đoạn-độc-lập)
- [5. Chi Tiết Các Thành Phần Cốt Lõi](#5-chi-tiết-các-thành-phần-cốt-lõi)
  - [5.1. Mô Hình Base: MediaFusion](#51-mô-hình-base-mediafusion)
  - [5.2. Dynamic RL Ensemble (PPO Agent)](#52-dynamic-rl-ensemble-ppo-agent)
  - [5.3. Tối Ưu Hóa Per-Class Post-Processing (17 Cores)](#53-tối-ưu-hóa-per-class-post-processing-17-cores)
- [6. Cấu Hình Huấn Luyện (`configs/MediaFusion.yaml`)](#6-cấu-hình-huấn-luyện-configsmediafusionyaml)
- [7. Đánh Giá & Xuất Kết Quả Nộp Bài](#7-đánh-giá--xuất-kết-quả-nộp-bài)

---

## 1. Giới Thiệu & Điểm Nổi Bật

**Reliable-Football (ActionAware)** là giải pháp toàn diện cho bài toán SoccerNet Action Spotting, kết hợp hài hòa giữa thị giác máy tính, xử lý âm thanh và học tăng cường:
1. **Kiến trúc MediaFusion**: Không gian thời gian đa phương thức kết hợp 6 dòng đặc trưng hình ảnh (Baidu 9344-d) cùng phổ âm thanh Log-Mel spectrogram (VGGish 128-d).
2. **Học biểu diễn với độ bất định (Uncertainty-Aware)**: Sử dụng Gaussian Log-Likelihood Loss để mô hình hóa độ lệch thời gian (displacement regression) kèm độ tin cậy.
3. **Dynamic RL Ensemble**: Huấn luyện PPO Agent dựa trên bộ lấy mẫu cân bằng ba tầng (`TriTierBalancedSampler`: 40% Event, 40% Hard Negative, 20% Clean Background), dự đoán trọng số softmax động theo ngữ cảnh cho từng frame.
4. **Per-Class NMS Tuning Song Song (17 Cores)**: Phân rã bài toán tối ưu Soft-NMS thành 17 tiến trình chạy song song trên CPU, tìm kiếm bộ 3 tham số $(W_c, \Theta_c, \text{Decay}_c)$ tối ưu riêng cho từng lớp, giúp bứt phá mạnh mẽ chỉ số **Tight a_mAP**.

---

## 2. Cấu Trúc Mã Nguồn

```text
ActionAware/
├── run_pipeline.sh              # 🚀 Master Script điều khiển toàn bộ pipeline
├── hadh_main.py                 # Điểm khởi chạy huấn luyện và kiểm thử mô hình Base
├── mediafusion_model.py         # Kiến trúc mạng MediaFusion
├── mediafusion_base.py          # Lớp cơ sở Space-Time Attention & Vector Gated Shift
├── mediafusion_dataset.py       # DataLoader chuẩn SoccerNet (Frames & Clips)
├── mediafusion_train.py         # Vòng lặp huấn luyện, NMS, Loss & Evaluation metrics
├── mediafusion_train_rl_ensemble.py # Môi trường huấn luyện RL Ensemble cơ bản
├── eval_checkpoint.sh           # Script đánh giá nhanh checkpoint độc lập (không dừng train)
├── class_embed.npy              # Vector nhúng đặc trưng 17 classes từ CLIP
├── configs/                     # Thư mục chứa cấu hình YAML
│   └── MediaFusion.yaml         # Cấu hình chính thức của MediaFusion
├── rl/                          # Module Tối Ưu Hóa & Reinforcement Learning
│   ├── run_pipeline_v2.sh       # Script chạy trọn gói luồng RL Ensemble & Tuning
│   ├── run_seed.sh              # Script chạy đánh giá đa seed (Multi-seed Robustness)
│   ├── train_rl.py              # Huấn luyện PPO Agent kết hợp động nhiều checkpoints
│   ├── tune_thresholds.py       # Quét lưới song song 17 CPU Cores tối ưu NMS parameters
│   ├── infer_rl.py              # Suy luận sliding-window và đánh giá trên Test split
│   ├── rl_env.py                # Môi trường Gym & TriTierBalancedSampler
│   └── best_params_v2_34_41_47.yaml # Bộ tham số NMS đã tối ưu cho 17 class
├── downloads/                   # Module tải dữ liệu tự động
│   ├── download_full.sh         # Tải Baidu features + Labels cho train/valid/test
│   ├── download_soccernet.py    # Python downloader an toàn đa luồng
│   └── download_videos.sh       # Tải video trận đấu (224p/720p)
└── feature_extractor/           # Module trích xuất đặc trưng bổ sung (CLIP, Mel, Dino)
```

---

## 3. Cài Đặt Môi Trường

Hệ thống yêu cầu môi trường Python 3.10+ cùng PyTorch và thư viện SoccerNet:

```bash
# Kích hoạt môi trường conda chuyên dụng
conda activate soccernet

# Hoặc cài đặt các thư viện cần thiết:
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu118
pip install SoccerNet stable-baselines3 gymnasium tqdm pyyaml pandas scikit-learn
```

---

## 4. Hướng Dẫn Chạy Full Luồng Qua Script `.sh`

Hệ thống cung cấp file master script **`./run_pipeline.sh`** tại thư mục gốc để quản lý toàn bộ vòng đời thử nghiệm.

### 4.1. Chạy 1-Click Toàn Bộ Pipeline
Để chạy toàn bộ quy trình từ tải dữ liệu, huấn luyện mô hình cơ sở, tối ưu RL Ensemble và đánh giá chính thức trên tập Test:

```bash
cd /workspace/ActionAware

# Cấp quyền thực thi (nếu chưa có)
chmod +x run_pipeline.sh eval_checkpoint.sh rl/*.sh downloads/*.sh

# Chạy toàn bộ pipeline
./run_pipeline.sh all
```

---

### 4.2. Chạy Từng Giai Đoạn Độc Lập

Bạn có thể chạy riêng rẽ từng bước theo nhu cầu nghiên cứu:

#### Bước 1: Tải dữ liệu SoccerNet
Tải tự động các vector đặc trưng Baidu và file nhãn `Labels-v2.json`:
```bash
./run_pipeline.sh download
```
*(Dữ liệu được lưu trữ tại `downloads/dataset/`)*

#### Bước 2: Huấn luyện Base Model
Huấn luyện mô hình MediaFusion với cấu hình trong `configs/MediaFusion.yaml`:
```bash
./run_pipeline.sh train MediaFusion
```
*(Checkpoints được tự động lưu tại `models/ASmodels/MediaFusion/`)*

#### Bước 3: Đánh giá nhanh một Checkpoint
Đánh giá checkpoint bất kỳ mà không làm gián đoạn tiến trình huấn luyện:
```bash
# Đánh giá checkpoint tốt nhất của MediaFusion trên tập Test
./run_pipeline.sh eval MediaFusion test

# Hoặc dùng trực tiếp eval_checkpoint.sh với checkpoint tùy chọn:
./eval_checkpoint.sh MediaFusion test models/ASmodels/MediaFusion/model_47.pth.tar
```

#### Bước 4: Chạy Luồng RL Ensemble & Per-Class Tuning
Khi đã có các checkpoint tốt (ví dụ: Epoch 34, 41, 47), kích hoạt luồng RL:

```bash
# Chạy trọn gói (Train RL Agent -> Parallel Grid Search 17 Cores -> Test Evaluation)
./run_pipeline.sh rl-all
```

Hoặc chia nhỏ từng bước trong luồng RL:
```bash
# 4a. Huấn luyện PPO Agent học trọng số kết hợp động
./run_pipeline.sh rl-train

# 4b. Quét song song trên 17 CPU Cores tìm bộ (Window, Threshold, Decay) tối ưu
./run_pipeline.sh rl-tune

# 4c. Suy luận trên tập Test với bộ tham số đã tuned
./run_pipeline.sh rl-test
```

#### Bước 5: Kiểm Tra Tính Ổn Định Đa Seed (Multi-Seed Validation)
Để chứng minh kết quả không phụ thuộc vào ngẫu nhiên:
```bash
bash rl/run_seed.sh 42 123 2026
```

---

## 5. Chi Tiết Các Thành Phần Cốt Lõi

### 5.1. Mô Hình Base: MediaFusion
- **Backbone Thị Giác**: Xử lý 6 visual streams từ Baidu qua module `VectorGatedShift` học cách dịch chuyển thông tin thời gian cục bộ.
- **Backbone Âm Thanh**: Trích xuất Log-Mel spectrogram qua mạng VGGish đã được chuẩn hóa.
- **Transformer Không - Thời Gian**: 8 tầng Encoder với cơ chế attention phân cấp (Hierarchical Space-Time Attention) và 4 tầng Decoder đối chiếu với các câu truy vấn ngữ nghĩa (`class_embed.npy`).

### 5.2. Dynamic RL Ensemble (PPO Agent)
- Thay vì cộng trung bình giản đơn ($1/M \sum P_m$), RL Agent quan sát vector đặc trưng đa chiều gồm xác suất dự đoán của từng mô hình, độ bất định và phân phối thời gian để xuất ra trọng số kết hợp softmax $w_t \in \mathbb{R}^M$.
- Cơ chế lấy mẫu **Tri-Tier Sampling** giúp mô hình tập trung vào các đoạn khó (Hard Negatives) và sự kiện (Events), triệt tiêu False Alarm ở các đoạn trận đấu tĩnh.

### 5.3. Tối Ưu Hóa Per-Class Post-Processing (17 Cores)
- Phân rã metric tổng thành 17 bài toán tối ưu độc lập.
- Không gian tìm kiếm: 12 mốc Threshold $\times$ 10 mốc Window $\times$ 2 dạng Decay (`pow2`, `gaussian`) = 240 cấu hình/class.
- Nhờ lưu cache xác suất dự đoán trên GPU trước, bước grid search trên 17 CPU Cores chỉ mất **~30–60 giây**.

---

## 6. Cấu Hình Huấn Luyện (`configs/MediaFusion.yaml`)

Các tham số quan trọng có thể điều chỉnh linh hoạt:

```yaml
# Kích thước chunk và bước lấy mẫu
chunk_size: 50           # Độ dài chunk (50 giây)
outputrate: 2            # 2 dự đoán / giây -> 100 frames / chunk
rC: 2                    # Bán kính nhãn phân loại (Classification tolerance)
rD: 3                    # Bán kính nhãn dịch chuyển (Displacement tolerance)

# Các nhánh dữ liệu
audio: true              # Bật nhánh âm thanh Log-Mel + VGGish
baidu: true              # Bật nhánh hình ảnh Baidu 6-streams

# Trọng số hàm mất mát
wC: 100                  # Hệ số trọng số phân loại
wD: 1                    # Hệ số trọng số displacement
focal: true              # Sử dụng Focal Loss

# Huấn luyện
BS: 16                   # Batch size
LR: 0.00005              # Learning rate
max_epochs: 53           # Số epoch tối đa
```

---

## 7. Đánh Giá & Xuất Kết Quả Nộp Bài

Sau khi chạy xong giai đoạn kiểm thử hoặc `rl-test`, hệ thống sẽ:
1. Tạo file dự đoán định dạng chuẩn JSON cho từng trận đấu.
2. Tự động đóng gói file nộp bài chính thức: `results_spotting_test.zip`.
3. Gọi trực tiếp bộ đánh giá chính thức của SoccerNet để in bảng kết quả chi tiết:
   - **Loose a_mAP** (Dung sai $\Delta \in [5, 60]$ giây)
   - **Tight a_mAP** (Dung sai $\Delta \in [1, 5]$ giây)
   - Điểm AP chi tiết của từng class trong 17 class hành động.
4. Báo cáo dạng JSON được lưu trữ tại:
   `models/RL_agents/.../eval_test/metrics_summary.json`

---

## 📄 License & Trích Dẫn
Dự án được xây dựng và phát triển trên chuẩn dữ liệu [SoccerNet](https://www.soccer-net.org/). Mọi đóng góp và thử nghiệm xin vui lòng tuân thủ các quy định về dữ liệu của ban tổ chức SoccerNet.
