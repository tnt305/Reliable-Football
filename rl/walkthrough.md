📂 Danh Sách Các File Đã Tạo Trong rl/


rl_env.py
:
SoccerNetRLDataset: Tự động phân loại 20.006 clips thành 3 nhóm (Event, Hard Negative, Clean Background).
TriTierBalancedSampler: Bốc mẫu theo tỷ lệ $40% - 40% - 20%$.
SoccerNetRLEnsembleEnv: Môi trường Gymnasium với quan sát $18M + 17$ chiều và hàm thưởng Relative Spotting Margin.


train_rl.py
:
Tự động trích xuất và cache dự đoán của base models 1 lần duy nhất vào RAM/Disk.
Huấn luyện PPO Agent trong 20.000 – 30.000 steps (chạy xong trong vòng ~1 phút).


tune_thresholds.py
:
Quét song song trên 17 CPU Cores cho 17 classes (mỗi class 140 tổ hợp) theo đúng 

tuning_plan.md
.
Tự động lưu bộ tham số tốt nhất vào rl/best_tuned_params.yaml.


infer_rl.py
:
Suy luận sliding-window với trọng số RL động.
Áp dụng Per-Class NMS window & threshold đã tuned.
Đóng gói file results_spotting_test.zip và gọi hàm evaluate() của SoccerNet để xuất báo cáo chính thức.


default_tuned_params.yaml
:
Bảng tham số khởi tạo mặc định cho 17 class.


run_pipeline.sh
:
Bash script 1-click điều khiển toàn bộ pipeline (đã được cấp quyền chmod +x).
🚀 Cách Chạy Thử Nghiệm
Bạn chỉ cần chạy 1 dòng lệnh trong terminal:

bash
cd /workspace/ActionAware
# Chạy trọn gói (Train RL -> Tune Thresholds -> Test Evaluation):
./rl/run_pipeline.sh all
# Hoặc chạy từng chặng độc lập:
./rl/run_pipeline.sh train   # Chỉ huấn luyện RL Agent
./rl/run_pipeline.sh tune    # Chỉ quét tìm Threshold & Window tối ưu
./rl/run_pipeline.sh test    # Chỉ chạy suy luận và đánh giá trên Test split
