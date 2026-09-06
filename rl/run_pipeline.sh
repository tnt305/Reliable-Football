#!/usr/bin/env bash
# ==============================================================================
# Script: rl/run_pipeline.sh
# Mục đích: Tự động hóa trọn gói quá trình Huấn luyện RL Ensemble, Tối ưu hóa
#           Threshold & Window theo từng Class, và Đánh giá trên tập Test.
#
# Cách dùng:
#   ./rl/run_pipeline.sh [STAGE] [OPTIONS]
#
# Các giai đoạn (STAGE):
#   all    : Chạy trọn gói (Train -> Tune -> Test) [Mặc định]
#   train  : Chỉ chạy huấn luyện RL Ensemble Agent
#   tune   : Chỉ chạy quét song song tối ưu Window & Threshold trên Valid
#   test   : Chỉ chạy suy luận và đánh giá trên Test split
# ==============================================================================

set -e

STAGE="${1:-all}"
CONFIG="configs/MediaFusion.yaml"
PYTHON_BIN="/root/miniconda3/envs/soccernet/bin/python"

if [ ! -f "$PYTHON_BIN" ]; then
    echo "⚠️ Không tìm thấy python trong conda env soccernet, sử dụng python mặc định..."
    PYTHON_BIN="python"
fi

echo "============================================================"
echo "⚡ SOCCERNET RL ENSEMBLE PIPELINE"
echo "   - Giai đoạn: $STAGE"
echo "   - Config:    $CONFIG"
echo "   - Python:    $PYTHON_BIN"
echo "============================================================"

# --- 1. GIAI ĐOẠN HUẤN LUYỆN RL ENSEMBLE ---
if [ "$STAGE" == "all" ] || [ "$STAGE" == "train" ]; then
    echo ""
    echo "▶️ [BƯỚC 1/3] HUẤN LUYỆN RL ENSEMBLE AGENT..."
    $PYTHON_BIN rl/train_rl.py \
        --config "$CONFIG" \
        --timesteps 30000 \
        --output_dir "models/RL_agents/MediaFusion_RL"
    echo "✅ [BƯỚC 1/3] Đã huấn luyện xong RL Agent!"
fi

# --- 2. GIAI ĐOẠN TỐI ƯU HÓA PER-CLASS THRESHOLDS & WINDOWS ---
if [ "$STAGE" == "all" ] || [ "$STAGE" == "tune" ]; then
    echo ""
    echo "▶️ [BƯỚC 2/3] PARALLEL GRID SEARCH 17 CLASSES (17 CORES)..."
    $PYTHON_BIN rl/tune_thresholds.py \
        --config "$CONFIG" \
        --agent "models/RL_agents/MediaFusion_RL/best_ppo_agent.zip" \
        --output_yaml "rl/best_tuned_params.yaml" \
        --cores 17
    echo "✅ [BƯỚC 2/3] Đã tìm thấy bộ tham số tối ưu tại: rl/best_tuned_params.yaml!"
fi

# --- 3. GIAI ĐOẠN SUY LUẬN & ĐÁNH GIÁ CHÍNH THỨC TRÊN TEST SPLIT ---
if [ "$STAGE" == "all" ] || [ "$STAGE" == "test" ]; then
    echo ""
    echo "▶️ [BƯỚC 3/3] SUY LUẬN & ĐÁNH GIÁ TRÊN TẬP TEST..."
    $PYTHON_BIN rl/infer_rl.py \
        --config "$CONFIG" \
        --split "test" \
        --agent "models/RL_agents/MediaFusion_RL/best_ppo_agent.zip" \
        --params "rl/best_tuned_params.yaml"
    echo "✅ [BƯỚC 3/3] Hoàn tất đánh giá trên tập Test!"
fi

echo ""
echo "🎉 PIPELINE ĐÃ HOÀN TẤT THÀNH CÔNG!"
