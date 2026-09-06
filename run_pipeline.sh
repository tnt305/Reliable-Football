#!/usr/bin/env bash
# ==============================================================================
# Reliable-Football (ActionAware) - Master Pipeline Script
# ==============================================================================
# Tự động hóa toàn bộ luồng công việc SoccerNet Action Spotting:
#   1. Download dữ liệu (Baidu Features + Labels)
#   2. Huấn luyện Base Model (MediaFusion)
#   3. Đánh giá Checkpoint đơn lẻ (Loose/Tight a_mAP)
#   4. Huấn luyện RL Dynamic Ensemble Agent
#   5. Tối ưu hóa tham số NMS per-class (17 CPU Cores)
#   6. Suy luận & Đánh giá chính thức trên tập Test
#
# Cách dùng:
#   ./run_pipeline.sh [COMMAND] [OPTIONS]
#
# Các lệnh khả dụng:
#   ./run_pipeline.sh all         # Chạy trọn gói từ A-Z
#   ./run_pipeline.sh download    # Tải dataset (Baidu Features + Labels)
#   ./run_pipeline.sh train       # Huấn luyện mô hình Base MediaFusion
#   ./run_pipeline.sh eval        # Đánh giá nhanh 1 checkpoint (ví dụ: ./run_pipeline.sh eval MediaFusion test)
#   ./run_pipeline.sh rl-train    # Huấn luyện RL Ensemble Agent
#   ./run_pipeline.sh rl-tune     # Grid search tối ưu Window & Threshold cho 17 class
#   ./run_pipeline.sh rl-test     # Đánh giá RL Ensemble + Tuned Post-processing trên Test split
#   ./run_pipeline.sh rl-all      # Chạy trọn gói luồng RL (Train RL -> Tune -> Test)
# ==============================================================================

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ------------------------------------------------------------------------------
# Cấu hình Môi trường & Python
# ------------------------------------------------------------------------------
CONDA_PYTHON="/root/miniconda3/envs/soccernet/bin/python"
if [ -f "$CONDA_PYTHON" ]; then
    PYTHON_BIN="$CONDA_PYTHON"
elif command -v python3 &>/dev/null; then
    PYTHON_BIN="$(command -v python3)"
else
    PYTHON_BIN="python"
fi

CONFIG_FILE="configs/MediaFusion.yaml"

COMMAND="${1:-help}"
ARG2="$2"
ARG3="$3"

# Màu sắc giao diện
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

print_header() {
    echo -e "${BLUE}======================================================================${NC}"
    echo -e "${GREEN}⚽ Reliable-Football Pipeline: $1${NC}"
    echo -e "${BLUE}======================================================================${NC}"
}

print_info() {
    echo -e "${YELLOW}ℹ️  $1${NC}"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

show_help() {
    print_header "HƯỚNG DẪN SỬ DỤNG"
    echo ""
    echo "Cú pháp: ./run_pipeline.sh [LỆNH] [TÙY CHỌN]"
    echo ""
    echo "Danh sách lệnh chính:"
    echo "  all                 Chạy toàn bộ quy trình từ tải data đến đánh giá hoàn tất"
    echo "  download            Tải Baidu Embeddings + Labels-v2.json (train, valid, test)"
    echo "  train [MODEL]       Huấn luyện mô hình Base (Mặc định: MediaFusion)"
    echo "  eval [MODEL] [SPLIT] Đánh giá nhanh 1 checkpoint (Mặc định: MediaFusion test)"
    echo "  rl-all              Chạy trọn gói luồng RL Ensemble (Train -> Tune -> Test)"
    echo "  rl-train            Huấn luyện RL Ensemble Agent (PPO)"
    echo "  rl-tune             Tối ưu hóa Window & Threshold song song trên 17 CPU Cores"
    echo "  rl-test             Đánh giá chính thức mô hình RL Ensemble trên tập Test"
    echo "  help                Hiển thị hướng dẫn này"
    echo ""
    echo "Ví dụ:"
    echo "  ./run_pipeline.sh download"
    echo "  ./run_pipeline.sh train MediaFusion"
    echo "  ./run_pipeline.sh eval MediaFusion test"
    echo "  ./run_pipeline.sh rl-all"
    echo ""
}

# ------------------------------------------------------------------------------
# 1. TẢI DỮ LIỆU SOCCERNET
# ------------------------------------------------------------------------------
run_download() {
    print_header "BƯỚC 1: TẢI DỮ LIỆU SOCCERNET"
    print_info "Kiểm tra và tải Baidu Features + Labels cho các split train, valid, test..."
    bash downloads/download_full.sh
    print_success "Dữ liệu đã sẵn sàng tại downloads/dataset/"
}

# ------------------------------------------------------------------------------
# 2. HUẤN LUYỆN BASE MODEL
# ------------------------------------------------------------------------------
run_train() {
    MODEL_NAME="${ARG2:-MediaFusion}"
    print_header "BƯỚC 2: HUẤN LUYỆN MÔ HÌNH BASE ($MODEL_NAME)"
    print_info "Bắt đầu huấn luyện mô hình..."
    $PYTHON_BIN hadh_main.py --model_name "$MODEL_NAME"
    print_success "Huấn luyện hoàn tất! Checkpoint tại models/ASmodels/$MODEL_NAME/"
}

# ------------------------------------------------------------------------------
# 3. ĐÁNH GIÁ CHECKPOINT
# ------------------------------------------------------------------------------
run_eval() {
    MODEL_NAME="${ARG2:-MediaFusion}"
    SPLIT="${ARG3:-test}"
    print_header "BƯỚC 3: ĐÁNH GIÁ CHECKPOINT ($MODEL_NAME trên tập $SPLIT)"
    ./eval_checkpoint.sh "$MODEL_NAME" "$SPLIT"
    print_success "Đánh giá hoàn tất!"
}

# ------------------------------------------------------------------------------
# 4. HUẤN LUYỆN RL DYNAMIC ENSEMBLE AGENT
# ------------------------------------------------------------------------------
run_rl_train() {
    print_header "BƯỚC 4: HUẤN LUYỆN RL ENSEMBLE AGENT"
    ./rl/run_pipeline_v2.sh train
    print_success "RL Agent đã được huấn luyện xong!"
}

# ------------------------------------------------------------------------------
# 5. TỐI ƯU HÓA PER-CLASS NMS PARAMETERS (17 CORES)
# ------------------------------------------------------------------------------
run_rl_tune() {
    print_header "BƯỚC 5: TỐI ƯU HÓA THAM SỐ PER-CLASS (17 CORES)"
    ./rl/run_pipeline_v2.sh tune
    print_success "Đã tìm ra bộ tham số tốt nhất tại rl/best_params_v2_34_41_47.yaml"
}

# ------------------------------------------------------------------------------
# 6. SUY LUẬN & ĐÁNH GIÁ CHÍNH THỨC TRÊN TẬP TEST
# ------------------------------------------------------------------------------
run_rl_test() {
    print_header "BƯỚC 6: SUY LUẬN & ĐÁNH GIÁ TRÊN TẬP TEST"
    ./rl/run_pipeline_v2.sh test
    print_success "Đã xuất báo cáo đánh giá và kết quả nộp bài!"
}

# ------------------------------------------------------------------------------
# CHẠY TRỌN GÓI LUỒNG RL
# ------------------------------------------------------------------------------
run_rl_all() {
    print_header "CHẠY TRỌN GÓI LUỒNG RL ENSEMBLE & TUNING"
    ./rl/run_pipeline_v2.sh all
    print_success "Toàn bộ luồng RL đã hoàn thành xuất sắc!"
}

# ------------------------------------------------------------------------------
# ĐIỀU HƯỚNG LỆNH
# ------------------------------------------------------------------------------
case "$COMMAND" in
    download)
        run_download
        ;;
    train)
        run_train
        ;;
    eval)
        run_eval
        ;;
    rl-train)
        run_rl_train
        ;;
    rl-tune)
        run_rl_tune
        ;;
    rl-test)
        run_rl_test
        ;;
    rl-all)
        run_rl_all
        ;;
    all)
        print_header "KHỞI CHẠY TOÀN BỘ LUỒNG PIPELINE (END-TO-END)"
        if [ ! -d "downloads/dataset/england_epl" ]; then
            run_download
        else
            print_info "Dữ liệu SoccerNet đã tồn tại, bỏ qua bước tải."
        fi
        run_train
        run_rl_all
        print_success "🎉 TOÀN BỘ PIPELINE ĐÃ HOÀN TẤT THÀNH CÔNG!"
        ;;
    help|--help|-h)
        show_help
        ;;
    *)
        print_error "Lệnh không hợp lệ: '$COMMAND'"
        echo ""
        show_help
        exit 1
        ;;
esac
