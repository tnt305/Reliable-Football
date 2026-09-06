#!/usr/bin/env bash
# ==============================================================================
# Setup & Download Script for SoccerNet Dataset
# Hỗ trợ tự động chuẩn bị môi trường và tải:
# - Video 224p/720p (từ tartotarto/soccernet-500-videos)
# - Baidu Embeddings (từ OpenSportsLab/SoccerNet-ActionSpotting-Features)
# - Labels-v2.json (chuẩn định dạng SoccerNet V2)
# ==============================================================================

set -e

# Đường dẫn thư mục script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Tìm và kích hoạt môi trường Python Conda
CONDA_PYTHON="/root/miniconda3/envs/soccernet/bin/python"
if [ -f "$CONDA_PYTHON" ]; then
    PYTHON_EXEC="$CONDA_PYTHON"
elif command -v python3 &>/dev/null; then
    PYTHON_EXEC="$(command -v python3)"
else
    echo "❌ Không tìm thấy Python! Vui lòng cài đặt Python trước."
    exit 1
fi

echo "=================================================================="
echo "⚡ KHỞI TẠO MÔI TRƯỜNG SOCCERNET DOWNLOADER"
echo "   Python: $PYTHON_EXEC"
echo "   Thư mục làm việc: $SCRIPT_DIR"
echo "=================================================================="

# Cài đặt / cập nhật các thư viện phụ thuộc cần thiết
echo "📦 Đang kiểm tra các gói thư viện cần thiết..."
$PYTHON_EXEC -m pip install -q --upgrade huggingface_hub tqdm numpy 2>/dev/null || true

# Cấp quyền thực thi cho python script
chmod +x "$SCRIPT_DIR/download_soccernet.py"

# Hiển thị hướng dẫn nếu không có tham số
if [ "$#" -eq 0 ]; then
    echo ""
    echo "📌 CÁCH SỬ DỤNG SETUP & DOWNLOAD SCRIPT:"
    echo "------------------------------------------------------------------"
    echo "1. Chạy test nhanh 1 trận (Video 224p + Baidu Embeddings + Labels):"
    echo "   ./setup.sh --test"
    echo ""
    echo "2. Tải toàn bộ split train (chỉ Features và Labels, bỏ qua video để tiết kiệm dung lượng):"
    echo "   ./setup.sh --split train --no_videos --output_dir ./dataset"
    echo ""
    echo "3. Tải toàn bộ split train (gồm cả Video 224p, Features, Labels):"
    echo "   ./setup.sh --split train --resolution 224p --output_dir ./dataset --max_workers 16"
    echo ""
    echo "4. Tải 10 trận mẫu để thử nghiệm huấn luyện:"
    echo "   ./setup.sh --split train --num_games 10 --output_dir ./dataset"
    echo ""
    echo "5. Tải tất cả các split (train, valid, test, challenge):"
    echo "   ./setup.sh --split all --output_dir ./dataset"
    echo "------------------------------------------------------------------"
    echo "Chạy mặc định: Tải test 1 trận mẫu ngay bây giờ..."
    echo ""
    $PYTHON_EXEC "$SCRIPT_DIR/download_soccernet.py" --split train --num_games 1 --output_dir "$SCRIPT_DIR/dataset"
    exit 0
fi

# Xử lý flag --test riêng
if [ "$1" == "--test" ]; then
    echo "🚀 Đang chạy kiểm tra tải 1 trận mẫu..."
    $PYTHON_EXEC "$SCRIPT_DIR/download_soccernet.py" --split train --num_games 1 --output_dir "$SCRIPT_DIR/dataset"
    exit 0
fi

# Chuyển tiếp toàn bộ tham số dòng lệnh sang download_soccernet.py
$PYTHON_EXEC "$SCRIPT_DIR/download_soccernet.py" "$@"
