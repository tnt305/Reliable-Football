#!/usr/bin/env bash
# ==============================================================================
# Script tải Video SoccerNet (224p) chuẩn cấu trúc để trích xuất CLIP / Mel / Dino
# ==============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_DIR="$SCRIPT_DIR/dataset"
HF_TOKEN="${HF_TOKEN:-}"
MAX_WORKERS=16
RESOLUTION="224p"

CONDA_PYTHON="/root/miniconda3/envs/soccernet/bin/python"
if [ -f "$CONDA_PYTHON" ]; then
    PYTHON_EXEC="$CONDA_PYTHON"
elif command -v python3 &>/dev/null; then
    PYTHON_EXEC="$(command -v python3)"
else
    echo "❌ Không tìm thấy Python!"
    exit 1
fi

echo "=================================================================="
echo "🚀 SOCCERNET VIDEO DOWNLOADER ($RESOLUTION)"
echo "   Output:      $OUTPUT_DIR"
echo "   Resolution:  $RESOLUTION"
echo "   Workers:     $MAX_WORKERS"
echo "=================================================================="

# Tải video cho tất cả các split (train, valid, test)
for SPLIT in train valid test; do
    echo ""
    echo "📥 Đang tải Video split: $SPLIT ..."
    $PYTHON_EXEC "$SCRIPT_DIR/download_soccernet.py" \
        --split "$SPLIT" \
        --output_dir "$OUTPUT_DIR" \
        --max_workers "$MAX_WORKERS" \
        --token "$HF_TOKEN" \
        --download_videos \
        --resolution "$RESOLUTION"
    echo "✅ Xong Video split: $SPLIT"
done

echo ""
echo "=================================================================="
echo "🎉 HOÀN TẤT TẢI TOÀN BỘ VIDEO TẠI: $OUTPUT_DIR"
echo "=================================================================="
