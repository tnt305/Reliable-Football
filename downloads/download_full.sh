#!/usr/bin/env bash
# ==============================================================================
# Script tải full Baidu Features + Labels cho tất cả splits SoccerNet
# Chỉ tải: baidu_soccer_embeddings + Labels-v2.json (không tải video)
# ==============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_DIR="$SCRIPT_DIR/dataset"
HF_TOKEN="${HF_TOKEN:-}"
MAX_WORKERS=32

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
echo "🚀 SOCCERNET FULL DOWNLOAD (Baidu Features + Labels)"
echo "   Output: $OUTPUT_DIR"
echo "   Workers: $MAX_WORKERS"
echo "=================================================================="

# Tải từng split
for SPLIT in train valid test; do
    echo ""
    echo "📥 Đang tải split: $SPLIT ..."
    $PYTHON_EXEC "$SCRIPT_DIR/download_soccernet.py" \
        --split "$SPLIT" \
        --output_dir "$OUTPUT_DIR" \
        --max_workers "$MAX_WORKERS" \
        --token "$HF_TOKEN"
    echo "✅ Xong split: $SPLIT"
done

echo ""
echo "=================================================================="
echo "✅ HOÀN TẤT! Dữ liệu tại: $OUTPUT_DIR"
echo "=================================================================="
