#!/usr/bin/env bash
# ==============================================================================
# Script tải toàn bộ Baidu Embeddings + Labels cho tất cả splits SoccerNet
# ==============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_EXEC="/root/miniconda3/envs/soccernet/bin/python"
OUTPUT_DIR="$SCRIPT_DIR/dataset"
HF_TOKEN="${HF_TOKEN:-}"
MAX_WORKERS=8

echo "=================================================================="
echo "⚡ SOCCERNET BAIDU EMBEDDINGS DOWNLOADER"
echo "   Output: $OUTPUT_DIR"
echo "   Workers: $MAX_WORKERS"
echo "=================================================================="

# Tải từng split: train, valid, test, challenge
for SPLIT in train valid test challenge; do
    echo ""
    echo "▶ Đang tải split: $SPLIT ..."
    $PYTHON_EXEC "$SCRIPT_DIR/download_soccernet.py" \
        --split "$SPLIT" \
        --output_dir "$OUTPUT_DIR" \
        --token "$HF_TOKEN" \
        --max_workers $MAX_WORKERS
done

echo ""
echo "=================================================================="
echo "✅ Hoàn tất tải toàn bộ dataset!"
echo "   Dữ liệu lưu tại: $OUTPUT_DIR"
echo "=================================================================="
