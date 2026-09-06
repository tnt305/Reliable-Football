#!/usr/bin/env bash
# ==============================================================================
# Script trích xuất CLIP-ViT Features siêu tốc trên 1 GPU 32GB VRAM
# Output: 1_CLIP_ViT.npy & 2_CLIP_ViT.npy
# 
# Support parameters từ master script:
#   --batch_size N      (default: 256 or auto-detect from GPU memory)
#   --num_workers N     (default: 8)
# ==============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATASET_PATH="downloads/dataset"

# Default values
BATCH_SIZE=256
NUM_WORKERS=8

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --batch_size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --num_workers)
            NUM_WORKERS="$2"
            shift 2
            ;;
        *)
            shift
            ;;
    esac
done

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
echo "⚡ CLIP-ViT EXTRACTOR (GPU ACCELERATED)"
echo "   Output:      1_CLIP_ViT.npy & 2_CLIP_ViT.npy"
echo "   Dataset:     $DATASET_PATH"
echo "   Batch Size:  $BATCH_SIZE"
echo "   Num Workers: $NUM_WORKERS"
echo "=================================================================="

$PYTHON_EXEC "$SCRIPT_DIR/clip_vit.py" \
    --dataset_path "$DATASET_PATH" \
    --batch_size "$BATCH_SIZE" \
    --num_workers "$NUM_WORKERS" \
    "$@"
