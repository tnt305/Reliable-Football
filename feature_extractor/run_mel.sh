#!/usr/bin/env bash
# ==============================================================================
# Script trích xuất Log-Mel Spectrogram (Audio Features) cho SoccerNet
# Output: audio1.npy (Hiệp 1) và audio2.npy (Hiệp 2)
# 
# Support parameters từ master script:
#   --batch_size N      (default: ignored for mel, but accepted for compatibility)
#   --num_workers N     (default: 8)
# ==============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATASET_PATH="downloads/dataset"

# Default values
NUM_WORKERS=8

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --batch_size)
            # Mel script doesn't use batch_size, but accept it for compatibility
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
echo "🚀 LOG-MEL EXTRACTOR (AUDIO - GPU ACCELERATED)"
echo "   Output:      audio1.npy & audio2.npy"
echo "   Dataset:     $DATASET_PATH"
echo "   Num Workers: $NUM_WORKERS"
echo "=================================================================="

$PYTHON_EXEC "$SCRIPT_DIR/mel.py" \
    --dataset_path "$DATASET_PATH" \
    --num_workers "$NUM_WORKERS" \
    "$@"
