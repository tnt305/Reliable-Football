#!/usr/bin/env bash
# ==============================================================================
# Script gộp CLIP-ViT Features vào Baidu Features
# Input:  {half}_baidu_soccer_embeddings.npy (8576) & {half}_CLIP_ViT.npy (768)
# Output: {half}_baidu_soccer_embeddings.npy (9344)
# ==============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATASET_PATH="downloads/dataset"
MERGE_SCRIPT="/workspace/ActionAware/merge_features.py"

# Parse command line arguments if any
while [[ $# -gt 0 ]]; do
    case $1 in
        --dataset_path)
            DATASET_PATH="$2"
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
echo "🔀 MERGE CLIP + BAIDU FEATURES"
echo "   Script:   $MERGE_SCRIPT"
echo "   Dataset:  $DATASET_PATH"
echo "   Target:   8576 + 768 -> 9344 dims"
echo "=================================================================="

$PYTHON_EXEC "$MERGE_SCRIPT" "$DATASET_PATH"
