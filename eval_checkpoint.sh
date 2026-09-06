#!/usr/bin/env bash
# ==============================================================================
# Script: eval_checkpoint.sh
# Mục đích: Tự động tạo config test tạm, snapshot checkpoint, chạy test độc lập
#           với tiến trình training, và tự động dọn dẹp (xóa) khi hoàn tất.
#
# Cách dùng:
#   ./eval_checkpoint.sh [MODEL_NAME] [SPLIT] [CHECKPOINT_PATH]
# Ví dụ:
#   ./eval_checkpoint.sh MediaFusion test
#   ./eval_checkpoint.sh MediaFusion test /path/to/custom_checkpoint.pth.tar
# ==============================================================================

set -e

MODEL_NAME="${1:-MediaFusion}"
SPLIT="${2:-test}"
CUSTOM_CKPT="$3"

CONFIG_DIR="configs"
MODELS_DIR="models"

SRC_CONFIG="${CONFIG_DIR}/${MODEL_NAME}.yaml"
if [ ! -f "$SRC_CONFIG" ]; then
    if [ -f "${CONFIG_DIR}/MediaFusion.yaml" ]; then
        SRC_CONFIG="${CONFIG_DIR}/MediaFusion.yaml"
    else
        echo "❌ Không tìm thấy file config gốc: $SRC_CONFIG"
        exit 1
    fi
fi

# Xác định file checkpoint cần test
if [ -n "$CUSTOM_CKPT" ]; then
    if [ -f "$CUSTOM_CKPT" ]; then
        SRC_CKPT="$CUSTOM_CKPT"
    elif [ -f "${MODELS_DIR}/type/${MODEL_NAME}/${CUSTOM_CKPT}" ]; then
        SRC_CKPT="${MODELS_DIR}/type/${MODEL_NAME}/${CUSTOM_CKPT}"
    elif [ -f "${MODELS_DIR}/ASmodels/${MODEL_NAME}/${CUSTOM_CKPT}" ]; then
        SRC_CKPT="${MODELS_DIR}/ASmodels/${MODEL_NAME}/${CUSTOM_CKPT}"
    else
        SRC_CKPT="$CUSTOM_CKPT"
    fi
else
    # Mặc định lấy checkpoint tốt nhất hiện tại của model
    if [ -f "${MODELS_DIR}/type/${MODEL_NAME}/model.pth.tar" ]; then
        SRC_CKPT="${MODELS_DIR}/type/${MODEL_NAME}/model.pth.tar"
    elif [ -f "${MODELS_DIR}/ASmodels/${MODEL_NAME}/model.pth.tar" ]; then
        SRC_CKPT="${MODELS_DIR}/ASmodels/${MODEL_NAME}/model.pth.tar"
    else
        SRC_CKPT="${MODELS_DIR}/type/${MODEL_NAME}/model.pth.tar"
    fi
fi

if [ ! -f "$SRC_CKPT" ]; then
    echo "❌ Không tìm thấy file checkpoint: $SRC_CKPT"
    exit 1
fi

# Tạo tên định danh duy nhất cho phiên test tạm
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
EVAL_NAME="${MODEL_NAME}_eval_${TIMESTAMP}"

TMP_CONFIG="${CONFIG_DIR}/${EVAL_NAME}.yaml"
TMP_ASMODELS_DIR="${MODELS_DIR}/ASmodels/${EVAL_NAME}"
TMP_RESULTS_DIR="${MODELS_DIR}/${EVAL_NAME}"

echo "============================================================"
echo "🚀 BẮT ĐẦU ĐÁNH GIÁ CHECKPOINT TỰ ĐỘNG"
echo "   - Model gốc:        $MODEL_NAME"
echo "   - Checkpoint:       $SRC_CKPT"
echo "   - Tập đánh giá:     $SPLIT"
echo "   - Phiên test tạm:   $EVAL_NAME"
echo "============================================================"

# Hàm dọn dẹp tự động (luôn chạy kể cả khi script kết thúc bình thường hoặc bị Ctrl+C)
cleanup() {
    echo ""
    echo "🧹 [Cleanup] Đang dọn dẹp các file/thư mục tạm..."
    if [ -f "$TMP_CONFIG" ]; then
        rm -f "$TMP_CONFIG"
        echo "   - Đã xóa config tạm: $TMP_CONFIG"
    fi
    if [ -d "$TMP_ASMODELS_DIR" ]; then
        rm -rf "$TMP_ASMODELS_DIR"
        echo "   - Đã xóa thư mục ASmodels tạm: $TMP_ASMODELS_DIR"
    fi
    echo "✅ Hoàn tất dọn dẹp."
}
trap cleanup EXIT INT TERM

# 1. Tạo file config tạm với test_only = True
echo "📝 Tạo file config tạm: $TMP_CONFIG ..."
python -c "
import yaml

with open('$SRC_CONFIG', 'r') as f:
    cfg = yaml.safe_load(f)

cfg['test_only'] = True
cfg['test_split'] = ['$SPLIT']

with open('$TMP_CONFIG', 'w') as f:
    yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
"

# 2. Tạo thư mục tạm và snapshot checkpoint (dùng copy để tránh xung đột ghi đè khi train đang chạy)
echo "📦 Snapshot checkpoint sang $TMP_ASMODELS_DIR/model.pth.tar ..."
mkdir -p "$TMP_ASMODELS_DIR"
cp "$SRC_CKPT" "$TMP_ASMODELS_DIR/model.pth.tar"

# 3. Chạy test
echo "▶️ Đang chạy đánh giá..."
python hadh_main.py --model_name "$EVAL_NAME"

echo ""
echo "🎉 Quá trình kiểm tra đã hoàn thành thành công!"
if [ -d "$TMP_RESULTS_DIR" ]; then
    echo "📁 Kết quả zip/json được lưu tại: $TMP_RESULTS_DIR"
fi
