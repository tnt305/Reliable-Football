#!/usr/bin/env bash
# ==============================================================================
# Pipeline RL Ensemble & Tuning cho v2 Checkpoints (model_34 + model_41 + model_47)
# ==============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$SCRIPT_DIR"

STAGE="${1:-all}"
CONFIG="configs/MediaFusion.yaml"
OUTPUT_DIR="models/RL_agents/MediaFusion_Ensemble_v2_34_41_47"
PARAMS_YAML="rl/best_params_v2_34_41_47.yaml"

MODELS="models/ASmodels/MediaFusion/model_34.pth.tar models/ASmodels/MediaFusion/model_41.pth.tar models/ASmodels/MediaFusion/model_47.pth.tar"

mkdir -p "$OUTPUT_DIR"

echo "============================================================"
echo "⚡ KHỞI ĐỘNG SOCCERNET RL ENSEMBLE PIPELINE (v2: Ep 34, 41, 47)"
echo "   - Giai đoạn: $STAGE"
echo "   - Config:    $CONFIG"
echo "   - Models:    $MODELS"
echo "   - Output:    $OUTPUT_DIR"
echo "============================================================"

# --- 1. GIAI ĐOẠN HUẤN LUYỆN RL ENSEMBLE AGENT ---
if [ "$STAGE" == "all" ] || [ "$STAGE" == "train" ]; then
    echo ""
    echo "▶️ [BƯỚC 1/3] HUẤN LUYỆN PPO ENSEMBLE AGENT..."
    python rl/train_rl.py \
        --config "$CONFIG" \
        --models $MODELS \
        --cache_path "$OUTPUT_DIR/valid_preds_cache.pkl" \
        --output_dir "$OUTPUT_DIR" \
        --timesteps 30000 \
        --force_cache
    echo "✅ [BƯỚC 1/3] Đã huấn luyện xong RL Agent tại: $OUTPUT_DIR/best_ppo_agent.zip"
fi

# --- 2. GIAI ĐOẠN TỐI ƯU HÓA PER-CLASS THRESHOLDS & WINDOWS ---
if [ "$STAGE" == "all" ] || [ "$STAGE" == "tune" ]; then
    echo ""
    echo "▶️ [BƯỚC 2/3] TỐI ƯU HÓA PER-CLASS WINDOWS & THRESHOLDS (TQDM TỪNG CLASS)..."
    python rl/tune_thresholds.py \
        --config "$CONFIG" \
        --agent "$OUTPUT_DIR/best_ppo_agent.zip" \
        --models $MODELS \
        --cache_probs "$OUTPUT_DIR/valid_probs_cache.pkl" \
        --output_yaml "$PARAMS_YAML" \
        --parallel \
        --cores 17
    echo "✅ [BƯỚC 2/3] Đã tối ưu hóa xong tham số và lưu tại: $PARAMS_YAML"
fi

# --- 3. GIAI ĐOẠN SUY LUẬN & ĐÁNH GIÁ CHÍNH THỨC TRÊN TEST SPLIT ---
if [ "$STAGE" == "all" ] || [ "$STAGE" == "test" ]; then
    echo ""
    echo "▶️ [BƯỚC 3/3] SUY LUẬN & ĐÁNH GIÁ CHÍNH THỨC TRÊN TẬP TEST..."
    python rl/infer_rl.py \
        --config "$CONFIG" \
        --split "test" \
        --agent "$OUTPUT_DIR/best_ppo_agent.zip" \
        --models $MODELS \
        --params "$PARAMS_YAML" \
        --output_dir "$OUTPUT_DIR/eval_test"
    echo "✅ [BƯỚC 3/3] Hoàn tất đánh giá trên tập Test!"
fi

echo ""
echo "🎉 PIPELINE ĐÃ HOÀN TẤT THÀNH CÔNG!"
