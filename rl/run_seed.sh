#!/usr/bin/env bash
# ==============================================================================
# Script Huấn luyện & Đánh giá RL Ensemble với Custom Seed
# ==============================================================================
# Cách dùng:
#   bash rl/run_seed.sh 2026
#   bash rl/run_seed.sh 42 123 2026 777
# ==============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$SCRIPT_DIR"

# Các đường dẫn mặc định
CONFIG="configs/MediaFusion.yaml"
BASE_CACHE="models/RL_agents/MediaFusion_Ensemble_v2_34_41_47/valid_preds_cache.pkl"
BEST_PARAMS="rl/best_params_v2_34_41_47.yaml"
MODELS="models/ASmodels/MediaFusion/model_34.pth.tar models/ASmodels/MediaFusion/model_41.pth.tar models/ASmodels/MediaFusion/model_47.pth.tar"
TIMESTEPS=30000
SPLIT="test"

# Lấy danh sách seeds từ tham số truyền vào (nếu không truyền mặc định chạy seed 2026)
SEEDS=("$@")
if [ ${#SEEDS[@]} -eq 0 ]; then
    SEEDS=(2026)
fi

echo "======================================================================"
echo "⚡ KHỞI CHẠY RL EXPERIMENTS VỚI SEED(S): ${SEEDS[*]}"
echo "   - Config:      $CONFIG"
echo "   - Base Cache:  $BASE_CACHE"
echo "   - Best Params: $BEST_PARAMS"
echo "   - Split:       $SPLIT"
echo "======================================================================"

for SEED in "${SEEDS[@]}"; do
    OUTPUT_DIR="models/RL_agents/seed_${SEED}"
    AGENT_PATH="$OUTPUT_DIR/best_ppo_agent.zip"
    EVAL_DIR="$OUTPUT_DIR/eval_${SPLIT}"

    mkdir -p "$OUTPUT_DIR"

    echo ""
    echo "======================================================================"
    echo "▶️ [SEED $SEED] 1/2. HUẤN LUYỆN PPO AGENT (30,000 steps)..."
    echo "======================================================================"
    python rl/train_rl.py \
        --config "$CONFIG" \
        --models $MODELS \
        --cache_path "$BASE_CACHE" \
        --output_dir "$OUTPUT_DIR" \
        --seed "$SEED" \
        --timesteps "$TIMESTEPS"

    echo "✅ [SEED $SEED] Đã lưu agent vào: $AGENT_PATH"

    echo ""
    echo "======================================================================"
    echo "▶️ [SEED $SEED] 2/2. SUY LUẬN & ĐÁNH GIÁ CHÍNH THỨC TRÊN TẬP $SPLIT..."
    echo "======================================================================"
    python rl/infer_rl.py \
        --config "$CONFIG" \
        --split "$SPLIT" \
        --agent "$AGENT_PATH" \
        --models $MODELS \
        --params "$BEST_PARAMS" \
        --output_dir "$EVAL_DIR"

    echo "✅ [SEED $SEED] Hoàn thành đánh giá tại: $EVAL_DIR"
done

echo ""
echo "======================================================================"
echo "🎉 TẤT CẢ SEED EXPERIMENTS ĐÃ HOÀN TẤT!"
echo "======================================================================"
for SEED in "${SEEDS[@]}"; do
    METRIC_JSON="models/RL_agents/seed_${SEED}/eval_${SPLIT}/metrics_summary.json"
    if [ -f "$METRIC_JSON" ]; then
        echo "🔹 [SEED $SEED] Kết quả:"
        python -c "
import json
with open('$METRIC_JSON') as f:
    d = json.load(f)
    print(f'   - Tight a-mAP: {d[\"overall\"][\"tight_a_mAP\"]:.2f}% | Loose a-mAP: {d[\"overall\"][\"loose_a_mAP\"]:.2f}%')
"
    fi
done
echo "======================================================================"
