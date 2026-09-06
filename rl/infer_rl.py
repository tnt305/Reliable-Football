import os
import sys
import time
import json
import zipfile
import logging
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from typing import Dict, List, Any

import yaml
import numpy as np
import torch
from torch.amp import autocast
from tqdm import tqdm
from stable_baselines3 import PPO

# Thêm đường dẫn gốc vào sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from SoccerNet.Evaluation.utils import AverageMeter, INVERSE_EVENT_DICTIONARY_V2
from SoccerNet.Evaluation.ActionSpotting import evaluate
from mediafusion_dataset import SoccerNetFramesTesting
from mediafusion_model import MediaFusion
from dataset import feats2clip
from mediafusion_train import get_spot_from_SNMS


def zip_results(zip_path: str, target_dir: str, filename: str = "results_spotting.json") -> None:
    """Nén toàn bộ các file results_spotting.json vào file ZIP chuẩn SoccerNet."""
    zipobj = zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED)
    rootlen = len(target_dir) + 1
    for base, _, files in os.walk(target_dir):
        for file in files:
            if file == filename:
                fn = os.path.join(base, file)
                zipobj.write(fn, fn[rootlen:])
    zipobj.close()


def run_inference(
    dataloader: Any,
    base_models: List[torch.nn.Module],
    rl_agent: PPO,
    cfg: Dict[str, Any],
    tuned_params_path: str,
    output_zip_path: str,
    output_json_folder: str,
    device: torch.device
) -> None:
    """Chạy suy luận toàn bộ dataset và lưu kết quả JSON & ZIP."""
    outputrate = cfg['outputrate']
    chunk_size = cfg['chunk_size']
    stride = cfg.get('test_stride', 8)
    num_classes = 17

    # 1. Đọc bộ tham số tuned per-class (window & threshold)
    logging.info(f"Đọc cấu hình hậu xử lý từ: {tuned_params_path}")
    with open(tuned_params_path, 'r') as f:
        tuned_cfg = yaml.safe_load(f)

    # Khởi tạo mảng window, threshold và decay
    windows = [8] * num_classes
    thresholds = [0.01] * num_classes
    decays = ['pow2'] * num_classes

    for item in tuned_cfg['classes']:
        c_id = item['id']
        windows[c_id] = item['window']
        thresholds[c_id] = item['threshold']
        decays[c_id] = item.get('decay', 'pow2')  # backward-compat nếu YAML cũ không có 'decay'

    for m in base_models:
        m.eval()
        m.to(device)

    batch_time = AverageMeter()
    end = time.time()

    logging.info(f"Bắt đầu suy luận RL Ensemble trên {len(dataloader)} trận...")

    with torch.no_grad():
        pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Suy luận {cfg.get('test_split', ['test'])[0]}", ncols=110)
        for i, (game_ID, data) in pbar:
            g_start = time.time()
            game_ID = game_ID[0]

            # Xử lý 2 hiệp đấu
            halves_data = []
            for featB_key, featA_key in [('featB1', 'featA1'), ('featB2', 'featA2')]:
                raw_B = data[featB_key].reshape(-1, data[featB_key].shape[-1])
                sec = raw_B.shape[0]
                clips_B = feats2clip(raw_B, stride=stride, clip_length=chunk_size)
                
                clips_A = None
                if cfg.get('audio', False) and featA_key in data and data[featA_key] is not None:
                    raw_A = data[featA_key].reshape(-1, data[featA_key].shape[-1])
                    if raw_A.shape[1] != 128 and raw_A.shape[0] == 128:
                        raw_A = raw_A.T
                    if raw_A.numel() > 0 and raw_A.shape[0] >= chunk_size * 100:
                        clips_A = feats2clip(raw_A, stride=stride * 100, clip_length=chunk_size * 100)

                halves_data.append((clips_B, clips_A, sec))

            timestamps_per_half = []

            for clips_B, clips_A, sec in halves_data:
                timestamp_long = np.zeros((sec * outputrate, num_classes), dtype=np.float32)
                num_clips = len(clips_B)
                BS = 32

                for b_start in range(0, num_clips, BS):
                    b_end = min(b_start + BS, num_clips)
                    batch_B = clips_B[b_start:b_end].to(device)
                    batch_A = clips_A[b_start:b_end].to(device) if clips_A is not None else None

                    # Dự đoán từ các base models
                    model_preds_C = []
                    model_preds_D = []
                    for m in base_models:
                        with autocast('cuda', dtype=torch.float16):
                            out = m(featsB=batch_B, featsA=batch_A, inference=True)
                        model_preds_C.append(out['preds'].float().cpu().numpy())
                        predD = out['predsD'][:, :, :, 0] if out['predsD'].ndim == 4 else out['predsD']
                        model_preds_D.append(predD.float().cpu().numpy())

                    # RL Agent sinh weights động cho từng clip trong batch
                    batch_s = b_end - b_start
                    for l in range(batch_s):
                        max_probs = [model_preds_C[m_idx][l, :, 1:].max(axis=0) for m_idx in range(len(base_models))]
                        uncs = [0.5] * len(base_models)
                        disagree = np.var(np.stack(max_probs, axis=0), axis=0)
                        obs = np.concatenate([np.stack(max_probs).flatten(), np.array(uncs, dtype=np.float32), disagree])
                        obs = np.clip(obs, 0.0, 1.0)

                        action, _ = rl_agent.predict(obs, deterministic=True)
                        exp_a = np.exp(action - np.max(action))
                        weights = exp_a / np.sum(exp_a)

                        # Trộn theo weights động
                        clip_pred_C = sum(weights[m_idx] * model_preds_C[m_idx][l] for m_idx in range(len(base_models)))
                        clip_pred_D = sum(weights[m_idx] * model_preds_D[m_idx][l] for m_idx in range(len(base_models)))

                        # Vectorized displacement update
                        initial_pos = (b_start + l) * stride * outputrate
                        min_thresh = min(thresholds)
                        mask = clip_pred_C[:, 1:] > min_thresh
                        j_idx, k_idx = np.where(mask)
                        if len(j_idx) > 0:
                            probs = clip_pred_C[j_idx, k_idx + 1]
                            # Lọc theo threshold riêng từng class
                            thresh_arr = np.array([thresholds[k] for k in k_idx])
                            pass_thresh = probs > thresh_arr
                            j_idx = j_idx[pass_thresh]
                            k_idx = k_idx[pass_thresh]
                            probs = probs[pass_thresh]

                            if len(j_idx) > 0:
                                displs = clip_pred_D[j_idx, k_idx + 1]
                                raw_pos = np.round(initial_pos + j_idx - displs).astype(np.int64)
                                valid_mask = (raw_pos >= 0) & (raw_pos < len(timestamp_long))
                                valid_pos = raw_pos[valid_mask]
                                valid_k = k_idx[valid_mask]
                                valid_probs = probs[valid_mask]
                                for pos, k, prob in zip(valid_pos, valid_k, valid_probs):
                                    if prob > timestamp_long[pos, k]:
                                        timestamp_long[pos, k] = prob

                timestamps_per_half.append(timestamp_long)

            g_elapsed = time.time() - g_start
            pbar.set_postfix({"game": game_ID.split("/")[-1][:25], "it_time": f"{g_elapsed:.1f}s"})

            # 4. Hậu xử lý Per-Class NMS và lưu JSON
            json_data = {"UrlLocal": game_ID, "predictions": []}
            for half_idx, timestamp in enumerate(timestamps_per_half):
                for l in range(num_classes):
                    spots = get_spot_from_SNMS(
                        timestamp[:, l],
                        window=windows[l] * outputrate,
                        thresh=thresholds[l],
                        decay=decays[l],
                    )
                    for spot in spots:
                        frame_index = int(spot[0])
                        confidence = float(spot[1])
                        seconds = int((frame_index / outputrate) % 60)
                        minutes = int((frame_index / outputrate) // 60)

                        json_data["predictions"].append({
                            "gameTime": f"{half_idx + 1} - {minutes}:{seconds:02d}",
                            "label": INVERSE_EVENT_DICTIONARY_V2[l],
                            "position": str(int((frame_index / outputrate) * 1000)),
                            "half": str(half_idx + 1),
                            "confidence": f"{confidence:.4f}"
                        })

            json_data["predictions"] = sorted(
                json_data["predictions"],
                key=lambda x: (int(x["half"]), int(x["position"]))
            )

            game_json_dir = os.path.join(output_json_folder, game_ID)
            os.makedirs(game_json_dir, exist_ok=True)
            with open(os.path.join(game_json_dir, "results_spotting.json"), 'w') as jf:
                json.dump(json_data, jf, indent=4)

            batch_time.update(time.time() - end)
            end = time.time()
            if (i + 1) % 10 == 0 or (i + 1) == len(dataloader):
                tqdm.write(f"[{i + 1}/{len(dataloader)}] Đã xử lý {i + 1} trận (Trung bình: {batch_time.avg:.2f}s/trận)")

    # 5. Nén kết quả thành ZIP
    logging.info(f"Đóng gói kết quả vào: {output_zip_path}")
    os.makedirs(os.path.dirname(output_zip_path), exist_ok=True)
    zip_results(output_zip_path, output_json_folder)


def main() -> None:
    parser = ArgumentParser(description="Inference & Đánh giá RL Ensemble cho SoccerNet", formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("--config", type=str, default="configs/MediaFusion.yaml", help="File config gốc")
    parser.add_argument("--split", type=str, default="test", help="Tập dữ liệu đánh giá: test hoặc valid")
    parser.add_argument("--agent", type=str, default="models/RL_agents/MediaFusion_RL/best_ppo_agent.zip", help="Đường dẫn file RL Agent")
    parser.add_argument("--models", nargs="+", type=str, default=None, help="Danh sách checkpoint các base models")
    parser.add_argument("--params", type=str, default=None, help="File YAML chứa bộ tham số tuned. Mặc định dùng best_tuned_params.yaml hoặc default_tuned_params.yaml")
    parser.add_argument("--output_dir", type=str, default=None, help="Thư mục lưu kết quả inference")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    cfg_path = args.config
    with open(cfg_path, 'r') as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() and cfg.get('gpu', 0) >= 0 else "cpu")

    # Xác định file tham số tuned
    tuned_params_path = args.params
    if not tuned_params_path:
        best_path = "rl/best_tuned_params.yaml"
        default_path = "rl/default_tuned_params.yaml"
        tuned_params_path = best_path if os.path.exists(best_path) else default_path

    # Load RL Agent
    if not os.path.exists(args.agent):
        logging.error(f"Không tìm thấy agent tại: {args.agent}")
        sys.exit(1)
    rl_agent = PPO.load(args.agent, device="cpu")

    # Load Base Models
    model_paths = args.models or [
        os.path.join(cfg['path_experiments'], "ASmodels", "MediaFusion", "model.pth.tar"),
        os.path.join(cfg['path_experiments'], "ASmodels", "MediaFusion", "model_34.pth.tar"),
        os.path.join(cfg['path_experiments'], "ASmodels", "MediaFusion", "model_41.pth.tar")
    ]
    base_models = []
    for p in model_paths:
        if os.path.exists(p):
            m = MediaFusion(chunk_size=cfg['chunk_size'], n_output=int(cfg['outputrate'] * cfg['chunk_size']),
                           baidu=cfg['baidu'], audio=cfg['audio'], model_cfg=cfg['model'])
            checkpoint = torch.load(p, map_location=device, weights_only=False)
            state = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
            m.load_state_dict(state)
            base_models.append(m)

    # Thư mục lưu kết quả
    base_out = args.output_dir or os.path.join(cfg['path_experiments'], "RL_agents", "MediaFusion_RL", f"eval_{args.split}")
    output_zip = os.path.join(base_out, f"results_spotting_{args.split}.zip")
    output_json = os.path.join(base_out, f"outputs_{args.split}")

    # Dataset & Dataloader
    dataset = SoccerNetFramesTesting(
        path_labels=cfg['path_labels'], path_baidu=cfg['path_baidu'], path_audio=cfg['path_audio'],
        split=[args.split], outputrate=cfg['outputrate'], chunk_size=cfg['chunk_size'],
        baidu=cfg['baidu'], audio=cfg['audio']
    )
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2)

    # 1. Chạy suy luận và đóng gói ZIP
    run_inference(dataloader, base_models, rl_agent, cfg, tuned_params_path, output_zip, output_json, device)

    # 2. Đánh giá SoccerNet metrics chính thức
    labels_path = cfg.get('path_labels', "downloads/dataset/")
    logging.info("Đang đánh giá Loose Metric...")
    res_l = evaluate(SoccerNet_path=labels_path, Predictions_path=output_zip,
                     split=args.split, prediction_file="results_spotting.json", version=2, metric="loose")

    logging.info("Đang đánh giá Tight Metric...")
    res_t = evaluate(SoccerNet_path=labels_path, Predictions_path=output_zip,
                     split=args.split, prediction_file="results_spotting.json", version=2, metric="tight")

    print("\n" + "="*70)
    print("🏆 KẾT QUẢ ĐÁNH GIÁ CHÍNH THỨC SOCCERNET RL ENSEMBLE")
    print("="*70)
    if res_l:
        print(f"  ⭐ Loose a_mAP: {res_l.get('a_mAP', 0.0) * 100:.2f}%")
    if res_t:
        print(f"  ⭐ Tight a_mAP: {res_t.get('a_mAP', 0.0) * 100:.2f}%")
    print("-"*70)

    if res_t and res_l and 'a_mAP_per_class' in res_t and 'a_mAP_per_class' in res_l:
        print(f"| {'ID':<3} | {'Class Name':<20} | {'Tight a-mAP':<15} | {'Loose a-mAP':<15} |")
        print("|" + "-"*5 + "|" + "-"*22 + "|" + "-"*17 + "|" + "-"*17 + "|")
        per_class_list = []
        for c_id in range(len(res_t['a_mAP_per_class'])):
            c_name = INVERSE_EVENT_DICTIONARY_V2.get(c_id, f"Class {c_id}")
            t_score = res_t['a_mAP_per_class'][c_id] * 100
            l_score = res_l['a_mAP_per_class'][c_id] * 100
            print(f"| {c_id:<3} | {c_name:<20} | {t_score:13.2f}% | {l_score:13.2f}% |")
            per_class_list.append({
                "id": c_id,
                "name": c_name,
                "tight_a_mAP": round(t_score, 2),
                "loose_a_mAP": round(l_score, 2)
            })
        print("="*70 + "\n")

        # Lưu file metrics_summary.json
        summary_path = os.path.join(base_out, "metrics_summary.json")
        summary_payload = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "split": args.split,
            "overall": {
                "loose_a_mAP": round(res_l.get('a_mAP', 0.0) * 100, 2),
                "tight_a_mAP": round(res_t.get('a_mAP', 0.0) * 100, 2),
            },
            "per_class": per_class_list
        }
        with open(summary_path, 'w') as sf:
            json.dump(summary_payload, sf, indent=4)
        logging.info(f"✅ Đã lưu bảng chi tiết metrics vào: {summary_path}")



if __name__ == "__main__":
    main()
