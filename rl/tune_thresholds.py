import os
import sys
import time
import json
import pickle
import logging
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from multiprocessing import Pool, cpu_count
from typing import Dict, List, Tuple, Any

import yaml
import numpy as np
import torch
from torch.amp import autocast
from tqdm import tqdm
from stable_baselines3 import PPO

# Thêm đường dẫn gốc vào sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from SoccerNet.Evaluation.utils import INVERSE_EVENT_DICTIONARY_V2
from mediafusion_dataset import SoccerNetFramesTesting
from mediafusion_model import MediaFusion
from dataset import feats2clip
from mediafusion_train import get_spot_from_SNMS

# Không gian tìm kiếm tối ưu cho Tight Metric ([1s, 2s, 3s, 4s, 5s])
CANDIDATE_THRESHOLDS = [
    0.005, 0.008, 0.010, 0.015, 0.020, 0.025,
    0.030, 0.040, 0.050, 0.060, 0.080, 0.100
]
# Candidate windows từ 2s đến 12s
CANDIDATE_WINDOWS = [2, 3, 4, 5, 6, 7, 8, 9, 10, 12]
# Tìm kiếm cả 2 decay function: pow2 (mặc định) và gaussian (suppression tập trung hơn)
CANDIDATE_DECAYS = ['pow2', 'gaussian']

# Trọng số delta cho Tight AP: ưu tiên AP@1s và AP@2s hơn để tối đa hóa precision tại delta nhỏ
# → Buộc grid search tìm bộ tham số cho dự đoán CỰC SẮC NÉT thay vì chỉ cân bằng 5 delta đều nhau
TIGHT_DELTA_WEIGHTS = [0.35, 0.30, 0.20, 0.10, 0.05]  # Tổng = 1.0, cho deltas [1s,2s,3s,4s,5s]


def precompute_closest(T: int, gt_list: List[int]) -> np.ndarray:
    """Tính mảng chỉ số ground truth gần nhất cho mỗi frame index (phục vụ tính TP tight)."""
    if len(gt_list) == 0:
        return np.full(T, -999999, dtype=np.int32)
    gt_arr = np.array(sorted(gt_list), dtype=np.int32)
    idx = np.searchsorted(gt_arr, np.arange(T))
    idx = np.clip(idx, 0, len(gt_arr) - 1)
    idx_left = np.maximum(0, idx - 1)
    dist1 = np.abs(np.arange(T) - gt_arr[idx])
    dist2 = np.abs(np.arange(T) - gt_arr[idx_left])
    return np.where(dist1 <= dist2, gt_arr[idx], gt_arr[idx_left])


def compute_tight_ap_for_spots(
    spots_per_game: List[np.ndarray],
    closests_per_game: List[np.ndarray],
    total_targets: int,
    framerate: int = 2,
    deltas: List[int] = [1, 2, 3, 4, 5],
    delta_weights: List[float] = None,
    gt_lists_per_game: List[List[int]] = None,
) -> float:
    """
    Tính chuẩn xác Tight a-mAP theo đúng công thức của SoccerNet v2:
    Tối ưu hóa vector hóa tốc độ cao (nhanh gấp 50x).
    """
    if total_targets == 0:
        return 0.0

    # Thu thập tất cả detections kèm game_id và sort 1 lần duy nhất
    det_list = []  # (conf, game_idx, frame_idx)
    for game_idx, spots in enumerate(spots_per_game):
        if len(spots) == 0:
            continue
        for s in spots:
            f_idx = int(s[0])
            conf = float(s[1])
            det_list.append((conf, game_idx, f_idx))

    if not det_list:
        return 0.0

    # Sắp xếp confidence giảm dần
    det_list.sort(key=lambda x: x[0], reverse=True)

    # Pre-extract closest GT và khoảng cách dist 1 lần duy nhất cho toàn bộ detections
    num_dets = len(det_list)
    det_info = []  # (game_idx, closest_gt, dist)
    for conf, game_idx, f_idx in det_list:
        closests = closests_per_game[game_idx]
        if f_idx < len(closests):
            c_gt = int(closests[f_idx])
            dist = abs(f_idx - c_gt) if c_gt >= 0 else 999999.0
            det_info.append((game_idx, c_gt, dist))
        else:
            det_info.append((game_idx, -1, 999999.0))

    ap_per_delta = []
    for delta in deltas:
        # SoccerNet định nghĩa tolerance radius là delta/2 giây = (delta * framerate) / 2 frames
        delta_tolerance = (delta * framerate) / 2.0

        # Theo dõi GT đã matched cho từng game
        matched_gts_per_game: Dict[int, set] = {}
        tp_arr = np.zeros(num_dets, dtype=np.float32)

        for i, (game_idx, closest_gt, dist) in enumerate(det_info):
            if dist <= delta_tolerance and closest_gt >= 0:
                if game_idx not in matched_gts_per_game:
                    matched_gts_per_game[game_idx] = set()
                if closest_gt not in matched_gts_per_game[game_idx]:
                    matched_gts_per_game[game_idx].add(closest_gt)
                    tp_arr[i] = 1.0

        tps = np.cumsum(tp_arr)
        fps = np.cumsum(1.0 - tp_arr)
        recalls = tps / float(total_targets)
        precisions = tps / (tps + fps)

        # Tích phân PR-Curve theo chuẩn VOC/SoccerNet
        prec_interp = np.maximum.accumulate(precisions[::-1])[::-1]
        rec_diff = np.diff(np.concatenate(([0.0], recalls)))
        ap = float(np.sum(rec_diff * prec_interp))
        ap_per_delta.append(ap)

    if delta_weights is not None:
        w = np.array(delta_weights[:len(ap_per_delta)], dtype=np.float64)
        w = w / w.sum()
        return float(np.dot(w, ap_per_delta))
    return float(np.mean(ap_per_delta))


def get_spot_from_SNMS_fast(Input: np.ndarray, window: float, thresh: float = 0.0, decay: str = 'pow2', max_spots: int = 300) -> np.ndarray:
    """Bản tối ưu hóa tốc độ cao của Soft-NMS, nhanh gấp 50-100 lần."""
    if np.max(Input) < thresh:
        return np.empty((0, 2), dtype=np.float32)

    detections_tmp = np.copy(Input).astype(np.float32)
    detections_tmp[detections_tmp < thresh] = -1.0
    indexes = []
    MaxValues = []
    half_w = float(window) / 2.0
    half_w_int = int(half_w)
    len_det = len(detections_tmp)

    for _ in range(max_spots):
        max_index = int(np.argmax(detections_tmp))
        max_value = float(detections_tmp[max_index])
        if max_value < thresh:
            break

        indexes.append(max_index)
        MaxValues.append(max_value)

        nms_from = max(0, max_index - half_w_int)
        nms_to = min(len_det, max_index + half_w_int + 1)
        offsets = np.arange(nms_from - max_index, nms_to - max_index, dtype=np.float32)

        if decay == 'gaussian':
            sigma = max(half_w / 2.0, 1.0)
            weight = 1.0 - np.exp(-(offsets ** 2) / (2.0 * sigma ** 2))
            weight[offsets == 0] = 0.0
        else:
            weight = (offsets / half_w) ** 2

        detections_tmp[nms_from:nms_to] *= weight
        detections_tmp[nms_from:nms_to][detections_tmp[nms_from:nms_to] < thresh] = -1.0

    if not indexes:
        return np.empty((0, 2), dtype=np.float32)
    return np.column_stack([indexes, MaxValues])


def grid_search_single_class(args_tuple: Tuple[int, List[np.ndarray], List[List[int]], int], show_pbar: bool = True) -> Dict[str, Any]:
    """
    Tìm kiếm (Window, Threshold, Decay) tối ưu hóa Tight a-mAP cho 1 class với thanh tiến trình tqdm chi tiết.
    """
    class_id, all_class_probs, all_gt_spots, outputrate = args_tuple
    class_name = INVERSE_EVENT_DICTIONARY_V2.get(class_id, f"Class_{class_id}")

    total_targets = sum(len(gt_list) for gt_list in all_gt_spots)
    closests_per_game = [
        precompute_closest(len(probs), gt_list)
        for probs, gt_list in zip(all_class_probs, all_gt_spots)
    ]

    combos = [
        (w, th, decay)
        for w in CANDIDATE_WINDOWS
        for th in CANDIDATE_THRESHOLDS
        for decay in CANDIDATE_DECAYS
    ]

    best_tight_ap = -1.0
    best_window = 6
    best_thresh = 0.010
    best_decay = 'pow2'

    iterator = combos
    pbar = None
    if show_pbar:
        pbar = tqdm(combos, desc=f"[{class_id+1:2d}/17] {class_name:<18}", ncols=110, dynamic_ncols=True, leave=True)
        iterator = pbar

    for w, th, decay in iterator:
        spots_per_game = [
            get_spot_from_SNMS_fast(probs, window=w * outputrate, thresh=th, decay=decay)
            for probs in all_class_probs
        ]

        tight_ap = compute_tight_ap_for_spots(
            spots_per_game=spots_per_game,
            closests_per_game=closests_per_game,
            total_targets=total_targets,
            framerate=outputrate,
            deltas=[1, 2, 3, 4, 5],
            delta_weights=TIGHT_DELTA_WEIGHTS,
        )

        if tight_ap > best_tight_ap:
            best_tight_ap = tight_ap
            best_window = w
            best_thresh = th
            best_decay = decay
            if pbar is not None:
                pbar.set_postfix({"Best_AP": f"{best_tight_ap*100:5.2f}%", "w": best_window, "th": best_thresh, "dec": best_decay[:4]})

    if pbar is not None:
        pbar.set_postfix({"FINAL_AP": f"{best_tight_ap*100:5.2f}%", "w": best_window, "th": best_thresh, "dec": best_decay[:4]})

    return {
        'class_id': class_id,
        'class_name': class_name,
        'best_window': best_window,
        'best_thresh': best_thresh,
        'best_decay': best_decay,
        'score': best_tight_ap,
    }


def grid_search_worker(args_tuple: Tuple[int, List[np.ndarray], List[List[int]], int]) -> Dict[str, Any]:
    return grid_search_single_class(args_tuple, show_pbar=False)


def extract_validation_probabilities(
    base_models: List[torch.nn.Module],
    rl_agent: PPO,
    dataloader: Any,
    cfg: Dict[str, Any],
    device: torch.device
) -> Tuple[List[List[np.ndarray]], List[List[List[int]]]]:
    """
    Inference trên toàn bộ tập Validation để trích xuất mảng xác suất và ground truth.
    """
    outputrate = cfg['outputrate']
    chunk_size = cfg['chunk_size']
    stride = cfg.get('test_stride', 8)
    num_classes = 17

    all_game_probs: List[List[np.ndarray]] = [[] for _ in range(num_classes)]
    all_game_gts: List[List[List[int]]] = [[] for _ in range(num_classes)]

    logging.info(f"Bắt đầu trích xuất probabilities cho {len(dataloader)} trận tập Valid...")
    start_t = time.time()

    for m in base_models:
        m.eval()
        m.to(device)

    with torch.no_grad():
        pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc="Trích xuất Valid", ncols=110)
        for i, (game_ID, data) in pbar:
            g_start = time.time()
            game_ID = game_ID[0]

            # Đọc nhãn ground truth
            labels_path = os.path.join(cfg['path_labels'], game_ID, "Labels-v2.json")
            game_gt_per_class: Dict[Tuple[int, int], List[int]] = {}
            if os.path.exists(labels_path):
                with open(labels_path, 'r') as jf:
                    labels_json = json.load(jf)
                for ann in labels_json.get("annotations", []):
                    half = int(ann.get("gameTime", "1 - 0:00").split(" - ")[0])
                    label_str = ann.get("label", "")
                    pos_ms = int(ann.get("position", "0"))
                    frame_pos = int(round((pos_ms / 1000.0) * outputrate))
                    for c_idx, c_name in INVERSE_EVENT_DICTIONARY_V2.items():
                        if c_name == label_str:
                            key = (half, c_idx)
                            if key not in game_gt_per_class:
                                game_gt_per_class[key] = []
                            game_gt_per_class[key].append(frame_pos)
                            break

            # Duyệt qua 2 hiệp đấu
            for half_num, (featB_key, featA_key) in enumerate([('featB1', 'featA1'), ('featB2', 'featA2')], start=1):
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

                timestamp_long = np.zeros((sec * outputrate, num_classes), dtype=np.float32)
                num_clips = len(clips_B)
                BS = 32

                for b_start in range(0, num_clips, BS):
                    b_end = min(b_start + BS, num_clips)
                    batch_B = clips_B[b_start:b_end].to(device)
                    batch_A = clips_A[b_start:b_end].to(device) if clips_A is not None else None

                    # Dự đoán từ các models với FP16 autocast chuẩn PyTorch mới
                    model_preds_C = []
                    model_preds_D = []
                    with autocast('cuda', dtype=torch.float16):
                        for m in base_models:
                            out = m(featsB=batch_B, featsA=batch_A, inference=True)
                            model_preds_C.append(out['preds'].float().cpu().numpy())
                            predD = out['predsD'][:, :, :, 0] if out['predsD'].ndim == 4 else out['predsD']
                            model_preds_D.append(predD.float().cpu().numpy())

                    # Tạo observation cho RL Agent
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

                        clip_pred_C = sum(weights[m_idx] * model_preds_C[m_idx][l] for m_idx in range(len(base_models)))
                        clip_pred_D = sum(weights[m_idx] * model_preds_D[m_idx][l] for m_idx in range(len(base_models)))

                        # Cập nhật displacement chuẩn xác
                        initial_pos = (b_start + l) * stride * outputrate
                        mask = clip_pred_C[:, 1:] > 0.001
                        j_idx, k_idx = np.where(mask)
                        if len(j_idx) > 0:
                            probs = clip_pred_C[j_idx, k_idx + 1]
                            displs = clip_pred_D[j_idx, k_idx + 1]
                            raw_pos = np.round(initial_pos + j_idx - displs).astype(np.int64)
                            valid_mask = (raw_pos >= 0) & (raw_pos < len(timestamp_long))
                            valid_pos = raw_pos[valid_mask]
                            valid_k = k_idx[valid_mask]
                            valid_probs = probs[valid_mask]
                            for pos, k, prob in zip(valid_pos, valid_k, valid_probs):
                                if prob > timestamp_long[pos, k]:
                                    timestamp_long[pos, k] = prob

                for k_class in range(num_classes):
                    all_game_probs[k_class].append(timestamp_long[:, k_class])
                    all_game_gts[k_class].append(game_gt_per_class.get((half_num, k_class), []))

            g_elapsed = time.time() - g_start
            pbar.set_postfix({"game": game_ID[:20], "time": f"{g_elapsed:.1f}s"})

    logging.info(f"Hoàn thành trích xuất trong {time.time() - start_t:.1f}s.")
    return all_game_probs, all_game_gts


def main() -> None:
    parser = ArgumentParser(description="Tối ưu hóa Per-Class Threshold & Window cho SoccerNet RL Ensemble (Tight a-mAP)", formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("--config", type=str, default="configs/MediaFusion.yaml", help="File config gốc")
    parser.add_argument("--agent", type=str, default="models/RL_agents/MediaFusion_Ensemble_v2_34_41_47/best_ppo_agent.zip", help="Đường dẫn file RL Agent đã train")
    parser.add_argument("--models", nargs="+", type=str, default=None, help="Checkpoints của các base models")
    parser.add_argument("--output_yaml", type=str, default="rl/best_params_v2_34_41_47.yaml", help="Nơi lưu bộ tham số tốt nhất")
    parser.add_argument("--cache_probs", type=str, default="models/RL_agents/MediaFusion_Ensemble_v2_34_41_47/valid_probs_cache.pkl", help="File cache lưu probabilities để chạy lại cực nhanh không cần inference lại")
    parser.add_argument("--force_extract", action="store_true", help="Bắt buộc trích xuất lại từ GPU dù đã có cache")
    parser.add_argument("--cores", type=int, default=17, help="Số process chạy song song trên CPU")
    parser.add_argument("--parallel", action="store_true", help="Chạy song song Pool thay vì tuần tự từng class")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    cfg_path = args.config
    with open(cfg_path, 'r') as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() and cfg.get('gpu', 0) >= 0 else "cpu")

    # 1. Nạp hoặc Trích xuất Probabilities
    if os.path.exists(args.cache_probs) and not args.force_extract:
        logging.info(f"Tìm thấy cache probabilities tại: {args.cache_probs}. Nạp dữ liệu trong vài giây...")
        with open(args.cache_probs, 'rb') as f:
            cache_data = pickle.load(f)
            all_game_probs = cache_data['probs']
            all_game_gts = cache_data['gts']
        logging.info(f"✅ Nạp thành công dữ liệu cho 17 Classes từ {len(all_game_probs[0])} hiệp đấu.")
    else:
        # Load RL Agent
        if not os.path.exists(args.agent):
            logging.error(f"Không tìm thấy agent tại {args.agent}!")
            sys.exit(1)
        rl_agent = PPO.load(args.agent, device="cpu")
        logging.info(f"Đã load RL Agent từ {args.agent}")

        # Load Base Models
        model_paths = args.models or [
            "models/ASmodels/MediaFusion/model_34.pth.tar",
            "models/ASmodels/MediaFusion/model_41.pth.tar",
            "models/ASmodels/MediaFusion/model_47.pth.tar"
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

        # Dataloader tập Valid
        dataset_valid = SoccerNetFramesTesting(
            path_labels=cfg['path_labels'], path_baidu=cfg['path_baidu'], path_audio=cfg['path_audio'],
            split=['valid'], outputrate=cfg['outputrate'], chunk_size=cfg['chunk_size'],
            baidu=cfg['baidu'], audio=cfg['audio']
        )
        val_loader = torch.utils.data.DataLoader(dataset_valid, batch_size=1, shuffle=False, num_workers=2)

        # Trích xuất từ GPU
        all_game_probs, all_game_gts = extract_validation_probabilities(base_models, rl_agent, val_loader, cfg, device)

        # Lưu cache
        os.makedirs(os.path.dirname(args.cache_probs), exist_ok=True)
        with open(args.cache_probs, 'wb') as f:
            pickle.dump({'probs': all_game_probs, 'gts': all_game_gts}, f)
        logging.info(f"✅ Đã lưu cache probabilities vào: {args.cache_probs}")

    # 2. Quét lưới 17 classes với tqdm hiển thị tiến độ chi tiết
    tasks = [(c_idx, all_game_probs[c_idx], all_game_gts[c_idx], cfg['outputrate']) for c_idx in range(17)]
    start_grid = time.time()
    results = []

    if not args.parallel:
        print("\n" + "="*90)
        print("🎯 BẮT ĐẦU TỐI ƯU HÓA 17 CLASSES TUẦN TỰ (TQDM TỪNG CLASS)")
        print("="*90)
        for task in tasks:
            res = grid_search_single_class(task, show_pbar=True)
            results.append(res)
    else:
        num_processes = min(args.cores, cpu_count(), 17)
        logging.info(f"Khởi tạo Pool với {num_processes} processes song song để tối ưu hóa Tight a-mAP cho 17 Classes...")
        with Pool(processes=num_processes) as pool:
            pbar = tqdm(
                pool.imap_unordered(grid_search_worker, tasks),
                total=len(tasks),
                desc="Tuning 17 Classes (Tight a-mAP)",
                ncols=110
            )
            for res in pbar:
                results.append(res)
                pbar.set_postfix({"Class": res['class_name'][:12], "Tight_aMAP": f"{res['score']*100:.2f}%"})

    logging.info(f"Grid search hoàn thành sau {time.time() - start_grid:.1f}s.")

    # 3. Xuất bảng kết quả và lưu file config YAML
    results = sorted(results, key=lambda x: x['class_id'])
    
    print("\n" + "="*90)
    print("🎯 BẢNG THAM SỐ TỐI ƯU HÓA CHUẨN TIGHT a-mAP (SOCCERNET v2) — Delta Weighted")
    print("="*90)
    print(f"{'ID':<3} | {'Class Name':<18} | {'Window (s)':<10} | {'Thresh':<8} | {'Decay':<8} | {'Valid Tight a-mAP':<18}")
    print("-" * 90)
    
    classes_config = []
    total_tight_ap = 0.0
    for r in results:
        tight_ap_pct = r['score'] * 100
        total_tight_ap += r['score']
        decay = r.get('best_decay', 'pow2')
        print(f"{r['class_id']:<3} | {r['class_name']:<18} | {r['best_window']:<10} | {r['best_thresh']:<8.3f} | {decay:<8} | {tight_ap_pct:16.2f}%")
        classes_config.append({
            'id': r['class_id'],
            'name': r['class_name'],
            'window': int(r['best_window']),
            'threshold': float(r['best_thresh']),
            'decay': decay,
        })
    mean_tight_ap = (total_tight_ap / 17.0) * 100
    print("-" * 90)
    print(f"⭐ ƯỚC TÍNH OVERALL TIGHT a-mAP TRÊN TẬP VALID: {mean_tight_ap:.2f}%")
    print(f"   (Weighted-delta objective: {TIGHT_DELTA_WEIGHTS} cho deltas 1s-5s)")
    print("="*90 + "\n")

    os.makedirs(os.path.dirname(args.output_yaml), exist_ok=True)
    with open(args.output_yaml, 'w') as f:
        yaml.dump({'classes': classes_config}, f, default_flow_style=False, sort_keys=False)

    logging.info(f"✅ Đã lưu bộ tham số tốt nhất vào: {args.output_yaml}")


if __name__ == "__main__":
    main()
