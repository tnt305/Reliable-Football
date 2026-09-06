import logging
import os
import time
from tqdm import tqdm
import torch
import numpy as np
import json
import zipfile
import yaml
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from datetime import datetime

# RL Imports
from stable_baselines3 import PPO

# MediaFusion Project Imports
from SoccerNet.Evaluation.utils import AverageMeter, INVERSE_EVENT_DICTIONARY_V2
from SoccerNet.Evaluation.ActionSpotting import evaluate
from mediafusion_dataset import SoccerNetFramesTesting
from mediafusion_model import MediaFusion
from dataset import feats2clip
# Import các hàm NMS từ mediafusion_train.py
from mediafusion_train import get_spot_from_NMS, get_spot_from_SNMS

# --- HÀM TESTSPOTTING ĐÃ ĐƯỢC CHỈNH SỬA CHO RL ENSEMBLE ---
def testSpottingRL(dataloader,
                   base_models,
                   rl_agent,
                   cfg,
                   model_name,
                   overwrite=True,
                   NMS_window=8,
                   NMS_threshold=0.5,
                   postprocessing='SNMS'):
    """
    Function for inference of the RL-ensembled action spotting models.
    """
    
    # Lấy thông tin từ config
    outputrate = cfg['outputrate']
    chunk_size = cfg['chunk_size']
    stride = cfg['chunk_size'] // 2 # Hoặc giá trị stride bạn dùng trong test
    device = next(rl_agent.policy.parameters()).device # Lấy device từ agent

    # Thiết lập đường dẫn output
    output_folder_name = f"outputs_{dataloader.dataset.split}"
    output_path = os.path.join(cfg['path_experiments'], "RL_agents", model_name, 
                               f'post_{postprocessing}_window_{NMS_window}')
    output_results_zip = os.path.join(output_path, f"results_spotting_{dataloader.dataset.split}.zip")
    
    if os.path.exists(output_results_zip) and not overwrite:
        logging.info(f"Results already exist at {output_results_zip}. Skipping inference.")
    else:
        # Chuyển các model cơ sở sang chế độ eval và đưa lên device
        for model in base_models:
            model.eval()
            model.to(device)

        batch_time = AverageMeter()
        data_time = AverageMeter()
        end = time.time()

        # --- Vòng lặp chính qua các game trong dataloader ---
        with tqdm(enumerate(dataloader), total=len(dataloader), ncols=150) as t:
            for i, (game_ID, data) in t:
                data_time.update(time.time() - end)
                game_ID = game_ID[0]

                # Chuẩn bị dữ liệu cho cả 2 hiệp
                # HIỆP 1
                featB_half1_full = data['featB1'].reshape(-1, data['featB1'].shape[-1])
                sec1 = featB_half1_full.shape[0]
                featB_half1_clips = feats2clip(featB_half1_full, stride=stride, clip_length=chunk_size)
                
                featA_half1_clips = None
                if cfg['audio']:
                    featA_half1_full = data['featA1'].reshape(-1, data['featA1'].shape[-1])
                    featA_half1_clips = feats2clip(featA_half1_full.T, stride=stride * 100, clip_length=chunk_size * 100)

                # HIỆP 2
                featB_half2_full = data['featB2'].reshape(-1, data['featB2'].shape[-1])
                sec2 = featB_half2_full.shape[0]
                featB_half2_clips = feats2clip(featB_half2_full, stride=stride, clip_length=chunk_size)

                featA_half2_clips = None
                if cfg['audio']:
                    featA_half2_full = data['featA2'].reshape(-1, data['featA2'].shape[-1])
                    featA_half2_clips = feats2clip(featA_half2_full.T, stride=stride * 100, clip_length=chunk_size * 100)

                # Khởi tạo mảng lưu kết quả dự đoán cho mỗi hiệp
                timestamp_long_half_1 = np.zeros((sec1 * outputrate, 17))
                timestamp_long_half_2 = np.zeros((sec2 * outputrate, 17))

                # --- XỬ LÝ DỰ ĐOÁN CHO TỪNG HIỆP ---
                for half_idx, (featB_clips, featA_clips, timestamp_long) in enumerate([
                    (featB_half1_clips, featA_half1_clips, timestamp_long_half_1),
                    (featB_half2_clips, featA_half2_clips, timestamp_long_half_2)
                ]):
                    
                    num_clips = len(featB_clips)
                    if num_clips == 0: continue

                    # Chia thành các batch nhỏ để xử lý
                    BS = 16 # Batch size cho inference
                    for b_start in range(0, num_clips, BS):
                        b_end = min(b_start + BS, num_clips)
                        batch_featB = featB_clips[b_start:b_end].clone().to(device)
                        batch_featA = featA_clips[b_start:b_end].clone().to(device) if cfg['audio'] else None
                        
                        # ----- ĐÂY LÀ PHẦN THAY ĐỔI CỐT LÕI -----
                        
                        # 1. Tạo observation cho RL agent
                        # Observation là đặc trưng của từng clip trong batch
                        if cfg['audio']:
                            obs_batch = torch.cat([batch_featB, batch_featA], dim=-1)
                        else:
                            obs_batch = batch_featB
                        
                        # 2. RL Agent dự đoán trọng số
                        # rl_agent.predict sẽ trả về (actions, states)
                        weights_batch, _ = rl_agent.predict(obs_batch.cpu().numpy(), deterministic=True)
                        
                        # Chuẩn hóa trọng số
                        weights_sum = weights_batch.sum(axis=1, keepdims=True)
                        weights_sum[weights_sum == 0] = 1 # Tránh chia cho 0
                        normalized_weights = weights_batch / weights_sum

                        # 3. Lấy dự đoán từ các model cơ sở và tính ensemble
                        with torch.no_grad():
                            all_preds_C = []
                            all_preds_D = []
                            for model in base_models:
                                output = model(featsB=batch_featB, featsA=batch_featA, inference=True)
                                all_preds_C.append(output['preds'])
                                all_preds_D.append(output['predsD'])

                            # Tính trung bình có trọng số
                            batch_size_curr = batch_featB.shape[0]
                            weighted_preds_C = torch.zeros_like(all_preds_C[0])
                            weighted_preds_D = torch.zeros_like(all_preds_D[0])

                            for i, weight_vec in enumerate(normalized_weights):
                                model_preds_C = all_preds_C[i]
                                model_preds_D = all_preds_D[i]
                                # weight_vec có shape (num_models,)
                                # Mở rộng để nhân với predictions
                                # Trọng số cho model thứ j là weight_vec[j]
                                for j in range(len(base_models)):
                                     # Dùng trọng số từ RL agent
                                     w = torch.from_numpy(normalized_weights[:, j:j+1]).float().to(device)
                                     w = w.unsqueeze(-1).unsqueeze(-1) # Mở rộng dims để broadcast
                                     weighted_preds_C += all_preds_C[j] * w
                                     weighted_preds_D += all_preds_D[j] * w

                        # 4. Gom kết quả vào mảng timestamp lớn
                        predC = weighted_preds_C.cpu().detach().numpy()
                        # Xử lý uncertainty nếu có
                        if cfg['model']['uncertainty']:
                            predD = weighted_preds_D[:, :, :, 0].cpu().detach().numpy()
                        else:
                            predD = weighted_preds_D.cpu().detach().numpy()

                        batch_s, nf, nc = predC.shape
                        for l in range(batch_s):
                            initial_pos = (b_start + l) * stride * outputrate
                            for j in range(nf):
                                for k in range(nc - 1): # Bỏ qua background class
                                    prob = predC[l, j, k + 1]
                                    if prob > NMS_threshold:
                                        rel_position = j - predD[l, j, k + 1]
                                        position = min(len(timestamp_long) - 1, max(0, int(round(initial_pos + rel_position))))
                                        timestamp_long[position, k] = max(timestamp_long[position, k], prob)
                
                # --- POST-PROCESSING và LƯU KẾT QUẢ ---
                # Chọn hàm NMS
                if postprocessing == 'NMS':
                    get_spot = get_spot_from_NMS
                    nms_windows_list = [NMS_window] * 17
                elif postprocessing == 'SNMS':
                    get_spot = get_spot_from_SNMS
                    # Các giá trị window tối ưu cho SNMS từ các nghiên cứu trước
                    nms_windows_list = [5, 7, 9, 12, 10, 14, 14, 5, 8, 8, 8, 8, 13, 5, 6, 6, 6]
                
                json_data = {"UrlLocal": game_ID, "predictions": []}

                for half_idx, timestamp_long in enumerate([timestamp_long_half_1, timestamp_long_half_2]):
                    for l in range(dataloader.dataset.num_classes):
                        spots = get_spot(timestamp_long[:, l], window=nms_windows_list[l] * outputrate, thresh=NMS_threshold)
                        
                        for spot in spots:
                            frame_index = int(spot[0])
                            confidence = spot[1]
                            seconds = int((frame_index / outputrate) % 60)
                            minutes = int((frame_index / outputrate) // 60)
                            
                            json_data["predictions"].append({
                                "gameTime": f"{half_idx+1} - {minutes}:{seconds}",
                                "label": INVERSE_EVENT_DICTIONARY_V2[l],
                                "position": str(int((frame_index / outputrate) * 1000)),
                                "half": str(half_idx + 1),
                                "confidence": str(confidence)
                            })
                
                # Lưu file JSON
                game_output_dir = os.path.join(output_path, output_folder_name, game_ID)
                os.makedirs(game_output_dir, exist_ok=True)
                with open(os.path.join(game_output_dir, "results_spotting.json"), 'w') as f:
                    json.dump(json_data, f, indent=4)

                batch_time.update(time.time() - end)
                end = time.time()
                t.set_description(f'Test (RL-Ensemble) {game_ID}: Time {batch_time.avg:.3f}s')

        # Nén thư mục kết quả
        def zipResults(zip_path, target_dir, filename="results_spotting.json"):
            with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipobj:
                rootlen = len(target_dir) + 1
                for base, dirs, files in os.walk(target_dir):
                    for file in files:
                        if file == filename:
                            fn = os.path.join(base, file)
                            zipobj.write(fn, fn[rootlen:])
        
        zipResults(zip_path=output_results_zip,
                   target_dir=os.path.join(output_path, output_folder_name))
        logging.info(f"Zipped results to {output_results_zip}")

    # Đánh giá kết quả
    if dataloader.dataset.split == "challenge":
        print("Visit eval.ai to evaluate performances on Challenge set")
        return None, None
    
    # Giả sử bạn có thư mục chứa ground-truth labels của SoccerNet
    labels_path = cfg.get('path_labels', '/path/to/soccernet/labels')
    
    results_l = evaluate(SoccerNet_path=labels_path, Predictions_path=output_results_zip,
                         split=dataloader.dataset.split, version=2, metric="loose")
    results_t = evaluate(SoccerNet_path=labels_path, Predictions_path=output_results_zip,
                         split=dataloader.dataset.split, version=2, metric="tight")

    return results_l, results_t


def main_test_rl(args, cfg):
    logging.info(f"--- Starting RL Ensemble Testing for {args.model_name} ---")
    device = torch.device("cuda" if torch.cuda.is_available() and cfg['gpu'] >= 0 else "cpu")

    # 1. Tải RL agent đã được huấn luyện
    rl_agent_path = os.path.join(cfg['path_experiments'], "RL_agents", args.model_name, "ppo_ensemble_final.zip")
    if not os.path.exists(rl_agent_path):
        logging.error(f"RL agent not found at {rl_agent_path}. Please train the agent first.")
        return
    rl_agent = PPO.load(rl_agent_path, device=device)
    logging.info(f"Loaded RL agent from {rl_agent_path}")

    # 2. Tải các mô hình MediaFusion cơ sở (base models)
    # LƯU Ý: Đường dẫn này phải giống với lúc bạn huấn luyện RL agent
    model_paths = [
        os.path.join(cfg['path_experiments'], "ASmodels", "MediaFusion_seed1", "model.pth.tar"),
        os.path.join(cfg['path_experiments'], "ASmodels", "MediaFusion_seed2", "model.pth.tar"),
        os.path.join(cfg['path_experiments'], "ASmodels", "MediaFusion_seed3", "model.pth.tar"),
    ]
    
    base_models = []
    for path in model_paths:
        if not os.path.exists(path):
            logging.error(f"Base model checkpoint not found at {path}.")
            return
        model = MediaFusion(chunk_size=cfg['chunk_size'], n_output=int(cfg['outputrate'] * cfg['chunk_size']), baidu=cfg['baidu'], audio=cfg['audio'], model_cfg=cfg['model'])
        checkpoint = torch.load(path, map_location=device)
        model.load_state_dict(checkpoint['state_dict'])
        base_models.append(model)
    logging.info(f"Loaded {len(base_models)} base models.")

    # 3. Tạo dataloader cho tập test
    for split in cfg['test_split']:
        logging.info(f"--- Testing on split: {split} ---")
        dataset_Test = SoccerNetFramesTesting(path_labels=cfg['path_labels'], path_baidu=cfg['path_baidu'], path_audio=cfg['path_audio'], split=split, outputrate=cfg['outputrate'], chunk_size=cfg['chunk_size'], baidu=cfg['baidu'], audio=cfg['audio'])
        test_loader = torch.utils.data.DataLoader(dataset_Test, batch_size=1, shuffle=False, num_workers=1, pin_memory=True)

        # 4. Chạy hàm test spotting với RL
        results_l, results_t = testSpottingRL(test_loader, base_models, rl_agent, cfg, args.model_name,
                                            NMS_threshold=cfg['NMS_threshold'],
                                            NMS_window=cfg.get('NMS_window', 8), # Lấy từ config, mặc định 8
                                            postprocessing=cfg.get('postprocessing', 'SNMS')) # Mặc định SNMS

        if results_l:
            logging.info("--- Loose Metric Results ---")
            logging.info(f"a_mAP: {results_l['a_mAP']:.4f}")
            logging.info(f"a_mAP visible: {results_l['a_mAP_visible']:.4f}")
            logging.info(f"a_mAP unshown: {results_l['a_mAP_unshown']:.4f}")

        if results_t:
            logging.info("--- Tight Metric Results ---")
            logging.info(f"a_mAP: {results_t['a_mAP']:.4f}")
            logging.info(f"a_mAP visible: {results_t['a_mAP_visible']:.4f}")
            logging.info(f"a_mAP unshown: {results_t['a_mAP_unshown']:.4f}")

if __name__ == '__main__':
    parser = ArgumentParser(description='MediaFusion RL Ensemble Testing', formatter_class=ArgumentDefaultsHelpFormatter)
    # Tên của agent đã được huấn luyện
    parser.add_argument('--model_name', required=True, type=str, help='Name of the trained RL agent model')
    args = parser.parse_args()

    numeric_level = getattr(logging, 'INFO', None)
    
    # Dùng file config của mô hình gốc để lấy các tham số
    args.config_file = os.path.join('configs', 'MediaFusion.yaml')
    with open(args.config_file, 'r') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    
    log_dir = os.path.join(cfg['path_experiments'], "RL_agents", args.model_name)
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"test_log_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log")
    
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s [%(levelname)-5.5s]  %(message)s",
        handlers=[logging.FileHandler(log_path), logging.StreamHandler()]
    )

    os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg['gpu'])
    start = time.time()
    main_test_rl(args, cfg)
    logging.info(f'Total Testing Time is {time.time()-start:.2f} seconds')