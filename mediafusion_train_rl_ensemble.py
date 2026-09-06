import torch
import time
import numpy as np
import os
import yaml
import logging
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from datetime import datetime

# RL Imports
from gymnasium import Env, spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback

# MediaFusion Project Imports
from mediafusion_dataset import SoccerNetFrames
from mediafusion_model import MediaFusion
from mediafusion_train import pred2vec, compute_mAP

class MediaFusionEnsembleEnv(Env):
    """
    Môi trường Reinforcement Learning để tìm trọng số ensemble cho các mô hình MediaFusion.
    """
    def __init__(self, models, dataloader, device, cfg):
        super(MediaFusionEnsembleEnv, self).__init__()
        
        self.models = [model.to(device) for model in models]
        self.dataloader = dataloader
        self.dataloader_iter = iter(self.dataloader)
        self.device = device
        self.cfg = cfg
        self.current_batch = None
        
        # Xác định không gian quan sát (observation space)
        # Ghép nối đặc trưng Baidu và Audio
        # Giả sử featB có 8576 chiều và featA có 128 chiều
        feat_dim = 8576 + (128 if self.cfg['audio'] else 0)
        obs_shape = (self.cfg['chunk_size'], feat_dim)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=obs_shape, dtype=np.float32
        )

        # Không gian hành động (action space): trọng số cho mỗi mô hình
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(len(self.models),), dtype=np.float32)

        logging.info("MediaFusionEnsembleEnv initialized.")
        logging.info(f"Observation shape: {obs_shape}")
        logging.info(f"Action shape: {(len(self.models),)}")


    def _get_obs(self):
        """Trích xuất và chuẩn bị observation từ batch hiện tại."""
        featB = self.current_batch['featB'].to(self.device)
        
        if self.cfg['audio']:
            featA = self.current_batch['featA'].to(self.device)
            # Lấy mẫu đầu tiên trong batch làm observation
            # Batch size của dataloader RL nên là 1
            obs = torch.cat([featB[0], featA[0]], dim=-1) 
        else:
            obs = featB[0]
            
        return obs.cpu().numpy()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Reset dataloader iterator
        self.dataloader_iter = iter(self.dataloader)
        try:
            self.current_batch = next(self.dataloader_iter)
        except StopIteration:
            logging.error("Dataloader is empty. Cannot reset environment.")
            # Xử lý trường hợp dataloader không có dữ liệu
            return np.zeros(self.observation_space.shape, dtype=np.float32), {}

        info = {}
        return self._get_obs(), info

    def step(self, action):
        # 1. Chuẩn hóa trọng số (action)
        weights = np.array(action)
        action_sum = weights.sum()
        if action_sum > 0:
            weights = weights / action_sum
        else:
            weights = np.ones_like(weights) / len(self.models) # Trọng số đều nhau nếu tổng = 0

        # 2. Lấy dự đoán từ các mô hình
        all_preds_C = []
        all_preds_D = []
        
        with torch.no_grad():
            featB = self.current_batch['featB'].to(self.device)
            featA = self.current_batch['featA'].to(self.device) if self.cfg['audio'] else None
            
            for model in self.models:
                model.eval()
                output = model(featsB=featB, featsA=featA, inference=True)
                all_preds_C.append(output['preds'])
                all_preds_D.append(output['predsD'])

        # 3. Tính dự đoán ensemble
        weighted_preds_C = torch.zeros_like(all_preds_C[0])
        weighted_preds_D = torch.zeros_like(all_preds_D[0])

        for i, weight in enumerate(weights):
            weighted_preds_C += all_preds_C[i] * weight
            weighted_preds_D += all_preds_D[i] * weight

        # 4. Tính toán REWARD (dựa trên mAP)
        labels = self.current_batch['labels']
        labels_D = self.current_batch['labels_displ']
        
        # Chuyển đổi labels và preds sang định dạng vector để tính mAP
        vec_labels = pred2vec([labels, labels_D], chunk_size=self.cfg['chunk_size'], outputrate=self.cfg['outputrate'], target=True)
        vec_preds = pred2vec([weighted_preds_C, weighted_preds_D], chunk_size=self.cfg['chunk_size'], outputrate=self.cfg['outputrate'], NMS=True)
        
        # Tính mAP làm reward. `compute_mAP` trả về (amap, amap_per_class)
        reward, _ = compute_mAP(vec_preds, vec_labels, metric="tight")

        # 5. Lấy observation tiếp theo
        terminated = False
        try:
            self.current_batch = next(self.dataloader_iter)
        except StopIteration:
            # Kết thúc episode khi hết dữ liệu
            terminated = True
            # Không cần observation mới khi kết thúc
            next_obs = np.zeros(self.observation_space.shape, dtype=np.float32)

        if not terminated:
             next_obs = self._get_obs()

        truncated = False # Không dùng truncated
        info = {'weights': weights}
        
        return next_obs, reward, terminated, truncated, info


def main_rl(args, cfg):
    # --- SETUP ---
    logging.info("Starting RL Ensemble Training")
    device = torch.device("cuda" if torch.cuda.is_available() and cfg['gpu'] >= 0 else "cpu")
    torch.manual_seed(cfg['seed'])
    np.random.seed(cfg['seed'])

    # --- TẢI CÁC MÔ HÌNH MEDIAFUSION ĐÃ ĐƯỢC HUẤN LUYỆN ---
    # Ví dụ: Tải 3 mô hình từ các checkpoints khác nhau
    # BẠN CẦN THAY ĐỔI ĐƯỜNG DẪN TỚI CÁC MODEL CỦA BẠN
    model_paths = [
        os.path.join(cfg['path_experiments'], "ASmodels", "MediaFusion_seed1", "model.pth.tar"),
        os.path.join(cfg['path_experiments'], "ASmodels", "MediaFusion_seed2", "model.pth.tar"),
        os.path.join(cfg['path_experiments'], "ASmodels", "MediaFusion_seed3", "model.pth.tar"),
    ]
    
    models = []
    for path in model_paths:
        if not os.path.exists(path):
            logging.error(f"Model checkpoint not found at {path}. Please provide valid paths.")
            return

        model = MediaFusion(chunk_size=cfg['chunk_size'], n_output=int(cfg['outputrate'] * cfg['chunk_size']), baidu=cfg['baidu'], audio=cfg['audio'], model_cfg=cfg['model'])
        checkpoint = torch.load(path, map_location=device)
        model.load_state_dict(checkpoint['state_dict'])
        model.eval()
        models.append(model)
    
    logging.info(f"Loaded {len(models)} models for ensembling.")

    # --- TẠO DATALOADER CHO VIỆC HUẤN LUYỆN RL ---
    # Dùng tập validation để huấn luyện agent RL
    # Batch size nên là 1 để mỗi step xử lý 1 mẫu
    dataset_Valid_RL = SoccerNetFrames(path_labels=cfg['path_labels'], path_store=cfg['path_store'], path_baidu=cfg['path_baidu'], path_audio=cfg['path_audio'], split=cfg['val_split'], chunk_size=cfg['chunk_size'], outputrate=cfg['outputrate'], stride=cfg['chunk_size'] // 2, rC=cfg['rC'], rD=cfg['rD'], store=cfg['store'], max_games=cfg['max_games'])
    
    rl_train_loader = torch.utils.data.DataLoader(dataset_Valid_RL, 
                            batch_size=1, # Quan trọng: batch size là 1
                            shuffle=True,
                            num_workers=cfg['num_workers'], 
                            pin_memory=True)

    # --- KHỞI TẠO MÔI TRƯỜNG VÀ AGENT ---
    env = MediaFusionEnsembleEnv(models, rl_train_loader, device, cfg)
    vec_env = DummyVecEnv([lambda: env]) # Bọc môi trường

    # Tạo thư mục lưu checkpoint của RL agent
    rl_save_path = os.path.join(cfg['path_experiments'], "RL_agents", cfg['model_name'])
    os.makedirs(rl_save_path, exist_ok=True)
    
    # Callback để lưu agent định kỳ
    checkpoint_callback = CheckpointCallback(
        save_freq=len(rl_train_loader) * 5,  # Lưu sau mỗi 5 epochs
        save_path=rl_save_path,
        name_prefix="ppo_ensemble",
        verbose=1
    )

    # Khởi tạo PPO agent với MlpPolicy
    rl_agent = PPO(
        'MlpPolicy',  # Chính xác cho đầu vào là vector đặc trưng
        vec_env,
        verbose=1,
        n_steps=len(rl_train_loader), # Số bước mỗi lần cập nhật = số lượng mẫu trong dataloader
        batch_size=64,
        n_epochs=10,
        device=device,
        tensorboard_log=os.path.join(rl_save_path, "tensorboard"),
    )

    # --- HUẤN LUYỆN AGENT ---
    total_timesteps = len(rl_train_loader) * 50 # Huấn luyện trong 50 epochs
    logging.info(f"Starting PPO training for {total_timesteps} timesteps.")
    rl_agent.learn(total_timesteps=total_timesteps, callback=checkpoint_callback)

    # Lưu agent cuối cùng
    rl_agent.save(os.path.join(rl_save_path, "ppo_ensemble_final.zip"))
    logging.info(f"RL agent training finished. Model saved to {rl_save_path}")


if __name__ == '__main__':
    # ... (giữ nguyên phần parser và logging từ main.py của bạn) ...
    parser = ArgumentParser(description='MediaFusion RL Ensemble Training', formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model_name', required=False, type=str, default='MediaFusion_RL', help='name of the RL ensemble model')
    args = parser.parse_args()

    numeric_level = getattr(logging, 'INFO', None)
    if not isinstance(numeric_level, int):
        raise ValueError('Invalid log level: %s' % 'INFO')
    
    # Dùng file config của mô hình gốc
    args.config_file = os.path.join('configs', 'MediaFusion.yaml')
    with open (args.config_file, 'r') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    cfg['model_name'] = args.model_name
    
    os.makedirs(os.path.join(cfg['path_experiments'], "RL_agents", args.model_name), exist_ok=True)
    log_path = os.path.join(cfg['path_experiments'], "RL_agents", args.model_name,
                            datetime.now().strftime('%Y-%m-%d_%H-%M-%S.log'))
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s",
        handlers=[logging.FileHandler(log_path), logging.StreamHandler()]
    )

    os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg['gpu'])
    start=time.time()
    main_rl(args, cfg)
    logging.info(f'Total Execution Time for RL training is {time.time()-start} seconds')