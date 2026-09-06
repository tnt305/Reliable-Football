import os
import sys
import time
import pickle
import logging
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from typing import List, Dict, Any

import yaml
import numpy as np
import torch
from torch.amp import autocast
from torch.utils.data import DataLoader
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from tqdm import tqdm

# Đảm bảo đường dẫn import từ thư mục gốc
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from mediafusion_dataset import SoccerNetFrames
from mediafusion_model import MediaFusion
from rl.rl_env import SoccerNetRLDataset, TriTierBalancedSampler, SoccerNetRLEnsembleEnv, DEFAULT_THRESHOLDS


class PPOProgressBarCallback(BaseCallback):
    """Thanh tiến trình tqdm hiển thị mượt mà tiến độ huấn luyện PPO và Mean Reward."""
    def __init__(self, total_timesteps: int, verbose: int = 0):
        super().__init__(verbose)
        self.total_timesteps = total_timesteps
        self.pbar = None

    def _on_training_start(self) -> None:
        self.pbar = tqdm(total=self.total_timesteps, desc="🤖 Huấn luyện PPO Agent", unit="step", dynamic_ncols=True)

    def _on_step(self) -> bool:
        self.pbar.update(1)
        if len(self.model.ep_info_buffer) > 0:
            rewards = [info['r'] for info in self.model.ep_info_buffer if 'r' in info]
            if rewards:
                self.pbar.set_postfix({'Mean Rew': f"{np.mean(rewards):+.3f}"})
        return True

    def _on_training_end(self) -> None:
        if self.pbar is not None:
            self.pbar.close()


def precompute_and_cache_predictions(
    models: List[torch.nn.Module],
    dataloader: DataLoader,
    cache_path: str,
    device: torch.device
) -> List[Dict[str, Any]]:
    """
    Chạy suy luận 1 lần duy nhất trên tập validation và cache dự đoán của tất cả models.
    """
    logging.info(f"Bắt đầu trích xuất và cache dự đoán cho {len(models)} models...")
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)

    for m in models:
        m.eval()
        m.to(device)

    cached_samples: List[Dict[str, Any]] = []
    start_time = time.time()

    with torch.no_grad():
        for data in tqdm(dataloader, desc="⚡ Trích xuất dự đoán tập Valid", dynamic_ncols=True):
            # Shape data: featB (B, chunk_size, 9344), featA (B, chunk_size*100, 128)
            featB = data['featB'].to(device)
            featA = data['featA'].to(device) if 'featA' in data else None
            labels = data['labels'].numpy()  # (B, T, 18)

            b_size = featB.shape[0]
            batch_preds_C = []
            batch_preds_D = []

            for m in models:
                with autocast('cuda', dtype=torch.float16):
                    output = m(featsB=featB, featsA=featA, inference=True)
                
                predC = output['preds'].float().cpu().numpy()  # (B, T, 18)
                predD = output['predsD'].float().cpu().numpy()  # (B, T, 18, 2) hoặc (B, T, 18)
                
                batch_preds_C.append(predC)
                batch_preds_D.append(predD)

            for b in range(b_size):
                sample_dict = {
                    'labels': labels[b].astype(np.float32),
                    'preds_C': [batch_preds_C[m_idx][b].astype(np.float16) for m_idx in range(len(models))],
                    'preds_D': [batch_preds_D[m_idx][b].astype(np.float16) for m_idx in range(len(models))]
                }
                cached_samples.append(sample_dict)

    logging.info(f"Lưu file cache dự đoán vào: {cache_path}")
    with open(cache_path, 'wb') as f:
        pickle.dump(cached_samples, f)

    return cached_samples


def main() -> None:
    parser = ArgumentParser(description="Huấn luyện RL Ensemble cho SoccerNet", formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("--config", type=str, default="configs/MediaFusion.yaml", help="Đường dẫn file config gốc")
    parser.add_argument("--models", nargs="+", type=str, default=None, help="Danh sách checkpoint của các base models")
    parser.add_argument("--cache_path", type=str, default="models/RL_agents/valid_preds_cache.pkl", help="Đường dẫn file cache")
    parser.add_argument("--force_cache", action="store_true", help="Ghi đè lại file cache nếu đã có")
    parser.add_argument("--timesteps", type=int, default=30000, help="Tổng số bước huấn luyện PPO")
    parser.add_argument("--output_dir", type=str, default="models/RL_agents/MediaFusion_RL", help="Thư mục lưu agent")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate cho PPO")
    parser.add_argument("--seed", type=int, default=None, help="Random seed cho huấn luyện (mặc định lấy từ config: 1)")
    args = parser.parse_args()

    cfg_path = args.config
    with open(cfg_path, 'r') as f:
        cfg = yaml.safe_load(f)

    seed = args.seed if args.seed is not None else cfg.get('seed', 1)

    # Set random seeds
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)]
    )

    device = torch.device("cuda" if torch.cuda.is_available() and cfg.get('gpu', 0) >= 0 else "cpu")
    logging.info(f"Thiết bị sử dụng: {device} | Random Seed: {seed}")

    # 1. Xác định danh sách checkpoint models
    model_paths = args.models
    if not model_paths:
        default_ckpts = [
            os.path.join(cfg['path_experiments'], "ASmodels", "MediaFusion", "model.pth.tar"),
            os.path.join(cfg['path_experiments'], "ASmodels", "MediaFusion", "model_34.pth.tar"),
            os.path.join(cfg['path_experiments'], "ASmodels", "MediaFusion", "model_41.pth.tar")
        ]
        model_paths = [p for p in default_ckpts if os.path.exists(p)]
        if not model_paths:
            logging.error("Không tìm thấy checkpoint mặc định nào trong models/ASmodels! Vui lòng chỉ định qua --models.")
            sys.exit(1)

    logging.info(f"Các models tham gia Ensemble ({len(model_paths)}):")
    for p in model_paths:
        logging.info(f"  - {p}")

    # 2. Kiểm tra hoặc tạo Cache dự đoán
    if os.path.exists(args.cache_path) and not args.force_cache:
        logging.info(f"Tìm thấy cache đã lưu tại {args.cache_path}. Tiến hành nạp trực tiếp...")
        with open(args.cache_path, 'rb') as f:
            cached_samples = pickle.load(f)
    else:
        # Load base models
        models = []
        for path in model_paths:
            m = MediaFusion(
                chunk_size=cfg['chunk_size'],
                n_output=int(cfg['outputrate'] * cfg['chunk_size']),
                baidu=cfg['baidu'],
                audio=cfg['audio'],
                model_cfg=cfg['model']
            )
            checkpoint = torch.load(path, map_location=device, weights_only=False)
            state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
            m.load_state_dict(state_dict)
            models.append(m)

        # Dataloader tập Valid
        dataset_valid = SoccerNetFrames(
            path_labels=cfg['path_labels'],
            path_store=cfg['path_store'],
            path_baidu=cfg['path_baidu'],
            path_audio=cfg['path_audio'],
            split=cfg['val_split'],
            chunk_size=cfg['chunk_size'],
            outputrate=cfg['outputrate'],
            stride=cfg['chunk_size'] // 2,
            rC=cfg['rC'],
            rD=cfg['rD'],
            store=cfg['store'],
            max_games=cfg['max_games']
        )
        val_loader = DataLoader(dataset_valid, batch_size=16, shuffle=False, num_workers=4, pin_memory=True)
        cached_samples = precompute_and_cache_predictions(models, val_loader, args.cache_path, device)

    # 3. Khởi tạo RL Dataset & Environment
    rl_dataset = SoccerNetRLDataset(cached_samples, thresholds=DEFAULT_THRESHOLDS)
    sampler = TriTierBalancedSampler(rl_dataset, total_samples=args.timesteps)
    env = SoccerNetRLEnsembleEnv(
        dataset=rl_dataset,
        sampler=sampler,
        num_models=len(model_paths),
        thresholds=DEFAULT_THRESHOLDS
    )
    vec_env = DummyVecEnv([lambda: env])

    # 4. Cấu hình PPO Agent (Contextual Bandit với gamma = 0.0)
    os.makedirs(args.output_dir, exist_ok=True)
    tb_log = os.path.join(args.output_dir, "tb_logs")
    
    agent = PPO(
        policy="MlpPolicy",
        env=vec_env,
        learning_rate=args.lr,
        n_steps=512,
        batch_size=64,
        n_epochs=5,
        gamma=0.0,            # Contextual Bandit: mỗi clip độc lập
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        verbose=0,
        tensorboard_log=tb_log,
        seed=seed,
        device="cpu"          # Chạy PPO trên CPU cực nhanh vì chỉ tính MLP nhỏ
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=5000,
        save_path=args.output_dir,
        name_prefix="ppo_checkpoint",
        verbose=0
    )
    pbar_callback = PPOProgressBarCallback(total_timesteps=args.timesteps)

    # 5. Huấn luyện Agent
    logging.info(f"Bắt đầu huấn luyện PPO Agent trong {args.timesteps} timesteps (Tối ưu chuẩn Tight Metric)...")
    start_train = time.time()
    agent.learn(total_timesteps=args.timesteps, callback=[checkpoint_callback, pbar_callback])
    logging.info(f"Huấn luyện hoàn tất trong {time.time() - start_train:.2f} giây!")

    # 6. Lưu mô hình cuối cùng
    final_model_path = os.path.join(args.output_dir, "best_ppo_agent.zip")
    agent.save(final_model_path)
    logging.info(f"✅ Đã lưu RL Agent tại: {final_model_path}")


if __name__ == "__main__":
    main()
