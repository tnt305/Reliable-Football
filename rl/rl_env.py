import logging
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import torch
from torch.utils.data import Dataset, Sampler
import gymnasium as gym
from gymnasium import spaces

# Trọng số nghịch đảo tần suất cho 17 class (từ tuning_plan.md)
CLASS_IMPORTANCE_WEIGHTS = np.array([
    5.0,  # 0: penalty (cực hiếm)
    1.2,  # 1: kick-off
    2.5,  # 2: goal (quan trọng)
    1.0,  # 3: substitution
    1.5,  # 4: offside
    1.0,  # 5: sh. on targ.
    1.0,  # 6: sh. off targ.
    0.8,  # 7: clearance
    0.5,  # 8: ball oop (rất nhiều)
    0.5,  # 9: throw in (rất nhiều)
    0.7,  # 10: foul
    1.0,  # 11: ind. fk
    1.2,  # 12: dir. fk
    1.0,  # 13: corner
    1.5,  # 14: yc
    5.0,  # 15: rc (cực hiếm)
    5.0,  # 16: 2nd yc (cực hiếm)
], dtype=np.float32)

DEFAULT_THRESHOLDS = np.array([
    0.005, 0.010, 0.020, 0.010, 0.010, 0.030, 0.030, 0.020,
    0.050, 0.050, 0.050, 0.010, 0.010, 0.020, 0.020, 0.001, 0.001
], dtype=np.float32)


class SoccerNetRLDataset(Dataset):
    """
    Dataset phục vụ huấn luyện RL Ensemble từ dữ liệu dự đoán đã được cache trước.
    """
    def __init__(self, samples: List[Dict[str, Any]], thresholds: Optional[np.ndarray] = None) -> None:
        super().__init__()
        self.samples = samples
        self.thresholds = thresholds if thresholds is not None else DEFAULT_THRESHOLDS

        self.event_indices: List[int] = []
        self.hard_neg_indices: List[int] = []
        self.clean_bg_indices: List[int] = []

        self._classify_samples()

    def _classify_samples(self) -> None:
        """Phân loại toàn bộ các samples thành 3 nhóm: Event, Hard Negative, Clean Background."""
        for idx, sample in enumerate(self.samples):
            labels = sample['labels']  # (T, 18)
            has_event = bool(labels[:, 1:].sum() > 0)

            if has_event:
                self.event_indices.append(idx)
            else:
                # Kiểm tra xem có model nào dự đoán vượt ngưỡng ở background không
                preds = sample['preds_C']  # List[np.ndarray] kích thước M x (T, 18)
                is_hard = False
                for p in preds:
                    max_prob = p[:, 1:].max(axis=0)  # (17,)
                    if np.any(max_prob > self.thresholds):
                        is_hard = True
                        break

                if is_hard:
                    self.hard_neg_indices.append(idx)
                else:
                    self.clean_bg_indices.append(idx)

        logging.info(
            f"SoccerNetRLDataset phân loại: {len(self.event_indices)} Events ({len(self.event_indices)/len(self.samples)*100:.1f}%), "
            f"{len(self.hard_neg_indices)} Hard Negatives ({len(self.hard_neg_indices)/len(self.samples)*100:.1f}%), "
            f"{len(self.clean_bg_indices)} Clean BGs ({len(self.clean_bg_indices)/len(self.samples)*100:.1f}%)."
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.samples[idx]


class TriTierBalancedSampler(Sampler):
    """
    Sampler cân bằng 3 tầng: 40% Event, 40% Hard Negative, 20% Clean Background.
    """
    def __init__(
        self,
        dataset: SoccerNetRLDataset,
        total_samples: int = 20000,
        p_event: float = 0.40,
        p_hard: float = 0.40,
        p_clean: float = 0.20
    ) -> None:
        super().__init__(data_source=dataset)
        self.dataset = dataset
        self.total_samples = total_samples
        self.p_event = p_event
        self.p_hard = p_hard
        self.p_clean = p_clean

    def __iter__(self):
        n_event = int(self.total_samples * self.p_event)
        n_hard = int(self.total_samples * self.p_hard)
        n_clean = self.total_samples - n_event - n_hard

        sampled_events = np.random.choice(self.dataset.event_indices, size=n_event, replace=True)
        
        # Nếu không có hard negative thì fallback sang event hoặc clean
        if len(self.dataset.hard_neg_indices) > 0:
            sampled_hard = np.random.choice(self.dataset.hard_neg_indices, size=n_hard, replace=True)
        else:
            sampled_hard = np.random.choice(self.dataset.event_indices, size=n_hard, replace=True)

        if len(self.dataset.clean_bg_indices) > 0:
            sampled_clean = np.random.choice(self.dataset.clean_bg_indices, size=n_clean, replace=True)
        else:
            sampled_clean = np.random.choice(self.dataset.event_indices, size=n_clean, replace=True)

        all_indices = np.concatenate([sampled_events, sampled_hard, sampled_clean])
        np.random.shuffle(all_indices)
        return iter(all_indices.tolist())

    def __len__(self) -> int:
        return self.total_samples


class SoccerNetRLEnsembleEnv(gym.Env):
    """
    Môi trường Gymnasium để huấn luyện RL Agent tìm trọng số Ensemble tối ưu theo từng clip.
    """
    metadata = {'render_modes': []}

    def __init__(
        self,
        dataset: SoccerNetRLDataset,
        sampler: Optional[TriTierBalancedSampler] = None,
        num_models: int = 2,
        thresholds: Optional[np.ndarray] = None
    ) -> None:
        super().__init__()
        self.dataset = dataset
        self.sampler = sampler if sampler is not None else TriTierBalancedSampler(dataset)
        self.sampler_iter = iter(self.sampler)
        self.num_models = num_models
        self.thresholds = thresholds if thresholds is not None else DEFAULT_THRESHOLDS

        # Không gian quan sát (Observation Space):
        # 1. Max probability của 17 class từ M models: M * 17
        # 2. Mean uncertainty của M models: M * 1
        # 3. Model disagreement (phương sai) cho 17 class: 17
        # Tổng số chiều = 17*M + M + 17 = 18*M + 17
        self.obs_dim = 18 * self.num_models + 17
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(self.obs_dim,), dtype=np.float32
        )

        # Không gian hành động (Action Space): vector liên tục M chiều
        self.action_space = spaces.Box(
            low=-2.0, high=2.0, shape=(self.num_models,), dtype=np.float32
        )

        self.current_sample: Optional[Dict[str, Any]] = None

    def _extract_obs(self, sample: Dict[str, Any]) -> np.ndarray:
        """Trích xuất vector quan sát đặc trưng cho clip."""
        preds_C = sample['preds_C']  # List M mảng (T, 18)
        preds_D = sample['preds_D']  # List M mảng (T, 18, 2) hoặc (T, 18)

        max_probs = []
        uncertainties = []

        for m in range(self.num_models):
            p = preds_C[m][:, 1:]  # (T, 17) bỏ background
            max_p = p.max(axis=0)   # (17,)
            max_probs.append(max_p)

            # Tính mean uncertainty từ kênh log-variance nếu có
            if preds_D[m].ndim == 3 and preds_D[m].shape[-1] >= 2:
                logvar = preds_D[m][:, 1:, 1]
                unc = float(np.mean(np.exp(np.clip(logvar, -5.0, 5.0))))
            else:
                unc = 0.5
            uncertainties.append(unc)

        # Tính disagreement (phương sai dự đoán giữa các models trên 17 classes)
        stack_probs = np.stack(max_probs, axis=0)  # (M, 17)
        disagreement = np.var(stack_probs, axis=0)  # (17,)

        obs = np.concatenate([
            stack_probs.flatten(),          # M * 17
            np.array(uncertainties, dtype=np.float32),  # M
            disagreement                    # 17
        ], axis=0).astype(np.float32)

        return np.clip(obs, 0.0, 1.0)

    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        try:
            sample_idx = next(self.sampler_iter)
        except StopIteration:
            self.sampler_iter = iter(self.sampler)
            sample_idx = next(self.sampler_iter)

        self.current_sample = self.dataset[sample_idx]
        obs = self._extract_obs(self.current_sample)
        return obs, {}

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        # 1. Softmax biến đổi action thành trọng số chuẩn tắc
        exp_a = np.exp(action - np.max(action))
        weights = exp_a / np.sum(exp_a)  # (M,)

        sample = self.current_sample
        labels = sample['labels']  # (T, 18)
        preds_C = sample['preds_C']  # List M mảng (T, 18)
        preds_D = sample['preds_D']  # List M mảng (T, 18, 2)

        # 2. Tính toán dự đoán Ensemble
        ens_pred_C = np.zeros_like(preds_C[0])
        ens_pred_D = np.zeros_like(preds_D[0])

        for m in range(self.num_models):
            ens_pred_C += weights[m] * preds_C[m]
            ens_pred_D += weights[m] * preds_D[m]

        # 3. Tính Reward Relative Spotting Margin
        has_event = bool(labels[:, 1:].sum() > 0)
        reward = 0.0

        if has_event:
            # Trích xuất danh sách các sự kiện thực tế duy nhất (t_center, c_idx)
            # Do nhãn mediafusion_dataset gán 1 cho toàn bộ dải [-rC, +rC], gom các frame liên tiếp thành 1 tâm duy nhất
            gt_events: List[Tuple[int, int]] = []
            for c in range(17):
                pos = np.where(labels[:, c + 1] > 0)[0]
                if len(pos) == 0:
                    continue
                diffs = np.diff(pos)
                splits = np.where(diffs > 1)[0] + 1
                for cluster in np.split(pos, splits):
                    if len(cluster) > 0:
                        gt_events.append((int(np.mean(cluster)), c))

            event_reward_sum = 0.0
            T = len(labels)
            frames_arr = np.arange(T)

            # Trích xuất displacement (kênh 0 nếu mô hình có ước lượng uncertainty)
            displ_ens = ens_pred_D[:, :, 0] if ens_pred_D.ndim == 3 else ens_pred_D

            for t_gt, c_idx in gt_events:
                class_w = CLASS_IMPORTANCE_WEIGHTS[c_idx]

                # Định vị vị trí hành động sau khi áp dụng hồi quy displacement: pred_pos = f - displ
                pred_pos_ens = np.round(frames_arr - displ_ens[:, c_idx + 1]).astype(np.int64)
                dist_ens = np.abs(pred_pos_ens - t_gt)
                prob_ens = ens_pred_C[:, c_idx + 1]

                # Tiêu chuẩn Tight: Dung sai <= 4 frames (2.0s tại 2 fps)
                tight_mask_ens = (dist_ens <= 4)
                peak_ens = float(np.max(prob_ens[tight_mask_ens])) if np.any(tight_mask_ens) else 0.0

                # Bonus Ultra-Tight: Sai số cực nhỏ <= 2 frames (1.0s tại 2 fps)
                ultra_mask_ens = (dist_ens <= 2)
                if np.any(ultra_mask_ens):
                    peak_ens += 0.2 * float(np.max(prob_ens[ultra_mask_ens]))

                # Tính điểm Tight tương ứng của từng Base Model
                best_single_peak = 0.0
                for m in range(self.num_models):
                    displ_m = preds_D[m][:, :, 0] if preds_D[m].ndim == 3 else preds_D[m]
                    pred_pos_m = np.round(frames_arr - displ_m[:, c_idx + 1]).astype(np.int64)
                    dist_m = np.abs(pred_pos_m - t_gt)
                    prob_m = preds_C[m][:, c_idx + 1]

                    tight_mask_m = (dist_m <= 4)
                    p_m = float(np.max(prob_m[tight_mask_m])) if np.any(tight_mask_m) else 0.0
                    ultra_mask_m = (dist_m <= 2)
                    if np.any(ultra_mask_m):
                        p_m += 0.2 * float(np.max(prob_m[ultra_mask_m]))
                    if p_m > best_single_peak:
                        best_single_peak = p_m

                # Thưởng độ cải thiện vượt trội trong vùng Tight so với model đơn lẻ tốt nhất
                diff = peak_ens - best_single_peak
                event_reward_sum += class_w * diff

            # Phạt nhẹ False Positive trên các frame còn lại của clip
            bg_mask = (labels[:, 1:].sum(axis=1) == 0)
            fp_penalty = 0.0
            if np.any(bg_mask):
                excess = np.maximum(0.0, ens_pred_C[bg_mask, 1:] - self.thresholds)
                fp_penalty = float(np.sum(excess ** 2))

            reward = event_reward_sum - 0.5 * fp_penalty

        else:
            # --- TRƯỜNG HỢP B: CLIP THUẦN BACKGROUND ---
            # Tính mức phạt False Positive của Ensemble vs các Single Models
            excess_ens = np.maximum(0.0, ens_pred_C[:, 1:] - self.thresholds)
            penalty_ens = float(np.sum(excess_ens ** 2))

            penalties_single = []
            for m in range(self.num_models):
                excess_m = np.maximum(0.0, preds_C[m][:, 1:] - self.thresholds)
                penalties_single.append(float(np.sum(excess_m ** 2)))

            worst_penalty = max(penalties_single)
            best_penalty = min(penalties_single)

            if worst_penalty > 0:
                # Nếu có model bị False Positive và Ensemble dập tắt được nó -> Thưởng
                reward = (worst_penalty - penalty_ens) * 0.5
            else:
                # Cả 3 model đều sạch
                reward = 0.0

        # Chuẩn hóa giới hạn reward trong khoảng an toàn [-2.0, 2.0]
        reward = float(np.clip(reward, -2.0, 2.0))

        # Contextual Bandit: mỗi clip là 1 quyết định độc lập
        terminated = True
        truncated = False
        info = {'weights': weights, 'has_event': has_event}

        # Trả về dummy observation khi kết thúc
        next_obs = np.zeros(self.obs_dim, dtype=np.float32)
        return next_obs, reward, terminated, truncated, info
