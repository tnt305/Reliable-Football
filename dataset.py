import os
import torch
import numpy as np
import random
import time
import math
import numpy as np
import torch
import logging
from tqdm import tqdm
import json
import pickle
from torch.utils.data import Dataset

from SoccerNet.utils import getListGames
from SoccerNet.Downloader import SoccerNetDownloader
from SoccerNet.Evaluation.utils import AverageMeter, EVENT_DICTIONARY_V2, INVERSE_EVENT_DICTIONARY_V2
from SoccerNet.Evaluation.utils import EVENT_DICTIONARY_V1, INVERSE_EVENT_DICTIONARY_V1


def feats2clip(feats, stride, clip_length, padding = "replicate_last", off=0):
    if padding =="zeropad":
        print("beforepadding", feats.shape)
        pad = feats.shape[0] - int(feats.shape[0]/stride)*stride
        print("pad need to be", clip_length-pad)
        m = torch.nn.ZeroPad2d((0, 0, clip_length-pad, 0))
        feats = m(feats)
        print("afterpadding", feats.shape)
        # nn.ZeroPad2d(2)

    idx = torch.arange(start=0, end=feats.shape[0]-1, step=stride)
    idxs = []
    for i in torch.arange(-off, clip_length-off):
        idxs.append(idx+i)
    idx = torch.stack(idxs, dim=1)

    if padding=="replicate_last":
        idx = idx.clamp(0, feats.shape[0]-1)
    # print(idx)
    return feats[idx,...]


class SoccerNetClips(Dataset):
    def __init__(self, 
                 path,
                 frames_features="baidu_soccer_embeddings.npy",
                 audio_features="AST.npy",
                 split=["train"], 
                 version=2,
                 framerate=2, 
                 window_size=25,
                 stride=25,
                 rC=2,
                 rD=3,
                 max_games=1000,
                 multimodal=False,
                 processed_data_path="/home/storage/thiendc/soccernet_dataloader",
                 store=False):
        
        self.path = f"{path}/{split[0]}"
        self.listGames = getListGames(split)
        self.frames_features = frames_features
        self.audio_features = audio_features
        self.framerate = framerate
        self.stride = stride
        self.window_size = window_size
        self.rC = rC
        self.rD = rD
        self.version = version
        self.multimodal = multimodal and (audio_features is not None)
        self.split = split[0]
        self.max_games = max_games
        self.groups = self.window_size // self.stride

        if self.version == 2:
            self.dict_event = EVENT_DICTIONARY_V2
            self.num_classes = 17
            self.labels_filename = "Labels-v2.json"
        else:
            raise NotImplementedError(f"Version {version} không hỗ trợ.")
            
        config_str = f"v{version}_w{window_size}_f{framerate}_s{stride}_{'mm' if self.multimodal else 'uni'}"
        self.processed_data_path = os.path.join(processed_data_path, self.split, config_str)
        self.path_list_file = os.path.join(self.processed_data_path, "clip_paths.pkl")
        
        print(f"DEBUG: Checking for file at: {self.path_list_file}")
        print(f"DEBUG: File exists? {os.path.exists(self.path_list_file)}")
        
        if store or not os.path.exists(self.path_list_file):
            logging.info(f"Bắt đầu tiền xử lý và lưu dữ liệu tại: '{self.processed_data_path}'")
            os.makedirs(self.processed_data_path, exist_ok=True)
            
            self.clip_paths = []
            self.clip_class_info = []
            
            # BIẾN NÀY KHÔNG CÓ TRONG CODE GỐC CỦA BẠN, TÔI THÊM VÀO ĐỂ LOGIC LƯU TRỮ HOẠT ĐỘNG
            # NÓ THAY THẾ CHO BIẾN `nclip` BỊ THIẾU
            global_clip_counter = 0

            n_games = 0
            for game in tqdm(self.listGames, desc=f"Preprocessing {self.split} set"):
                n_games += 1
                if n_games > self.max_games:
                    break
                game_path = os.path.join(self.path, game)
                # === XỬ LÝ HIỆP 1 ===
                try:

                    feat_half1_np_raw = np.load(os.path.join(game_path, "1_" + self.frames_features))
                    # Làm phẳng tất cả các chiều trừ chiều cuối cùng.
                    # Giả sử chiều cuối cùng là chiều feature và có kích thước là 2048.
                    feat_half1_np = feat_half1_np_raw.reshape(-1, feat_half1_np_raw.shape[-1])

                    # Thêm một bước kiểm tra để đảm bảo an toàn
                    # if feat_half1_np.shape[-1] != 2048:
                    #     logging.error(f"Lỗi shape feature cho game {game}, hiệp 1. Kích thước chiều cuối là {feat_half1_np.shape[-1]}, mong đợi 2048.")
                    #     continue # Bỏ qua game này nếu shape không đúng
                    audio_feat_half1_np = np.load(os.path.join(game_path, "1_" + self.audio_features)) if self.multimodal else None
                except FileNotFoundError as e:
                    logging.warning(f"Không tìm thấy file feature cho hiệp 1 game {game}: {e}. Bỏ qua game.")
                    continue
                
                # 1. Cắt clip VIDEO
                clips_half1 = feats2clip(torch.from_numpy(feat_half1_np), 
                                         stride=self.stride, clip_length=self.window_size)

                # 2. Cắt clip ÂM THANH
                if self.multimodal:
                    log_mel_full_1 = torch.from_numpy(audio_feat_half1_np)
                    audio_clips_half1 = feats2clip(log_mel_full_1, 
                                                   stride=self.stride * 100, 
                                                   clip_length=self.window_size * 100)
                    
                    # 3. ĐỒNG BỘ SỐ LƯỢNG CLIP (QUAN TRỌNG NHẤT)
                    num_clips_1 = min(len(clips_half1), len(audio_clips_half1))
                    clips_half1 = clips_half1[:num_clips_1]
                    audio_clips_half1 = audio_clips_half1[:num_clips_1]
                else:
                    num_clips_1 = len(clips_half1)
                    audio_clips_half1 = [None] * num_clips_1 # Tạo list placeholder

                # 4. Tạo nhãn cho hiệp 1
                labels_data = json.load(open(os.path.join(game_path, self.labels_filename)))
                label_time_dim = self.window_size * self.framerate
                labels_half1 = np.zeros((num_clips_1, label_time_dim, self.num_classes + 1), dtype=np.float32)
                labels_half1[:, :, 0] = 1
                label_half1_displ = np.full((num_clips_1, label_time_dim, self.num_classes + 1), 1000, dtype=np.float32)

                # === XỬ LÝ HIỆP 2 (TƯƠNG TỰ HIỆP 1) ===
                try:
                    feat_half2_np_raw = np.load(os.path.join(game_path, "2_" + self.frames_features))
                    feat_half2_np = feat_half2_np_raw.reshape(-1, feat_half2_np_raw.shape[-1])

                    # if feat_half2_np.shape[-1] != 2048:
                    #     logging.error(f"Lỗi shape feature cho game {game}, hiệp 2. Kích thước chiều cuối là {feat_half2_np.shape[-1]}, mong đợi 2048.")
                    #     continue
                    audio_feat_half2_np = np.load(os.path.join(game_path, "2_" + self.audio_features)) if self.multimodal else None
                except FileNotFoundError as e:
                    logging.warning(f"Không tìm thấy file feature cho hiệp 2 game {game}: {e}. Bỏ qua hiệp 2.")
                    # Xử lý tiếp với chỉ hiệp 1 nếu cần, hoặc bỏ qua cả game
                    feat_half2_np, audio_feat_half2_np = None, None

                if feat_half2_np is not None:
                    clips_half2 = feats2clip(torch.from_numpy(feat_half2_np), 
                                             stride=self.stride, clip_length=self.window_size)
                    if self.multimodal:
                        log_mel_full_2 = torch.from_numpy(audio_feat_half2_np)
                        audio_clips_half2 = feats2clip(log_mel_full_2, 
                                                       stride=self.stride * 100, 
                                                       clip_length=self.window_size * 100)
                        num_clips_2 = min(len(clips_half2), len(audio_clips_half2))
                        clips_half2 = clips_half2[:num_clips_2]
                        audio_clips_half2 = audio_clips_half2[:num_clips_2]
                    else:
                        num_clips_2 = len(clips_half2)
                        audio_clips_half2 = [None] * num_clips_2
                    
                    labels_half2 = np.zeros((num_clips_2, label_time_dim, self.num_classes + 1), dtype=np.float32)
                    labels_half2[:, :, 0] = 1
                    label_half2_displ = np.full((num_clips_2, label_time_dim, self.num_classes + 1), 1000, dtype=np.float32)
                else:
                    clips_half2, audio_clips_half2, labels_half2, label_half2_displ, num_clips_2 = [], [], [], [], 0

                # 5. Gán nhãn cho cả 2 hiệp
                for annotation in labels_data["annotations"]:
                    half = int(annotation["gameTime"][0])
                    event = annotation["label"]
                    if event not in self.dict_event: continue
                    label_idx = self.dict_event[event]
                    
                    position_sec = int(annotation["position"]) / 1000
                    start_clip_idx = max(0, math.floor((position_sec - self.window_size) / self.stride) + 1)
                    end_clip_idx = math.floor(position_sec / self.stride)

                    for l in range(start_clip_idx, end_clip_idx + 1):
                        if half == 1 and 0 <= l < num_clips_1:
                            target_labels, target_displ = labels_half1, label_half1_displ
                        elif half == 2 and 0 <= l < num_clips_2:
                            target_labels, target_displ = labels_half2, label_half2_displ
                        else:
                            continue

                        clip_start_sec = l * self.stride
                        position_in_clip_label_frame = int((position_sec - clip_start_sec) * self.framerate)
                        rC_label_frame = self.rC * self.framerate
                        rD_label_frame = self.rD * self.framerate
                        
                        min_rel_pos = max(0, position_in_clip_label_frame - rC_label_frame)
                        max_rel_pos = min(label_time_dim, position_in_clip_label_frame + rC_label_frame + 1)
                        target_labels[l, min_rel_pos:max_rel_pos, 0] = 0
                        target_labels[l, min_rel_pos:max_rel_pos, label_idx + 1] = 1

                        min_rel_pos_d = max(0, position_in_clip_label_frame - rD_label_frame)
                        max_rel_pos_d = min(label_time_dim, position_in_clip_label_frame + rD_label_frame + 1)
                        if max_rel_pos_d > min_rel_pos_d:
                            displ = position_in_clip_label_frame - np.arange(min_rel_pos_d, max_rel_pos_d)
                            target_displ[l, min_rel_pos_d:max_rel_pos_d, label_idx+1] = displ


                # 6. LƯU TRỮ (THAY THẾ HOÀN TOÀN KHỐI CODE LỖI)
                game_clips_info = [
                    (clips_half1, audio_clips_half1, labels_half1, label_half1_displ),
                    (clips_half2, audio_clips_half2, labels_half2, label_half2_displ)
                ]
                
                temp_clip_paths = []
                temp_clip_class_info = []
                
                for v_clips, a_clips, lbls, dspls in game_clips_info:
                    for i in range(len(v_clips)):
                        clip_dir = os.path.join(self.processed_data_path, f"clip_{global_clip_counter}")
                        os.makedirs(clip_dir, exist_ok=True)
                        base_path = os.path.join(clip_dir, "data")

                        np.save(f"{base_path}_video.npy", v_clips[i].numpy())
                        np.save(f"{base_path}_labels.npy", lbls[i])
                        np.save(f"{base_path}_displ.npy", dspls[i])
                        if self.multimodal and a_clips[i] is not None:
                            np.save(f"{base_path}_audio.npy", a_clips[i].numpy())
                        
                        temp_clip_paths.append(base_path)
                        
                        # THU THẬP THÔNG TIN LỚP
                        present_classes = np.where(np.any(lbls[i, :, 1:] == 1, axis=0))[0]
                        if len(present_classes) == 0:
                            # Nếu không có event nào, gán nó vào lớp background (index num_classes)
                            present_classes = np.array([self.num_classes])
                        
                        temp_clip_class_info.append(present_classes.tolist())
                        global_clip_counter += 1
                        
                # 7. SẮP XẾP THEO GROUP (giữ nguyên logic)
                num_clips_game = len(temp_clip_paths)
                num_clips_per_group = math.ceil(num_clips_game / self.groups)
                for i in range(num_clips_per_group):
                    for g in range(self.groups):
                        idx = i * self.groups + g
                        if idx < num_clips_game:
                            self.clip_paths.append(temp_clip_paths[idx])
                            self.clip_class_info.append(temp_clip_class_info[idx])
                
                # 7. SẮP XẾP THEO GROUP (QUAN TRỌNG)
                num_clips_game = len(temp_clip_paths)
                num_clips_per_group = math.ceil(num_clips_game / self.groups)
                for i in range(num_clips_per_group):
                    for g in range(self.groups):
                        idx = i * self.groups + g
                        if idx < num_clips_game:
                            self.clip_paths.append(temp_clip_paths[idx])

            # Lưu danh sách đường dẫn
            with open(self.path_list_file, 'wb') as f:
                pickle.dump({"paths": self.clip_paths, "class_info": self.clip_class_info}, f)
            logging.info(f"Đã tiền xử lý và lưu thành công {len(self.clip_paths)} clip.")
            

        else: # Tải lại nếu đã tồn tại
            with open(self.path_list_file, 'rb') as f:
                data_stored = pickle.load(f)
                self.clip_paths = data_stored["paths"]
                self.clip_class_info = data_stored["class_info"]
            logging.info(f"Đã tải lại {len(self.clip_paths)} clip từ {self.path_list_file}.")

    def _get_index(self, index):
        if self.split == 'train':
            group = np.random.randint(0, self.groups)
        else:
            group = 0
        actual_index = index * self.groups + group
        # if actual_index >= len(self.clip_paths):
        #     actual_index = len(self.clip_paths) - 1
        actual_index = min(actual_index, len(self.clip_paths) - 1)
        return actual_index

    def __getitem__(self, index):
        actual_index = self._get_index(index)
        clip_base_path = self.clip_paths[actual_index]
        data = {}
        try:
            data['video'] = torch.from_numpy(np.load(f"{clip_base_path}_video.npy"))
            data['labels'] = torch.from_numpy(np.load(f"{clip_base_path}_labels.npy"))
            data['labels_displ'] = torch.from_numpy(np.load(f"{clip_base_path}_displ.npy"))
            if self.multimodal:
                data['audio'] = torch.from_numpy(np.load(f"{clip_base_path}_audio.npy"))
        except FileNotFoundError as e:
            logging.error(f"Lỗi tải file cho clip tại {clip_base_path}: {e}")
        return data

    def __len__(self):
        return len(self.clip_paths) // self.groups


class SoccerNetClipsTesting(Dataset):
    def __init__(self, path, 
                frames_features="baidu_soccer_embeddings.npy", 
                audio_features="AST.npy",
                split=["test"], 
                version=2, 
                framerate=2, 
                window_size=25,
                multimodal=False, 
                processed_data_path="/home/storage/thiendc/soccernet_dataloader", 
                ):
        self.path = f"{path}/{split[0]}"
        self.listGames = getListGames(split)
        self.frames_features = frames_features
        self.audio_features = audio_features
        self.window_size_frame = window_size*framerate
        self.framerate = framerate
        self.version = version
        self.multimodal = multimodal and (audio_features is not None)
        self.processed_data_path = processed_data_path
        self.stride = 1/framerate
        
        self.split=split
        if version == 2:
            self.dict_event = EVENT_DICTIONARY_V2
            self.num_classes = 17
            self.labels="Labels-v2.json"

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            feat_half1 (np.array): features for the 1st half.
            feat_half2 (np.array): features for the 2nd half.
            label_half1 (np.array): labels (one-hot) for the 1st half.
            label_half2 (np.array): labels (one-hot) for the 2nd half.
        """
        data = dict()
        # Load features
        feat_half1 = np.load(os.path.join(self.path, self.listGames[index], "1_" + self.frames_features))
        feat_half2 = np.load(os.path.join(self.path, self.listGames[index], "2_" + self.frames_features))

        data['video_half1'] = feat_half1
        data['video_half2'] = feat_half2

        if self.multimodal:
            feat_half1_audio = np.load(os.path.join(self.path, self.listGames[index], "1_" + self.audio_features))
            feat_half2_audio = np.load(os.path.join(self.path, self.listGames[index], "2_" + self.audio_features))

            data['audio_half1'] = feat_half1_audio
            data['audio_half2'] = feat_half2_audio

        return self.listGames[index], data

    def __len__(self):
        return len(self.listGames)

# decoupled_samplers

def create_class_balanced_sampler(class_info, num_classes):
    """
    Tạo một WeightedRandomSampler để cân bằng các lớp.

    Args:
        class_info (list[list[int]]): Danh sách chứa các lớp có trong mỗi clip.
        num_classes (int): Tổng số lớp (không bao gồm background).
    """
    # Đếm số lần xuất hiện của mỗi lớp (bao gồm cả background)
    class_counts = np.zeros(num_classes + 1)
    for classes_in_clip in class_info:
        for c in classes_in_clip:
            class_counts[c] += 1
    
    # Tránh chia cho 0 nếu một lớp nào đó không có mẫu nào
    class_counts[class_counts == 0] = 1

    # Trọng số của mỗi lớp là 1 / số lần xuất hiện
    class_weights = 1. / torch.tensor(class_counts, dtype=torch.float)
    
    # Tính trọng số cho mỗi MẪU (clip)
    # Trọng số của một clip sẽ là trọng số của lớp hiếm nhất có trong nó.
    sample_weights = []
    for classes_in_clip in class_info:
        # Nếu clip rỗng (lỗi), gán trọng số tối thiểu
        if not classes_in_clip:
            sample_weights.append(torch.min(class_weights).item())
            continue
        weights_of_classes_in_clip = class_weights[classes_in_clip]
        # Lấy trọng số lớn nhất (tương ứng với lớp hiếm nhất)
        sample_weights.append(torch.max(weights_of_classes_in_clip).item())

    sampler = torch.utils.data.WeightedRandomSampler(
        weights=torch.DoubleTensor(sample_weights),
        num_samples=len(sample_weights),
        replacement=True # Lấy mẫu có thay thế là cần thiết cho việc cân bằng
    )
    return sampler