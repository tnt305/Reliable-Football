import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import cv2
from tqdm import tqdm
import torch
import torch.nn.functional as F
import numpy as np
from torchvision.ops import box_convert, roi_align
from PIL import Image
import ffmpeg
import torchvision.transforms as transforms
import warnings
warnings.filterwarnings("ignore")

from groundingdino.util.inference import load_model, load_image, predict, annotate
from groundingdino.util.misc import nested_tensor_from_tensor_list

class GroundingDINOFeatureExtractor:
    def __init__(self, model_config_path, model_checkpoint_path, device="cuda",):
        self.device = device
        self.model = load_model(model_config_path, model_checkpoint_path, device)
        self.model.eval()
        self.roi_output_size = (7, 7)
        self.final_feature_dim = 1024

    def filter_boxes(self, boxes, logits, phrases, image_shape, text_prompt="",
                    max_area_ratio=0.8, max_dimension_ratio=0.8, max_boxes=None):
        """
        Lọc bounding boxes theo các điều kiện:
        - Diện tích < 80% image area
        - Width/Height < 80% image dimensions
        - Ưu tiên boxes có trong text prompt trước, sau đó mới đến boxes khác
        - Chỉ lấy top 10 boxes
        """
        original_count = len(boxes)
        if len(boxes) == 0:
            return boxes, logits, phrases, original_count
        
        h, w = image_shape[:2]
        image_area = h * w
        
        # Parse text prompt để lấy keywords
        prompt_keywords = []
        if text_prompt:
            # Split bởi dấu chấm và clean up
            keywords = [keyword.strip().lower() for keyword in text_prompt.split('.') if keyword.strip()]
            prompt_keywords = keywords
        
        # Convert boxes từ cxcywh sang xyxy để tính diện tích
        boxes_scaled = boxes * torch.tensor([w, h, w, h], device=boxes.device)
        boxes_xyxy = box_convert(boxes_scaled, in_fmt="cxcywh", out_fmt="xyxy")
        
        valid_indices = []
        for i, box in enumerate(boxes_xyxy):
            x1, y1, x2, y2 = box.cpu().numpy()
            
            # Tính diện tích và kích thước
            box_width = x2 - x1
            box_height = y2 - y1
            box_area = box_width * box_height
            
            # Kiểm tra các điều kiện
            area_ratio = box_area / image_area
            width_ratio = box_width / w
            height_ratio = box_height / h
            
            if (area_ratio < max_area_ratio and 
                width_ratio < max_dimension_ratio and 
                height_ratio < max_dimension_ratio):
                valid_indices.append(i)
        
        if len(valid_indices) == 0:
            return torch.empty(0, 4, device=boxes.device), torch.empty(0, device=boxes.device), [], original_count
        
        # Lọc theo indices hợp lệ
        filtered_boxes = boxes[valid_indices]
        filtered_logits = logits[valid_indices]
        filtered_phrases = [phrases[i] for i in valid_indices]
        
        # Ưu tiên sorting: prompt keywords trước, sau đó confidence
        if len(filtered_logits) > max_boxes and prompt_keywords:
            priority_indices = []
            other_indices = []
            
            for i, phrase in enumerate(filtered_phrases):
                phrase_lower = phrase.lower()
                # Check if phrase contains any prompt keyword
                is_priority = any(keyword in phrase_lower or phrase_lower in keyword 
                                for keyword in prompt_keywords)
                
                if is_priority:
                    priority_indices.append(i)
                else:
                    other_indices.append(i)
            
            # Sort priority group by confidence
            if priority_indices:
                priority_logits = filtered_logits[priority_indices]
                _, priority_sorted_indices = torch.sort(priority_logits, descending=True)
                priority_indices = [priority_indices[i] for i in priority_sorted_indices.cpu().numpy()]
            
            # Sort other group by confidence
            if other_indices:
                other_logits = filtered_logits[other_indices]
                _, other_sorted_indices = torch.sort(other_logits, descending=True)
                other_indices = [other_indices[i] for i in other_sorted_indices.cpu().numpy()]
            
            # Combine: priority first, then others, up to max_boxes
            final_indices = priority_indices + other_indices
            final_indices = final_indices[:max_boxes]
            
            # Apply final selection
            filtered_boxes = filtered_boxes[final_indices]
            filtered_logits = filtered_logits[final_indices]
            filtered_phrases = [filtered_phrases[i] for i in final_indices]
            
        elif len(filtered_logits) > max_boxes:
            # Fallback: sort by confidence only
            _, sorted_indices = torch.sort(filtered_logits, descending=True)
            top_indices = sorted_indices[:max_boxes]
            
            filtered_boxes = filtered_boxes[top_indices]
            filtered_logits = filtered_logits[top_indices]
            filtered_phrases = [filtered_phrases[i] for i in top_indices.cpu().numpy()]
        
        return filtered_boxes, filtered_logits, filtered_phrases, original_count

    def extract_backbone_features(self, image_tensor_no_batch):
        """
        Trích xuất feature maps từ backbone.
        """
        with torch.no_grad():
            nested_tensor = nested_tensor_from_tensor_list([image_tensor_no_batch])
            feature_maps_tuple = self.model.backbone(nested_tensor)
            return feature_maps_tuple

    def roi_align_features(self, feature_maps_tuple, boxes, image_shape):
        """
        Áp dụng ROI Align. Xử lý trường hợp đầu ra là list.
        """
        # Lấy phần tử cuối cùng từ tuple đầu ra của backbone
        last_element = feature_maps_tuple[-1]

        # KIỂM TRA: Nếu phần tử đó là list, lấy tensor đầu tiên trong list
        # Đây là bước quan trọng để xử lý cấu trúc dữ liệu không nhất quán
        if isinstance(last_element, list):
            feature_maps = last_element[0]
        else:
            feature_maps = last_element

        if len(boxes) == 0:
            C = feature_maps.shape[1]
            return torch.zeros(0, C, *self.roi_output_size, device=self.device)
        
        original_h, original_w = image_shape
        feature_h, feature_w = feature_maps.shape[2], feature_maps.shape[3]
        
        spatial_scale = feature_h / original_h
        
        # Prepare boxes for ROI Align: add batch indices and ensure CUDA
        batch_indices = torch.zeros(len(boxes), 1, device=self.device, dtype=boxes.dtype)
        roi_boxes = torch.cat([batch_indices, boxes.to(self.device)], dim=1)  # (N, 5)
        
        roi_features = roi_align(
            feature_maps,
            roi_boxes,
            output_size=self.roi_output_size,
            spatial_scale=spatial_scale,
            sampling_ratio=-1
        )
        
        return roi_features


class VideoFeatureExtractor:
    def __init__(self, model_config_path, model_checkpoint_path, device="cuda", max_entities=10):
        """
        Video feature extractor using GroundingDINO + ROI Align
        """
        self.device = device
        self.image_extractor = GroundingDINOFeatureExtractor(
            model_config_path, model_checkpoint_path, device
        )
        self.feature_dim = 1024
        self.max_entities = max_entities
        
    def extract_features_from_frame_array(self, frame_array, text_prompt, 
                                        box_threshold=0.3, text_threshold=0.3, 
                                        ):
        """
        Extract features trực tiếp từ frame array sử dụng memory buffer
        """
        # Convert RGB to PIL Image
        pil_image = Image.fromarray(frame_array.astype(np.uint8))
        
        # Sử dụng GroundingDINO transforms trực tiếp
        import groundingdino.datasets.transforms as T
        
        transform = T.Compose([
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        
        # Transform image
        image_transformed, _ = transform(pil_image, None)
        
        # Predict using model
        boxes, logits, phrases = predict(
            model=self.image_extractor.model,
            image=image_transformed,
            caption=text_prompt,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            device=self.image_extractor.device,
            remove_combined = True
        )
        
        # Áp dụng filtering cho boxes với text prompt
        h, w = frame_array.shape[:2]
        boxes, logits, phrases, _ = self.image_extractor.filter_boxes(
            boxes, logits, phrases, (h, w), text_prompt, 
            max_boxes=self.max_entities
        )
        print("Đây là số lượng boundingbox {}".format(len(boxes)))
        # Extract backbone features
        feature_maps_tuple = self.image_extractor.extract_backbone_features(
            image_transformed.to(self.image_extractor.device)
        )
        
        # ===== THAY ĐỔI 3: Không tổng hợp, trả về mảng features và mask =====
        if len(boxes) > 0:
            boxes_scaled = boxes * torch.tensor([w, h, w, h], device=boxes.device)
            boxes_xyxy = box_convert(boxes_scaled, in_fmt="cxcywh", out_fmt="xyxy")
            roi_features = self.image_extractor.roi_align_features(feature_maps_tuple, boxes_xyxy, (h, w))
            
            # Pool và flatten để có (num_detected, feature_dim)
            pooled_features = F.adaptive_avg_pool2d(roi_features, (1, 1)).squeeze(-1).squeeze(-1)
            
            # Đảm bảo feature dim nhất quán
            if pooled_features.size(1) != self.feature_dim:
                if pooled_features.size(1) > self.feature_dim:
                    pooled_features = F.adaptive_avg_pool1d(pooled_features.unsqueeze(1), self.feature_dim).squeeze(1)
                else:
                    padding = torch.zeros(pooled_features.size(0), self.feature_dim - pooled_features.size(1), device=self.device)
                    pooled_features = torch.cat([pooled_features, padding], dim=1)
        else:
            # Nếu không có box nào, tạo mảng rỗng
            pooled_features = torch.empty(0, self.feature_dim, device=self.image_extractor.device)
        
        # Tạo mảng features và mask có kích thước cố định
        num_detected = pooled_features.shape[0]
        output_features = torch.zeros(self.max_entities, self.feature_dim, device=self.device)
        output_mask = torch.ones(self.max_entities, device=self.device, dtype=torch.bool) # True = padding

        if num_detected > 0:
            output_features[:num_detected] = pooled_features
            output_mask[:num_detected] = False # False = real data
        
        return {
            'entity_features': output_features.cpu().numpy(),
            'entity_mask': output_mask.cpu().numpy()
        }
        
    def get_video_info(self, video_path):
        """
        Lấy thông tin về video
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        # Lấy thông tin video
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration_seconds = frame_count / fps
        
        cap.release()
        
        return {
            'fps': fps,
            'total_frames': frame_count,
            'duration_seconds': duration_seconds,
            'frames_per_second_sampling': int(fps)  # Để lấy 1 frame/giây
        }
    
    def extract_frames_at_1fps(self, video_path):
        """
        Extract frames từ video với cv2 tối ưu (không temp files)
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration_seconds = int(total_frames / fps)
        
        frames = []
        frame_timestamps = []
        
        # Extract 1 frame per second
        for second in range(duration_seconds):
            frame_number = int(second * fps)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            
            ret, frame = cap.read()
            if not ret:
                break
            
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)
            frame_timestamps.append(second)
        
        cap.release()
        return frames, frame_timestamps
    
    def process_video_to_features(self, video_path, text_prompt, 
                                box_threshold=0.3, text_threshold=0.3):
        frames, _ = self.extract_frames_at_1fps(video_path)
        
        if not frames:
            print(f"Warning: No frames extracted from {video_path}. Skipping.")
            return None
        
        list_of_feature_arrays = []
        list_of_mask_arrays = []
        
        # ===== SỬA ĐỔI Ở ĐÂY =====
        # Lặp trực tiếp qua 'frames', không cần enumerate ở đây nữa
        # tqdm sẽ tự động theo dõi tiến trình
        for frame in tqdm(frames, desc=f"Processing {os.path.basename(video_path)}"):
            try:
                # Bây giờ 'frame' chắc chắn là một mảng NumPy
                result = self.extract_features_from_frame_array(
                    frame, text_prompt, box_threshold, text_threshold
                )
                list_of_feature_arrays.append(result['entity_features'])
                list_of_mask_arrays.append(result['entity_mask'])
            except Exception as e:
                # ... (phần xử lý lỗi không đổi)
                print(f"Error processing a frame in {video_path}: {e}. Using zero features.")
                list_of_feature_arrays.append(np.zeros((self.max_entities, self.feature_dim)))
                list_of_mask_arrays.append(np.ones((self.max_entities), dtype=bool))

        # ===== THAY ĐỔI 4: Stack và trả về hai mảng lớn =====
        final_features = np.stack(list_of_feature_arrays, axis=0)
        final_masks = np.stack(list_of_mask_arrays, axis=0)

        return {
            'features': final_features, # Shape: (T, max_entities, dim)
            'masks': final_masks      # Shape: (T, max_entities)
        }

def has_exactly_two_mkv_files(directory: str) -> bool:
    count = 0
    with os.scandir(directory) as entries:
        for entry in entries:
            # Check if the current entry is a file and ends with ".mkv"
            if entry.is_file() and entry.name.endswith(".mkv"):
                count += 1
                # Exit early if more than two such files are found
                if count > 2:
                    return False
    return count == 2
def list_end_directories(path: str) -> list:
    """
    Recursively traverse a directory and 
    return all paths that are files (end of branch).
    """
    end_files = []
    for root, _, files in os.walk(path):
        for file in files:
            if file.startswith('SoccerNetV2'):                
                pass
            else:
                refined_path = os.path.join(root, file)
                # print(refined_path)
                # if has_exactly_two_mkv_files(refined_path):
                end_files.append(refined_path)
    end_files = ["/".join(file.split("/")[:-1]) for file in end_files]
    end_files = [item for item in end_files if has_exactly_two_mkv_files(item)]
    end_files = sorted(list(set(end_files)))
    return end_files
# ===== VIDEO PROCESSING EXAMPLE =====
if __name__ == "__main__":
    # Initialize video extractor
    video_extractor = VideoFeatureExtractor(
        model_config_path="/home/thiendc/projects/video_summarization/train_soccernet/grounding_dino/config/GroundingDINO_SwinB_cfg.py",
        model_checkpoint_path="/home/thiendc/projects/video_summarization/train_soccernet/groundingdino_swinb_cogcoor.pth",
        device="cuda"
    )

    # Video settings
    TEXT_PROMPT = "goal. penalty. red card. yellow card. referee. close-up player. spectator"

    # VIDEO_PATH = "/home/thiendc/projects/video_summarization/v5/data/soccernet/test/england_epl/2014-2015/2015-05-17 - 18-00 Manchester United 1 - 1 Arsenal/1_224p.mkv"
    # Process video

    games = []
    list_games = list_end_directories("/home/storage/thiendc/soccernet/train")
    for item in list_games:
        item_1 = os.path.join("/home/storage/thiendc/soccernet/train", item, "1_224p.mkv")
        item_2 = os.path.join("/home/storage/thiendc/soccernet/train", item, "2_224p.mkv")
        games.append(item_1)
        games.append(item_2)
    print("Total games: ", len(games))
    games = games[:100]
    
    for game in tqdm(games):
        video_path = game
            
        # Extract video number and directory
        video_dir = os.path.dirname(video_path)         # "/home/.../2015-05-17 - 18-00 Manchester United 1 - 1 Arsenal/"
        video_filename = os.path.basename(video_path)   # "1_224p.mkv"
        video_number = video_filename.split('_')[0]
        output_filename = f'{video_number}_local_entities.npz'
        output_path = os.path.join(video_dir, output_filename)
        
        result = video_extractor.process_video_to_features(
            video_path=video_path,
            text_prompt=TEXT_PROMPT,
            box_threshold=0.3,
            text_threshold=0.3,
        )
        if result:
            np.savez_compressed(
                output_path,
                features=result['features'],
                masks=result['masks']
            )
            print(f"Processed video. Features shape: {result['features'].shape}, Masks shape: {result['masks'].shape}")
            print(f"Saved to {output_path}")