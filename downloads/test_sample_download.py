import os
import json
import numpy as np
from huggingface_hub import hf_hub_download

def test_download_sample():
    output_dir = "./sample_test_download"
    os.makedirs(output_dir, exist_ok=True)
    
    # Trận mẫu thử nghiệm
    game_path_video = "england_epl/2014-2015/2015-02-21 - 18-00 Chelsea 1 - 1 Burnley"
    game_path_features = "england_epl/2014-2015/2015-02-21_-_18-00_Chelsea_1_-_1_Burnley"
    
    print("=" * 60)
    print("🚀 BẮT ĐẦU TEST TẢI SAMPLE DỮ LIỆU SOCCERNET")
    print("=" * 60)

    # 1. Tải Video 224p từ repo tartotarto/soccernet-500-videos
    print("\n[1/3] Đang tải Video 224p (1_224p.mkv)...")
    video_file = f"{game_path_video}/1_224p.mkv"
    video_local_path = hf_hub_download(
        repo_id="tartotarto/soccernet-500-videos",
        repo_type="dataset",
        filename=video_file,
        local_dir=output_dir
    )
    video_size_mb = os.path.getsize(video_local_path) / (1024 * 1024)
    print(f"  ✓ Video 224p đã tải thành công: {video_local_path}")
    print(f"  ✓ Dung lượng: {video_size_mb:.2f} MB")

    # 2. Tải Baidu Embeddings từ repo OpenSportsLab/SoccerNet-ActionSpotting-Features
    print("\n[2/3] Đang tải Baidu Embeddings (1_baidu_soccer_embeddings.npy)...")
    feature_file = f"baidu_soccer_embeddings/train/{game_path_features}/1_baidu_soccer_embeddings.npy"
    feat_local_path = hf_hub_download(
        repo_id="OpenSportsLab/SoccerNet-ActionSpotting-Features",
        repo_type="dataset",
        filename=feature_file,
        local_dir=output_dir
    )
    feat_data = np.load(feat_local_path)
    feat_size_mb = os.path.getsize(feat_local_path) / (1024 * 1024)
    print(f"  ✓ Baidu Embeddings đã tải thành công: {feat_local_path}")
    print(f"  ✓ Shape: {feat_data.shape}, Dtype: {feat_data.dtype}, Size: {feat_size_mb:.2f} MB")

    # 3. Tải & trích xuất Labels-v2.json từ annotations.json
    print("\n[3/3] Đang lấy nhãn Labels-v2.json...")
    ann_file = "baidu_soccer_embeddings/train/annotations.json"
    ann_local_path = hf_hub_download(
        repo_id="OpenSportsLab/SoccerNet-ActionSpotting-Features",
        repo_type="dataset",
        filename=ann_file,
        local_dir=output_dir
    )
    with open(ann_local_path, "r", encoding="utf-8") as f:
        ann_data = json.load(f)

    # Trích xuất annotations cho trận Chelsea 1 - 1 Burnley
    game_annotations = []
    for v in ann_data.get("videos", []):
        if game_path_features in v.get("path", ""):
            game_annotations.extend(v.get("annotations", []))

    label_output_path = os.path.join(output_dir, game_path_video, "Labels-v2.json")
    os.makedirs(os.path.dirname(label_output_path), exist_ok=True)
    labels_v2 = {
        "UrlLocal": f"{game_path_video}/",
        "UrlYoutube": "",
        "annotations": game_annotations
    }
    with open(label_output_path, "w", encoding="utf-8") as f:
        json.dump(labels_v2, f, indent=4)
    
    print(f"  ✓ Labels-v2.json đã được tạo thành công: {label_output_path}")
    print(f"  ✓ Số lượng action annotations: {len(game_annotations)}")

    print("\n" + "=" * 60)
    print("🎉 TẤT CẢ 3 THÀNH PHẦN ĐÃ ĐƯỢC TẢI VÀ KIỂM TRA HỢP LỆ THÀNH CÔNG!")
    print("=" * 60)

if __name__ == "__main__":
    test_download_sample()
