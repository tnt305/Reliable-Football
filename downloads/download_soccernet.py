#!/usr/bin/env python3
"""
SoccerNet Action Spotting Dataset Downloader
Tải tự động Videos (224p/720p), Baidu Embeddings và Labels-v2.json chuẩn format SoccerNet.
"""

import os
import sys
import json
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from huggingface_hub import hf_hub_download

REPO_FEATURES = "OpenSportsLab/SoccerNet-ActionSpotting-Features"
REPO_VIDEOS = "tartotarto/soccernet-500-videos"
HF_TOKEN = os.environ.get("HF_TOKEN", None)

def convert_hf_path_to_standard(hf_game_rel_path: str) -> str:
    """
    Chuyển tên thư mục từ format HuggingFace (dùng dấu gạch dưới)
    về chuẩn format SoccerNet (dùng khoảng trắng).
    Ví dụ: england_epl/2014-2015/2015-02-21_-_18-00_Chelsea_1_-_1_Burnley
       ->  england_epl/2014-2015/2015-02-21 - 18-00 Chelsea 1 - 1 Burnley
    """
    parts = hf_game_rel_path.split("/")
    if len(parts) >= 3:
        league = parts[0]
        season = parts[1]
        game_name = parts[2].replace("_-_", " - ").replace("_", " ")
        return os.path.join(league, season, game_name)
    return hf_game_rel_path.replace("_-_", " - ").replace("_", " ")

def download_file_safe(repo_id: str, filename: str, target_filepath: str, output_dir: str = "./dataset") -> bool:
    """Tải 1 file từ HuggingFace và lưu vào đúng đường dẫn target_filepath."""
    import shutil
    try:
        # Kiểm tra file đã tồn tại và có dung lượng thực (> 10KB)
        if os.path.exists(target_filepath) and os.path.getsize(target_filepath) > 10240:
            return True

        os.makedirs(os.path.dirname(target_filepath), exist_ok=True)
        # Không dùng local_dir để tránh path mismatch giữa gạch dưới và khoảng trắng
        # hf_hub_download trả về path trong HF cache, sau đó copy sang target
        cached_path = hf_hub_download(
            repo_id=repo_id,
            repo_type="dataset",
            filename=filename,
            token=HF_TOKEN,
        )
        # Resolve symlink để lấy file thực trong HF blob cache
        real_path = os.path.realpath(cached_path)
        file_size = os.path.getsize(real_path) if os.path.exists(real_path) else 0
        if file_size <= 1024:
            print(f"⚠️ File tải về quá nhỏ ({file_size} bytes) - có thể là LFS pointer: {real_path}", file=sys.stderr)
            return False
        if os.path.exists(target_filepath):
            os.remove(target_filepath)
        shutil.copyfile(real_path, target_filepath)
        print(f"  ✓ {os.path.basename(target_filepath)} ({file_size / 1024 / 1024:.1f} MB)")
        return True
    except Exception as e:
        print(f"⚠️ Lỗi tải {filename}: {e}", file=sys.stderr)
        return False

def process_game(game_info: dict, args: argparse.Namespace) -> tuple:
    """Xử lý tải trọn gói 1 trận đấu (Video, Baidu Features, Labels-v2.json)."""
    std_game_path = game_info["std_game_path"]
    hf_game_path = game_info["hf_game_path"]
    split = game_info["split"]
    annotations = game_info["annotations"]
    
    target_game_dir = os.path.join(args.output_dir, std_game_path)
    os.makedirs(target_game_dir, exist_ok=True)
    
    success = True
    
    # 1. Tạo Labels-v2.json từ annotations
    label_file = os.path.join(target_game_dir, "Labels-v2.json")
    if not os.path.exists(label_file) or os.path.getsize(label_file) == 0:
        try:
            data = {
                "UrlLocal": f"{std_game_path}/",
                "UrlYoutube": "",
                "annotations": annotations
            }
            with open(label_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=4)
        except Exception as e:
            print(f"⚠️ Lỗi tạo {label_file}: {e}", file=sys.stderr)
            success = False

    # 2. Tải Baidu Embeddings - path trong repo dùng dấu gạch dưới (hf_game_path)
    if args.download_features:
        for half in [1, 2]:
            feat_hf = f"baidu_soccer_embeddings/{split}/{hf_game_path}/{half}_baidu_soccer_embeddings.npy"
            feat_target = os.path.join(target_game_dir, f"{half}_baidu_soccer_embeddings.npy")
            if not download_file_safe(REPO_FEATURES, feat_hf, feat_target, args.output_dir):
                success = False

    # 3. Tải Video (224p hoặc 720p) - path trong repo dùng khoảng trắng (std_game_path)
    if args.download_videos:
        res = args.resolution
        for half in [1, 2]:
            vid_hf = f"{std_game_path}/{half}_{res}.mkv"
            vid_target = os.path.join(target_game_dir, f"{half}_{res}.mkv")
            if not download_file_safe(REPO_VIDEOS, vid_hf, vid_target, args.output_dir):
                success = False

    return std_game_path, success

def main():
    parser = argparse.ArgumentParser(description="SoccerNet Action Spotting Downloader")
    parser.add_argument("--split", type=str, default="train", choices=["train", "valid", "test", "challenge", "all", "trainvalidtest"],
                        help="Split dữ liệu cần tải (mặc định: train). 'trainvalidtest' để tải train+valid+test")
    parser.add_argument("--output_dir", type=str, default="./dataset",
                        help="Thư mục đích để lưu dataset (mặc định: ./dataset)")
    parser.add_argument("--token", type=str, default=None,
                        help="HuggingFace token (mặc định: dùng HF_TOKEN trong code)")
    parser.add_argument("--resolution", type=str, default="224p", choices=["224p", "720p"],
                        help="Độ phân giải video (mặc định: 224p)")
    parser.add_argument("--download_videos", action="store_true", default=False,
                        help="Tải video trận đấu (mặc định: False)")
    parser.add_argument("--download_features", action="store_true", default=True,
                        help="Tải Baidu Embeddings (mặc định: True)")
    parser.add_argument("--no_features", dest="download_features", action="store_false",
                        help="Bỏ qua tải features")
    parser.add_argument("--max_workers", type=int, default=8,
                        help="Số luồng tải đồng thời (mặc định: 8)")
    parser.add_argument("--num_games", type=int, default=None,
                        help="Giới hạn số trận cần tải (mặc định: tải toàn bộ split)")
    
    args = parser.parse_args()

    if args.split == "all":
        splits = ["train", "valid", "test", "challenge"]
    elif args.split == "trainvalidtest":
        splits = ["train", "valid", "test"]
    else:
        splits = [args.split]
    
    print("=" * 65)
    print("🚀 SOCCERNET DATASET DOWNLOADER")
    print(f"   Splits:             {splits}")
    print(f"   Output Directory:   {os.path.abspath(args.output_dir)}")
    print(f"   Download Features:  {args.download_features} (Baidu Embeddings)")
    print(f"   Download Videos:    {args.download_videos} (Resolution: {args.resolution if args.download_videos else 'N/A'})")
    # Override token nếu có
    if args.token:
        global HF_TOKEN
        HF_TOKEN = args.token
    print(f"   Max Workers:        {args.max_workers}")
    if args.num_games:
        print(f"   Limit Games:        {args.num_games}")
    print("=" * 65)

    all_games_to_download = []

    for spl in splits:
        print(f"\n📥 Đang lấy danh sách trận đấu cho split [{spl}]...")
        ann_filename = f"baidu_soccer_embeddings/{spl}/annotations.json"
        try:
            ann_path = hf_hub_download(repo_id=REPO_FEATURES, repo_type="dataset", filename=ann_filename, token=HF_TOKEN)
            with open(ann_path, "r", encoding="utf-8") as f:
                ann_data = json.load(f)
        except Exception as e:
            print(f"❌ Không lấy được annotations cho split {spl}: {e}", file=sys.stderr)
            continue

        # Gom nhóm annotations theo game
        games_dict = {}
        for video in ann_data.get("videos", []):
            vid_path = video.get("path", "")
            # vid_path dạng: england_epl/2014-2015/2015-02-21_-_18-00_Chelsea_1_-_1_Burnley/1_baidu_soccer_embeddings.npy
            dir_path = os.path.dirname(vid_path)
            if dir_path not in games_dict:
                games_dict[dir_path] = []
            games_dict[dir_path].extend(video.get("annotations", []))

        for hf_game_path, annotations in games_dict.items():
            std_game_path = convert_hf_path_to_standard(hf_game_path)
            all_games_to_download.append({
                "split": spl,
                "hf_game_path": hf_game_path,
                "std_game_path": std_game_path,
                "annotations": annotations
            })

    if args.num_games and args.num_games > 0:
        all_games_to_download = all_games_to_download[:args.num_games]

    total_games = len(all_games_to_download)
    print(f"🎯 Tổng cộng có {total_games} trận đấu cần xử lý.")

    if total_games == 0:
        print("Không tìm thấy trận đấu nào phù hợp. Kết thúc.")
        return

    # Tiến hành tải song song
    success_count = 0
    failed_games = []

    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {executor.submit(process_game, game, args): game for game in all_games_to_download}
        
        with tqdm(total=total_games, desc="Downloading SoccerNet Games", unit="game") as pbar:
            for future in as_completed(futures):
                game_path, ok = future.result()
                if ok:
                    success_count += 1
                else:
                    failed_games.append(game_path)
                pbar.update(1)

    print("\n" + "=" * 65)
    print(f"✅ Hoàn tất! Thành công: {success_count}/{total_games} trận.")
    if failed_games:
        print(f"❌ Có {len(failed_games)} trận gặp lỗi:")
        for g in failed_games[:10]:
            print(f"   - {g}")
    print(f"📂 Toàn bộ dữ liệu được lưu chuẩn tại: {os.path.abspath(args.output_dir)}")
    print("=" * 65)

if __name__ == "__main__":
    main()
