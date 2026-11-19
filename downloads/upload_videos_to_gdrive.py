#!/usr/bin/env python3
"""
Upload SoccerNet 224p videos + Labels-v2.json to Google Drive via rclone.
Supports two modes:
  - local: upload games already present in --local-dir (e.g. downloads/dataset).
  - hf:    stream-download each of the 500 games from HuggingFace, upload to Drive,
           and delete local temp + HF cache right away (resume via completed_upload.txt).
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

_DEFAULT_HF_HOME = "/home/storage/thiendc/soccernet_clip_frames/_hf_cache"
os.makedirs(_DEFAULT_HF_HOME, exist_ok=True)
os.environ.setdefault("HF_HOME", _DEFAULT_HF_HOME)
os.environ.setdefault("HF_HUB_CACHE", os.path.join(_DEFAULT_HF_HOME, "hub"))
os.environ.setdefault("HF_XET_CACHE", os.path.join(_DEFAULT_HF_HOME, "xet"))

from huggingface_hub import hf_hub_download
from tqdm import tqdm

REPO_FEATURES = "OpenSportsLab/SoccerNet-ActionSpotting-Features"
REPO_VIDEOS = "tartotarto/soccernet-500-videos"


def convert_hf_path_to_standard(hf_game_rel_path: str) -> str:
    parts = hf_game_rel_path.split("/")
    if len(parts) >= 3:
        return os.path.join(parts[0], parts[1], parts[2].replace("_-_", " - ").replace("_", " "))
    return hf_game_rel_path.replace("_-_", " - ").replace("_", " ")


def rclone_upload_dir(local_dir: Path, remote_dir: str, transfers: int = 4) -> bool:
    cmd = [
        "rclone", "copy", str(local_dir), remote_dir,
        "--include", "*_224p.mkv",
        "--include", "Labels-v2.json",
        "--transfers", str(transfers),
        "--retries", "3",
        "--stats", "5s",
        "--stats-one-line",
        "-v",
    ]
    res = subprocess.run(cmd)
    if res.returncode != 0:
        tqdm.write(f"  [WARN] rclone upload failed for {local_dir} (code={res.returncode})")
        return False
    return True


def upload_from_local(local_root: Path, remote_root: str, transfers: int, delete_local: bool) -> None:
    label_files = sorted(local_root.rglob("Labels-v2.json"))
    print(f"Found {len(label_files)} local games under {local_root}.")
    for lbl in tqdm(label_files, desc="Upload local -> GDrive", unit="game"):
        game_dir = lbl.parent
        rel = game_dir.relative_to(local_root)
        remote_dir = f"{remote_root.rstrip('/')}/{rel.as_posix()}"
        ok = rclone_upload_dir(game_dir, remote_dir, transfers=transfers)
        if ok and delete_local:
            shutil.rmtree(game_dir, ignore_errors=True)


def upload_from_hf(remote_root: str, tmp_root: Path, transfers: int, splits: List[str]) -> None:
    token = os.environ.get("HF_TOKEN", None)
    hf_cache_dir = tmp_root / "_hf_cache"
    completed_file = tmp_root / "completed_upload.txt"
    tmp_root.mkdir(parents=True, exist_ok=True)
    hf_cache_dir.mkdir(parents=True, exist_ok=True)

    completed = set()
    if completed_file.exists():
        completed = {l.strip() for l in completed_file.read_text(encoding="utf-8").splitlines() if l.strip()}

    games_map: Dict[str, List[dict]] = {}
    for split in splits:
        print(f"Loading annotations.json ({split})...")
        ann_path = hf_hub_download(
            repo_id=REPO_FEATURES,
            repo_type="dataset",
            filename=f"baidu_soccer_embeddings/{split}/annotations.json",
            token=token,
            cache_dir=str(hf_cache_dir),
        )
        with open(ann_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for v in data.get("videos", []):
            dir_path = os.path.dirname(v.get("path", ""))
            std_path = convert_hf_path_to_standard(dir_path)
            games_map.setdefault(std_path, []).extend(v.get("annotations", []))

    pending = [g for g in sorted(games_map.keys()) if g not in completed]
    print(f"Total games: {len(games_map)} | Pending upload: {len(pending)}")

    with open(completed_file, "a", encoding="utf-8") as cf:
        for std_game in tqdm(pending, desc="Stream HF -> GDrive", unit="game"):
            game_tmp = tmp_root / std_game
            game_tmp.mkdir(parents=True, exist_ok=True)
            hf_paths_to_clean: List[str] = []

            try:
                with open(game_tmp / "Labels-v2.json", "w", encoding="utf-8") as lf:
                    json.dump({"UrlLocal": f"{std_game}/", "annotations": games_map[std_game]}, lf, indent=2)

                for half in (1, 2):
                    mkv_name = f"{half}_224p.mkv"
                    try:
                        cached = hf_hub_download(
                            repo_id=REPO_VIDEOS,
                            repo_type="dataset",
                            filename=f"{std_game}/{mkv_name}",
                            token=token,
                            cache_dir=str(hf_cache_dir),
                        )
                        real = os.path.realpath(cached)
                        hf_paths_to_clean.extend([cached, real])
                        if os.path.exists(real) and os.path.getsize(real) > 1024:
                            shutil.copyfile(real, game_tmp / mkv_name)
                    except Exception as e:
                        tqdm.write(f"  [WARN] Missing {std_game}/{mkv_name}: {e}")

                remote_dir = f"{remote_root.rstrip('/')}/{std_game}"
                if rclone_upload_dir(game_tmp, remote_dir, transfers=transfers):
                    cf.write(std_game + "\n")
                    cf.flush()
            finally:
                shutil.rmtree(game_tmp, ignore_errors=True)
                for p in set(hf_paths_to_clean):
                    try:
                        if p and os.path.islink(p):
                            os.unlink(p)
                        elif p and os.path.isfile(p):
                            os.remove(p)
                    except OSError:
                        pass


def main() -> None:
    parser = argparse.ArgumentParser(description="Upload SoccerNet 224p videos to Google Drive via rclone.")
    parser.add_argument("--mode", choices=["local", "hf"], default="local", help="local: upload existing folder; hf: stream 500 games from HF.")
    parser.add_argument("--remote", type=str, default="drive:SoccerNet_224p", help="rclone destination, e.g. drive:SoccerNet_224p or thiendc:SoccerNet_224p")
    parser.add_argument("--local-dir", type=str, default="downloads/dataset", help="Local directory for --mode local.")
    parser.add_argument("--tmp-dir", type=str, default="/home/storage/thiendc/_tmp_gdrive_upload", help="Temp directory for --mode hf.")
    parser.add_argument("--transfers", type=int, default=4)
    parser.add_argument("--delete-local", action="store_true", help="Delete local game folder after successful upload (--mode local).")
    parser.add_argument("--splits", nargs="+", default=["train", "valid", "test"])
    args = parser.parse_args()

    if args.mode == "local":
        upload_from_local(Path(args.local_dir), args.remote, args.transfers, args.delete_local)
    else:
        upload_from_hf(args.remote, Path(args.tmp_dir), args.transfers, args.splits)


if __name__ == "__main__":
    main()
