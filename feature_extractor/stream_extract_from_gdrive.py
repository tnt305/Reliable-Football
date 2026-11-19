#!/usr/bin/env python3
"""
Stream-pull SoccerNet .mkv + Labels-v2.json from Google Drive via rclone,
extract CLIP (.npy, 512-d projected) and Audio Log-Mel (.npy) on GPU,
save to out_dir/<game>/, and delete temporary .mkv files immediately.
Supports automatic resume via completed_extract.txt.
"""

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import List

import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from feature_extractor.clip_vit import FastCLIPExtractor, decode_video_ffmpeg
from feature_extractor.mel import compute_log_mel_gpu, extract_audio_pcm_pipe


def run_rclone(args: List[str]) -> subprocess.CompletedProcess:
    return subprocess.run(["rclone"] + args, capture_output=True, text=True)


def list_gdrive_games(remote_root: str) -> List[str]:
    """List relative game directories on Google Drive containing *_224p.mkv or Labels-v2.json."""
    res = run_rclone(["lsf", "-R", "--files-only", "--include", "*_224p.mkv", remote_root])
    if res.returncode != 0:
        raise RuntimeError(f"rclone lsf failed: {res.stderr.strip()}")
    games = sorted({os.path.dirname(line.strip()) for line in res.stdout.splitlines() if line.strip()})
    return [g for g in games if g]


def process_game(
    remote_root: str,
    rel_game: str,
    tmp_root: Path,
    out_root: Path,
    clip_extractor: FastCLIPExtractor,
    res_tag: str,
    target_fps: int,
) -> bool:
    tmp_game_dir = tmp_root / rel_game
    out_game_dir = out_root / rel_game
    tmp_game_dir.mkdir(parents=True, exist_ok=True)
    out_game_dir.mkdir(parents=True, exist_ok=True)

    try:
        # 1. Pull only videos + Labels-v2.json for this game from Google Drive
        src = f"{remote_root.rstrip('/')}/{rel_game}"
        dl = run_rclone([
            "copy", src, str(tmp_game_dir),
            "--include", f"*_{res_tag}.mkv",
            "--include", "Labels-v2.json",
            "--transfers", "4",
        ])
        if dl.returncode != 0:
            tqdm.write(f"  [WARN] rclone copy failed for {rel_game}: {dl.stderr.strip()}")
            return False

        # Copy Labels-v2.json to permanent feature directory
        lbl_src = tmp_game_dir / "Labels-v2.json"
        if lbl_src.exists():
            shutil.copy2(lbl_src, out_game_dir / "Labels-v2.json")

        # 2. Extract CLIP (.npy) + Audio Log-Mel (.npy) for each half
        extracted_any = False
        for half in (1, 2):
            mkv_path = tmp_game_dir / f"{half}_{res_tag}.mkv"
            if not mkv_path.exists():
                continue

            # CLIP features
            clip_out = out_game_dir / f"{half}_CLIP_ViT.npy"
            if not clip_out.exists():
                frames = decode_video_ffmpeg(str(mkv_path), target_fps=target_fps)
                if len(frames) > 0:
                    feats = clip_extractor.extract_features(frames)
                    np.save(clip_out, feats)
                    extracted_any = True

            # Audio Log-Mel features
            audio_out = out_game_dir / f"audio{half}.npy"
            if not audio_out.exists():
                pcm = extract_audio_pcm_pipe(str(mkv_path))
                if pcm is not None and len(pcm) > 0:
                    mel = compute_log_mel_gpu(pcm)
                    np.save(audio_out, mel)
                    extracted_any = True

        return extracted_any or (out_game_dir / "1_CLIP_ViT.npy").exists()

    finally:
        # 3. Always delete temporary .mkv files right away
        shutil.rmtree(tmp_game_dir, ignore_errors=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Stream-extract CLIP & Audio .npy from Google Drive via rclone.")
    parser.add_argument("--remote", type=str, required=True, help="rclone remote path, e.g. gdrive:SoccerNet/videos")
    parser.add_argument("--out-dir", type=str, default="/home/storage/thiendc/soccernet_features_512d")
    parser.add_argument("--clip-model", type=str, default="/home/storage/thiendc/checkpoints/clip_l14_contrastive/best_model")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--fps", type=int, default=1)
    parser.add_argument("--res", type=str, default="224p")
    parser.add_argument("--no-projection", action="store_true", help="Disable CLIP projection head (not recommended for 512d).")
    args = parser.parse_args()

    out_root = Path(args.out_dir)
    tmp_root = out_root / "_tmp_mkv"
    completed_file = out_root / "completed_extract.txt"
    out_root.mkdir(parents=True, exist_ok=True)
    tmp_root.mkdir(parents=True, exist_ok=True)

    completed = set()
    if completed_file.exists():
        completed = {l.strip() for l in completed_file.read_text(encoding="utf-8").splitlines() if l.strip()}

    print(f"Listing games from {args.remote} ...")
    all_games = list_gdrive_games(args.remote)
    pending = [g for g in all_games if g not in completed]
    print(f"Found {len(all_games)} games total, {len(pending)} pending.")

    if not pending:
        print("Nothing to extract.")
        return

    clip_extractor = FastCLIPExtractor(
        model_name=args.clip_model,
        batch_size=args.batch_size,
        use_projection=not args.no_projection,
    )

    with open(completed_file, "a", encoding="utf-8") as cf:
        pbar = tqdm(pending, desc="Extract GDrive->NPY", unit="game")
        for rel_game in pbar:
            t0 = time.time()
            pbar.set_postfix_str(rel_game.split("/")[-1][:32])
            ok = process_game(
                remote_root=args.remote,
                rel_game=rel_game,
                tmp_root=tmp_root,
                out_root=out_root,
                clip_extractor=clip_extractor,
                res_tag=args.res,
                target_fps=args.fps,
            )
            if ok:
                cf.write(rel_game + "\n")
                cf.flush()
            tqdm.write(f"[{'OK' if ok else 'SKIP'}] {rel_game} ({time.time() - t0:.1f}s)")


if __name__ == "__main__":
    main()
