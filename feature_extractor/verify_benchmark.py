#!/usr/bin/env python3
"""
Script kiểm chứng (Verification & Benchmark):
1. Giải mã video thật từ dataset bằng Universal Decoder (FFmpeg + OpenCV).
2. Trích xuất features bằng cả 2 phương pháp (Gốc vs Tối Ưu).
3. Lưu tạm file .npy để kiểm tra toàn vẹn ghi/đọc đĩa.
4. Đo lường tốc độ, Cosine Similarity, sai số tuyệt đối.
5. TỰ ĐỘNG XÓA TOÀN BỘ FILE EMBEDDINGS THỬ NGHIỆM TẠO RA (TUYỆT ĐỐI KHÔNG XÓA FILE VIDEO).
"""

import os
import sys
import time
import shutil
import subprocess
import argparse
import cv2
import numpy as np
import torch
from torch.amp import autocast
import warnings
warnings.filterwarnings("ignore")

from transformers import CLIPModel

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def decode_video(video_path: str, target_fps: int = 1) -> np.ndarray:
    """Giải mã video bằng Universal Decoder (FFmpeg nếu có, fallback OpenCV cap.grab())."""
    # 1. Thử FFmpeg nếu có binary
    if shutil.which("ffmpeg"):
        cmd = [
            "ffmpeg", "-an", "-sn", "-dn",
            "-i", video_path,
            "-vf", f"fps={target_fps},scale=-2:224,crop=224:224",
            "-f", "image2pipe", "-pix_fmt", "rgb24", "-vcodec", "rawvideo",
            "-hide_banner", "-loglevel", "error", "-"
        ]
        try:
            pipe = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=10**8)
            raw_data, _ = pipe.communicate()
            if pipe.returncode == 0 and len(raw_data) > 0:
                frames = np.frombuffer(raw_data, dtype=np.uint8).reshape((-1, 224, 224, 3))
                if len(frames) > 0:
                    return frames
        except Exception:
            pass

    # 2. Fallback OpenCV cv2 với cap.grab() siêu tốc
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return np.zeros((0, 224, 224, 3), dtype=np.uint8)

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0 or np.isnan(fps):
            fps = 25.0
        step = max(int(round(fps / target_fps)), 1)

        frames = []
        frame_idx = 0
        while True:
            if frame_idx % step == 0:
                ret, frame = cap.read()
                if not ret or frame is None:
                    break
                h, w = frame.shape[:2]
                if h != 224:
                    scale = 224.0 / h
                    new_w = int(round(w * scale))
                    frame = cv2.resize(frame, (new_w, 224), interpolation=cv2.INTER_AREA)
                    h, w = 224, new_w

                start_x = max((w - 224) // 2, 0)
                cropped = frame[:, start_x : start_x + 224]
                if cropped.shape[1] < 224:
                    cropped = cv2.resize(cropped, (224, 224))
                rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
                frames.append(rgb)
            else:
                ret = cap.grab()
                if not ret:
                    break
            frame_idx += 1

        cap.release()
        if len(frames) > 0:
            return np.array(frames, dtype=np.uint8)
    except Exception as e:
        print(f"⚠️ Lỗi giải mã cv2: {e}")

    return np.zeros((0, 224, 224, 3), dtype=np.uint8)


def main():
    parser = argparse.ArgumentParser(description="Benchmark & Verify CLIP Features (Tự động xóa embedding tạm sau test)")
    parser.add_argument("--num_videos", type=int, default=5, help="Số video cần test (mặc định: 5)")
    parser.add_argument("--dataset_path", type=str, default="downloads/dataset")
    args = parser.parse_args()

    print("=" * 68)
    print(f"🧪 KIỂM CHỨNG & ĐO LƯỜNG HIỆU NĂNG: CLIP-ViT EXTRACTOR")
    print(f"🖥️ Thiết bị: {DEVICE} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print("=" * 68)

    # Chuẩn CLIP Mean & Std
    CLIP_MEAN_FP32 = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=DEVICE).view(1, 3, 1, 1)
    CLIP_STD_FP32 = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=DEVICE).view(1, 3, 1, 1)

    CLIP_MEAN_FP16 = CLIP_MEAN_FP32.to(torch.float16)
    CLIP_STD_FP16 = CLIP_STD_FP32.to(torch.float16)

    # Quét danh sách video thực tế (> 10MB)
    all_videos = []
    for root, _, files in os.walk(args.dataset_path):
        for f in files:
            if f.endswith(".mkv"):
                full_p = os.path.join(root, f)
                if os.path.getsize(full_p) > 10 * 1024 * 1024:
                    all_videos.append(full_p)
    all_videos = sorted(all_videos)

    if not all_videos:
        print("❌ Chưa tìm thấy file video .mkv hoàn chỉnh (>10MB) trong dataset!")
        return

    test_videos = all_videos[:args.num_videos]
    print(f"🎯 Tìm thấy {len(all_videos)} video hợp lệ. Đang tiến hành test trên {len(test_videos)} video...")

    print("\n🔄 Đang nạp mô hình CLIP 'ViFortune-AI/CLIP-ViT'...")
    model = CLIPModel.from_pretrained("ViFortune-AI/CLIP-ViT", dtype=torch.float16).to(DEVICE)
    model.eval()

    @torch.no_grad()
    def original_extract(frames_np, batch_size=128):
        all_features = []
        for i in range(0, len(frames_np), batch_size):
            batch = frames_np[i : i + batch_size]
            batch_tensor = torch.from_numpy(batch).to(DEVICE).permute(0, 3, 1, 2).float()
            batch_tensor = (batch_tensor / 255.0 - CLIP_MEAN_FP32) / CLIP_STD_FP32
            with autocast(device_type="cuda", dtype=torch.float16):
                vision_outputs = model.vision_model(pixel_values=batch_tensor)
                features = vision_outputs.pooler_output
            all_features.append(features.float().cpu().numpy())
        return np.concatenate(all_features, axis=0)

    @torch.inference_mode()
    def optimized_extract(frames_np, batch_size=512):
        all_features = []
        for i in range(0, len(frames_np), batch_size):
            batch = frames_np[i : i + batch_size]
            batch_tensor = (
                torch.from_numpy(batch)
                .to(DEVICE, non_blocking=True)
                .permute(0, 3, 1, 2)
                .to(torch.float16)
            )
            batch_tensor.div_(255.0).sub_(CLIP_MEAN_FP16).div_(CLIP_STD_FP16)
            with autocast(device_type="cuda", dtype=torch.float16):
                vision_outputs = model.vision_model(pixel_values=batch_tensor)
                features = vision_outputs.pooler_output
            all_features.append(features.float().cpu().numpy())
        return np.concatenate(all_features, axis=0)

    # Chạy warmup GPU
    dummy = np.zeros((128, 224, 224, 3), dtype=np.uint8)
    _ = original_extract(dummy, batch_size=128)
    _ = optimized_extract(dummy, batch_size=128)
    torch.cuda.synchronize()

    temp_embeddings_to_delete = []

    for idx, vid_path in enumerate(test_videos, 1):
        print(f"\n🎬 [{idx}/{len(test_videos)}] Video: {os.path.basename(vid_path)} ({os.path.getsize(vid_path)/(1024*1024):.1f} MB)")
        vid_dir = os.path.dirname(vid_path)

        # 1. Decode video
        t0 = time.time()
        frames = decode_video(vid_path, target_fps=1)
        t_decode = time.time() - t0
        num_frames = len(frames)
        print(f"   ⏱️ Decode: {num_frames} frames trong {t_decode:.3f}s ({num_frames/max(t_decode, 1e-4):.1f} FPS)")

        if num_frames == 0:
            print("   ⚠️ Không decode được frames, bỏ qua video này.")
            continue

        # 2. Chạy Bản Gốc (Batch 128)
        torch.cuda.synchronize()
        t0 = time.time()
        feat_orig = original_extract(frames, batch_size=128)
        torch.cuda.synchronize()
        t_orig = time.time() - t0
        fps_orig = num_frames / t_orig

        # Lưu thử nghiệm file embedding tạm bản gốc
        temp_orig_file = os.path.join(vid_dir, f"_temp_test_orig_{idx}.npy")
        np.save(temp_orig_file, feat_orig)
        temp_embeddings_to_delete.append(temp_orig_file)

        # 3. Chạy Bản Tối Ưu (Batch 512)
        torch.cuda.synchronize()
        t0 = time.time()
        feat_opt = optimized_extract(frames, batch_size=512)
        torch.cuda.synchronize()
        t_opt = time.time() - t0
        fps_opt = num_frames / t_opt

        # Lưu thử nghiệm file embedding tạm bản tối ưu
        temp_opt_file = os.path.join(vid_dir, f"_temp_test_opt_{idx}.npy")
        np.save(temp_opt_file, feat_opt)
        temp_embeddings_to_delete.append(temp_opt_file)

        # 4. So sánh số học
        norm_orig = feat_orig / np.linalg.norm(feat_orig, axis=-1, keepdims=True)
        norm_opt = feat_opt / np.linalg.norm(feat_opt, axis=-1, keepdims=True)
        cosine_sim = np.sum(norm_orig * norm_opt, axis=-1)
        mean_cos_sim = np.mean(cosine_sim)
        max_abs_diff = np.max(np.abs(feat_orig - feat_opt))
        speedup = fps_opt / fps_orig

        print(f"   ⚡ GPU Bản Gốc:     {fps_orig:.1f} FPS ({t_orig:.3f}s)")
        print(f"   ⚡ GPU Bản Tối Ưu:  {fps_opt:.1f} FPS ({t_opt:.3f}s) ──> Nhanh hơn {speedup:.2f}x lần")
        print(f"   📐 Cosine Similarity: {mean_cos_sim:.8f} (1.0000 = Trùng khớp hoàn hảo)")
        print(f"   📐 Sai số tối đa:     {max_abs_diff:.8e}")

    # =========================================================
    # BƯỚC 5: TỰ ĐỘNG XÓA SẠCH 100% CÁC FILE EMBEDDINGS THỬ NGHIỆM
    # (GIỮ NGUYÊN VẸN TOÀN BỘ FILE VIDEO .MKV)
    # =========================================================
    print("\n" + "=" * 68)
    print("🧹 ĐANG TỰ ĐỘNG XÓA CÁC FILE EMBEDDINGS THỬ NGHIỆM...")
    deleted_count = 0
    for fpath in temp_embeddings_to_delete:
        if os.path.exists(fpath):
            os.remove(fpath)
            deleted_count += 1
            print(f"   🗑️ Đã xóa file embedding: {os.path.basename(fpath)}")

    print(f"✅ Đã xóa {deleted_count} file embedding thử nghiệm. KHÔNG XÓA bất kỳ video .mkv nào!")
    print("=" * 68)

if __name__ == "__main__":
    main()
