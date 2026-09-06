#!/usr/bin/env python3
# ==============================================================================
# Siêu tốc CLIP Feature Extractor (Tối ưu hóa tối đa cho 1 GPU 32GB VRAM)
# - Pipeline Producer-Consumer đa luồng chuẩn: 8 CPU workers decode song song
# - Giới hạn RAM an toàn: decoded_queue maxsize=8 (~3GB RAM)
# - GPU 100% bão hòa tính toán liên tục, không bị nghẽn decode
# ==============================================================================

import os
import sys
import subprocess
import argparse
import queue
import threading
import shutil
import numpy as np
import torch
from torch.amp import autocast
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")
try:
    import transformers.utils.versions
    transformers.utils.versions.require_version = lambda *args, **kwargs: True
except Exception:
    pass

from transformers import CLIPModel

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

# Chuẩn ImageNet / CLIP Mean & Std trên GPU dưới dạng FP16
CLIP_MEAN = torch.tensor(
    [0.48145466, 0.4578275, 0.40821073], device=DEVICE, dtype=torch.float16
).view(1, 3, 1, 1)
CLIP_STD = torch.tensor(
    [0.26862954, 0.26130258, 0.27577711], device=DEVICE, dtype=torch.float16
).view(1, 3, 1, 1)


def decode_video_ffmpeg(video_path: str, target_fps: int = 1) -> np.ndarray:
    """Giải mã video bằng FFmpeg pipe trực tiếp vào RAM (kèm fallback OpenCV)."""
    if shutil.which("ffmpeg"):
        cmd = [
            "ffmpeg",
            "-an", "-sn", "-dn",
            "-i", video_path,
            "-vf", f"fps={target_fps},scale=-2:224,crop=224:224",
            "-f", "image2pipe",
            "-pix_fmt", "rgb24",
            "-vcodec", "rawvideo",
            "-hide_banner",
            "-loglevel", "error",
            "-"
        ]
        try:
            pipe = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                bufsize=10**8,
            )
            raw_data, _ = pipe.communicate()
            if pipe.returncode == 0 and len(raw_data) > 0:
                frames = np.frombuffer(raw_data, dtype=np.uint8).reshape((-1, 224, 224, 3))
                if len(frames) > 0:
                    return frames
        except Exception:
            pass

    # Fallback OpenCV cv2
    try:
        import cv2
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
    except Exception:
        pass

    return np.zeros((0, 224, 224, 3), dtype=np.uint8)


class FastCLIPExtractor:

    def __init__(
        self,
        model_name: str = "ViFortune-AI/CLIP-ViT",
        batch_size: int = 512,
    ):
        self.batch_size = batch_size
        print(f"🔄 Đang tải mô hình CLIP: '{model_name}'...")

        self.model = CLIPModel.from_pretrained(
            model_name,
            dtype=torch.float16,
        ).to(DEVICE)
        self.model.eval()

        self.embed_dim = self.model.config.vision_config.hidden_size  # 768
        gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
        gpu_mem = (
            torch.cuda.get_device_properties(0).total_memory / (1024**3)
            if torch.cuda.is_available()
            else 0
        )
        print(
            f"✅ Sẵn sàng trích xuất trên: {DEVICE} ({gpu_name} - {gpu_mem:.1f} GB VRAM)"
        )
        print(f"   - Feature Dim: {self.embed_dim} | Batch Size: {batch_size}")

    @torch.inference_mode()
    def extract_features(self, frames_np: np.ndarray) -> np.ndarray:
        """Inference siêu tốc qua Vision Model trên GPU (Inference Mode + In-place FP16)."""
        total_frames = frames_np.shape[0]
        if total_frames == 0:
            return np.zeros((0, self.embed_dim), dtype=np.float32)

        all_features = []
        for i in range(0, total_frames, self.batch_size):
            batch = frames_np[i : i + self.batch_size]

            # Chuyển batch trực tiếp lên GPU (non_blocking) và cast sang float16
            batch_tensor = (
                torch.from_numpy(batch)
                .to(DEVICE, non_blocking=True)
                .permute(0, 3, 1, 2)
                .to(torch.float16)
            )

            # Chuẩn hóa in-place trên GPU
            batch_tensor.div_(255.0).sub_(CLIP_MEAN).div_(CLIP_STD)

            # Chạy qua Vision Transformer lấy [CLS] pooler_output (768D)
            with autocast(device_type="cuda", dtype=torch.float16):
                vision_outputs = self.model.vision_model(
                    pixel_values=batch_tensor
                )
                features = vision_outputs.pooler_output

            all_features.append(features.float().cpu().numpy())

        return np.concatenate(all_features, axis=0)


def async_writer_thread(write_queue: queue.Queue):
    """Luồng ghi file .npy ngầm để GPU không phải chờ ổ cứng."""
    while True:
        item = write_queue.get()
        if item is None:
            write_queue.task_done()
            break
        output_path, features_np = item
        try:
            np.save(output_path, features_np)
        except Exception as e:
            print(f"❌ Lỗi khi lưu {output_path}: {e}")
        finally:
            write_queue.task_done()


def main():
    parser = argparse.ArgumentParser(description="Trích xuất CLIP Features siêu tốc trên 1 GPU 32GB")
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="downloads/dataset",
        help="Đường dẫn đến thư mục dataset chứa video .mkv",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="ViFortune-AI/CLIP-ViT",
        help="HuggingFace model ID hoặc local path của mô hình CLIP",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=512,
        help="Batch size trên GPU (mặc định: 512)",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=8,
        help="Số luồng CPU decode video FFmpeg chạy song song (mặc định: 8)",
    )
    args = parser.parse_args()

    extractor = FastCLIPExtractor(
        model_name=args.model_name, batch_size=args.batch_size
    )

    print("🔍 Đang quét danh sách video...")
    games = []
    for root, _, files in os.walk(args.dataset_path):
        for file in files:
            if file.endswith(".mkv"):
                full_p = os.path.join(root, file)
                if os.path.getsize(full_p) > 10 * 1024 * 1024:
                    games.append(full_p)
    games = sorted(games)
    print(f"🎯 Tìm thấy {len(games)} files .mkv hợp lệ")

    # Lọc danh sách video cần xử lý (bỏ qua những video đã có file .npy)
    pending_tasks = []
    for video_path in games:
        video_dir = os.path.dirname(video_path)
        video_filename = os.path.basename(video_path)
        video_number = video_filename.split("_")[0]
        output_path = os.path.join(video_dir, f"{video_number}_CLIP_ViT.npy")

        if not (os.path.exists(output_path) and os.path.getsize(output_path) > 0):
            pending_tasks.append((video_path, output_path))

    total_pending = len(pending_tasks)
    print(f"⏩ Cần xử lý {total_pending}/{len(games)} videos (đã bỏ qua {len(games) - total_pending} video đã tồn tại)")

    if total_pending == 0:
        print("🎉 Tất cả video đã được trích xuất hoàn tất!")
        return

    # Khởi tạo Queue và Worker ngầm cho Writer
    write_queue = queue.Queue(maxsize=16)
    writer = threading.Thread(target=async_writer_thread, args=(write_queue,), daemon=True)
    writer.start()

    # Hàng đợi decode video song song (Buffer tối đa 8 video trong RAM để tránh tràn RAM)
    max_queue_size = 8
    decoded_queue = queue.Queue(maxsize=max_queue_size)
    task_queue = queue.Queue()

    for task in pending_tasks:
        task_queue.put(task)

    def decode_worker():
        while True:
            try:
                task = task_queue.get_nowait()
            except queue.Empty:
                break
            video_path, output_path = task
            frames = decode_video_ffmpeg(video_path, target_fps=1)
            decoded_queue.put((video_path, output_path, frames))
            task_queue.task_done()

    # Chạy num_workers threads decode đồng thời
    worker_threads = []
    for _ in range(args.num_workers):
        t = threading.Thread(target=decode_worker, daemon=True)
        t.start()
        worker_threads.append(t)

    def sentinel_feeder():
        for t in worker_threads:
            t.join()
        decoded_queue.put(None)

    threading.Thread(target=sentinel_feeder, daemon=True).start()

    # GPU Consumer Loop
    pbar = tqdm(total=total_pending, desc="⚡ GPU Processing (CLIP Features)", unit="video")
    while True:
        item = decoded_queue.get()
        if item is None:
            decoded_queue.task_done()
            break

        video_path, output_path, frames = item
        if len(frames) > 0:
            # 1. Trích xuất đặc trưng trên GPU siêu tốc
            features_np = extractor.extract_features(frames)

            # 2. Đưa vào hàng đợi ghi đĩa ngầm (không làm chậm GPU)
            write_queue.put((output_path, features_np))
        else:
            print(f"\n⚠️ Không decode được frames từ {os.path.basename(video_path)}, bỏ qua.")

        decoded_queue.task_done()
        pbar.update(1)

    pbar.close()

    # Đợi ghi đĩa hoàn tất
    print("💾 Đang hoàn tất lưu các file .npy vào đĩa...")
    write_queue.put(None)
    writer.join()

    print("\n🎉 HOÀN THÀNH TRÍCH XUẤT TOÀN BỘ DATASET!")


if __name__ == "__main__":
    main()
