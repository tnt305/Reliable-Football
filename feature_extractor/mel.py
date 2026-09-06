#!/usr/bin/env python3
# ==============================================================================
# GPU Log-Mel Spectrogram Extractor (Dùng PyTorch + Librosa, không cần torchaudio)
# ==============================================================================

import os
import sys
import subprocess
import argparse
import numpy as np
import torch
import librosa
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import warnings

warnings.filterwarnings("ignore")

# Cấu hình SoccerNet / AST
TARGET_SR = 16000
N_FFT = 400
HOP_LENGTH = 160
N_MELS = 128
LOG_OFFSET = 1e-10
MAX_LOG_VALUE = 7430.77
DB_RANGE = 80.0

# Sử dụng GPU RTX A4000
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Tạo Mel Filterbank bằng librosa và chuyển lên GPU
mel_basis = librosa.filters.mel(
    sr=TARGET_SR, n_fft=N_FFT, n_mels=N_MELS, htk=False, norm="slaney"
)
MEL_BASIS_GPU = torch.from_numpy(mel_basis).float().to(DEVICE)
WINDOW_GPU = torch.hann_window(N_FFT).to(DEVICE)


def extract_audio_pcm_pipe(video_path: str, target_sr: int = 16000) -> np.ndarray:
    """Đọc audio stream qua RAM bằng FFmpeg pipe"""
    command = [
        "ffmpeg",
        "-i",
        video_path,
        "-hide_banner",
        "-loglevel",
        "error",
        "-vn",
        "-acodec",
        "pcm_s16le",
        "-ar",
        str(target_sr),
        "-ac",
        "1",
        "-f",
        "s16le",
        "pipe:1",
    ]
    try:
        proc = subprocess.Popen(
            command, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL
        )
        raw_audio, _ = proc.communicate()
        if proc.returncode != 0 or len(raw_audio) == 0:
            return None
        return (
            np.frombuffer(raw_audio, dtype=np.int16).astype(np.float32)
            / 32768.0
        )
    except Exception:
        return None


def compute_log_mel_gpu(audio_np: np.ndarray) -> np.ndarray:
    """Tính toán Mel-Spectrogram trên GPU RTX A4000 bằng PyTorch STFT"""
    waveform = torch.from_numpy(audio_np).float().to(DEVICE)

    with torch.no_grad():
        # 1. STFT trên GPU
        stft = torch.stft(
            waveform,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH,
            win_length=N_FFT,
            window=WINDOW_GPU,
            center=True,
            pad_mode="reflect",
            normalized=False,
            onesided=True,
            return_complex=True,
        )

        # 2. Power Spectrogram (magnitude squared)
        spectrogram = stft.abs() ** 2.0  # [Freq_bins, Frames]

        # 3. Nhân ma trận với Mel Basis trên GPU
        mel_spec = torch.matmul(MEL_BASIS_GPU, spectrogram)  # [n_mels, Frames]
        mel_tensor = mel_spec.T  # [Frames, n_mels]

        # 4. Chuẩn hóa Log-Mel
        log_offset_t = torch.tensor(LOG_OFFSET, device=DEVICE)
        max_log_val_t = torch.tensor(MAX_LOG_VALUE, device=DEVICE)

        log_mel_tensor = 10.0 * (
            torch.log10(torch.maximum(log_offset_t, mel_tensor))
            - torch.log10(max_log_val_t)
        )
        max_val = torch.max(log_mel_tensor)
        normalized_log_mel = torch.maximum(log_mel_tensor, max_val - DB_RANGE)

    return normalized_log_mel.cpu().numpy()


def process_video(video_file_path: str) -> bool:
    video_filename = os.path.basename(video_file_path)
    video_number = video_filename.split("_")[0]
    vid_dir = os.path.dirname(video_file_path)
    
    # Định dạng audio1.npy và audio2.npy
    output_audio_path = os.path.join(vid_dir, f"audio{video_number}.npy")

    # Bỏ qua nếu đã xử lý
    if os.path.exists(output_audio_path) and os.path.getsize(output_audio_path) > 0:
        return True

    audio_np = extract_audio_pcm_pipe(video_file_path, TARGET_SR)

    # Video không có tiếng
    if audio_np is None or len(audio_np) == 0:
        zero_tensor = np.zeros((100, N_MELS), dtype=np.float32)
        np.save(output_audio_path, zero_tensor)
        return True

    # Pad audio nếu quá ngắn so với n_fft
    min_length = N_FFT + 1
    if len(audio_np) < min_length:
        audio_np = np.pad(audio_np, (0, min_length - len(audio_np)), mode="constant")

    # Tính toán trên GPU
    log_mel = compute_log_mel_gpu(audio_np)
    np.save(output_audio_path, log_mel)
    return True


def main():
    parser = argparse.ArgumentParser(description="Log-Mel Extractor tăng tốc GPU")
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="downloads/dataset",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=8,
        help="Số luồng đọc audio song song",
    )
    args = parser.parse_args()

    print("==================================================================")
    print(f"🚀 LOG-MEL EXTRACTOR (GPU ACCELERATED)")
    print(
        f"   🖥️  Thiết bị: {DEVICE} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})"
    )
    print(f"   📂 Dataset: {args.dataset_path}")
    print(f"   ⚡ Workers: {args.num_workers}")
    print("==================================================================")

    games = []
    for root, _, files in os.walk(args.dataset_path):
        for file in files:
            if file.endswith(".mkv"):
                games.append(os.path.join(root, file))

    games = sorted(games)
    print(f"🎯 Tìm thấy {len(games)} files .mkv")

    # Dùng đa luồng CPU đọc audio gối đầu cho GPU xử lý
    with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
        list(
            tqdm(
                executor.map(process_video, games),
                total=len(games),
                desc="⚡ Audio Processing",
                unit="video",
            )
        )

    print("\n🎉 HOÀN THÀNH XỬ LÝ TOÀN BỘ DATASET!")


if __name__ == "__main__":
    main()
