#!/usr/bin/env bash
# ==============================================================================
# Reliable-Football - Full End-to-End Setup & Feature Extraction Script
#
# Luồng mặc định (chạy trọn gói khi gọi `bash setup.sh`):
#   1. Cài đặt system tools (wget, curl, git, gcc, ffmpeg, tmux, libgl1, libsndfile1)
#   2. Cài đặt Miniconda + tạo env `soccernet` (Python 3.12) + `uv`
#   3. Cài PyTorch (tự động nhận diện CUDA/GPU) -> cài `requirements.txt`
#   4. Tải đầy đủ SoccerNet: Video 224p + Baidu Embeddings + Labels-v2.json (train/valid/test)
#      (+ dọn HF dataset cache sau khi tải xong để tiết kiệm ~258GB ổ đĩa)
#   5. Extract Audio Log-Mel (`audio1.npy`, `audio2.npy`)
#   6. Extract CLIP-ViT `ViFortune-AI/CLIP-ViT/best` (`1_CLIP_ViT.npy`, `2_CLIP_ViT.npy`)
#      với GPU NVDEC (`-hwaccel cuda`) + auto OOM fallback — KHÔNG merge vào Baidu
#
# Tùy chọn CLI:
#   --setup-only      Chỉ cài đặt môi trường & thư viện (bỏ qua download & extract)
#   --skip-download   Bỏ qua bước tải Video + Baidu
#   --skip-extract    Bỏ qua bước extract Audio & CLIP-ViT
#
# Biến môi trường tùy chọn:
#   HF_TOKEN=...                 Token HuggingFace (nếu cần)
#   DOWNLOAD_WORKERS=16          Số luồng tải HuggingFace (mặc định: 16)
#   AUDIO_WORKERS=<auto>         Số luồng extract Audio (mặc định: auto theo số CPU core, tối đa 8)
#   CLIP_WORKERS=<auto>          Số luồng decode video cho CLIP (mặc định: auto theo CPU core, 2..4)
#   CLIP_BATCH_SIZE=<auto>       Batch size GPU cho CLIP ViT-L/14 (mặc định: auto theo VRAM trống)
#   CLEAN_HF_DATASET_CACHE=1     Xóa cache dataset HF sau khi tải xong (1=bật, 0=tắt)
# ==============================================================================

set -eo pipefail

# ==========================================
# CONFIG
# ==========================================

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONDA_DIR="/root/miniconda3"
ENV_NAME="soccernet"
PYTHON_VERSION="3.12"
MINICONDA_INSTALLER="/tmp/Miniconda3-latest-Linux-x86_64.sh"

DATASET_DIR="$PROJECT_DIR/downloads/dataset"
LOG_DIR="/tmp/reliable_football_logs"
mkdir -p "$LOG_DIR"

HF_TOKEN="${HF_TOKEN:-}"
DOWNLOAD_WORKERS="${DOWNLOAD_WORKERS:-16}"
CLEAN_HF_DATASET_CACHE="${CLEAN_HF_DATASET_CACHE:-1}"

SKIP_DOWNLOAD=0
SKIP_EXTRACT=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --setup-only)
            SKIP_DOWNLOAD=1
            SKIP_EXTRACT=1
            shift
            ;;
        --skip-download)
            SKIP_DOWNLOAD=1
            shift
            ;;
        --skip-extract)
            SKIP_EXTRACT=1
            shift
            ;;
        -h|--help)
            sed -n '2,30p' "$0"
            exit 0
            ;;
        *)
            echo "[WARNING] Unknown option ignored: $1"
            shift
            ;;
    esac
done

# ==========================================
# HELPERS
# ==========================================

print_header() {
    echo ""
    echo "=========================================="
    echo "$1"
    echo "=========================================="
}

check_gpu_free_mem_mb() {
    if command -v nvidia-smi >/dev/null 2>&1; then
        nvidia-smi --query-gpu=memory.free --format=csv,nounits,noheader | head -1 | tr -d '[:space:]'
    else
        echo "0"
    fi
}

get_initial_clip_batch_size() {
    local gpu_mem="${1:-0}"
    if [ "$gpu_mem" -gt 28000 ]; then
        echo "256"
    elif [ "$gpu_mem" -gt 18000 ]; then
        echo "128"
    elif [ "$gpu_mem" -gt 10000 ]; then
        echo "64"
    else
        echo "32"
    fi
}

is_oom_error() {
    local log_file="$1"
    grep -qiE "cuda out of memory|out of memory|cublas_status_alloc_failed|ENOMEM" "$log_file"
}

run_clip_with_oom_fallback() {
    local initial_batch="$1"
    local num_workers="$2"
    local batch_size="$initial_batch"
    local attempt=1

    while [ "$batch_size" -ge 8 ]; do
        local log_file="$LOG_DIR/clip_best_batch${batch_size}.log"
        echo "[INFO] CLIP-ViT (best) attempt $attempt: batch_size=$batch_size, num_workers=$num_workers"

        set +e
        bash "$PROJECT_DIR/feature_extractor/run_clip.sh" \
            --subfolder best \
            --batch_size "$batch_size" \
            --num_workers "$num_workers" \
            2>&1 | tee "$log_file"
        local exit_code=${PIPESTATUS[0]}
        set -e

        if [ "$exit_code" -eq 0 ]; then
            echo "[OK] CLIP-ViT (best) extraction completed successfully."
            return 0
        fi

        # Dọn các file 0-byte nếu bị cắt ngang giữa chừng (giữ lại file hoàn chỉnh để resume)
        find "$DATASET_DIR" -type f -name "*_CLIP_ViT.npy" -size 0 -delete 2>/dev/null || true

        if is_oom_error "$log_file"; then
            batch_size=$((batch_size / 2))
            num_workers=$((num_workers > 2 ? num_workers - 1 : 1))
            attempt=$((attempt + 1))
            if [ "$batch_size" -ge 8 ]; then
                echo "[WARNING] OOM detected! Retrying with batch_size=$batch_size, num_workers=$num_workers ..."
                sleep 2
            fi
        else
            echo "[ERROR] CLIP-ViT extraction failed (non-OOM). Check log: $log_file"
            return "$exit_code"
        fi
    done

    echo "[ERROR] CLIP-ViT extraction exhausted all OOM fallback attempts (min batch_size=8)."
    return 1
}

# ==========================================
# START
# ==========================================

print_header "Reliable-Football - Full Server Setup & Feature Pipeline"
echo "[INFO] Project directory : $PROJECT_DIR"
echo "[INFO] Dataset directory : $DATASET_DIR"
echo "[INFO] CPU cores         : $(nproc)"

# ==========================================
# STEP 1 - CHECK ROOT
# ==========================================

print_header "STEP 1 - Checking permissions"

if [ "$(id -u)" -ne 0 ]; then
    echo "[ERROR] Please run this script as root."
    exit 1
fi
echo "[OK] Running as root."

# ==========================================
# STEP 2 - CHECK / INSTALL SYSTEM TOOLS
# ==========================================

print_header "STEP 2 - Checking system tools"

APT_UPDATED=0
apt_update_once() {
    if [ "$APT_UPDATED" -eq 0 ]; then
        echo "[INFO] Updating apt package list..."
        apt-get update -y
        APT_UPDATED=1
    fi
}

install_apt_pkg_if_missing() {
    local check_cmd="$1"
    local pkg_name="$2"
    if [ -n "$check_cmd" ] && command -v "$check_cmd" >/dev/null 2>&1; then
        echo "[OK] $pkg_name found: $(command -v "$check_cmd")"
    elif dpkg -s "$pkg_name" >/dev/null 2>&1; then
        echo "[OK] $pkg_name already installed."
    else
        echo "[INFO] Installing $pkg_name..."
        apt_update_once
        DEBIAN_FRONTEND=noninteractive apt-get install -y "$pkg_name"
    fi
}

install_apt_pkg_if_missing "wget" "wget"
install_apt_pkg_if_missing "curl" "curl"
install_apt_pkg_if_missing "git" "git"
install_apt_pkg_if_missing "gcc" "build-essential"
install_apt_pkg_if_missing "ffmpeg" "ffmpeg"
install_apt_pkg_if_missing "tmux" "tmux"
install_apt_pkg_if_missing "" "libgl1"
install_apt_pkg_if_missing "" "libglib2.0-0"
install_apt_pkg_if_missing "" "libsndfile1"

# ==========================================
# STEP 3 - CHECK / INSTALL CONDA
# ==========================================

print_header "STEP 3 - Checking Conda"

if [ -f "$CONDA_DIR/etc/profile.d/conda.sh" ]; then
    echo "[OK] Miniconda found at: $CONDA_DIR"
elif command -v conda >/dev/null 2>&1; then
    CONDA_DIR="$(conda info --base)"
    echo "[OK] Conda found at: $CONDA_DIR"
else
    echo "[INFO] Conda not found. Installing Miniconda..."
    wget --show-progress \
        https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
        -O "$MINICONDA_INSTALLER"
    bash "$MINICONDA_INSTALLER" -b -p "$CONDA_DIR"
    rm -f "$MINICONDA_INSTALLER"
    "$CONDA_DIR/bin/conda" init bash
    echo "[OK] Miniconda installed successfully."
fi

# ==========================================
# STEP 4 - LOAD CONDA & ACCEPT TOS
# ==========================================

print_header "STEP 4 - Loading Conda"

if [ -f "$CONDA_DIR/etc/profile.d/conda.sh" ]; then
    # shellcheck disable=SC1091
    source "$CONDA_DIR/etc/profile.d/conda.sh"
else
    echo "[ERROR] Cannot find: $CONDA_DIR/etc/profile.d/conda.sh"
    exit 1
fi

echo "[OK] Conda loaded: $(conda --version)"

conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main 2>/dev/null || true
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r 2>/dev/null || true

# ==========================================
# STEP 5 - CREATE & ACTIVATE ENVIRONMENT
# ==========================================

print_header "STEP 5 - Conda environment ($ENV_NAME)"

if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    echo "[OK] Environment '$ENV_NAME' already exists."
else
    echo "[INFO] Creating environment '$ENV_NAME' (Python $PYTHON_VERSION)..."
    conda create -n "$ENV_NAME" "python=$PYTHON_VERSION" -y
    echo "[OK] Environment '$ENV_NAME' created."
fi

conda activate "$ENV_NAME"
PYTHON_BIN="$(which python)"
echo "[OK] Active env: $CONDA_DEFAULT_ENV | Python: $($PYTHON_BIN --version) ($PYTHON_BIN)"

# ==========================================
# STEP 6 - INSTALL UV & PACKAGING TOOLS
# ==========================================

print_header "STEP 6 - Checking UV & pip"

export PATH="/root/.local/bin:$HOME/.local/bin:$PATH"

if ! command -v uv >/dev/null 2>&1; then
    echo "[INFO] Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="/root/.local/bin:$HOME/.local/bin:$PATH"
fi

if ! command -v uv >/dev/null 2>&1; then
    echo "[ERROR] uv installation failed."
    exit 1
fi
echo "[OK] uv ready: $(uv --version)"

"$PYTHON_BIN" -m pip install --upgrade pip setuptools wheel

# ==========================================
# STEP 7 - GPU & PYTORCH SETUP (TRƯỚC requirements.txt để tránh bị cài đè CPU torch)
# ==========================================

print_header "STEP 7 - GPU & PyTorch Setup"

cd "$PROJECT_DIR"

if command -v nvidia-smi >/dev/null 2>&1; then
    CUDA_DRIVER_VER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1 || echo "unknown")
    GPU_NAME=$(nvidia-smi --query-gpu=gpu_name --format=csv,noheader | head -1 || echo "NVIDIA GPU")
    CUDA_MAX_RAW=$(nvidia-smi | grep -o "CUDA Version: [0-9]*\.[0-9]*" | head -1 | awk '{print $3}' || echo "")

    echo "[INFO] GPU detected     : $GPU_NAME"
    echo "[INFO] Driver Version   : $CUDA_DRIVER_VER"
    echo "[INFO] Max CUDA Support : ${CUDA_MAX_RAW:-unknown}"

    CUDA_TAG="cu124"
    if [ -n "$CUDA_MAX_RAW" ]; then
        CUDA_MAJOR=$(echo "$CUDA_MAX_RAW" | cut -d. -f1)
        CUDA_MINOR=$(echo "$CUDA_MAX_RAW" | cut -d. -f2)

        if [ "$CUDA_MAJOR" -ge 13 ]; then
            CUDA_TAG="cu124"
        elif [ "$CUDA_MAJOR" -eq 12 ]; then
            if [ "$CUDA_MINOR" -ge 4 ]; then
                CUDA_TAG="cu124"
            else
                CUDA_TAG="cu121"
            fi
        elif [ "$CUDA_MAJOR" -eq 11 ]; then
            CUDA_TAG="cu118"
        fi
    fi

    echo "[INFO] Installing PyTorch ($CUDA_TAG)..."
    uv pip install --python "$PYTHON_BIN" \
        torch torchvision \
        --index-url "https://download.pytorch.org/whl/$CUDA_TAG"

    "$PYTHON_BIN" -c "import torch; assert torch.cuda.is_available(), 'CUDA unavailable'; print('✅ PyTorch CUDA:', torch.__version__, '|', torch.cuda.get_device_name(0))" || {
        echo "[WARNING] PyTorch CUDA check failed with $CUDA_TAG. Retrying with cu121 fallback..."
        uv pip install --python "$PYTHON_BIN" --reinstall \
            torch torchvision \
            --index-url https://download.pytorch.org/whl/cu121
        "$PYTHON_BIN" -c "import torch; assert torch.cuda.is_available(), 'CUDA still unavailable'; print('✅ PyTorch CUDA:', torch.__version__, '|', torch.cuda.get_device_name(0))"
    }
else
    echo "[WARNING] No NVIDIA GPU detected. Installing CPU PyTorch..."
    uv pip install --python "$PYTHON_BIN" \
        torch torchvision \
        --index-url https://download.pytorch.org/whl/cpu
fi

# ==========================================
# STEP 8 - INSTALL REQUIREMENTS.TXT
# ==========================================

print_header "STEP 8 - Installing requirements.txt"

if [ -f "$PROJECT_DIR/requirements.txt" ]; then
    uv pip install --python "$PYTHON_BIN" -r "$PROJECT_DIR/requirements.txt"
    echo "[OK] requirements.txt installed."
else
    echo "[ERROR] requirements.txt not found in $PROJECT_DIR"
    exit 1
fi

# Verify các thư viện cốt lõi & CUDA sau khi cài requirements.txt
echo "[INFO] Verifying core imports..."
"$PYTHON_BIN" - <<'PYEOF'
import torch, transformers, librosa, cv2, huggingface_hub, safetensors, numpy, SoccerNet
print(f"  torch={torch.__version__} | cuda={torch.cuda.is_available()}")
print(f"  transformers={transformers.__version__} | huggingface_hub={huggingface_hub.__version__}")
print(f"  librosa={librosa.__version__} | cv2={cv2.__version__} | numpy={numpy.__version__}")
PYEOF

# ==========================================
# STEP 9 - DOWNLOAD DATASET (VIDEO 224p + BAIDU + LABELS)
# ==========================================

if [ "$SKIP_DOWNLOAD" -eq 0 ]; then
    print_header "STEP 9 - Downloading SoccerNet (Video 224p + Baidu + Labels)"
    echo "[INFO] Splits  : train, valid, test (500 games = 1000 halves)"
    echo "[INFO] Output  : $DATASET_DIR"
    echo "[INFO] Workers : $DOWNLOAD_WORKERS"

    DL_ARGS=(
        --split trainvalidtest
        --output_dir "$DATASET_DIR"
        --download_videos
        --resolution 224p
        --max_workers "$DOWNLOAD_WORKERS"
    )
    if [ -n "$HF_TOKEN" ]; then
        DL_ARGS+=(--token "$HF_TOKEN")
    fi

    "$PYTHON_BIN" "$PROJECT_DIR/downloads/download_soccernet.py" "${DL_ARGS[@]}"

    MKV_COUNT=$(find "$DATASET_DIR" -type f -name "*_224p.mkv" | wc -l)
    BAIDU_COUNT=$(find "$DATASET_DIR" -type f -name "*_baidu_soccer_embeddings.npy" | wc -l)
    echo "[OK] Download summary: MKV=$MKV_COUNT | Baidu=$BAIDU_COUNT"

    if [ "$MKV_COUNT" -eq 0 ]; then
        echo "[ERROR] No .mkv files downloaded. Aborting before extraction."
        exit 1
    fi

    if [ "$CLEAN_HF_DATASET_CACHE" -eq 1 ]; then
        echo "[INFO] Cleaning redundant HuggingFace dataset cache (~258GB) to save disk space..."
        rm -rf "$HOME/.cache/huggingface/hub/datasets--OpenSportsLab--SoccerNet-ActionSpotting-Features" \
               "$HOME/.cache/huggingface/hub/datasets--tartotarto--soccernet-500-videos"
        echo "[OK] HF dataset cache cleaned. Disk free:"
        df -h "$PROJECT_DIR" | tail -1
    fi
else
    print_header "STEP 9 - Skipping dataset download (--skip-download / --setup-only)"
fi

# ==========================================
# STEP 10 - EXTRACT AUDIO FEATURES (LOG-MEL)
# ==========================================

if [ "$SKIP_EXTRACT" -eq 0 ]; then
    CPU_CORES=$(nproc)

    if [ -z "${AUDIO_WORKERS:-}" ]; then
        if [ "$CPU_CORES" -ge 8 ]; then
            AUDIO_WORKERS=8
        else
            AUDIO_WORKERS="$CPU_CORES"
        fi
    fi

    print_header "STEP 10 - Extracting Audio Features (Log-Mel -> audio1.npy, audio2.npy)"
    echo "[INFO] AUDIO_WORKERS=$AUDIO_WORKERS (CPU cores: $CPU_CORES)"

    cd "$PROJECT_DIR"
    bash "$PROJECT_DIR/feature_extractor/run_mel.sh" \
        --num_workers "$AUDIO_WORKERS" \
        2>&1 | tee "$LOG_DIR/mel_extract.log"

    AUDIO_COUNT=$(find "$DATASET_DIR" -type f -name "audio[1-2].npy" | wc -l)
    echo "[OK] Audio extraction complete: $AUDIO_COUNT audio*.npy files."

    # ==========================================
    # STEP 11 - EXTRACT CLIP-ViT BEST (KHÔNG MERGE)
    # ==========================================

    if [ -z "${CLIP_WORKERS:-}" ]; then
        if [ "$CPU_CORES" -ge 8 ]; then
            CLIP_WORKERS=4
        elif [ "$CPU_CORES" -ge 4 ]; then
            CLIP_WORKERS=3
        else
            CLIP_WORKERS=2
        fi
    fi

    GPU_FREE_MB=$(check_gpu_free_mem_mb)
    if [ -z "${CLIP_BATCH_SIZE:-}" ]; then
        CLIP_BATCH_SIZE=$(get_initial_clip_batch_size "$GPU_FREE_MB")
    fi

    print_header "STEP 11 - Extracting CLIP-ViT 'best' (ViT-L/14 -> 768D, NO MERGE)"
    echo "[INFO] Model           : ViFortune-AI/CLIP-ViT (subfolder=best)"
    echo "[INFO] GPU Free VRAM   : ${GPU_FREE_MB} MB"
    echo "[INFO] CLIP_BATCH_SIZE : $CLIP_BATCH_SIZE (auto OOM fallback enabled)"
    echo "[INFO] CLIP_WORKERS    : $CLIP_WORKERS (GPU NVDEC -hwaccel cuda enabled)"
    echo "[INFO] Merge into Baidu: DISABLED (theo yêu cầu)"

    cd "$PROJECT_DIR"
    run_clip_with_oom_fallback "$CLIP_BATCH_SIZE" "$CLIP_WORKERS"

    CLIP_COUNT=$(find "$DATASET_DIR" -type f -name "*_CLIP_ViT.npy" | wc -l)
    echo "[OK] CLIP-ViT (best) extraction complete: $CLIP_COUNT *_CLIP_ViT.npy files."
else
    print_header "STEP 10-11 - Skipping feature extraction (--skip-extract / --setup-only)"
fi

# ==========================================
# FINAL SUMMARY
# ==========================================

print_header "ALL STEPS COMPLETED SUCCESSFULLY"
echo "Project dir : $PROJECT_DIR"
echo "Conda env   : $ENV_NAME ($PYTHON_BIN)"
if [ -d "$DATASET_DIR" ]; then
    echo "Dataset stats ($DATASET_DIR):"
    echo "  - Video (.mkv)   : $(find "$DATASET_DIR" -type f -name "*_224p.mkv" | wc -l)"
    echo "  - Baidu (.npy)   : $(find "$DATASET_DIR" -type f -name "*_baidu_soccer_embeddings.npy" | wc -l)"
    echo "  - Audio (.npy)   : $(find "$DATASET_DIR" -type f -name "audio[1-2].npy" | wc -l)"
    echo "  - CLIP-ViT (.npy): $(find "$DATASET_DIR" -type f -name "*_CLIP_ViT.npy" | wc -l) (chưa merge)"
fi
echo ""
echo "To activate conda manually in a new shell:"
echo "  source $CONDA_DIR/etc/profile.d/conda.sh && conda activate $ENV_NAME && cd $PROJECT_DIR"
echo ""
