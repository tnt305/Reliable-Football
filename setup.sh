#!/usr/bin/env bash

set -e

# ==========================================
# CONFIG
# ==========================================

PROJECT_DIR="/workspace/Viettel-Media-Fusion"

CONDA_DIR="/root/miniconda3"
ENV_NAME="soccernet"
PYTHON_VERSION="3.12"

MINICONDA_INSTALLER="/tmp/Miniconda3-latest-Linux-x86_64.sh"

# ==========================================
# HELPER
# ==========================================

print_header() {
    echo ""
    echo "=========================================="
    echo "$1"
    echo "=========================================="
}

# ==========================================
# START
# ==========================================

print_header "Viettel Media Fusion - Server Setup"

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
# STEP 2 - CHECK SYSTEM TOOLS
# ==========================================

print_header "STEP 2 - Checking system tools"

APT_UPDATED=0

apt_update_once() {
    if [ "$APT_UPDATED" -eq 0 ]; then
        echo "[INFO] Updating apt package list..."
        apt-get update
        APT_UPDATED=1
    fi
}

if ! command -v wget >/dev/null 2>&1; then
    echo "[INFO] wget not found. Installing..."
    apt_update_once
    apt-get install -y wget
else
    echo "[OK] wget found: $(command -v wget)"
fi

if ! command -v curl >/dev/null 2>&1; then
    echo "[INFO] curl not found. Installing..."
    apt_update_once
    apt-get install -y curl
else
    echo "[OK] curl found: $(command -v curl)"
fi

if ! command -v git >/dev/null 2>&1; then
    echo "[INFO] git not found. Installing..."
    apt_update_once
    apt-get install -y git
else
    echo "[OK] git found: $(command -v git)"
fi

if ! command -v build-essential >/dev/null 2>&1; then
    echo "[INFO] Installing build-essential..."
    apt_update_once
    apt-get install -y build-essential
fi

if ! command -v ffmpeg >/dev/null 2>&1; then
    echo "[INFO] ffmpeg not found. Installing..."
    apt_update_once
    apt-get install -y ffmpeg
else
    echo "[OK] ffmpeg found: $(command -v ffmpeg)"
fi

if ! command -v tmux >/dev/null 2>&1; then
    echo "[INFO] tmux not found. Installing..."
    apt_update_once
    apt-get install -y tmux
else
    echo "[OK] tmux found: $(command -v tmux)"
fi

# ==========================================
# STEP 3 - CHECK / INSTALL CONDA
# ==========================================

print_header "STEP 3 - Checking Conda"

if [ -f "$CONDA_DIR/etc/profile.d/conda.sh" ]; then
    echo "[OK] Miniconda found at:"
    echo "     $CONDA_DIR"

elif command -v conda >/dev/null 2>&1; then
    echo "[OK] Conda found:"
    command -v conda

else
    echo "[INFO] Conda not found."
    echo "[INFO] Installing Miniconda..."

    cd /tmp

    wget \
        --show-progress \
        https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
        -O "$MINICONDA_INSTALLER"

    bash "$MINICONDA_INSTALLER" \
        -b \
        -p "$CONDA_DIR"

    rm -f "$MINICONDA_INSTALLER"

    echo "[INFO] Initializing Conda..."

    "$CONDA_DIR/bin/conda" init bash

    echo "[OK] Miniconda installed successfully."
fi

# ==========================================
# STEP 4 - LOAD CONDA
# ==========================================

print_header "STEP 4 - Loading Conda"

if [ -f "$CONDA_DIR/etc/profile.d/conda.sh" ]; then
    source "$CONDA_DIR/etc/profile.d/conda.sh"
else
    echo "[ERROR] Cannot find:"
    echo "        $CONDA_DIR/etc/profile.d/conda.sh"
    exit 1
fi

echo "[OK] Conda loaded."
echo "[INFO] Conda version:"
conda --version

# ==========================================
# STEP 5 - ACCEPT CONDA TOS
# ==========================================

print_header "STEP 5 - Accepting Conda Terms of Service"

echo "[INFO] Checking/accepting Anaconda channel Terms of Service..."

conda tos accept \
    --override-channels \
    --channel https://repo.anaconda.com/pkgs/main \
    || true

conda tos accept \
    --override-channels \
    --channel https://repo.anaconda.com/pkgs/r \
    || true

echo "[OK] Conda Terms of Service handled."

# ==========================================
# STEP 6 - CREATE ENVIRONMENT
# ==========================================

print_header "STEP 6 - Checking Conda environment"

if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then

    echo "[OK] Environment '$ENV_NAME' already exists."

else

    echo "[INFO] Environment '$ENV_NAME' not found."
    echo "[INFO] Creating environment..."
    echo "[INFO] Python version: $PYTHON_VERSION"

    conda create \
        -n "$ENV_NAME" \
        "python=$PYTHON_VERSION" \
        -y

    echo "[OK] Environment '$ENV_NAME' created successfully."
fi

# ==========================================
# STEP 7 - ACTIVATE ENVIRONMENT
# ==========================================

print_header "STEP 7 - Activating environment"

conda activate "$ENV_NAME"

echo "[OK] Active environment:"
echo "     $CONDA_DEFAULT_ENV"

echo ""
echo "[INFO] Python version:"
python --version

echo ""
echo "[INFO] Python path:"
which python

# ==========================================
# STEP 8 - INSTALL UV
# ==========================================

print_header "STEP 8 - Checking UV"

# Add common uv installation paths
export PATH="/root/.local/bin:$PATH"

if command -v uv >/dev/null 2>&1; then

    echo "[OK] uv already installed."
    echo "[INFO] uv version:"
    uv --version

else

    echo "[INFO] uv not found."
    echo "[INFO] Installing uv..."

    curl -LsSf https://astral.sh/uv/install.sh | sh

    export PATH="/root/.local/bin:$PATH"

    if ! command -v uv >/dev/null 2>&1; then
        echo "[ERROR] uv installation failed."
        exit 1
    fi

    echo "[OK] uv installed successfully."
    echo "[INFO] uv version:"
    uv --version
fi

# ==========================================
# STEP 9 - CHECK PROJECT
# ==========================================

print_header "STEP 9 - Checking project directory"

if [ ! -d "$PROJECT_DIR" ]; then
    echo "[ERROR] Project directory does not exist:"
    echo "        $PROJECT_DIR"
    exit 1
fi

cd "$PROJECT_DIR"

echo "[OK] Project directory:"
pwd

# ==========================================
# STEP 10 - UPDATE PYTHON PACKAGING TOOLS
# ==========================================

print_header "STEP 10 - Updating Python packaging tools"

python -m pip install \
    --upgrade \
    pip \
    setuptools \
    wheel

echo "[OK] Python packaging tools updated."

# ==========================================
# STEP 11 - INSTALL PROJECT DEPENDENCIES
# ==========================================

print_header "STEP 11 - Installing project dependencies"

if [ -f "$PROJECT_DIR/requirements.txt" ]; then

    echo "[INFO] requirements.txt found."

    echo "[INFO] Installing dependencies with uv..."

    uv pip install \
        --python "$(which python)" \
        -r requirements.txt

    echo "[OK] requirements.txt installed successfully."

else

    echo "[WARNING] requirements.txt not found."
fi

# ==========================================
# STEP 11.5 - DETECT GPU & INSTALL COMPATIBLE PYTORCH (CUDA AUTO-DETECT)
# ==========================================

print_header "STEP 11.5 - GPU & PyTorch Setup"

if command -v nvidia-smi >/dev/null 2>&1; then
    CUDA_DRIVER_VER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1 || echo "unknown")
    GPU_NAME=$(nvidia-smi --query-gpu=gpu_name --format=csv,noheader | head -1 || echo "NVIDIA GPU")
    CUDA_MAX_RAW=$(nvidia-smi | grep -o "CUDA Version: [0-9]*\.[0-9]*" | head -1 | awk '{print $3}' || echo "")

    echo "[INFO] GPU detected: $GPU_NAME"
    echo "[INFO] Driver Version: $CUDA_DRIVER_VER"
    echo "[INFO] Max supported CUDA: $CUDA_MAX_RAW"

    # Tự động map sang index wheel phù hợp nhất của PyTorch
    # Tránh cài bản CUDA quá cao khiến driver báo 'driver too old'
    CUDA_TAG="cu121"
    if [ -n "$CUDA_MAX_RAW" ]; then
        CUDA_MAJOR=$(echo "$CUDA_MAX_RAW" | cut -d. -f1)
        CUDA_MINOR=$(echo "$CUDA_MAX_RAW" | cut -d. -f2)

        if [ "$CUDA_MAJOR" -ge 13 ]; then
            CUDA_TAG="cu124"  # cu124 tương thích tối đa và ổn định nhất
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

    echo "[INFO] Installing PyTorch with tag: $CUDA_TAG..."
    python -m pip install \
        torch \
        torchvision \
        --index-url "https://download.pytorch.org/whl/$CUDA_TAG"

    echo "[INFO] Verifying PyTorch GPU support..."
    python -c "import torch; assert torch.cuda.is_available(), 'CUDA is not available'; print('✅ PyTorch GPU Ready:', torch.cuda.get_device_name(0))" || {
        echo "[WARNING] PyTorch CUDA check failed with $CUDA_TAG. Retrying with cu121 fallback..."
        python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
        python -c "import torch; print('CUDA status:', torch.cuda.is_available())"
    }

    echo "[OK] PyTorch GPU setup verified."
else
    echo "[INFO] No NVIDIA GPU detected. Installing CPU version of PyTorch..."
    python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
fi

# ==========================================
# STEP 12 - INSTALL CURRENT PROJECT
# ==========================================

print_header "STEP 12 - Checking project package configuration"

if [ -f "$PROJECT_DIR/pyproject.toml" ]; then

    echo "[INFO] pyproject.toml found."
    echo "[INFO] Installing current project..."

    uv pip install \
        --python "$(which python)" \
        -e .

    echo "[OK] Project installed successfully."

elif [ -f "$PROJECT_DIR/setup.py" ]; then

    echo "[INFO] setup.py found."
    echo "[INFO] Installing current project..."

    uv pip install \
        --python "$(which python)" \
        -e .

    echo "[OK] Project installed successfully."

else

    echo "[INFO] No pyproject.toml or setup.py found."
    echo "[INFO] Skipping editable project installation."
fi

# ==========================================
# STEP 13 - FINAL CHECK
# ==========================================

print_header "STEP 13 - Final verification"

echo "Project directory:"
echo "  $PROJECT_DIR"

echo ""

echo "Conda:"
conda --version

echo ""

echo "Environment:"
echo "  $CONDA_DEFAULT_ENV"

echo ""

echo "Python:"
python --version

echo ""

echo "Python path:"
which python

echo ""

echo "Pip:"
python -m pip --version

echo ""

echo "UV:"
uv --version

echo ""

print_header "SETUP COMPLETED SUCCESSFULLY"

echo "To activate manually after SSH:"
echo ""
echo "  source $CONDA_DIR/etc/profile.d/conda.sh"
echo "  conda activate $ENV_NAME"
echo "  cd $PROJECT_DIR"
echo ""