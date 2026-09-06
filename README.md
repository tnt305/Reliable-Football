# ⚽ Reliable-Football (ActionAware)

**English** | [Tiếng Việt](README_vi.md)

> **Space-Time Attention Transformer with KnowledegeBank for SoccerNet Action Spotting**  
> A high-accuracy multi-modal (Audio-Visual) action spotting system benchmarked on the **SoccerNet v2 (17 Classes)** challenge.

---

## 📌 Table of Contents
- [1. Overview & Key Highlights](#1-overview--key-highlights)
- [2. Repository Structure](#2-repository-structure)
- [3. Environment Setup](#3-environment-setup)
- [4. End-to-End Execution Guide via Shell Scripts](#4-end-to-end-execution-guide-via-shell-scripts)
  - [4.1. 1-Click Master Pipeline](#41-1-click-master-pipeline)
  - [4.2. Running Individual Stages](#42-running-individual-stages)
- [5. Core Architectural Components](#5-core-architectural-components)
  - [5.1. Base Architecture: MediaFusion](#51-base-architecture-mediafusion)
  - [5.2. Dynamic RL Ensemble (PPO Agent)](#52-dynamic-rl-ensemble-ppo-agent)
  - [5.3. Parallel Per-Class Post-Processing Tuning (17 CPU Cores)](#53-parallel-per-class-post-processing-tuning-17-cpu-cores)
- [6. Configuration Guide (`configs/MediaFusion.yaml`)](#6-configuration-guide-configsmediafusionyaml)
- [7. Evaluation & Submission Packaging](#7-evaluation--submission-packaging)
- [8. License & Acknowledgments](#8-license--acknowledgments)

---

## 1. Overview & Key Highlights

**Reliable-Football (ActionAware)** delivers an end-to-end deep learning framework designed to solve the temporal action spotting problem in professional soccer broadcasts:
1. **Multi-Modal MediaFusion Architecture**: Seamlessly fuses 6 pre-extracted visual streams (Baidu embeddings, 9344-d) with acoustic features (Log-Mel spectrograms processed through VGGish, 128-d).
2. **Uncertainty-Aware Displacement Regression**: Employs Gaussian Log-Likelihood Loss to predict temporal offsets from action centers alongside calibrated uncertainty estimates.
3. **Context-Aware Dynamic RL Ensemble**: Leverages a Proximal Policy Optimization (PPO) agent trained on a balanced tri-tier dataset (`TriTierBalancedSampler`: 40% Event, 40% Hard Negative, 20% Clean Background) to dynamically predict softmax weighting vectors across multiple model checkpoints per time-step.
4. **17-Core Parallel Per-Class Optimization**: Deconstructs global Soft-NMS into 17 independent search tasks across CPU cores to find optimal $(W_c, \Theta_c, \text{Decay}_c)$ triplets, dramatically elevating the critical **Tight a_mAP** metric without degrading Loose a_mAP.

---

## 2. Repository Structure

```text
ActionAware/
├── run_pipeline.sh              # 🚀 Master pipeline runner script (All-in-One automation)
├── hadh_main.py                 # Primary entry point for training & standalone testing
├── mediafusion_model.py         # MediaFusion deep neural network definition
├── mediafusion_base.py          # Space-Time Attention layers & Vector Gated Shift
├── mediafusion_dataset.py       # SoccerNet DataLoader (Frames, Clips, Audio, Baidu)
├── mediafusion_train.py         # Training loop, Soft-NMS, Loss functions & metrics
├── mediafusion_train_rl_ensemble.py # Baseline RL ensemble training environment
├── eval_checkpoint.sh           # Safe background evaluation tool (eval without halting training)
├── class_embed.npy              # 17-class text feature embeddings from CLIP
├── configs/                     # YAML configuration directory
│   └── MediaFusion.yaml         # Official MediaFusion training & test parameters
├── rl/                          # Reinforcement Learning & Optimization Subsystem
│   ├── run_pipeline_v2.sh       # Full-flow RL script (Train -> Tune -> Test)
│   ├── run_seed.sh              # Multi-seed validation script for robustness testing
│   ├── train_rl.py              # PPO Agent trainer for dynamic checkpoint weighting
│   ├── tune_thresholds.py       # Parallel grid search across 17 CPU cores for NMS tuning
│   ├── infer_rl.py              # Sliding-window inference with RL weights & evaluation
│   ├── rl_env.py                # Gymnasium environment & TriTierBalancedSampler
│   └── best_params_v2_34_41_47.yaml # Optimized per-class parameters for top checkpoints
├── downloads/                   # Automated dataset download utilities
│   ├── download_full.sh         # Fast downloader for Baidu features + Labels
│   ├── download_soccernet.py    # Multi-threaded Hugging Face dataset downloader
│   └── download_videos.sh       # Optional match video downloader (224p/720p)
└── feature_extractor/           # Custom feature extractors (CLIP, Mel, Grounding DINO)
```

---

## 3. Environment Setup

The repository is built for Linux environments with CUDA acceleration:

```bash
# Activate existing conda environment
conda activate soccernet

# Or install dependencies manually:
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu118
pip install SoccerNet stable-baselines3 gymnasium tqdm pyyaml pandas scikit-learn
```

---

## 4. End-to-End Execution Guide via Shell Scripts

The system includes a master bash script **`./run_pipeline.sh`** at the project root for unified pipeline management.

### 4.1. 1-Click Master Pipeline
To run the entire workflow from downloading data, training the base model, optimizing the RL ensemble, and generating final test metrics:

```bash
cd /workspace/ActionAware

# Grant executable permissions
chmod +x run_pipeline.sh eval_checkpoint.sh rl/*.sh downloads/*.sh

# Run all stages sequentially
./run_pipeline.sh all
```

---

### 4.2. Running Individual Stages

You can also run each stage independently as needed:

#### Stage 1: Download SoccerNet Features & Labels
Downloads Baidu embeddings and `Labels-v2.json` for `train`, `valid`, and `test` splits:
```bash
./run_pipeline.sh download
```
*(Files are stored under `downloads/dataset/`)*

#### Stage 2: Train Base MediaFusion Model
Trains the multi-modal transformer using parameters defined in `configs/MediaFusion.yaml`:
```bash
./run_pipeline.sh train MediaFusion
```
*(Checkpoints are automatically saved to `models/ASmodels/MediaFusion/`)*

#### Stage 3: Fast Standalone Checkpoint Evaluation
Evaluates any model checkpoint without interrupting ongoing training processes:
```bash
# Evaluate best checkpoint on the test split
./run_pipeline.sh eval MediaFusion test

# Or evaluate a specific checkpoint directly via eval_checkpoint.sh:
./eval_checkpoint.sh MediaFusion test models/ASmodels/MediaFusion/model_47.pth.tar
```

#### Stage 4: Run RL Dynamic Ensemble & Per-Class Post-Processing
When you have multiple converged checkpoints (e.g., Epochs 34, 41, 47), run the RL pipeline:

```bash
# Run complete RL pipeline (Train PPO Agent -> 17-Core Grid Search -> Test Evaluation)
./run_pipeline.sh rl-all
```

Or execute granular sub-steps:
```bash
# 4a. Train PPO Agent to learn adaptive fusion weights
./run_pipeline.sh rl-train

# 4b. Perform parallel grid search over 17 CPU cores to optimize NMS parameters
./run_pipeline.sh rl-tune

# 4c. Run sliding-window inference on Test split using tuned parameters
./run_pipeline.sh rl-test
```

#### Stage 5: Multi-Seed Robustness Validation
To verify stability across different random initializations:
```bash
bash rl/run_seed.sh 42 123 2026
```

---

## 5. Core Architectural Components

### 5.1. Base Architecture: MediaFusion
- **Visual Branch**: Processes 6 Baidu visual embedding streams via `VectorGatedShift` modules that learn local temporal displacements.
- **Audio Branch**: Extracts Log-Mel spectrograms passed through a pre-trained VGGish backbone.
- **Space-Time Transformer**: 8-layer Transformer Encoder with hierarchical temporal and modality attention, paired with a 4-layer Transformer Decoder querying semantic class embeddings (`class_embed.npy`).

### 5.2. Dynamic RL Ensemble (PPO Agent)
- Replaces static uniform averaging ($1/M \sum P_m$) with dynamic weighting $w_t \in \mathbb{R}^M$.
- Observation vector of size $(18 \times M + 17)$ captures model confidence, historical class distributions, and uncertainty metrics.
- **Tri-Tier Sampling** ensures robust training across sparse positive events and challenging hard negative scenes.

### 5.3. Parallel Per-Class Post-Processing Tuning (17 CPU Cores)
- Decomposes the global $a\_mAP$ optimization into 17 independent sub-problems.
- Search space per class: 12 Thresholds $\times$ 10 Window sizes $\times$ 2 Decay kernels (`pow2`, `gaussian`) = 240 combinations.
- By caching raw model probabilities from the GPU once, evaluating 4,080 combinations across 17 CPU cores completes in **~30–60 seconds**.

---

## 6. Configuration Guide (`configs/MediaFusion.yaml`)

Key parameters in `configs/MediaFusion.yaml`:

```yaml
# Temporal Chunking & Resolution
chunk_size: 50           # Temporal window length (50 seconds)
outputrate: 2            # Predictions per second -> 100 frames / chunk
rC: 2                    # Radius for classification target window
rD: 3                    # Radius for displacement target window

# Input Streams
audio: true              # Enable Log-Mel + VGGish acoustic branch
baidu: true              # Enable Baidu 6-stream visual branch

# Loss Configuration
wC: 100                  # Weight for classification loss
wD: 1                    # Weight for displacement regression loss
focal: true              # Enable Focal Loss for class imbalance

# Training Parameters
BS: 16                   # Batch size
LR: 0.00005              # Base learning rate
max_epochs: 53           # Maximum training epochs
```

---

## 7. Evaluation & Submission Packaging

Upon evaluation completion, the framework:
1. Generates standard SoccerNet detection JSON files per match.
2. Packages predictions into the official submission archive: `results_spotting_test.zip`.
3. Calls the official SoccerNet Evaluation API to output a full performance breakdown:
   - **Loose a_mAP** ($\Delta \in [5, 60]$ seconds)
   - **Tight a_mAP** ($\Delta \in [1, 5]$ seconds)
   - Class-by-class Average Precision (AP) for all 17 classes.
4. Stores machine-readable summary metrics in:
   `models/RL_agents/.../eval_test/metrics_summary.json`

---

## 8. License & Acknowledgments
This repository is developed for and benchmarked against the [SoccerNet](https://www.soccer-net.org/) Action Spotting challenge dataset. Please adhere to the official SoccerNet data terms and conditions when utilizing this codebase.
