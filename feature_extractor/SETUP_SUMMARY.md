# Summary: Feature Extraction with OOM Fallback & Auto-Cleanup

## What Was Created

### 1. Master Script: `run_all_with_fallback.sh`
**Location**: `/workspace/ActionAware/feature_extractor/run_all_with_fallback.sh`

**Features**:
- ✅ GPU memory auto-detection (uses `nvidia-smi`)
- ✅ Intelligent batch size selection based on available VRAM
- ✅ Runs CLIP extraction first, then MEL extraction sequentially
- ✅ OOM detection and automatic cleanup of created `.npy` files
- ✅ Progressive fallback: 256 → 128 → 64 → 32
- ✅ Detailed logging to `/tmp/feature_extractor_logs/`
- ✅ Color-coded console output (info, success, warnings, errors)

### 2. Updated Scripts
**Modified `run_clip.sh` and `run_mel.sh`**:
- ✅ Added support for `--batch_size` parameter
- ✅ Added support for `--num_workers` parameter
- ✅ Backward compatible (works with or without parameters)

## How It Works

### Process Flow

```
START
  ↓
Check GPU Memory
  ↓
Calculate Initial Batch Size
  ├─ GPU > 25GB  → 256 / 8 workers
  ├─ GPU > 15GB  → 128 / 4 workers
  ├─ GPU > 8GB   → 64 / 2 workers
  └─ GPU ≤ 8GB   → 32 / 1 worker
  ↓
Run CLIP Extraction (Attempt N)
  ├─ Success? → Continue
  ├─ OOM? → Delete created .npy files → Reduce batch size → Retry
  └─ Other error? → Exit
  ↓
Run MEL Extraction (Attempt N)
  ├─ Success? → Continue
  ├─ OOM? → Delete created .npy files → Reduce batch size → Retry
  └─ Other error? → Exit
  ↓
END (Success or all retries exhausted)
```

### File Cleanup Logic

**What gets deleted on OOM:**

CLIP extraction creates:
```
{video_number}_CLIP_ViT.npy  (e.g., 1_CLIP_ViT.npy, 2_CLIP_ViT.npy)
```

MEL extraction creates:
```
audio1.npy, audio2.npy           (main audio features)
{1,2}_AST.npy                    (AST model features)
```

**Cleanup mechanism:**
1. Create timestamp file before running: `/tmp/clip_before_$$.tmp`
2. Run extraction script
3. If OOM detected:
   - Find all `.npy` files modified **after** timestamp
   - Delete only those files (preserves previous successful runs)
   - Reduce batch size and retry

**Example:**
```bash
# First run, attempt 1 (batch_size=256)
# OOM detected, cleanup creates:
# ❌ /dataset/match1/1_CLIP_ViT.npy (deleted)
# ❌ /dataset/match1/2_CLIP_ViT.npy (deleted)
# ❌ /dataset/match2/1_CLIP_ViT.npy (deleted)
# ...

# Second run, attempt 2 (batch_size=128)
# Now has more GPU memory available
# ✅ Succeeds and writes all files
```

## Usage

### Run Everything

```bash
cd /workspace/ActionAware/feature_extractor
./run_all_with_fallback.sh
```

### Monitor Progress

```bash
# Terminal 1: Watch CLIP logs
tail -f /tmp/feature_extractor_logs/clip_batch*.log

# Terminal 2: Watch MEL logs
tail -f /tmp/feature_extractor_logs/mel_batch*.log

# Terminal 3: Monitor GPU
watch -n 1 nvidia-smi
```

### Manual Run Specific Script

```bash
# Run CLIP only with custom batch size
./run_clip.sh --batch_size 128 --num_workers 4

# Run MEL only
./run_mel.sh --num_workers 4
```

## Key Advantages

| Feature | Benefit |
|---------|---------|
| **GPU Memory Auto-Detect** | No manual configuration needed |
| **OOM Handling** | Script survives GPU memory exhaustion |
| **Auto-Cleanup** | No corrupted partial files left behind |
| **Progressive Fallback** | Tries multiple configurations automatically |
| **File Tracking** | Only deletes newly created files, preserves old data |
| **Detailed Logging** | Easy debugging with timestamped logs |
| **Color Output** | Easy to spot errors and progress |

## Log Files

All logs saved to: `/tmp/feature_extractor_logs/`

```
clip_batch256.log    → First CLIP attempt (batch_size=256)
clip_batch128.log    → Second CLIP attempt (batch_size=128)
clip_batch64.log     → Third CLIP attempt (batch_size=64)
clip_batch32.log     → Fourth CLIP attempt (batch_size=32)

mel_batch256.log     → First MEL attempt (batch_size=256)
mel_batch128.log     → Second MEL attempt (batch_size=128)
... (same pattern as CLIP)
```

View logs:
```bash
# View all CLIP attempts
cat /tmp/feature_extractor_logs/clip_*.log

# View specific attempt
cat /tmp/feature_extractor_logs/clip_batch128.log

# Follow in real-time
tail -f /tmp/feature_extractor_logs/clip_batch256.log
```

## Example Scenarios

### Scenario 1: Sufficient GPU Memory
```
GPU Memory: 32GB
↓
Initial batch_size=256
↓
CLIP: Attempt 1 (batch_size=256) → ✅ Success
MEL: Attempt 1 (batch_size=256) → ✅ Success
↓
All features extracted in one pass
```

### Scenario 2: OOM on First Attempt
```
GPU Memory: 20GB
↓
Initial batch_size=256
↓
CLIP: Attempt 1 (batch_size=256) → ❌ OOM
       Cleanup created files
       Attempt 2 (batch_size=128) → ✅ Success
MEL: Attempt 1 (batch_size=128) → ✅ Success
↓
Features extracted after fallback
```

### Scenario 3: Multiple Fallbacks
```
GPU Memory: 6GB
↓
Initial batch_size=32
↓
CLIP: Attempt 1 (batch_size=32) → ✅ Success
MEL: Attempt 1 (batch_size=32) → ✅ Success
↓
Smaller batch sizes from the start, no fallback needed
```

## Troubleshooting

### Issue: "nvidia-smi: command not found"
**Solution**: Script defaults to batch_size=256. If memory issues occur, manually set batch_size lower.

### Issue: "OOM detected in clip extraction" (repeated failures)
**Solution**: 
- Check available disk space: `df -h`
- Check GPU memory: `nvidia-smi`
- Try reducing batch_size manually: `./run_clip.sh --batch_size 32 --num_workers 1`

### Issue: Files keep getting deleted
**Solution**: Check if files are being created successfully. The cleanup only deletes files during OOM, not on success.

### Issue: Want to re-extract all features
**Solution**: Delete existing features first
```bash
find downloads/dataset \
    \( -name "*_CLIP_ViT.npy" -o -name "audio*.npy" -o -name "*_AST.npy" \) \
    -delete
```

## Files Created/Modified

```
✨ NEW FILES:
   └─ run_all_with_fallback.sh        (Master script with OOM handling)
   └─ README_FALLBACK.md              (Detailed documentation)

📝 MODIFIED FILES:
   └─ run_clip.sh                     (Added parameter support)
   └─ run_mel.sh                      (Added parameter support)
```

## Next Steps

1. **Test the master script**:
   ```bash
   ./run_all_with_fallback.sh
   ```

2. **Monitor the logs**:
   ```bash
   tail -f /tmp/feature_extractor_logs/clip_batch*.log
   ```

3. **After features are extracted**, use them for training:
   ```bash
   python /workspace/ActionAware/train.py
   ```

## Support

For detailed usage information, see: `README_FALLBACK.md`

For questions about individual scripts:
- CLIP extraction: `run_clip.sh` / `clip_vit.py`
- MEL extraction: `run_mel.sh` / `mel.py`

---

**Created**: 2026-09-02  
**Type**: Feature Extraction Orchestration  
**Status**: ✅ Ready to use
