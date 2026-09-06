# Quick Start: Feature Extraction with OOM Fallback

## ✅ Setup Complete

All scripts are now ready to use with automatic OOM handling and cleanup.

## Files Modified/Enhanced

```
✨ ENHANCED:
   feature_extract.sh             Main script - with OOM fallback & cleanup
   run_clip.sh                    Added --batch_size and --num_workers params
   run_mel.sh                     Added --batch_size and --num_workers params

📚 DOCUMENTATION:
   README_FALLBACK.md             Full documentation
   SETUP_SUMMARY.md               Setup overview
   QUICK_START.md                 This file
```

## Usage

### Automatic (Recommended) ⭐

Run with auto GPU detection and OOM fallback:

```bash
cd /workspace/ActionAware/feature_extractor
./feature_extract.sh
```

That's it! The script will:
- ✅ Detect GPU memory
- ✅ Choose optimal batch size
- ✅ Run CLIP extraction
- ✅ Run MEL extraction
- ✅ If OOM: cleanup files and retry with smaller batch size

### Manual (If you need specific batch sizes)

```bash
# Run CLIP only
./run_clip.sh --batch_size 128 --num_workers 4

# Run MEL only
./run_mel.sh --num_workers 4

# Or run both with custom sizes (no auto-fallback)
./run_clip.sh --batch_size 64 --num_workers 2
./run_mel.sh --num_workers 2
```

## Monitor Progress

**Terminal 1 - Watch extraction logs:**
```bash
tail -f /tmp/feature_extractor_logs/clip_batch*.log
```

**Terminal 2 - Monitor GPU:**
```bash
watch -n 1 nvidia-smi
```

**Terminal 3 - Check created files:**
```bash
find downloads/dataset \
    -name "*.npy" -mmin -5 | wc -l  # Files created in last 5 minutes
```

## What Happens on OOM

```
Running CLIP (batch_size=256)
  ↓
❌ GPU Out of Memory
  ↓
🧹 Cleanup: Delete partially created files
  ↓
Retry CLIP (batch_size=128, num_workers=4)
  ↓
✅ Success!
```

## Log Files

```
/tmp/feature_extractor_logs/
├── clip_batch256.log    → Attempt 1 (batch=256)
├── clip_batch128.log    → Attempt 2 (batch=128)
├── clip_batch64.log     → Attempt 3 (batch=64)
├── clip_batch32.log     → Attempt 4 (batch=32)
├── mel_batch256.log     → Attempt 1 (batch=256)
├── mel_batch128.log     → Attempt 2 (batch=128)
├── mel_batch64.log      → Attempt 3 (batch=64)
└── mel_batch32.log      → Attempt 4 (batch=32)
```

## GPU Memory Thresholds

```
GPU Memory              Initial Batch Size    Workers
──────────────────────────────────────────────────────
> 25 GB                 256                   8
> 15 GB                 128                   4
> 8 GB                  64                    2
≤ 8 GB                  32                    1
```

## Output Files

**CLIP extraction creates:**
```
{video_number}_CLIP_ViT.npy
Example: /dataset/match1/1_CLIP_ViT.npy
```

**MEL extraction creates:**
```
audio{1,2}.npy           # Main mel-spectrogram features
{1,2}_AST.npy            # AST model features
Example: /dataset/match1/audio1.npy
```

## Troubleshooting

### Q: Script says "OOM detected" repeatedly
**A:** Your GPU memory is too low. Try:
```bash
# Manual run with very low batch size
./run_clip.sh --batch_size 16 --num_workers 1
./run_mel.sh --num_workers 1
```

### Q: "nvidia-smi not found"
**A:** Script defaults to batch_size=256. If OOM occurs, manually reduce:
```bash
./feature_extract.sh  # Will fallback automatically if OOM
```

### Q: Files keep getting deleted
**A:** This is normal during OOM fallback. Files are recreated in the next attempt.

### Q: Want to re-extract everything?
**A:** Delete existing features:
```bash
find downloads/dataset \
    \( -name "*_CLIP_ViT.npy" -o -name "audio*.npy" -o -name "*_AST.npy" \) \
    -delete

# Then run again
./feature_extract.sh
```

### Q: How do I know if extraction succeeded?
**A:** Check:
```bash
# Count extracted files
find downloads/dataset \
    -name "*_CLIP_ViT.npy" -o -name "audio*.npy" | wc -l

# Check file sizes (they shouldn't be 0 or very small)
find downloads/dataset \
    -name "*_CLIP_ViT.npy" -exec ls -lh {} \; | head -5
```

## Advanced

### Change Fallback Strategy

Edit `feature_extract.sh`, in `run_with_fallback()` function:

```bash
# Change reduction from 50% to 33%
batch_size=$((batch_size * 2 / 3))  # Instead of / 2
```

### Change Dataset Path

Edit the script and change:
```bash
DATASET_PATH="/your/custom/dataset/path"
```

### Change Log Location

Edit the script and change:
```bash
LOG_DIR="/your/custom/log/path"
```

## Next Steps

1. **Run extraction:**
   ```bash
   cd /workspace/ActionAware/feature_extractor
   ./feature_extract.sh
   ```

2. **Monitor progress:**
   ```bash
   tail -f /tmp/feature_extractor_logs/clip_batch*.log
   ```

3. **After features are ready**, use for training:
   ```bash
   cd /workspace/ActionAware
   python train.py --features clip mel
   ```

## Documentation

- **Full Guide**: `README_FALLBACK.md` - Detailed documentation
- **Setup Overview**: `SETUP_SUMMARY.md` - Setup overview
- **This File**: `QUICK_START.md` - Quick reference

---

**Ready to extract features!** 🚀

```bash
cd /workspace/ActionAware/feature_extractor
./feature_extract.sh
```

**Questions?** Check `README_FALLBACK.md` for detailed documentation.
