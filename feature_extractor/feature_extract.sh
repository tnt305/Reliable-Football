#!/usr/bin/env bash
# ==============================================================================
# Master Script: Run CLIP + MEL extractors with OOM fallback & cleanup
# 
# Features:
#   - Check GPU memory before running
#   - Track created files before execution
#   - On OOM: Delete created files and retry with lower batch size
#   - Support auto-fallback: 256 -> 128 -> 64 -> 32
# ==============================================================================

set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATASET_PATH="downloads/dataset"
LOG_DIR="/tmp/feature_extractor_logs"
mkdir -p "$LOG_DIR"

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# ==============================================================================
# UTILITY FUNCTIONS
# ==============================================================================

log_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

log_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

log_warn() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

log_error() {
    echo -e "${RED}❌ $1${NC}"
}

check_gpu_memory() {
    if command -v nvidia-smi &>/dev/null; then
        nvidia-smi --query-gpu=memory.free --format=csv,nounits,noheader | head -1
    else
        log_warn "nvidia-smi not found, assuming sufficient memory"
        echo "20000"
    fi
}

get_initial_batch_size() {
    local gpu_mem=$1
    
    if [ "$gpu_mem" -gt 25000 ]; then
        echo "256"
    elif [ "$gpu_mem" -gt 15000 ]; then
        echo "128"
    elif [ "$gpu_mem" -gt 8000 ]; then
        echo "64"
    else
        echo "32"
    fi
}

# Find all .npy files created after a certain timestamp
find_created_files() {
    local script_type=$1  # "clip" or "mel"
    local before_timestamp=$2
    
    # Find all files modified after the timestamp
    find "$DATASET_PATH" -type f -newer "$before_timestamp" 2>/dev/null | while read -r file; do
        case $script_type in
            clip)
                if [[ "$file" =~ _CLIP_ViT\.npy$ ]]; then
                    echo "$file"
                fi
                ;;
            mel)
                if [[ "$file" =~ audio[1-2]\.npy$ ]]; then
                    echo "$file"
                fi
                ;;
        esac
    done
}

# Delete all created files
cleanup_created_files() {
    local script_type=$1
    local before_timestamp=$2
    
    log_warn "Cleaning up files created during execution..."
    
    find_created_files "$script_type" "$before_timestamp" | while read -r file; do
        if [ -f "$file" ]; then
            log_warn "Removing: $file"
            rm -f "$file"
        fi
    done
    
    log_success "Cleanup completed"
}

# Check if error is OOM
is_oom_error() {
    local log_file=$1
    grep -qi "out of memory\|cuda out of memory\|pytorch\|torch.cuda\|RuntimeError.*memory\|ENOMEM" "$log_file"
}

# ==============================================================================
# MAIN EXECUTION
# ==============================================================================

run_feature_extractor() {
    local script_type=$1  # "clip" or "mel"
    local batch_size=$2
    local num_workers=$3
    
    local script_name="run_${script_type}.sh"
    local script_path="$SCRIPT_DIR/$script_name"
    local log_file="$LOG_DIR/${script_type}_batch${batch_size}.log"
    
    if [ ! -f "$script_path" ]; then
        log_error "Script not found: $script_path"
        return 1
    fi
    
    log_info "Running $script_name with batch_size=$batch_size, num_workers=$num_workers"
    
    # Create timestamp before running
    local before_file="/tmp/${script_type}_before_$$.tmp"
    touch "$before_file"
    
    # Run script with parameters (if applicable)
    if bash "$script_path" \
        --batch_size "$batch_size" \
        --num_workers "$num_workers" \
        2>&1 | tee "$log_file"; then
        
        log_success "$script_type extraction completed successfully"
        rm -f "$before_file"
        return 0
    else
        # Check if it's OOM
        if is_oom_error "$log_file"; then
            log_error "OOM detected in $script_type extraction"
            cleanup_created_files "$script_type" "$before_file"
            rm -f "$before_file"
            return 2  # Indicate OOM (need retry)
        else
            log_error "$script_type extraction failed (not OOM)"
            rm -f "$before_file"
            return 1  # Other error
        fi
    fi
}

run_merge() {
    local script_name="run_merge.sh"
    local script_path="$SCRIPT_DIR/$script_name"
    local log_file="$LOG_DIR/merge.log"
    
    if [ ! -f "$script_path" ]; then
        log_error "Script not found: $script_path"
        return 1
    fi
    
    log_info "Running $script_name --dataset_path $DATASET_PATH"
    if bash "$script_path" --dataset_path "$DATASET_PATH" 2>&1 | tee "$log_file"; then
        log_success "Feature merge completed successfully"
        return 0
    else
        log_error "Feature merge failed. Check log: $log_file"
        return 1
    fi
}

# Run with exponential fallback (only retry on OOM, exit immediately on other errors)
run_with_fallback() {
    local script_type=$1
    local initial_batch=$2
    
    local batch_size=$initial_batch
    local num_workers=8
    local attempt=1
    
    while [ "$batch_size" -ge 8 ]; do
        log_info "Attempt $attempt: batch_size=$batch_size, num_workers=$num_workers"
        
        run_feature_extractor "$script_type" "$batch_size" "$num_workers"
        local exit_code=$?
        
        if [ $exit_code -eq 0 ]; then
            return 0  # Success
        elif [ $exit_code -eq 1 ]; then
            log_error "Non-OOM error in $script_type extraction, aborting fallback"
            return 1  # Non-OOM error: do not retry
        fi
        
        # exit_code == 2 -> OOM: reduce resources and retry
        batch_size=$((batch_size / 2))
        num_workers=$((num_workers / 2))
        if [ "$num_workers" -lt 1 ]; then
            num_workers=1
        fi
        
        attempt=$((attempt + 1))
        
        if [ "$batch_size" -ge 8 ]; then
            log_warn "OOM detected, retrying with batch_size=$batch_size, num_workers=$num_workers ..."
            sleep 2
        fi
    done
    
    log_error "All fallback attempts exhausted for $script_type (minimum batch_size=8)"
    return 1
}

# ==============================================================================
# MAIN FLOW
# ==============================================================================

main() {
    log_info "========== Feature Extraction Master Script =========="
    log_info "Dataset: $DATASET_PATH"
    log_info "Log Dir: $LOG_DIR"
    log_info ""
    
    # Check GPU memory
    local gpu_mem=$(check_gpu_memory)
    log_info "GPU Memory Available: ${gpu_mem}MB"
    
    # Get initial batch size
    local initial_batch=$(get_initial_batch_size "$gpu_mem")
    log_info "Initial batch size: $initial_batch"
    log_info ""
    
    # Run CLIP extraction
    log_info "========== Starting CLIP Extraction =========="
    if ! run_with_fallback "clip" "$initial_batch"; then
        log_error "CLIP extraction failed after all retries"
        return 1
    fi
    log_success "CLIP extraction completed"
    echo ""
    
    # Run MEL extraction
    log_info "========== Starting MEL Extraction =========="
    if ! run_with_fallback "mel" "$initial_batch"; then
        log_error "MEL extraction failed after all retries"
        return 1
    fi
    log_success "MEL extraction completed"
    echo ""
    
    # Run MERGE Features (CLIP + Baidu -> 9344 dims)
    log_info "========== Starting Feature Merge (CLIP + Baidu) =========="
    if ! run_merge; then
        log_error "Feature merge failed"
        return 1
    fi
    log_success "Feature merge completed"
    echo ""
    
    log_success "========== All extractions & merge completed successfully =========="
    return 0
}

# Run main
main
exit $?
