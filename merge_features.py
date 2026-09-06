import os
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

def has_exactly_two_mkv_files(directory: str) -> bool:
    count = 0
    with os.scandir(directory) as entries:
        for entry in entries:
            if entry.is_file() and entry.name.endswith(".mkv"):
                count += 1
                if count > 2:
                    return False
    return count == 2

def list_end_directories(path: str) -> list:
    end_files = []
    for root, _, files in os.walk(path):
        for file in files:
            if file.startswith("SoccerNetV2"):
                pass
            else:
                refined_path = os.path.join(root, file)
                end_files.append(refined_path)
    end_files = ["/".join(file.split("/")[:-1]) for file in end_files]
    # end_files = [item for item in end_files if has_exactly_two_mkv_files(item)]
    end_files = sorted(list(set(end_files)))
    return end_files


def merge_features(base_path):
    print(f"Scanning games in {base_path}...")
    list_games = list_end_directories(base_path)
    print(f"Found {len(list_games)} games.")
    
    for game_path in tqdm(list_games, desc="Processing games"):
        game_rel_path = game_path.replace(base_path, "").strip("/")
        
        for half in [1, 2]:
            # Define filenames
            baidu_name = f"{half}_baidu_soccer_embeddings.npy"
            clip_name = f"{half}_CLIP_ViT.npy"
            
            # We want to overwrite the baidu file
            baidu_file = os.path.join(game_path, baidu_name)
            clip_file = os.path.join(game_path, clip_name)

            # Check inputs exist
            if not os.path.exists(baidu_file):
                continue
            if not os.path.exists(clip_file):
                print(f"Missing CLIP features for {game_rel_path} (Half {half})")
                continue
            
            try:
                # Load Baidu features
                feat_b = np.load(baidu_file)
                
                # Reshape Baidu if needed (sometimes saved as (T, 1, D))
                if feat_b.ndim == 3:
                    feat_b = feat_b.squeeze(1)
                
                # Check for double-merge (Safety)
                # Original Baidu is 8576, CLIP is 768. Combined is 9344.
                # If dimension > 9000, probably already merged.
                if feat_b.shape[-1] > 9000:
                    if os.path.exists(clip_file):
                        os.remove(clip_file)
                    continue
                
                # Try to load CLIP features
                try:
                    feat_c = np.load(clip_file)
                except (ValueError, OSError) as e:
                    # CLIP file corrupted - create zero embedding
                    print(f"⚠️  CLIP file corrupted for {game_rel_path} Half {half}: {e}")
                    T_b = feat_b.shape[0]
                    
                    # Create zero CLIP embedding with correct shape (T_b, 768)
                    print(f"   Creating zero CLIP embedding with shape ({T_b}, 768)")
                    feat_c = np.zeros((T_b, 768), dtype=np.float32)
                    
                    # Save the zero embedding to replace corrupted file
                    np.save(clip_file, feat_c)
                    print(f"   Saved zero CLIP to: {clip_file}")

                # Get dimensions
                T_b = feat_b.shape[0]
                T_c = feat_c.shape[0]
                
                # Handle dimension mismatch
                if T_b != T_c:
                    # If CLIP has 1 more frame, trim it
                    if T_c == T_b + 1:
                        print(f"Trimming CLIP for {game_rel_path} Half {half}: {T_c} -> {T_b}")
                        feat_c = feat_c[:-1]  # Remove last frame
                        T_c = feat_c.shape[0]
                    # If difference is larger, replace with zero embedding
                    else:
                        print(f"⚠️  Large mismatch for {game_rel_path} Half {half}: Baidu {T_b} vs CLIP {T_c}.")
                        print(f"   Replacing CLIP with zero embedding ({T_b}, 768)")
                        feat_c = np.zeros((T_b, 768), dtype=np.float32)
                        
                        # Save the zero embedding to replace incorrect file
                        np.save(clip_file, feat_c)
                        print(f"   Saved zero CLIP to: {clip_file}")

                # Concatenate: [T, D_baidu + D_clip]
                feat_merged = np.concatenate([feat_b, feat_c], axis=-1).astype(np.float32)
                
                # Overwrite Baidu file
                np.save(baidu_file, feat_merged)
                
                # Xóa file CLIP_ViT.npy sau khi đã gộp thành công
                if os.path.exists(clip_file):
                    os.remove(clip_file)
                
            except Exception as e:
                print(f"❌ Error processing {game_rel_path} Half {half}: {e}")

if __name__ == "__main__":
    import sys
    SOCCERNET_PATH = sys.argv[1] if len(sys.argv) > 1 else "downloads/dataset"
    merge_features(SOCCERNET_PATH)
