import os
import re
import torch
import pandas as pd
import numpy as np
from transformers import CLIPTextModel, CLIPTokenizer


def has_exactly_two_mkv_files(directory: str) -> bool:
    count = 0
    with os.scandir(directory) as entries:
        for entry in entries:
            # Check if the current entry is a file and ends with ".mkv"
            if entry.is_file() and entry.name.endswith(".mkv"):
                count += 1
                # Exit early if more than two such files are found
                if count > 2:
                    return False
    return count == 2

def list_end_directories(path: str) -> list:
    """
    Recursively traverse a directory and 
    return all paths that are files (end of branch).
    """
    end_files = []
    for root, _, files in os.walk(path):
        for file in files:
            if file.startswith('SoccerNetV2'):                
                pass
            else:
                refined_path = os.path.join(root, file)
                end_files.append(refined_path)
    end_files = ["/".join(file.split("/")[:-1]) for file in end_files]
    end_files = [item for item in end_files if has_exactly_two_mkv_files(item)]
    end_files = sorted(list(set(end_files)))
    return end_files
##########################

data_dir = "/home/thiendc/projects/video_summarization/train_soccernet/"
df_list = []

# Bước 1: Lọc riêng các file CSV hợp lệ
csv_files = [
    f for f in os.listdir(data_dir)
    if f.startswith("validation_epoch") and f.endswith(".csv")
]

# Bước 2: Định nghĩa key để sort theo số epoch
def extract_epoch(filename):
    match = re.search(r"validation_epoch_(\d+)", filename)
    return int(match.group(1)) if match else -1  # hoặc 0

# Bước 3: Sort theo số epoch
def validation_summary(data_dir, output_path):
    for item in sorted(csv_files, key=extract_epoch):
        file_path = os.path.join(data_dir, item)

        df = pd.read_csv(file_path, index_col=0)

        # Parse giá trị amap
        match = re.search(r"amap_(\d+\.\d+)", item)
        amap_value = float(match.group(1)) if match else None

        # Thêm cột
        df['Tight-amap'] = amap_value

        df_list.append(df)

    # Gộp tất cả lại
    df_all = pd.concat(df_list, ignore_index=True)
    df_all.to_csv(output_path, index=False)

#########################

# Thay đường dẫn model CLIP local của bạn:
model_name = "/home/thiendc/projects/video_summarization/v5/openvocab_clipvit/best_model"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def create_text_embeddings(model_name, device, output_path):
    with open("texts.txt", "r") as f:
        prompts = [line.strip() for line in f.readlines() if line.strip()]
    print(f"Loaded {len(prompts)} prompts")
    # ==== LOAD MODEL ====
    print("Loading CLIP tokenizer and text model...")
    tokenizer = CLIPTokenizer.from_pretrained(model_name)
    text_model = CLIPTextModel.from_pretrained(model_name).to(device)
    text_model.eval()

    # ==== ENCODE ====
    print("Encoding prompts...")
    with torch.no_grad():
        inputs = tokenizer(prompts, padding=True, return_tensors="pt").to(device)
        text_features = text_model(**inputs)
        class_embed = text_features.pooler_output.detach().cpu().numpy()

    print(f"Embeddings shape: {class_embed.shape}")

    # ==== SAVE ====
    np.save(output_path, class_embed)
    print(f"Saved embeddings to {output_path} ✅")