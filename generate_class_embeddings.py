from transformers import CLIPTokenizer, CLIPTextModelWithProjection
import torch
import numpy as np
import os
from tqdm import tqdm

def generate_class_embeddings():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = "ViFortune-AI/CLIP-ViT"
    classes_file = "classes.txt"
    output_file = "class_embed.npy"

    print(f"Loading CLIP Text model and tokenizer from {model_name}...")
    # Sử dụng CLIPTextModelWithProjection để lấy đúng projector ra 512 chiều
    tokenizer = CLIPTokenizer.from_pretrained(model_name)
    model = CLIPTextModelWithProjection.from_pretrained(model_name).to(device)
    model.eval()

    # Kiểm tra dimension của model
    embed_dim = model.config.projection_dim
    print(f"CLIP Text Projection dimension: {embed_dim}")

    if not os.path.exists(classes_file):
        print(f"Error: {classes_file} not found.")
        return

    # Read classes
    with open(classes_file, "r") as f:
        class_lines = [line.strip() for line in f.readlines() if line.strip()]

    print(f"Found {len(class_lines)} class descriptions.")

    all_embeddings = []

    with torch.no_grad():
        for text in tqdm(class_lines, desc="Extracting CLIP Text embeddings"):
            inputs = tokenizer(text=[text], return_tensors="pt", padding=True, truncation=True).to(device)
            
            # Lấy embedding từ CLIP Text Encoder qua lớp Projection
            outputs = model(**inputs)
            text_features = outputs.text_embeds # Đây là feature đã qua projection (512d)
            
            # Normalize embeddings
            text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)
            
            all_embeddings.append(text_features.cpu().numpy())

    class_embeddings = np.vstack(all_embeddings)
    
    # Final check and save
    print(f"Final embeddings shape: {class_embeddings.shape}")
    np.save(output_file, class_embeddings)
    print(f"Class embeddings saved to {output_file}")

if __name__ == "__main__":
    generate_class_embeddings()
