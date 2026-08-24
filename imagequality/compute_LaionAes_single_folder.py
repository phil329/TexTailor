import os
import argparse
import torch
import torch.nn as nn
import open_clip
from PIL import Image
from tqdm import tqdm
import numpy as np
import json


# ---------------------
# 1. Load CLIP Model and Preprocess
# ---------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
model, _, preprocess = open_clip.create_model_and_transforms(
    'ViT-L-14',
    device=device,
)

# ---------------------
# 2. Load Aesthetic Predictor Weights
# ---------------------
aesthetic_model = nn.Linear(768, 1)
s = torch.load('./aesthetic-predictor/sa_0_4_vit_l_14_linear.pth')
aesthetic_model.load_state_dict(s)
aesthetic_model = aesthetic_model.to(device)
aesthetic_model.eval()

# ---------------------
# 3. Compute Aesthetic Score for a Single Image
# ---------------------
def compute_aesthetic_score(image_path):
    try:
        image = Image.open(image_path).convert("RGB")
        image_input = preprocess(image).unsqueeze(0).to(device)

        with torch.no_grad():
            image_features = model.encode_image(image_input)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            score = aesthetic_model(image_features)[0].item()

        return score
    except Exception as e:
        print(f"[Warning] skipping {image_path}: {e}")
        return None

# ---------------------
# 4. Find all leaf folders (containing only images, no subdirectories)
# ---------------------
def find_leaf_folders(root_folder):
    leaf_folders = []
    for dirpath, dirnames, filenames in os.walk(root_folder):
        # If there are no subdirectories, consider it a leaf folder
        if not dirnames:
            image_files = [
                f for f in filenames
                if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))
            ]
            if len(image_files) > 0:
                leaf_folders.append(dirpath)
    return leaf_folders

# ---------------------
# 5. Compute and Save Subfolder Scores
# ---------------------
def compute_and_save_subfolder_scores(subfolder_path, root_folder, output_dir="results_json"):
    os.makedirs(output_dir, exist_ok=True)

    # Naming the JSON file with a relative path to avoid name conflicts
    relative_name = os.path.relpath(subfolder_path, root_folder).replace(os.sep, "_")
    save_path = os.path.join(output_dir, f"{relative_name}.json")
    
    if os.path.exists(save_path):
        print(f'{save_path} exists, skip!!!!!!')
        return
    
    results = {}
    scores = []

    image_files = [
        f for f in os.listdir(subfolder_path)
        if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))
    ]

    for filename in tqdm(image_files, desc=f"Processing {os.path.relpath(subfolder_path, root_folder)}"):
        filepath = os.path.join(subfolder_path, filename)
        if os.path.isfile(filepath):
            score = compute_aesthetic_score(filepath)
            if score is not None:
                results[filepath] = score
                scores.append(score)

    avg_score = float(np.mean(scores)) if scores else None

    save_data = {
        "folder": subfolder_path,
        "average_score": avg_score,
        "images": results
    }

    

    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(save_data, f, indent=4, ensure_ascii=False)

    if avg_score is not None:
        print(f"📂 {subfolder_path} Average Aesthetic Score = {avg_score:.2f} -> Save to {save_path}")
    else:
        print(f"📂 {subfolder_path} No valid images found")

# ---------------------
# 6. Main Program
# ---------------------
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Compute Aesthetic Scores for Subfolders")
    parser.add_argument("--root_folder", type=str, required=True, help="Root folder path")
    parser.add_argument("--output_dir", type=str, default=None, help="Output folder path (optional)")

    args = parser.parse_args()
    
    output_dir = args.output_dir or f'./aesthetic_scores/{os.path.basename(args.root_folder)}'
    
    leaf_folders = find_leaf_folders(args.root_folder)
    print(f"Found {len(leaf_folders)} leaf folders")

    for sub in leaf_folders:
        compute_and_save_subfolder_scores(sub, args.root_folder, output_dir=output_dir)
