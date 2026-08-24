import os
import json
import argparse
from tqdm import tqdm
import numpy as np

import torch
from PIL import Image
from hpsv2.src.open_clip import create_model_and_transforms, get_tokenizer



# ---------------------
# 1. Initialize HPSv2 Model
# ---------------------
# Setup the args and prepare the model and tokenizer
args = {
    'model': 'ViT-H-14',
    'precision': 'amp',
    'checkpoint': 'xswu/HPSv2/HPS_v2_compressed.pt'  # replace with your checkpoint path
}

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model, preprocess_train, preprocess_val = create_model_and_transforms(
    args['model'],
    # 'laion2B-s32B-b79K',
    precision=args['precision'],
    device=device,
    jit=False,
    force_quick_gelu=False,
    force_custom_text=False,
    force_patch_dropout=False,
    force_image_size=None,
    pretrained_image=False,
    image_mean=None,
    image_std=None,
    light_augmentation=True,
    aug_cfg={},
    output_dict=True,
    with_score_predictor=False,
    with_region_predictor=False
)

checkpoint = torch.load(args['checkpoint'])
model.load_state_dict(checkpoint['state_dict'])
tokenizer = get_tokenizer(args['model'])
model.eval()



# ---------------------
# 2. Find all leaf folders
# ---------------------
def find_leaf_folders(root_folder):
    leaf_folders = []
    for dirpath, dirnames, filenames in os.walk(root_folder):
        if not dirnames:
            image_files = [
                f for f in filenames
                if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))
            ]
            if image_files:
                leaf_folders.append(dirpath)
    return leaf_folders

# ---------------------
# 3. Compute and Save Subfolder Scores (Support Batch Inference)
# ---------------------
def compute_and_save_subfolder_scores(subfolder_path, root_folder, output_dir="results_json", batch_size=8):
    os.makedirs(output_dir, exist_ok=True)
    
    relative_name = os.path.relpath(subfolder_path, root_folder).replace(os.sep, "_")
    save_path = os.path.join(output_dir, f"{relative_name}.json")
    
    if os.path.exists(save_path):
        print(f'{save_path} exists, skip!')
        return
    
    results = {}
    scores = []

    image_files = [
        f for f in os.listdir(subfolder_path)
        if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))
    ]
    if not image_files:
        print(f"📂 {subfolder_path} No valid images found")
        return

    # Process in batches
    for i in tqdm(range(0, len(image_files), batch_size), desc=f"Processing {os.path.relpath(subfolder_path, root_folder)}"):
        batch_files = image_files[i:i+batch_size]
        batch_paths = [os.path.join(subfolder_path, f) for f in batch_files]
        batch_prompts = [os.path.splitext(f)[0] for f in batch_files]  # Remove extension for prompt

        try:
            # Process multiple images
            image_tensors = [
                preprocess_val(Image.open(p)).unsqueeze(0) for p in batch_paths
            ]
            images = torch.cat(image_tensors, dim=0).to(device)   # [B, C, H, W]

            # Process multiple texts
            texts = tokenizer(batch_prompts).to(device)

            # Compute HPS
            with torch.no_grad():
                outputs = model(images, texts)
                image_features, text_features = outputs["image_features"], outputs["text_features"]
                logits_per_image = outputs["logit_scale"] * image_features @ text_features.T   # [B, B]

            # Get the diagonal as the paired scores, 
            # which represent the similarity scores between each image and its corresponding text
            # logits_per_image: [B, B]
            hps_scores = torch.diagonal(logits_per_image).cpu().numpy()
            
            # print('hps_scores : ',hps_scores)
        except Exception as e:
            print(f"[Warning] Batch processing failed {batch_paths}: {e}")
            continue

        for filepath, score in zip(batch_paths, hps_scores):
            # score = reward[0].item()
            # print('score : ',type(score),score)
            results[filepath] = score.item()
            scores.append(score)

    avg_score = float(np.mean(scores)) if scores else None
    std_score = float(np.std(scores, ddof=1)) if len(scores) > 1 else 0.0

    save_data = {
        "folder": subfolder_path,
        "average_score": avg_score,
        "std_score": std_score,
        "images": results
    }

    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(save_data, f, indent=4, ensure_ascii=False)

    if avg_score is not None:
        print(f"📂 {subfolder_path} Average HPSv2 Score = {avg_score:.2f}, std = {std_score:.2f} -> Save to {save_path}")

# ---------------------
# 4. Main Program
# ---------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute HPSv2 Scores for Subfolders")
    parser.add_argument("--root_folder", type=str, required=True, help="Root folder path")
    parser.add_argument("--output_dir", type=str, default=None, help="Output folder path (optional)")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for inference")

    args = parser.parse_args()
    output_dir = args.output_dir or f'./hpsv2_score_results/{os.path.basename(args.root_folder)}'
    
    leaf_folders = find_leaf_folders(args.root_folder)
    print(f"Found {len(leaf_folders)} leaf folders")

    for sub in leaf_folders:
        compute_and_save_subfolder_scores(sub, args.root_folder, output_dir=output_dir, batch_size=args.batch_size)
