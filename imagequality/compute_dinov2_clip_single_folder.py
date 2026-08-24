import os
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
import pandas as pd
from transformers import AutoImageProcessor, AutoModel, CLIPProcessor, CLIPModel

# -----------------------------
# Parameter Parsing
# -----------------------------
parser = argparse.ArgumentParser(description="Compute DINOv2 and CLIP-Text similarity scores")
parser.add_argument("--image_folder", type=str, required=True, help="Folder containing images to compare")
parser.add_argument("--baseline_image", type=str, required=True, help="Path to baseline image")
parser.add_argument("--caption", type=str, required=True, help="Caption for CLIP-Text similarity")
parser.add_argument("--output_csv", type=str, default="image_similarity_scores.csv", help="Output CSV file path")
parser.add_argument("--device", type=str, default="cuda", help="Device: cuda or cpu")
args = parser.parse_args()

device = args.device if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# -----------------------------
# Initialize DINOv2
# -----------------------------
dinov2_processor = AutoImageProcessor.from_pretrained('facebook/dinov2-base')
dinov2_model = AutoModel.from_pretrained('facebook/dinov2-base').to(device)
dinov2_model.eval()

def extract_dinov2_features(image_path):
    img = Image.open(image_path).convert("RGB")
    inputs = dinov2_processor(images=img, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = dinov2_model(**inputs)
        feat = outputs.last_hidden_state[:, 0, :]  # CLS token
        feat = F.normalize(feat, p=2, dim=-1)
    return feat.squeeze()

# -----------------------------
# Initialize CLIP
# -----------------------------
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
clip_model.eval()

def compute_clip_score(image_path, text):
    img = Image.open(image_path).convert("RGB")
    inputs = clip_processor(text=[text], images=img, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        outputs = clip_model(**inputs)
        return outputs.logits_per_image.item()

# -----------------------------
# Extract Baseline Features
# -----------------------------
print("Extracting baseline DINOv2 features...")
baseline_feat = extract_dinov2_features(args.baseline_image)

# -----------------------------
# Iterate Through Image Folder
# -----------------------------
results = []
image_files = [f for f in os.listdir(args.image_folder) if f.lower().endswith((".png",".jpg",".jpeg"))]

for img_name in tqdm(image_files, desc="Processing images"):
    img_path = os.path.join(args.image_folder, img_name)

    # DINOv2 Similarity
    feat = extract_dinov2_features(img_path)
    dinov2_sim = F.cosine_similarity(baseline_feat, feat, dim=0).item()

    # CLIP-Text Similarity
    clip_score = compute_clip_score(img_path, args.caption)

    results.append({
        "image": img_name,
        "dinov2_similarity": dinov2_sim,
        "clip_score": clip_score
    })

# -----------------------------
# Save Results
# -----------------------------
df = pd.DataFrame(results)
df.to_csv(args.output_csv, index=False)
print(f"Done! Results saved to {args.output_csv}")
