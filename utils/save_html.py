import os
from pathlib import Path
import argparse
from typing import List, Optional
from visualizer import HtmlPageVisualizer, load_image
import re

def extract_index_from_filename(filename: str) -> int:
    """Get index from filename for sorting"""
    # Extract the first sequence of digits found in the filename
    match = re.search(r'(\d+)', filename)
    if match:
        return int(match.group(1))
    return 0  # Default to 0 if no digits found

def get_image_files(folder_path: str, max_images: int = 100) -> List[str]:
    """Get image files from a folder, sorted by numeric index in filename"""
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}
    image_files = []
    
    try:
        for file_path in Path(folder_path).iterdir():
            if file_path.is_file() and file_path.suffix.lower() in image_extensions:
                image_files.append(str(file_path))
    except Exception as e:
        print(f"Error reading folder {folder_path}: {e}")

    # Sort by numeric index in filename
    image_files.sort(key=lambda x: extract_index_from_filename(os.path.basename(x)))
    
    # Limit to max_images
    if max_images > 0:
        image_files = image_files[:max_images]
    
    return image_files

input_path = "./outputs/remove_layers"


subfolders = os.listdir(input_path)
num_rows = len(subfolders)
num_cols=60
html = HtmlPageVisualizer(num_rows=num_rows, num_cols=num_cols)


# assign columns to each subfolder
for row_idx, subfolder in enumerate(subfolders):
    # import pdb; pdb.set_trace()
    # folder_name = subfolder.name
    print(f"Processing folder {subfolder}")
    folder_name = subfolder
    # Get images in this folder
    subfolder = os.path.join(input_path, subfolder)
    image_files = get_image_files(f"{subfolder}")
    # Set content for each row
    for col_idx, image_file in enumerate(image_files):
        image_name = image_file.split('/')[-1].replace('.png', '')
        image = load_image(path=image_file, resize=True)
        print(image_file, row_idx, col_idx)
        if col_idx == 0:
            text=f"{folder_name}_{image_name}"
        else:
            text=f"{image_name}"
        html.set_cell(row_idx, col_idx, 
                        text=text, 
                        image=image
                        )
html.save(f'{input_path}/Qwen_remove_layers_visualize.html')