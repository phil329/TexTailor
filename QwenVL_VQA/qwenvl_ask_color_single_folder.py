import asyncio
import httpx
import os
import time
import pandas as pd
import json
import base64
from io import BytesIO
from PIL import Image
import argparse
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

from qwen_vl_utils import to_rgb, smart_resize, MIN_PIXELS, MAX_PIXELS, IMAGE_FACTOR


headers = {
    "Accept": "application/json",
    "Content-Type": "application/json"
}

BASE_URL = "Your_API_Endpoint_Here"
MODEL_NAME = "Your_Model_Name_Here"
TIMEOUT = 60.0  # seconds for connection timeout


helpings = """
You are given an image, its caption, and a set of objects with their expected colors.

Your task:
- For each object: check if the color in the caption matches the actual color in the image.
- If the color matches, return "Yes".
- If the color does not match or the object is not visible, return "No".

Rules:
- Output ONLY a single valid JSON object.
- The JSON keys must be exactly the provided object names.
- The values must be strictly "Yes" or "No".
- Do not generate any other words.
- Do not add explanations, extra text, or formatting outside the JSON.

Example:
Caption: "A red chair with golden armrests."
Objects: {"chair": "red", "armrests": "gold"}
Output: {"chair": "Yes", "armrests": "No"}
"""


def pil_to_base64(img: Image.Image) -> str:
    buf = BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


async def ask_one(prompt, image, client: httpx.AsyncClient) -> str:
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": helpings},
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image}"}},
                    {"type": "text", "text": prompt}
                ]
            }
        ],
        "max_output_tokens": 256,
    }
    try:
        resp = await client.post(BASE_URL, headers=headers, json=payload)
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]
    except Exception as e:
        return f"Error: {e}"


async def batch_process(prompt_list, image_list, max_concurrency: int = 10):
    start = time.time()
    semaphore = asyncio.Semaphore(max_concurrency)

    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        async def sem_task(prompt, image):
            async with semaphore:
                return await ask_one(prompt, image, client)

        tasks = [sem_task(p, i) for p, i in zip(prompt_list, image_list)]
        responses = await asyncio.gather(*tasks)

    duration = time.time() - start
    print(f"\nProcessed {len(prompt_list)} images in {duration:.2f} seconds.")
    return responses


def read_image_and_process(image_path):
    img = Image.open(image_path).convert("RGB")
    img = to_rgb(img)
    width, height = img.size
    resized_height, resized_width = smart_resize(
        height, width,
        factor=IMAGE_FACTOR,
        min_pixels=MIN_PIXELS,
        max_pixels=MAX_PIXELS,
    )
    img = img.resize((resized_width, resized_height))
    img_b64 = pil_to_base64(img)
    
    return img_b64




def is_valid_response(resp: str) -> bool:
    if isinstance(resp, float) and pd.isna(resp):  # NaN
        return False
    if isinstance(resp, str) and resp.startswith("Error"):
        return False
    try:
        data = json.loads(resp)
        return isinstance(data, dict)
    except Exception:
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ask object color in images")
    parser.add_argument("--data_path", type=str, required=True, help="Path to image data root")
    parser.add_argument("--prompt", type=str, required=True, help="CSV file with columns: image, prompt, object")
    parser.add_argument('--objects', type=str, help='Comma-separated object names')
    parser.add_argument('--colors', type=str, help='Comma-separated colors for objects')
    parser.add_argument('--result_folder',type=str)
    parser.add_argument("--batchsize", type=int, default=8, help="Batch size")
    args = parser.parse_args()
    
    args.objects = args.objects.split(",")
    args.colors = args.colors.split(",")
    
    
    if len(args.objects) != len(args.colors):
        print(f"Error: Number of objects ({len(args.objects)}) must match number of colors ({len(args.colors)})")
        print("Objects:", args.objects)
        print("Colors:", args.colors)
    
        
    args.data_path = os.path.join(args.data_path, '_'.join(args.prompt.split()))
    
    
    # Find all image files 'png', 'jpg', 'jpeg'
    image_names = []
    for root, dirs, files in os.walk(args.data_path):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_names.append(os.path.join(root, file))
    # image_names maybe baseline.png or layer_{i}.png
    # I want to sort them by the number in the filename if possible
    
    image_names.sort(key=lambda x: int(''.join(filter(str.isdigit, os.path.basename(x)))) if any(c.isdigit() for c in os.path.basename(x)) else float('inf'))
    # image_names = sorted(image_names)
    
    print(f"Found {len(image_names)} images in {args.data_path}")
    # print(image_names[:10])
    
    os.makedirs(args.result_folder, exist_ok=True)
    
    save_path = os.path.join(args.result_folder, f"{'_'.join(args.prompt.split())}.csv")
    
    # If the file exists, read it
    if os.path.exists(save_path):
        df = pd.read_csv(save_path)
        prev_results = dict(zip(df["image"], df["response"]))
    else:
        prev_results = {}
    
    
    
    # Initialize final_results with previous results or None
    final_results = [prev_results.get(os.path.basename(p), None) for p in image_names]
    
    
    # Find indices that need reprocessing
    bad_indices = [i for i, resp in enumerate(final_results) if not is_valid_response(resp)]
    print(f"Need to reprocess {len(bad_indices)} / {len(image_names)} images")
    
    
    if bad_indices:
        # only read and process images that need to be reprocessed
        # images = [read_image_and_process(p) for p in tqdm(image_names, desc="Reading and processing images")]
        
        images = [read_image_and_process(image_names[i]) for i in tqdm(bad_indices, desc="Reading and processing images")]
        
        
        # images = [read_image_and_process(p) for p in tqdm(image_names, desc="Reading and processing images")]
        
        objects_with_colors = [f'"{obj}": "{col}"' for obj, col in zip(args.objects, args.colors)]
        objects_str = '{' + ', '.join(objects_with_colors) + '}'
        
        
        for i in tqdm(range(0, len(bad_indices), args.batchsize), desc="Reprocessing"):
            # batch_idx = bad_indices[i:i+args.batchsize]
            # batch_images = [images[j] for j in batch_idx]
            
            batch_idx = bad_indices[i:min(i+args.batchsize, len(bad_indices))]
            batch_images = [images[j] for j in range(i, min(i+args.batchsize, len(bad_indices)))]
            
            
            batch_questions = [
                f'Caption: "{args.prompt}"\nObjects: {objects_str}'
                for _ in batch_idx
            ]
            results = asyncio.run(batch_process(batch_questions, batch_images, max_concurrency=args.batchsize))
            
            for j, idx in enumerate(batch_idx):
                final_results[idx] = results[j]
                print(f"✅ {os.path.basename(image_names[idx])} updated: {results[j]}")
    
        # save results to csv
        df = pd.DataFrame({
            "image": [os.path.basename(p) for p in image_names],
            "response": final_results
        })
        df.to_csv(save_path, index=False)
        
        
        print("\n🎉 All images processed.")
    else:
        print(f'🎉 All images already have valid results for {save_path}')
