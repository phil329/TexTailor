from diffusers import DiffusionPipeline
import torch
import os


positive_magic = {
    "en": "Ultra HD, 4K, cinematic composition.", # for english prompt
}
model_name = "black-forest-labs/FLUX.1-dev"
model_path = "./pretrained_models/FLUX.1-dev"

# Load the pipeline
if torch.cuda.is_available():
    torch_dtype = torch.bfloat16
    device = "cuda"
else:
    torch_dtype = torch.float32
    device = "cpu"

pipe = DiffusionPipeline.from_pretrained(model_path, torch_dtype=torch_dtype)
pipe = pipe.to(device)

os.makedirs("outputs/flux_strength", exist_ok=True)
os.makedirs("outputs/flux_remove", exist_ok=True)
os.makedirs("outputs/flux_empty", exist_ok=True)



with open('files/prompts_90.txt', 'r', encoding='utf-8') as file:
     prompts = [line.strip() for line in file]


negative_prompt = " " # Recommended if you don't use a negative prompt.

for prompt in prompts:
    file_name = prompt.split()
    image_name = "_".join(file_name)
    os.makedirs(f"outputs/flux_baseline/{image_name}", exist_ok=True)
    os.makedirs(f"outputs/flux_remove/{image_name}", exist_ok=True)
    os.makedirs(f"outputs/flux_remove/{image_name}", exist_ok=True)
    os.makedirs(f"outputs/flux_empty/{image_name}", exist_ok=True)
    save_name = prompt
    image_baseline = pipe(
        prompt=prompt + positive_magic["en"],
        negative_prompt=negative_prompt,
        width=1024,
        height=1024,
        num_inference_steps=50,
        guidance_scale=3.5,
        generator=torch.Generator("cpu").manual_seed(0)
    ).images[0]
    image_baseline.save(f"outputs/flux_baseline/{image_name}/baseline.png")
    print(prompt, image_name)

    for i in range(58):

        image_1 = pipe(
            prompt=prompt + positive_magic["en"],
            negative_prompt=negative_prompt,
            width=1024,
            height=1024,
            num_inference_steps=50,
            guidance_scale=3.5,
            generator=torch.Generator("cpu").manual_seed(0),
            modulated_layers=i,
            modulated_scales=2.0,
            modulated_ways="strength"
        ).images[0]
        image_1.save(f"outputs/flux_strength/{image_name}/layer_{i}.png")

        image_2 = pipe(
            prompt=prompt + positive_magic["en"],
            negative_prompt=negative_prompt,
            width=1024,
            height=1024,
            num_inference_steps=50,
            guidance_scale=3.5,
            generator=torch.Generator("cpu").manual_seed(0),
            removed_layers=[i]
        ).images[0]
        image_2.save(f"outputs/flux_remove/{image_name}/layer_{i}.png")

        image_3 = pipe(
            prompt=prompt + positive_magic["en"],
            negative_prompt=negative_prompt,
            width=1024,
            height=1024,
            num_inference_steps=50,
            guidance_scale=3.5,
            generator=torch.Generator("cpu").manual_seed(0),
            modulated_layers=i,
            modulated_ways="empty"
        ).images[0]
        image_3.save(f"outputs/flux_empty/{image_name}/layer_{i}.png")