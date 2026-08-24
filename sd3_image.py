from diffusers import DiffusionPipeline
import torch
import os


positive_magic = {
    "en": "Ultra HD, 4K, cinematic composition.", # for english prompt
}
model_name = "stabilityai/stable-diffusion-3.5-large"
model_path = "./pretrained_models/stable-diffusion-3.5-large"

# Load the pipeline
if torch.cuda.is_available():
    torch_dtype = torch.bfloat16
    device = "cuda"
else:
    torch_dtype = torch.float32
    device = "cpu"

pipe = DiffusionPipeline.from_pretrained(model_path, torch_dtype=torch_dtype)
pipe = pipe.to(device)

os.makedirs("outputs/SD3_strength", exist_ok=True)
os.makedirs("outputs/SD3_remove", exist_ok=True)
os.makedirs("outputs/SD3_empty", exist_ok=True)



with open('files/prompts_90.txt', 'r', encoding='utf-8') as file:
     prompts = [line.strip() for line in file]


negative_prompt = " " # Recommended if you don't use a negative prompt.

for prompt in prompts:
    file_name = prompt.split()
    image_name = "_".join(file_name)
    os.makedirs(f"outputs/SD3_strength/{image_name}", exist_ok=True)
    os.makedirs(f"outputs/SD3_remove/{image_name}", exist_ok=True)
    os.makedirs(f"outputs/SD3_empty/{image_name}", exist_ok=True)
    print(prompt, image_name)
    for i in range(38):
        # image = pipe(
        #     prompt=prompt + positive_magic["en"],
        #     negative_prompt=negative_prompt,
        #     width=width,
        #     height=height,
        #     num_inference_steps=50,
        #     true_cfg_scale=4.0,
        #     generator=torch.Generator(device="cuda").manual_seed(42),
        #     skpped_layers=i,
        # ).images[0]
        # image.save(f"outputs/remove_layers/{image_name}_{i}.png")

        image_empty = pipe(
            prompt=prompt + positive_magic["en"],
            negative_prompt=negative_prompt,
            width=1024,
            height=1024,
            num_inference_steps=28,
            true_cfg_scale=7.0,
            modulated_layers=i,
            modulated_ways="empty"
        ).images[0]
        image_empty.save(f"outputs/SD3_empty/{image_name}_{i}.png")

        image_1 = pipe(
            prompt=prompt + positive_magic["en"],
            negative_prompt=negative_prompt,
            width=1024,
            height=1024,
            num_inference_steps=28,
            guidance_scale=7.0,
            modulated_layers=i,
            modulated_scales=2.0,
            modulated_ways="strength"
        ).images[0]
        image_1.save(f"outputs/SD3_strength/{image_name}/layer_{i}.png")

        image_2 = pipe(
            prompt=prompt + positive_magic["en"],
            negative_prompt=negative_prompt,
            width=1024,
            height=1024,
            num_inference_steps=28,
            guidance_scale=7.0,
            removed_layers=i
        ).images[0]
        image_2.save(f"outputs/SD3_remove/{image_name}/layer_{i}.png")