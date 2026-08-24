from diffusers import DiffusionPipeline
import torch

model_name = "Qwen/Qwen-Image"
model_path = "./pretrained_models/Qwen-Image"

# Load the pipeline
if torch.cuda.is_available():
    torch_dtype = torch.bfloat16
    device = "cuda"
else:
    torch_dtype = torch.float32
    device = "cpu"

pipe = DiffusionPipeline.from_pretrained(model_path, torch_dtype=torch_dtype)
pipe = pipe.to(device)

positive_magic = {
    "en": "Ultra HD, 4K, cinematic composition.", # for english prompt
    "zh": "超清，4K，电影级构图" # for chinese prompt
}

# Generate image
prompt = '''A coffee shop entrance features a chalkboard sign reading "Qwen Coffee 😊 $2 per cup," with a neon light beside it displaying "通义千问". Next to it hangs a poster showing a beautiful Chinese woman, and beneath the poster is written "π≈3.1415926-53589793-23846264-33832795-02384197".'''


with open('files/prompts_90.txt', 'r', encoding='utf-8') as file:
     prompts = [line.strip() for line in file]


negative_prompt = " " # Recommended if you don't use a negative prompt.


# Generate with different aspect ratios
aspect_ratios = {
    "1:1": (1328, 1328),
    "16:9": (1664, 928),
    "9:16": (928, 1664),
    "4:3": (1472, 1104),
    "3:4": (1104, 1472),
    "3:2": (1584, 1056),
    "2:3": (1056, 1584),
}

width, height = aspect_ratios["16:9"]

for prompt in prompts:
    file_name = prompt.split()
    image_name = "_".join(file_name)
    print(prompt, image_name)
    for i in range(60):
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

        # image = pipe(
        #     prompt=prompt + positive_magic["en"],
        #     negative_prompt=negative_prompt,
        #     width=width,
        #     height=height,
        #     num_inference_steps=50,
        #     true_cfg_scale=4.0,
        #     generator=torch.Generator(device="cuda").manual_seed(42),
        #     modulated_layers=i,
        #     modulated_ways="empty"
        # ).images[0]
        # image.save(f"outputs/remove_conditions/{image_name}_{i}.png")

        image = pipe(
            prompt=prompt + positive_magic["en"],
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            num_inference_steps=50,
            true_cfg_scale=4.0,
            generator=torch.Generator(device="cuda").manual_seed(42),
            modulated_layers=i,
            modulated_scales=2.0,
            modulated_ways="strength"
        ).images[0]
        image.save(f"outputs/strength_conditions_v3/{image_name}_{i}.png")