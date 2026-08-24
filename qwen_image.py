import os
import argparse
from typing import List, Tuple

import torch
import torch.distributed as dist
from diffusers import DiffusionPipeline


ASPECT_RATIOS = {
    "1:1": (1328, 1328),
    "16:9": (1664, 928),
    "9:16": (928, 1664),
    "4:3": (1472, 1104),
    "3:4": (1104, 1472),
    "3:2": (1584, 1056),
    "2:3": (1056, 1584),
}


def setup_distributed() -> Tuple[int, int, str]:
    """Initialize torch.distributed and return (rank, world_size, device)."""
    if not dist.is_initialized():
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend)

    rank = dist.get_rank()
    world_size = dist.get_world_size()

    if torch.cuda.is_available():
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
        device = f"cuda:{local_rank}"
    else:
        device = "cpu"

    return rank, world_size, device


def broadcast_dirs_creation(rank: int, dirs: List[str]):
    if rank == 0:
        for d in dirs:
            os.makedirs(d, exist_ok=True)
    dist.barrier()


def parse_args():
    parser = argparse.ArgumentParser(description="Distributed Qwen-Image generation")
    parser.add_argument("--model_path", type=str, default="./pretrained_models/Qwen-Image")
    parser.add_argument("--prompts_file", type=str, default="files/prompts_90.txt")
    parser.add_argument("--out_empty", type=str, default="outputs/Qwen_empty")
    parser.add_argument("--out_strength", type=str, default="outputs/Qwen_strength")
    parser.add_argument("--out_remove", type=str, default="outputs/Qwen_remove")
    parser.add_argument("--aspect", type=str, default="16:9", choices=list(ASPECT_RATIOS.keys()))
    parser.add_argument("--num_layers", type=int, default=60)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_prompts(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        prompts = [line.strip() for line in f if line.strip()]
    return prompts


def shard(items: List[str], rank: int, world_size: int) -> List[str]:
    return items[rank::world_size]


def main():
    args = parse_args()

    rank, world_size, device = setup_distributed()
    print(rank, world_size, device)
    torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    positive_magic = {
        "en": "Ultra HD, 4K, cinematic composition.",
    }

    width, height = ASPECT_RATIOS[args.aspect]

    # Create root output dirs once
    broadcast_dirs_creation(rank, [args.out_empty, args.out_strength])

    # Load pipeline per rank
    pipe = DiffusionPipeline.from_pretrained(args.model_path, torch_dtype=torch_dtype)
    pipe = pipe.to(device)

    prompts = load_prompts(args.prompts_file)
    prompts_shard = shard(prompts, rank, world_size)

    negative_prompt = "nsfw, paintings, cartoon, anime, sketches, worst quality, low quality, normal quality, lowres, watermark, monochrome, grayscale, ugly, blurry, Tan skin, dark skin, black skin, skin spots, skin blemishes, age spot, glans, disabled, bad anatomy, amputation, bad proportions, twins, missing body, fused body, extra head, poorly drawn face, bad eyes, deformed eye, unclear eyes, cross-eyed, long neck, malformed limbs, extra limbs, extra arms, missing arms, bad tongue, strange fingers, mutated hands, missing hands, poorly drawn hands, extra hands, fused hands, connected hand, bad hands, missing fingers, extra fingers, 4 fingers, 3 fingers, deformed hands, extra legs, bad legs, many legs, more than two legs, bad feet, extra feets"

    for prompt in prompts_shard:
        file_name = prompt.split()
        image_name = "_".join(file_name)
        out_dir_empty = os.path.join(args.out_empty, image_name)
        out_dir_strength = os.path.join(args.out_strength, image_name)
        out_dir_remove = os.path.join(args.out_remove, image_name)

        # Each rank may touch different prompts; ensure dir existence without race
        os.makedirs(out_dir_empty, exist_ok=True)
        os.makedirs(out_dir_strength, exist_ok=True)
        os.makedirs(out_dir_remove, exist_ok=True)

        if rank == 0:
            print(f"[rank {rank}] Processing prompt: {prompt} -> {image_name}")

        for i in range(args.num_layers):
            gen = torch.Generator(device=device)
            gen.manual_seed(args.seed + i + rank * 1000)

            image_1 = pipe(
                prompt=prompt + positive_magic["en"],
                negative_prompt=negative_prompt,
                width=width,
                height=height,
                num_inference_steps=args.steps,
                true_cfg_scale=4.0,
                generator=gen,
                modulated_layers=i,
                modulated_ways="empty",
            ).images[0]
            image_1.save(os.path.join(out_dir_empty, f"layer_{i}.png"))

            image_2 = pipe(
                prompt=prompt + positive_magic["en"],
                negative_prompt=negative_prompt,
                width=width,
                height=height,
                num_inference_steps=args.steps,
                true_cfg_scale=4.0,
                generator=gen,
                modulated_layers=i,
                modulated_scales=2.0,
                modulated_ways="strong",
            ).images[0]
            image_2.save(os.path.join(out_dir_strength, f"layer_{i}.png"))

            image_3 = pipe(
                prompt=prompt + positive_magic["en"],
                negative_prompt=negative_prompt,
                width=width,
                height=height,
                num_inference_steps=args.steps,
                true_cfg_scale=4.0,
                generator=gen,
                removed_layers=i,
            ).images[0]
            image_3.save(os.path.join(out_dir_remove, f"layer_{i}.png"))

    dist.barrier()
    if rank == 0:
        print("All ranks finished.")


if __name__ == "__main__":
    main()

