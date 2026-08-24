from diffusers import DiffusionPipeline
import torch
import os
import argparse
import pandas as pd
import torch.multiprocessing as mp

positive_magic = {
    "en": "Ultra HD, 4K, cinematic composition.", # for english prompt
}
negative_prompt = " "


model_path = "./pretrained_models/FLUX.1-dev"


# ========== single prompt processing ==========
def process_prompt(pipe, prompt, number=None, device_id=0,output_folder='outputs',manual_seed=42):
    file_name = prompt.split()
    if number is not None:
        file_name.append(str(number))
    image_name = "_".join(file_name)

    os.makedirs(f"{output_folder}/flux_strength_seed{manual_seed}/{image_name}", exist_ok=True)
    os.makedirs(f"{output_folder}/flux_remove_seed{manual_seed}/{image_name}", exist_ok=True)
    os.makedirs(f"{output_folder}/flux_empty_seed{manual_seed}/{image_name}", exist_ok=True)
    

    filename_0 = f"{output_folder}/flux_strength_seed{manual_seed}/{image_name}/baseline.png"
    if not os.path.exists(filename_0):
        image_0 = pipe(
            prompt=prompt + positive_magic["en"],
            negative_prompt=negative_prompt,
            width=1024,
            height=1024,
            num_inference_steps=50,
            guidance_scale=3.5,
            generator=torch.Generator(device=f"cuda:{device_id}").manual_seed(manual_seed),
        ).images[0]
        image_0.save(filename_0)
        image_0.save(f"{output_folder}/flux_remove_seed{manual_seed}/{image_name}/baseline.png")
        image_0.save(f"{output_folder}/flux_empty_seed{manual_seed}/{image_name}/baseline.png")
        
        print(f"[GPU {device_id}] Generating {image_name} - baseline")
    else:
        print(f"[GPU {device_id}] {image_name} - baseline exists, skipping")
    

    for i in range(58):
        print(f"[GPU {device_id}] Generating {image_name} - layer {i}")

        # flux_strength
        filename_1 = f"{output_folder}/flux_strength_seed{manual_seed}/{image_name}/layer_{i}.png"
        if not os.path.exists(filename_1):
            image_1 = pipe(
                prompt=prompt + positive_magic["en"],
                negative_prompt=negative_prompt,
                width=1024,
                height=1024,
                num_inference_steps=50,
                guidance_scale=3.5,
                generator=torch.Generator(device=f"cuda:{device_id}").manual_seed(manual_seed),
                modulated_layers=i,
                modulated_ways="strength"
            ).images[0]
            image_1.save(filename_1)
        else:
            print(f"[GPU {device_id}] {image_name} - layer {i} strength exists, skipping")

        # flux_remove
        filename_2 = f"{output_folder}/flux_remove_seed{manual_seed}/{image_name}/layer_{i}.png"
        if not os.path.exists(filename_2):
            image_2 = pipe(
                prompt=prompt + positive_magic["en"],
                negative_prompt=negative_prompt,
                width=1024,
                height=1024,
                num_inference_steps=50,
                guidance_scale=3.5,
                generator=torch.Generator(device=f"cuda:{device_id}").manual_seed(manual_seed),
                removed_layer=i
            ).images[0]
            image_2.save(filename_2)
        else:
            print(f"[GPU {device_id}] {image_name} - layer {i} remove exists, skipping")

        # flux_empty
        filename_3 = f"{output_folder}/flux_empty_seed{manual_seed}/{image_name}/layer_{i}.png"
        if not os.path.exists(filename_3):
            image_3 = pipe(
                prompt=prompt + positive_magic["en"],
                negative_prompt=negative_prompt,
                width=1024,
                height=1024,
                num_inference_steps=50,
                guidance_scale=3.5,
                generator=torch.Generator(device=f"cuda:{device_id}").manual_seed(manual_seed),
                modulated_layers=i,
                modulated_ways="empty"
            ).images[0]
            image_3.save(filename_3)
        else:
            print(f"[GPU {device_id}] {image_name} - layer {i} empty exists, skipping")


# ========== every GPU has its own worker ==========
def gpu_worker(device_id, task_queue,output_folder="outputs", manual_seed=42):
    print(f"[GPU {device_id}] Worker started")

    # Moving model loading inside the worker to avoid CUDA OOM on multi-GPU setups
    pipe = DiffusionPipeline.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        use_safetensors=True
    )
    pipe = pipe.to(f"cuda:{device_id}")

    while True:
        item = task_queue.get()
        if item is None:  # End signal
            break
        prompt, number = item
        process_prompt(pipe, prompt, number, device_id,output_folder,manual_seed)


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser("Inference on Qwen Image", add_help=False)
    parser.add_argument("--output_folder", type=str,default='outputs', help="folder to save output images")
    parser.add_argument('--prompt_file',type=str,help="file to savce prompts and labels")
    parser.add_argument('--manual_seed',type=int,default=42)
    args = parser.parse_args()
    
    
    mp.set_start_method("spawn")
    
    os.makedirs(f"{args.output_folder}/flux_strength_seed{args.manual_seed}", exist_ok=True)
    os.makedirs(f"{args.output_folder}/flux_remove_seed{args.manual_seed}", exist_ok=True)
    os.makedirs(f"{args.output_folder}/flux_empty_seed{args.manual_seed}", exist_ok=True)


    # read prompts CSV
    df = pd.read_csv(args.prompt_file)
    prompts = df["caption"].tolist()
    
    if "number" in df.columns:
        numbers = df["number"].tolist()
    else:
        numbers = [None] * len(prompts)

    num_gpus = torch.cuda.device_count()
    print(f"Detected {num_gpus} GPUs.")

    # Establish a queue
    task_queue = mp.Queue()

    # Start GPU worker
    processes = []
    for i in range(num_gpus):
        p = mp.Process(target=gpu_worker, args=(i, task_queue,args.output_folder,args.manual_seed))
        p.start()
        processes.append(p)

    # Put tasks into the queue, workers will pick them up
    for prompt, number in zip(prompts, numbers):
        task_queue.put((prompt, number))

    # Send end signal
    for _ in range(num_gpus):
        task_queue.put(None)

    for p in processes:
        p.join()

    print("All prompts processed.")
