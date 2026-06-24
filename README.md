<div align="center">


<h1>TexTailor: Inference-Time Textual Guidance Tailoring for Multimodal Diffusion Transformers | ECCV 2026</h1>

<div>
<a href="https://github.com/phil329" target="_blank">Binglei Li</a> <sup>1,2</sup> | 
<a href="https://github.com/kobeshegu" target="_blank">Mengping Yang </a><sup>1,3,*</sup> |
<a href="https://scholar.google.com/citations?user=XprTQQ8AAAAJ&hl=en" target="_blank">Zhiyu Tan </a><sup>1,3</sup> | <br>
<a href="https://cs.fudan.edu.cn/3f/f9/c25909a278521/page.htm" target="_blank">Junping Zhang </a><sup>1,#</sup> |
<a href="https://ai3.fudan.edu.cn/info/1088/1694.htm" target="_blank">Hao Li </a><sup>1,2,3,,#</sup> 
</div>
<br>
<div>
    <sup></sup><sup>1</sup> Fudan University  <sup>2</sup> Shanghai Innovation Institute  <sup>3</sup> Shanghai Academy of AI for Science
</div>
<div>
    <sup>*</sup> Project Lead &nbsp;&nbsp;&nbsp; <sup>#</sup> Corresponding Authors  
</div>

[![arXiv](https://img.shields.io/badge/arXiv-2601.02211-b31b1b.svg)](https://arxiv.org/abs/2601.02211) [![ECCV](https://img.shields.io/badge/ECCV-2026-4b44ce.svg)](https://arxiv.org/abs/2601.02211)

The repository contains the code for the `TexTailor` method presented in the paper: *TexTailor: Inference-Time Textual Guidance Tailoring for Multimodal Diffusion Transformers **(ECCV 2026)***.



![image](./asserts/teaser.png)

<div align="left"> 

## 🛠️ Method Overview
This repository provides code and resources for analyzing and improving Multimodal Diffusion Transformers (MMDiT) in text-to-image generation and editing. We introduce a systematic pipeline to investigate the roles of different blocks and their interactions with textual conditions in MMDiT-based models such as FLUX and Qwen Image.

Key features:
- Block-wise analysis: Remove, disable, or enhance textual hidden-states at specific blocks to study their impact.
- Insights: Early blocks capture semantic information, later blocks render finer details, and selective enhancement of textual conditions improves semantic attributes.
- Training-free strategies: Methods for better text alignment, precise image editing, and faster inference.
- Performance: Our approach improves T2I-Combench and GenEval scores without sacrificing synthesis quality.

Refer to the documentation and examples to get started with block analysis, editing, and acceleration for your own diffusion models.

## 🚀 Getting Started
### Environment Requirement 🌍

We recommend using Python 3.8+ and PyTorch 1.12+ with CUDA support. The environment is compatible with `diffusers==0.35.0` or you can install the local version of diffusers in this repo.

```shell
pip install -r requirements.txt

cd diffusers
pip install -e .
```

### Minimal Example for Inference 🐍
We provide a minimal example for inference using the FLUX model.The only thing you need to modify is the inference parameters `modulated_layers` and `modulated_scales` for enhancing text alignment. `modulated_phases` can also be set for token-level enhancement.


```python

import torch
from diffusers import FluxPipeline 
# make sure import the local version of diffusers


pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16)
pipe.enable_model_cpu_offload() #save some VRAM by offloading the model to CPU. Remove this if you have enough GPU power

prompt = "A cat and dog playing together in the park, photorealistic, high quality, 4k"

image = pipe(
    prompt,
    height=1024,
    width=1024,
    guidance_scale=3.5,
    num_inference_steps=50,
    max_sequence_length=512,
    removed_layers=None,
    generator=torch.Generator("cpu").manual_seed(0)
    modulated_layers=[2,7,12,17,22], # improve non-spatial text alignment  
    modulated_scales=1.5,   # Optional[float, List[float]] = 1.5,
    modulated_phases=None,  # phases to enhance, Optional[List[str]] = None, using sentence-level is None
    # modulated_ways=modulated_ways # using 'empty' for probing analysis
).images[0]
image.save("flux-dev-enhance.png")

```

### Non-intrusive Controller (No Source Modifications) 🔌

As an alternative to the modified pipeline API above, we provide standalone controllers that work with the **original, unmodified** diffusers library via PyTorch native hooks.  No source-code edits are required.

| File | Class | Target Model |
| ---- | ----- | ------------ |
| `controllers/flux_text_controller.py` | `FluxTextEmbeddingController` | FLUX.1-dev / FLUX.1-schnell |
| `controllers/sd3_text_controller.py` | `SD3TextEmbeddingController` | Stable Diffusion 3.0 / 3.5 |
| `controllers/qwen_text_controller.py` | `QwenTextEmbeddingController` | Qwen-Image |

All three controllers share the same interface.  The recommended usage is via the context manager, which automatically removes all hooks after generation:

```python
import torch
from diffusers import FluxPipeline
from controllers.flux_text_controller import FluxTextEmbeddingController

pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16)
pipe.enable_model_cpu_offload()
controller = FluxTextEmbeddingController()

prompt = "A cat and dog playing together in the park, photorealistic, high quality, 4k"

# Strengthen text conditioning at selected blocks
with controller.control(
    pipe.transformer,
    modulated_layers=[2, 7, 12, 17, 22],
    modulated_scales=1.5,
    modulated_ways="strengthen",
):
    image = pipe(prompt, height=1024, width=1024, guidance_scale=3.5,
                 num_inference_steps=50).images[0]
image.save("flux-dev-enhance.png")
```

Three operations are supported at each block:

| Operation | Effect |
| --------- | ------ |
| `strengthen` | Multiply `encoder_hidden_states` by `modulated_scales` before the block (amplify or attenuate text guidance) |
| `empty` | Zero out `encoder_hidden_states` before the block (disables text conditioning at that layer) |
| `remove` | Skip the block entirely — no computation, saves memory and time |

Token-level control is available for `strengthen` via `modulated_ranges`:

```python
# Only scale tokens at positions [0, 20] (inclusive) — useful for phrase-level enhancement
with controller.control(
    pipe.transformer,
    modulated_layers=[5],
    modulated_scales=[1.5],
    modulated_ranges=[(0, 20)],
    modulated_ways="strengthen",
):
    image = pipe(prompt, ...).images[0]
```

For the full parameter reference, block-index conventions, and SD3 / Qwen specifics, see [`controllers/readme_controller.md`](controllers/readme_controller.md).

### Image Editing ✂️

We do not have time to provide a minimal example for editing. For editing, we implement our proposed enhance techniques on selected blocks on [StableFlow](https://github.com/snap-research/stable-flow).

### Acceleration ⚡

We provide a script `flux_teacache_ours.py` for acceleration using our methods compatible with Teacache on FLUX model.  You can run the script as follows for acceleration evaluation. If you want to remove or modulate different layers, just change the `layers` and `layers2` variables.

```shell
export layers="5 10 15 20 25 30 35 40 45 50 55"
export layers2="30 40 50"
python flux_teacache_ours.py \
    --prompt_file ./T2I_CompBench_sampled_160prompts.txt \
    --save_dir ./teacache_results/flux_noteacache0.4_160_wocfg_${layers// /_}_remove${layers2// /_} \
    --removed_cfg_layers ${layers} --removed_layers ${layers2} \
    --enable_teacache --teacache_strength 0.4
```

## Parameters that can be modified for different applications

The main changes are the following parameters added to the `__call__` function of the SD3, FLUX, and Qwen Image pipelines:

| Parameter | Type / Values | Default | Description |
| --------- | ------------- | ------- | ----------- |
| `removed_layers` | `Optional[Union[int, List[int]]]` | `None` | Layers to **remove** — blocks in this list are skipped during inference. |
| `modulated_layers` | `Optional[Union[int, List[int]]]` | `None` | Layers where **text conditions are modulated**. |
| `modulated_scales` | `Optional[Union[float, List[float]]]` | `1.5` | **Scaling factor(s)** controlling the strength of modulation. Only effective if `modulated_layers` is set. |
| `modulated_phases` | `Optional[Union[str, List[str]]]` | `None` | **Target phrases** for token-level enhancement. Only effective if `modulated_layers` is set. |
| `modulated_ways` | `"strengthen"` \| `"empty"` | `"strengthen"` | Defines **how** to modulate: `"strengthen"` amplifies text conditions, `"empty"` zeros them for probing analysis. |

## Datasets 📂 and Evaluation 🥇

For the probing analysis, we use the datasets in `./prompts` folders, which have been filtered by human checking. The datasets include:

- `prompts_number_filterd.txt`: A subset of prompts focusing on numerical attributes, used to evaluate the model's ability to accurately render numbers in generated images.
- `prompts_object_color_filtered.txt`: A sub collection of prompts emphasizing object colors, designed to assess the model's performance in capturing and reproducing color details in images.
- `prompts_object_position_filtered.txt`: A subdataset of prompts centered around spatial relationships, aimed at evaluating how well the model understands and represents spatial arrangements in generated images.

We also provide the QwenVL-2.5 VQA code in `./QwenVL_VQA` for probing analysis. For evaluation, we use T2I-CompBench++ and GenEval.

## Acknowledgements

We would like to thank the following open-source projects and their contributors for providing benchmarks and tools that facilitated our research: [T2I-CompBench++](https://github.com/Karine-Huang/T2I-CompBench), [GenEval](https://github.com/djghosh13/geneval), [CountGD](https://github.com/niki-amini-naieni/CountGD), [Stable Flow](https://github.com/snap-research/stable-flow), [Teacache](https://github.com/ali-vilab/TeaCache), and [Diffusers](https://github.com/huggingface/diffusers).


## Citation
If you find this work useful in your research, please cite:
```
@article{li2026textailor,
  title={{TexTailor}: Inference-Time Textual Guidance Tailoring for Multimodal Diffusion Transformers},
  author={Binglei Li and Mengping Yang and Zhiyu Tan and Junping Zhang and Hao Li},
  journal={arXiv preprint arXiv:2601.02211},
  year={2026}
}
```