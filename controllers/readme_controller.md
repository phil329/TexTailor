# Text Embedding Controller

A non-intrusive text embedding controller that operates at the Transformer Block level of diffusion models to precisely manipulate `encoder_hidden_states` (text conditioning). **No modifications to the diffusers source code are required** — implemented via PyTorch's native hook mechanism.

Supports three models:

| File | Class | Target Model |
| ---- | ----- | ------------ |
| `flux_text_controller.py` | `FluxTextEmbeddingController` | FLUX.1-dev / FLUX.1-schnell |
| `sd3_text_controller.py` | `SD3TextEmbeddingController` | Stable Diffusion 3.0 / 3.5 |
| `qwen_text_controller.py` | `QwenTextEmbeddingController` | Qwen-Image |

---

## How It Works

All three models iterate over Transformer blocks sequentially. Each block processes both image tokens and text tokens (`encoder_hidden_states`) together in joint attention. This controller intercepts and modifies `encoder_hidden_states` **before** each block is called, enabling precise control over text conditioning at specific layers.

### Three Operations

| Operation | Effect | Mechanism |
| --------- | ------ | --------- |
| **remove** | Skip the specified block entirely, performing no computation | Temporarily replaces `block.forward` with a passthrough function |
| **empty** | Zero out text embeddings before the specified block | `register_forward_pre_hook` |
| **strengthen** | Multiply text embeddings by a scale factor before the specified block | `register_forward_pre_hook` |

> **remove** truly skips computation (saves memory and time); **empty** makes the layer blind to all text information; **strengthen** amplifies (scale > 1) or attenuates (scale < 1) text guidance at that layer.

---

## Block Structure by Model

### FLUX

```
transformer_blocks[0..18]           Dual-stream blocks (dual-stream DiT), 19 total
single_transformer_blocks[0..37]    Single-stream blocks, 38 total

Global index convention (same as the intrusive version):
  global 0  ~ 18  → transformer_blocks[global_idx]
  global 19 ~ 56  → single_transformer_blocks[global_idx - 19]
```

### Stable Diffusion 3 / 3.5

```
transformer_blocks[0..23]   JointTransformerBlock, 24 total

Note:
  The last block (index=23) has context_pre_only=True,
  so its output has encoder_hidden_states=None (by SD3 design).
  The controller handles this transparently — on remove, it correctly returns (None, hidden_states).
```

### Qwen-Image

```
transformer_blocks[0..59]   QwenImageTransformerBlock, 60 total

Note:
  Block calls include an extra encoder_hidden_states_mask argument.
  The controller only modifies encoder_hidden_states; the mask is left unchanged.
```

---

## Quick Start

### Requirements

```
Python >= 3.8
PyTorch >= 2.0   # register_forward_pre_hook requires with_kwargs=True
diffusers        # stock version, no source modifications needed
```

### Import

```python
from flux_text_controller import FluxTextEmbeddingController
from sd3_text_controller   import SD3TextEmbeddingController
from qwen_text_controller  import QwenTextEmbeddingController
```

---

## Usage

All three controllers share an identical interface. Examples below use Flux.

### Recommended: Context Manager (automatic attach / detach)

```python
import torch
from diffusers import FluxPipeline
from flux_text_controller import FluxTextEmbeddingController

pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16).to("cuda")
controller = FluxTextEmbeddingController()

with controller.control(
    pipe.transformer,
    removed_layers=[5, 10],          # skip block 5 and block 10
    modulated_layers=[3, 7],         # modify text conditioning before block 3 and block 7
    modulated_scales=[2.0, 3.0],     # multiply by 2.0 and 3.0 respectively
    modulated_ways="strengthen",     # "empty" or "strengthen"
):
    image = pipe("a beautiful landscape", num_inference_steps=20).images[0]
```

> After the `with` block exits, all hooks and forward replacements are automatically removed and the transformer is fully restored.

### Manual attach / detach

```python
controller.attach(
    pipe.transformer,
    modulated_layers=[3],
    modulated_ways="empty",
)

image = pipe("a beautiful landscape", num_inference_steps=20).images[0]

controller.detach()   # must be called manually, otherwise hooks remain active for subsequent inference
```

---

## Parameter Reference

All controllers share the same parameter signature:

```python
controller.attach(
    transformer,          # the model's transformer instance (pipe.transformer)
    removed_layers,       # int or List[int], global block indices to skip
    modulated_layers,     # int or List[int], global block indices where text conditioning is modified
    modulated_scales,     # float or List[float], scale factor for each modulated layer
                          #   only effective with strengthen; defaults to 2.0 when None
    modulated_ranges,     # (start, end) or [(s0,e0), ...], token-level range(s) to operate on (optional)
                          #   only effective with strengthen; None applies to all tokens
    modulated_ways,       # str, "empty" (zero out) or "strengthen" (scale), default "empty"
)
```

### `removed_layers` vs `modulated_layers`

Both can be used simultaneously and **do not interfere with each other**:

```python
with controller.control(
    pipe.transformer,
    removed_layers=[2, 4],           # blocks 2 and 4 are skipped entirely
    modulated_layers=[6, 8],         # blocks 6 and 8 have text embeddings scaled before execution
    modulated_scales=1.5,            # broadcast: all modulated layers use scale 1.5
    modulated_ways="strengthen",
):
    ...
```

---

## Examples

### empty: Remove text conditioning entirely at a layer

```python
# When block 3 executes, encoder_hidden_states is all zeros → visual generation at this layer is unguided by text
with controller.control(pipe.transformer, modulated_layers=[3], modulated_ways="empty"):
    image = pipe(prompt, ...).images[0]
```

### strengthen: Amplify text guidance for all tokens

```python
# Text embeddings × 2.5 before block 5
with controller.control(
    pipe.transformer,
    modulated_layers=[5],
    modulated_scales=[2.5],
    modulated_ways="strengthen",
):
    image = pipe(prompt, ...).images[0]
```

### strengthen: Amplify a specific token range only

```python
# Before block 5, multiply only token positions [0, 20] by 3.0; all other tokens unchanged
with controller.control(
    pipe.transformer,
    modulated_layers=[5],
    modulated_scales=[3.0],
    modulated_ranges=[(0, 20)],      # closed interval, both endpoints inclusive
    modulated_ways="strengthen",
):
    image = pipe(prompt, ...).images[0]
```

### Multiple token ranges

```python
# Amplify two separate ranges [0,5] and [15,25] using the same scale
with controller.control(
    pipe.transformer,
    modulated_layers=[3],
    modulated_scales=[2.0],
    modulated_ranges=[(0, 5), (15, 25)],
    modulated_ways="strengthen",
):
    image = pipe(prompt, ...).images[0]
```

### remove: Skip specified blocks

```python
# Blocks 5 and 10 are bypassed; hidden_states pass through unchanged to the next block
with controller.control(pipe.transformer, removed_layers=[5, 10]):
    image = pipe(prompt, ...).images[0]
```

### Combined usage (remove + strengthen + empty)

```python
with controller.control(
    pipe.transformer,
    removed_layers=[2],
    modulated_layers=[5, 8],
    modulated_scales=[2.0, 0.5],     # block 5 amplified, block 8 attenuated
    modulated_ways="strengthen",
):
    image = pipe(prompt, ...).images[0]
```

---

## FLUX Global Index Reference

FLUX has two types of blocks (dual-stream and single-stream) addressed via a unified global index:

```
Dual-stream blocks (FluxTransformerBlock):
  global 0  → transformer_blocks[0]
  global 1  → transformer_blocks[1]
  ...
  global 18 → transformer_blocks[18]

Single-stream blocks (FluxSingleTransformerBlock):
  global 19 → single_transformer_blocks[0]
  global 20 → single_transformer_blocks[1]
  ...
  global 56 → single_transformer_blocks[37]
```

Example — operating on the 5th single-stream block:

```python
# single_transformer_blocks[5] corresponds to global index 19 + 5 = 24
with controller.control(pipe.transformer, modulated_layers=[24], modulated_scales=[2.0], modulated_ways="strengthen"):
    ...
```

---

## Notes

1. **Always pair `detach` calls**: When using the manual form, forgetting to call `detach()` leaves hooks active and affects all subsequent inference. The `with controller.control(...)` context manager is recommended as it guarantees automatic cleanup.

2. **`removed_layers` truly skips computation**: Unlike `empty`, `remove` does not execute the block's forward pass, saving the compute cost of those layers.

3. **`modulated_ranges` has no effect with `empty`**: `modulated_ranges` only applies when `modulated_ways="strengthen"`; `empty` always zeros out all tokens.

4. **scale < 1 attenuates guidance**: `modulated_scales=0.0` is equivalent to `empty`; `modulated_scales=0.5` halves the text guidance strength.

5. **PyTorch version requirement**: `register_forward_pre_hook` with `with_kwargs=True` requires PyTorch >= 2.0.

6. **Compatible with gradient checkpointing**: Hooks are registered at the `nn.Module.__call__` level of each block. Gradient checkpointing is controlled by the transformer's own forward pass — the two are independent and do not conflict.
