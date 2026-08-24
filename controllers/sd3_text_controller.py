"""
sd3_text_controller.py
======================
Non-intrusive text-embedding controller for SD3Transformer2DModel
(Stable Diffusion 3 / 3.5).

Works with the **original, unmodified** diffusers SD3Transformer2DModel and
StableDiffusion3Pipeline — no source-code edits required.

Three operations are supported (matching the intrusive version):

  remove     : Skip specified blocks entirely (pass hidden_states /
                encoder_hidden_states through unchanged, no computation).

  empty      : Zero out encoder_hidden_states **before** a block runs.

  strengthen : Multiply encoder_hidden_states by `scale` before a block,
                optionally restricted to a token sub-range.

SD3 block-list structure
------------------------
  SD3.0 : 24 JointTransformerBlock layers in transformer_blocks[0..23]
  SD3.5 Large: same 24 blocks + dual-attention in 13 of them

  The **last** block has context_pre_only=True, meaning it sets
  encoder_hidden_states = None in its output.  This controller handles
  that edge case transparently for both the skip and pre-hook operations.

Index convention
----------------
  index  0 … num_layers-1  →  transformer_blocks[index]

Usage
-----
    from sd3_text_controller import SD3TextEmbeddingController

    controller = SD3TextEmbeddingController()

    # ── context-manager form (recommended) ───────────────────────────
    with controller.control(
        pipe.transformer,
        removed_layers=[5, 10],
        modulated_layers=[3, 7],
        modulated_scales=[2.0, 3.0],
        modulated_ranges=[(0, 20)],        # optional token range(s)
        modulated_ways="strengthen",       # "empty" | "strengthen"
    ):
        output = pipe(prompt=..., ...)

    # ── manual form ──────────────────────────────────────────────────
    controller.attach(pipe.transformer, removed_layers=[3], modulated_ways="empty")
    output = pipe(prompt=..., ...)
    controller.detach()
"""

import functools
from contextlib import contextmanager
from typing import List, Optional, Tuple, Union

import torch


# ─────────────────────────────────────────────────────────────────────────────
# helpers
# ─────────────────────────────────────────────────────────────────────────────

def _to_int_list(x) -> Optional[List[int]]:
    if x is None:
        return None
    if isinstance(x, int):
        return [x]
    return list(x)


def _to_float_list(scales, n: int) -> List[float]:
    if scales is None:
        return [2.0] * n
    if isinstance(scales, (int, float)):
        return [float(scales)] * n
    return [float(s) for s in scales]


def _to_range_list(ranges) -> Optional[List[Tuple[int, int]]]:
    """Accept (start, end) or [(s0,e0), (s1,e1), ...]."""
    if ranges is None:
        return None
    if isinstance(ranges, tuple) and len(ranges) == 2 and isinstance(ranges[0], int):
        return [ranges]
    return list(ranges)


# ─────────────────────────────────────────────────────────────────────────────
# controller
# ─────────────────────────────────────────────────────────────────────────────

class SD3TextEmbeddingController:
    """
    Non-intrusive text-embedding controller for SD3Transformer2DModel.

    Attaches PyTorch forward hooks / temporary forward replacements to the
    transformer's individual blocks.  The transformer's own forward() method
    is never touched.

    Special case — context_pre_only block (last block in SD3)
    ----------------------------------------------------------
    The last JointTransformerBlock in SD3 has context_pre_only=True.
    Its forward() sets encoder_hidden_states = None and returns (None, hidden_states).

    • empty / strengthen on this block still work correctly: zeroing or scaling
      encoder_hidden_states before the block affects the norm1_context
      computation; since the block discards enc anyway, the overall pipeline
      output is unchanged in meaning but the internal attention differs.

    • remove on this block returns (encoder_hidden_states_unchanged, hidden_states)
      instead of (None, hidden_states).  The SD3 transformer forward loop
      does not use encoder_hidden_states after the last block, so this is safe.
    """

    def __init__(self) -> None:
        self._hooks: list = []
        self._patched_blocks: list = []

    # ------------------------------------------------------------------ #
    # index → block mapping
    # ------------------------------------------------------------------ #

    @staticmethod
    def _get_block(transformer, idx: int) -> torch.nn.Module:
        blocks = transformer.transformer_blocks
        if idx < 0 or idx >= len(blocks):
            raise IndexError(
                f"Block index {idx} out of range (num_layers={len(blocks)})."
            )
        return blocks[idx]

    # ------------------------------------------------------------------ #
    # hook / forward factories
    # ------------------------------------------------------------------ #

    @staticmethod
    def _make_empty_pre_hook():
        """Zero out encoder_hidden_states before the block."""
        def hook(module, args, kwargs):
            enc = kwargs.get("encoder_hidden_states")
            if enc is not None:
                kwargs["encoder_hidden_states"] = torch.zeros_like(enc)
                return args, kwargs
        return hook

    @staticmethod
    def _make_strengthen_pre_hook(
        scale: float,
        token_ranges: Optional[List[Tuple[int, int]]] = None,
    ):
        """Scale encoder_hidden_states (optionally per-token-range) before the block."""
        def hook(module, args, kwargs):
            enc = kwargs.get("encoder_hidden_states")
            if enc is not None:
                if token_ranges is not None:
                    enc = enc.clone()
                    for start, end in token_ranges:
                        enc[:, start : end + 1, :] = enc[:, start : end + 1, :] * scale
                    kwargs["encoder_hidden_states"] = enc
                else:
                    kwargs["encoder_hidden_states"] = enc * scale
                return args, kwargs
        return hook

    @staticmethod
    def _make_skip_forward(block: torch.nn.Module):
        """
        Replace block.forward with a passthrough.

        For the context_pre_only (last) block the real forward would return
        (None, hidden_states).  We return (encoder_hidden_states, hidden_states)
        instead — the SD3 transformer never uses encoder_hidden_states after the
        last block, so this is safe.
        """
        is_context_pre_only = getattr(block, "context_pre_only", False)

        @functools.wraps(block.forward)
        def skip_forward(*args, **kwargs):
            hidden_states = kwargs.get("hidden_states")
            if hidden_states is None and args:
                hidden_states = args[0]
            encoder_hidden_states = kwargs.get("encoder_hidden_states")
            if encoder_hidden_states is None and len(args) > 1:
                encoder_hidden_states = args[1]
            # Mirror what the real last-block would return so the unpacking
            # `enc, img = block(...)` still works in all cases.
            if is_context_pre_only:
                return None, hidden_states
            return encoder_hidden_states, hidden_states

        return skip_forward

    # ------------------------------------------------------------------ #
    # public API
    # ------------------------------------------------------------------ #

    def attach(
        self,
        transformer: torch.nn.Module,
        removed_layers: Optional[Union[int, List[int]]] = None,
        modulated_layers: Optional[Union[int, List[int]]] = None,
        modulated_scales: Optional[Union[float, List[float]]] = None,
        modulated_ranges: Optional[Union[Tuple[int, int], List[Tuple[int, int]]]] = None,
        modulated_ways: str = "empty",
    ) -> None:
        """
        Attach hooks/patches to the transformer's blocks.

        Parameters
        ----------
        transformer      : SD3Transformer2DModel instance.
        removed_layers   : Block indices to skip entirely.
        modulated_layers : Block indices to apply empty/strengthen.
        modulated_scales : Per-layer scale factors (strengthen only).
                           Single float → broadcast to all modulated layers.
        modulated_ranges : Token-range(s) for selective strengthening.
                           (start, end) or [(s0,e0), (s1,e1), ...]
        modulated_ways   : "empty" | "strengthen"
        """
        self.detach()

        removed_layers = _to_int_list(removed_layers)
        modulated_layers = _to_int_list(modulated_layers)
        modulated_ranges = _to_range_list(modulated_ranges)

        if modulated_layers is not None:
            modulated_scales = _to_float_list(modulated_scales, len(modulated_layers))

        # ── remove ────────────────────────────────────────────────────
        if removed_layers:
            for idx in removed_layers:
                block = self._get_block(transformer, idx)
                self._patched_blocks.append((block, block.forward))
                block.forward = self._make_skip_forward(block)

        # ── empty / strengthen ────────────────────────────────────────
        if modulated_layers:
            for i, idx in enumerate(modulated_layers):
                block = self._get_block(transformer, idx)
                if modulated_ways == "empty":
                    hook_fn = self._make_empty_pre_hook()
                else:
                    scale = modulated_scales[i]
                    hook_fn = self._make_strengthen_pre_hook(scale, modulated_ranges)

                handle = block.register_forward_pre_hook(hook_fn, with_kwargs=True)
                self._hooks.append(handle)

    def detach(self) -> None:
        """Remove all hooks and restore original block.forward methods."""
        for handle in self._hooks:
            handle.remove()
        self._hooks.clear()

        for block, original_forward in self._patched_blocks:
            block.forward = original_forward
        self._patched_blocks.clear()

    @contextmanager
    def control(
        self,
        transformer: torch.nn.Module,
        removed_layers=None,
        modulated_layers=None,
        modulated_scales=None,
        modulated_ranges=None,
        modulated_ways: str = "empty",
    ):
        """
        Context manager — automatically attaches before and detaches after.

        Example
        -------
        with controller.control(pipe.transformer, removed_layers=[5, 10]):
            output = pipe(prompt=..., ...)
        """
        try:
            self.attach(
                transformer,
                removed_layers=removed_layers,
                modulated_layers=modulated_layers,
                modulated_scales=modulated_scales,
                modulated_ranges=modulated_ranges,
                modulated_ways=modulated_ways,
            )
            yield self
        finally:
            self.detach()


# ─────────────────────────────────────────────────────────────────────────────
# quick usage demo
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")

    from diffusers import StableDiffusion3Pipeline
    import torch

    MODEL_ID = "stabilityai/stable-diffusion-3.5-large"   # adjust as needed

    pipe = StableDiffusion3Pipeline.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16)
    pipe = pipe.to("cuda")

    controller = SD3TextEmbeddingController()

    prompt = "a beautiful landscape"

    # ── example 1: remove blocks 5 and 10 ────────────────────────────
    with controller.control(pipe.transformer, removed_layers=[5, 10]):
        img1 = pipe(prompt, num_inference_steps=28, guidance_scale=7.0).images[0]
    img1.save("sd3_removed.png")

    # ── example 2: empty text embedding at block 3 ───────────────────
    with controller.control(
        pipe.transformer,
        modulated_layers=[3],
        modulated_ways="empty",
    ):
        img2 = pipe(prompt, num_inference_steps=28, guidance_scale=7.0).images[0]
    img2.save("sd3_empty.png")

    # ── example 3: strengthen all tokens ×2 at blocks 3 and 7 ────────
    with controller.control(
        pipe.transformer,
        modulated_layers=[3, 7],
        modulated_scales=[2.0, 2.0],
        modulated_ways="strengthen",
    ):
        img3 = pipe(prompt, num_inference_steps=28, guidance_scale=7.0).images[0]
    img3.save("sd3_strengthen.png")

    # ── example 4: strengthen only tokens 0-20 at block 3 ────────────
    with controller.control(
        pipe.transformer,
        modulated_layers=[3],
        modulated_scales=[3.0],
        modulated_ranges=[(0, 20)],
        modulated_ways="strengthen",
    ):
        img4 = pipe(prompt, num_inference_steps=28, guidance_scale=7.0).images[0]
    img4.save("sd3_strengthen_token_range.png")

    print("Done. Saved sd3_removed.png / sd3_empty.png / sd3_strengthen.png / sd3_strengthen_token_range.png")
