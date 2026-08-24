"""
flux_text_controller.py
=======================
Non-intrusive text-embedding controller for FluxTransformer2DModel.

Works with the **original, unmodified** diffusers FluxTransformer2DModel and
FluxPipeline — no source-code edits required.

Three operations are supported (matching the intrusive version):

  remove     : Skip specified blocks entirely (pass hidden_states /
                encoder_hidden_states through unchanged, no computation).

  empty      : Zero out encoder_hidden_states **before** a block runs,
                removing text conditioning at that layer.

  strengthen : Multiply encoder_hidden_states by `scale` before a block,
                optionally restricted to a token sub-range.

Index convention (same as the intrusive version)
-------------------------------------------------
  global idx  0 … N_double-1          → transformer_blocks[idx]
  global idx  N_double … N_double+N_single-1 → single_transformer_blocks[idx - N_double]

  Flux-dev defaults: N_double = 19, N_single = 38  (total 57 blocks)

Usage
-----
    from flux_text_controller import FluxTextEmbeddingController

    controller = FluxTextEmbeddingController()

    # ── context-manager form (recommended) ────────────────────────────
    with controller.control(
        pipe.transformer,
        removed_layers=[5, 10],            # skip these blocks entirely
        modulated_layers=[3, 7],           # apply empty / strengthen here
        modulated_scales=[2.0, 3.0],       # one scale per modulated layer
        modulated_ranges=[(0, 20)],        # optional: token-level range(s)
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

class FluxTextEmbeddingController:
    """
    Non-intrusive text-embedding controller for FluxTransformer2DModel.

    Attaches PyTorch forward hooks / temporary forward replacements to the
    transformer's individual blocks — the transformer's own forward() method
    is never touched.
    """

    def __init__(self) -> None:
        self._hooks: list = []                   # hook handles to remove on detach
        self._patched_blocks: list = []          # (block, original_forward) pairs

    # ------------------------------------------------------------------ #
    # index → block mapping
    # ------------------------------------------------------------------ #

    @staticmethod
    def _get_block(transformer, global_idx: int) -> torch.nn.Module:
        """
        Map a global block index to the actual nn.Module.
        Flux has two separate block lists; single-stream blocks start after
        all double-stream blocks.
        """
        double_blocks = transformer.transformer_blocks
        num_double = len(double_blocks)
        if global_idx < num_double:
            return double_blocks[global_idx]
        single_blocks = transformer.single_transformer_blocks
        local_idx = global_idx - num_double
        if local_idx >= len(single_blocks):
            raise IndexError(
                f"Global block index {global_idx} out of range "
                f"(N_double={num_double}, N_single={len(single_blocks)})."
            )
        return single_blocks[local_idx]

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
        Replace block.forward with a passthrough that returns
        (encoder_hidden_states, hidden_states) unchanged, skipping all
        computation inside the block.
        """
        @functools.wraps(block.forward)
        def skip_forward(*args, **kwargs):
            hidden_states = kwargs.get("hidden_states")
            if hidden_states is None and args:
                hidden_states = args[0]
            encoder_hidden_states = kwargs.get("encoder_hidden_states")
            if encoder_hidden_states is None and len(args) > 1:
                encoder_hidden_states = args[1]
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
        transformer      : FluxTransformer2DModel instance.
        removed_layers   : Global block indices to skip entirely.
        modulated_layers : Global block indices to apply empty/strengthen.
        modulated_scales : Per-layer scale factors (strengthen only).
                           Single float → broadcast to all modulated layers.
                           None with modulated_ways="empty" → ignored.
        modulated_ranges : Token-range(s) for selective strengthening.
                           (start, end) or [(s0,e0), (s1,e1), ...]
        modulated_ways   : "empty" | "strengthen"
        """
        self.detach()  # clear any previous state

        removed_layers = _to_int_list(removed_layers)
        modulated_layers = _to_int_list(modulated_layers)
        modulated_ranges = _to_range_list(modulated_ranges)

        if modulated_layers is not None:
            modulated_scales = _to_float_list(modulated_scales, len(modulated_layers))

        # ── remove: replace block.forward with a passthrough ──────────
        if removed_layers:
            for idx in removed_layers:
                block = self._get_block(transformer, idx)
                self._patched_blocks.append((block, block.forward))
                block.forward = self._make_skip_forward(block)

        # ── empty / strengthen: register pre-hook ─────────────────────
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
# quick usage demo  (run this file directly to verify)
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")

    from diffusers import FluxPipeline
    import torch

    MODEL_ID = "black-forest-labs/FLUX.1-dev"   # adjust as needed

    pipe = FluxPipeline.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16)
    pipe = pipe.to("cuda")

    controller = FluxTextEmbeddingController()

    prompt = "a beautiful landscape"

    # ── example 1: remove blocks 5 and 10 ────────────────────────────
    with controller.control(pipe.transformer, removed_layers=[5, 10]):
        img1 = pipe(prompt, num_inference_steps=20, guidance_scale=3.5).images[0]
    img1.save("flux_removed.png")

    # ── example 2: empty text embedding at block 3 ───────────────────
    with controller.control(
        pipe.transformer,
        modulated_layers=[3],
        modulated_ways="empty",
    ):
        img2 = pipe(prompt, num_inference_steps=20, guidance_scale=3.5).images[0]
    img2.save("flux_empty.png")

    # ── example 3: strengthen all tokens ×2 at blocks 3 and 7 ────────
    with controller.control(
        pipe.transformer,
        modulated_layers=[3, 7],
        modulated_scales=[2.0, 2.0],
        modulated_ways="strengthen",
    ):
        img3 = pipe(prompt, num_inference_steps=20, guidance_scale=3.5).images[0]
    img3.save("flux_strengthen.png")

    # ── example 4: strengthen only tokens 0-20 at single-stream block 20
    #    (global index = 19 + 20 = 39) ──────────────────────────────────
    with controller.control(
        pipe.transformer,
        modulated_layers=[39],
        modulated_scales=[3.0],
        modulated_ranges=[(0, 20)],
        modulated_ways="strengthen",
    ):
        img4 = pipe(prompt, num_inference_steps=20, guidance_scale=3.5).images[0]
    img4.save("flux_strengthen_token_range.png")

    print("Done. Saved flux_removed.png / flux_empty.png / flux_strengthen.png / flux_strengthen_token_range.png")
