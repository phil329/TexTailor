"""
qwen_text_controller.py
=======================
Non-intrusive text-embedding controller for QwenImageTransformer2DModel.

Works with the **original, unmodified** diffusers QwenImageTransformer2DModel
and the QwenImage pipeline — no source-code edits required.

Three operations are supported (matching the intrusive version):

  remove     : Skip specified blocks entirely (pass hidden_states /
                encoder_hidden_states through unchanged, no computation).

  empty      : Zero out encoder_hidden_states **before** a block runs.

  strengthen : Multiply encoder_hidden_states by `scale` before a block,
                optionally restricted to a token sub-range.

QwenImage block-list structure
-------------------------------
  60 QwenImageTransformerBlock layers in transformer_blocks[0..59].
  All blocks are dual-stream and return (encoder_hidden_states, hidden_states).

QwenImage block call signature (from the transformer's forward loop)
---------------------------------------------------------------------
  encoder_hidden_states, hidden_states = block(
      hidden_states=hidden_states,
      encoder_hidden_states=encoder_hidden_states,
      encoder_hidden_states_mask=encoder_hidden_states_mask,   # <-- extra arg
      temb=temb,
      image_rotary_emb=image_rotary_emb,
      joint_attention_kwargs=attention_kwargs,
  )

  The mask is carried through skip_forward unchanged so the outer loop
  can continue to use it normally.

Index convention
----------------
  index  0 … 59  →  transformer_blocks[index]

Usage
-----
    from qwen_text_controller import QwenTextEmbeddingController

    controller = QwenTextEmbeddingController()

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

class QwenTextEmbeddingController:
    """
    Non-intrusive text-embedding controller for QwenImageTransformer2DModel.

    Attaches PyTorch forward hooks / temporary forward replacements to the
    transformer's individual blocks.  The transformer's own forward() method
    is never touched.

    QwenImage specifics
    -------------------
    • All 60 blocks are dual-stream — they always return
      (encoder_hidden_states, hidden_states).

    • The block call includes an extra `encoder_hidden_states_mask` kwarg.
      The pre-hook targets encoder_hidden_states only; the mask is untouched.

    • The skip_forward passthrough returns the mask-unchanged
      (encoder_hidden_states, hidden_states) tuple expected by the outer loop.
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
        """
        Zero out encoder_hidden_states before the block.
        encoder_hidden_states_mask is left intact so attention masking still
        applies to the (now-zero) embedding.
        """
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
        (encoder_hidden_states, hidden_states) unchanged.

        Note: QwenImage's transformer forward loop expects exactly this
        two-element tuple.  The encoder_hidden_states_mask kwarg is not
        returned since it is not part of the block's return value.
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
        transformer      : QwenImageTransformer2DModel instance.
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

    # adjust the import path for your QwenImage pipeline
    from diffusers import QwenImagePipeline   # or your actual pipeline class
    import torch

    MODEL_ID = "Qwen/Qwen2.5-VL-Image"   # adjust as needed

    pipe = QwenImagePipeline.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16)
    pipe = pipe.to("cuda")

    controller = QwenTextEmbeddingController()

    prompt = "a beautiful landscape"

    # ── example 1: remove blocks 5 and 10 ────────────────────────────
    with controller.control(pipe.transformer, removed_layers=[5, 10]):
        img1 = pipe(prompt, num_inference_steps=20).images[0]
    img1.save("qwen_removed.png")

    # ── example 2: empty text embedding at block 3 ───────────────────
    with controller.control(
        pipe.transformer,
        modulated_layers=[3],
        modulated_ways="empty",
    ):
        img2 = pipe(prompt, num_inference_steps=20).images[0]
    img2.save("qwen_empty.png")

    # ── example 3: strengthen all tokens ×2 at blocks 3 and 7 ────────
    with controller.control(
        pipe.transformer,
        modulated_layers=[3, 7],
        modulated_scales=[2.0, 2.0],
        modulated_ways="strengthen",
    ):
        img3 = pipe(prompt, num_inference_steps=20).images[0]
    img3.save("qwen_strengthen.png")

    # ── example 4: strengthen only tokens 0-20 at block 30 ───────────
    with controller.control(
        pipe.transformer,
        modulated_layers=[30],
        modulated_scales=[3.0],
        modulated_ranges=[(0, 20)],
        modulated_ways="strengthen",
    ):
        img4 = pipe(prompt, num_inference_steps=20).images[0]
    img4.save("qwen_strengthen_token_range.png")

    print("Done. Saved qwen_removed.png / qwen_empty.png / qwen_strengthen.png / qwen_strengthen_token_range.png")
