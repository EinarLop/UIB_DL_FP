"""Helper functions for LABF_2026 — student version (`_stv`).

Public surface:
    - DEFAULT_NEGATIVE                          shared negative prompt
    - clip_score(image, text, model, pp)        CLIP-as-critic cosine similarity
    - show_image_grid(images, titles, ...)      matplotlib visualisation
    - show_mask_overlay(image, mask, ...)       visualise a mask over an image
    - generate_with_lora(pipe, prompts, ...)    text-to-image generation wrapper
    - TrainingState                             dataclass returned by setup_lora_training
    - setup_lora_training(pipe, lora_config, lr)  inject LoRA + optimiser + scheduler
    - sample_batch(dataset, state, rng, augment)  one (image_tensor, prompt) batch
    - save_lora_adapter(state, output_dir)      atomic save of trained adapter
    - delete_trained_adapter(state)             cleanup so subsequent loads start pristine

This is the `_stv` companion of `helper_LF.py`. The differences are:

1. The monolithic `train_lora_unet` has been decomposed into
   ``setup_lora_training`` + ``sample_batch`` + ``save_lora_adapter`` +
   ``delete_trained_adapter``, so the per-step diffusion training body
   becomes the student's responsibility (see the LoRA training cell of
   the lab notebook).
2. ``postprocess_mask`` (the binary-mask alternative) is removed — the
   lab uses the soft heatmap directly and the function was unused.

The helpers ``clip_score``, ``show_image_grid``, ``show_mask_overlay``,
and ``generate_with_lora`` are byte-identical to their counterparts in
the original ``helper_LF.py``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

import matplotlib.pyplot as plt

if TYPE_CHECKING:                       # editor support without runtime import cost
    from diffusers import (
        AutoencoderKL,
        DDPMScheduler,
        StableDiffusionPipeline,
        UNet2DConditionModel,
    )
    from transformers import CLIPTextModel, CLIPTokenizer


# ---------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------

# Default negative prompt shared by Task A (inpainting) and Task B
# (text-to-image generation). Pushes the diffusion sampler away from the
# most common failure modes (low-quality artefacts, watermarks, deformed
# anatomy) without injecting domain-specific bias.
DEFAULT_NEGATIVE = (
    "blurry, low quality, watermark, distorted, deformed, ugly"
)


# ---------------------------------------------------------------------
# CLIP-as-critic scoring
# ---------------------------------------------------------------------

def clip_score(
    image: Image.Image,
    text: str,
    clip_model,
    clip_preprocess,
) -> float:
    """Cosine similarity between a CLIP-encoded image and text.

    Both image and text embeddings are L2-normalised before the dot
    product. Return value is a scalar in [-1, +1]; typical "good match"
    scores for descriptive prompts fall in [0.20, 0.35].
    """
    device = next(clip_model.parameters()).device
    with torch.no_grad():
        img = clip_preprocess(image).unsqueeze(0).to(device)
        z_img = clip_model.encode_image(img)
        # clip.tokenize is lazy-imported to keep this module importable
        # even before CLIP is available (e.g. in unit tests).
        import clip as _clip
        tokens = _clip.tokenize([text]).to(device)
        z_txt = clip_model.encode_text(tokens)

        z_img = F.normalize(z_img.float(), dim=-1)
        z_txt = F.normalize(z_txt.float(), dim=-1)
        return (z_img @ z_txt.T).squeeze().item()


# ---------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------

def show_image_grid(
    images: list[Image.Image],
    titles: list[str],
    rows: int = 1,
    cols: int | None = None,
    title: str | None = None,
    figsize_per_cell: float = 3.5,
) -> None:
    """Display a grid of PIL images with per-cell titles."""
    if cols is None:
        cols = (len(images) + rows - 1) // rows
    if rows * cols < len(images):
        raise ValueError(
            f"show_image_grid: rows*cols ({rows*cols}) < len(images) "
            f"({len(images)})"
        )
    fig, axes = plt.subplots(
        rows, cols, figsize=(cols * figsize_per_cell, rows * figsize_per_cell)
    )
    axes = np.atleast_2d(axes)
    for i, (img, caption) in enumerate(zip(images, titles)):
        r, c = i // cols, i % cols
        ax = axes[r, c]
        ax.imshow(img)
        ax.set_title(caption, fontsize=9)
        ax.axis("off")
    for j in range(len(images), rows * cols):
        r, c = j // cols, j % cols
        axes[r, c].axis("off")
    if title:
        fig.suptitle(title, fontsize=12)
    plt.tight_layout()
    plt.show()


def show_mask_overlay(
    image: Image.Image,
    mask: np.ndarray,
    title: str = "",
    alpha: float = 0.45,
    colour: tuple[int, int, int] = (255, 255, 0),  # yellow
) -> None:
    """Show the original image and a coloured overlay of the mask."""
    arr = np.array(image, dtype=np.float32)
    overlay = arr.copy()
    m = mask.astype(bool)
    overlay[m] = (1 - alpha) * overlay[m] + alpha * np.array(colour, dtype=np.float32)
    overlay = overlay.clip(0, 255).astype(np.uint8)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(image); axes[0].set_title("image"); axes[0].axis("off")
    axes[1].imshow(overlay); axes[1].set_title("mask overlay"); axes[1].axis("off")
    if title:
        fig.suptitle(title, fontsize=12)
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------
# Image preprocessing for VAE encode
# ---------------------------------------------------------------------

# Pillow >= 10 deprecated bare ``Image.BICUBIC`` / ``Image.FLIP_LEFT_RIGHT``
# in favour of the ``Resampling`` / ``Transpose`` enums. ``getattr`` keeps
# the helper compatible with both old and new Pillow without emitting
# DeprecationWarnings on every call.
_BICUBIC = getattr(Image, "Resampling", Image).BICUBIC
_FLIP_LR = getattr(Image, "Transpose", Image).FLIP_LEFT_RIGHT


def _prepare_image_for_vae(
    img: Image.Image,
    size: int = 512,
    flip: bool = False,
) -> torch.Tensor:
    """Resize + centre-crop to ``size`` and normalise to [-1, +1].

    When ``flip`` is True the image is also mirrored horizontally — a
    cheap augmentation used during LoRA training to avoid overfitting
    to the small personalisation set.
    """
    scale = size / min(img.size)
    new_size = (int(img.width * scale), int(img.height * scale))
    img = img.resize(new_size, _BICUBIC)
    left = (img.width - size) // 2
    top = (img.height - size) // 2
    img = img.crop((left, top, left + size, top + size))
    if flip:
        img = img.transpose(_FLIP_LR)
    arr = np.array(img, dtype=np.float32) / 127.5 - 1.0     # [-1, +1]
    return torch.from_numpy(arr).permute(2, 0, 1)            # (3, H, W)


# ---------------------------------------------------------------------
# LoRA training — split helpers (the per-step body lives in the notebook)
# ---------------------------------------------------------------------

@dataclass
class TrainingState:
    """Container for the objects the per-step LoRA training body needs.

    The notebook builds this via ``setup_lora_training(pipe, ...)`` and
    then drives the training loop directly. The methods on this class
    (``encode_text``, ``attach_grad``, ``assert_grad_flowing``) wrap
    plumbing details that aren't pedagogically meaningful for the
    student to write themselves: the tokeniser/text-encoder chain,
    the gradient-checkpoint requirement that inputs must require_grad,
    and a sanity check that LoRA gradients are flowing.

    Attributes
    ----------
    pipe : the SD pipeline (kept for caller convenience).
    vae, text_encoder, unet, tokenizer : the four sub-models extracted
        from the pipe. The U-Net has a LoRA adapter named ``"default"``
        injected; the others are frozen.
    optimizer : an AdamW over the trainable LoRA parameters only.
    noise_scheduler : a DDPMScheduler matching the pipe's scheduler config
        (used for ``add_noise`` and ``num_train_timesteps``; the pipe's
        own scheduler — typically a fast deterministic sampler — is for
        inference, not training).
    base_dtype : the dtype the frozen U-Net base weights use (typically
        FP16). LoRA params have been re-cast to FP32 on top of this
        for numerical stability. Cast frozen tensors to ``base_dtype``;
        do not re-cast LoRA params.
    device : the torch.device on which the pipe lives.
    trainable_params : the list of LoRA parameters that receive
        gradient. Use this for ``torch.nn.utils.clip_grad_norm_``.
    """

    pipe: "StableDiffusionPipeline"
    vae: "AutoencoderKL"
    text_encoder: "CLIPTextModel"
    unet: "UNet2DConditionModel"
    tokenizer: "CLIPTokenizer"
    optimizer: Optional[torch.optim.Optimizer]
    noise_scheduler: "DDPMScheduler"
    base_dtype: torch.dtype
    device: torch.device
    trainable_params: list = field(default_factory=list)
    _grad_check_done: bool = field(default=False, init=False, repr=False)

    def encode_text(self, prompt: str) -> torch.Tensor:
        """Tokenise + text-encode a prompt; return the cross-attention
        hidden states ready to be passed to ``self.unet``.

        The text encoder is frozen (the encoder forward runs under
        ``no_grad``), but the resulting tensor is then marked
        ``requires_grad=True`` so the U-Net's gradient-checkpointed
        cross-attention blocks can build an autograd graph through it
        — without that, the K/V LoRA branch is detached during the
        recompute pass and never receives gradient.
        """
        tokens = self.tokenizer(
            prompt, padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True, return_tensors="pt",
        ).input_ids.to(self.device)
        with torch.no_grad():
            embeds = self.text_encoder(tokens)[0].to(dtype=self.base_dtype)
        embeds.requires_grad_(True)
        return embeds

    def attach_grad(self, x: torch.Tensor) -> torch.Tensor:
        """Mark ``x`` as requiring gradient and return it.

        Required for inputs that flow into a gradient-checkpointed
        forward pass (the U-Net, here): without ``requires_grad=True``
        on the input, the checkpoint mechanism does not build an
        autograd graph through the recompute, the LoRA matmul outputs
        end up detached, and the LoRA parameters never receive gradient.
        """
        x.requires_grad_(True)
        return x

    def assert_grad_flowing(self) -> None:
        """Raise if no LoRA-B parameter received gradient.

        Call this exactly once, on step 1, after the first
        ``loss.backward()``. Catches the most common silent failure
        mode: ``target_modules`` does not match any module name in the
        U-Net, so PEFT registers no adapters and training quietly does
        nothing.

        ``lora_B`` is checked rather than ``lora_A`` because ``lora_B``
        is initialised to zero, so ``d(loss)/d(lora_A) = 0`` on step 1
        by construction; ``lora_B`` always receives gradient when
        forward and backward are wired correctly.

        The check uses ``any`` rather than ``all`` because individual
        deep-block LoRA grads can be very small and may underflow to
        exactly zero on low-entropy inputs even when the wiring is
        correct. The ``any`` form still catches global failure modes.
        """
        if self._grad_check_done:
            return
        lora_b_params = [
            (n, p) for n, p in self.unet.named_parameters()
            if p.requires_grad and "lora_B" in n
        ]
        firing = [
            n for n, p in lora_b_params
            if p.grad is not None and torch.is_nonzero(p.grad.abs().sum())
        ]
        if not firing:
            raise RuntimeError(
                f"No LoRA-B parameter received any gradient on step 1 "
                f"(checked {len(lora_b_params)} lora_B weights). "
                f"Likely causes: target_modules wrong, requires_grad "
                f"flags wrong, gradient-checkpointing detaching the "
                f"whole UNet, or the UNet was not put in train() mode "
                f"(unet.training={self.unet.training})."
            )
        zero_count = len(lora_b_params) - len(firing)
        if zero_count > len(lora_b_params) * 0.5:
            # Informational, not an error. With deeply-stacked attention
            # blocks under FP16 autocast, individual gradients can
            # underflow to exactly zero on visually homogeneous inputs.
            # The training is still wired correctly; with textured photos
            # this count drops sharply.
            print(
                f"[LoRA] note: {zero_count}/{len(lora_b_params)} "
                f"lora_B params have zero grad on step 1. This typically "
                f"reflects low-entropy training inputs (solid colours, "
                f"near-uniform images) underflowing in deep blocks under "
                f"FP16 autocast — not a wiring error. Texture-rich photos "
                f"reduce the count significantly."
            )
        self._grad_check_done = True


def setup_lora_training(
    pipe,
    lora_config,
    lr: float,
) -> TrainingState:
    """Inject the LoRA, freeze everything else, build the optimiser.

    Validates that ``pipe`` is a 4-channel SD-2.x text-to-image
    pipeline (``in_channels == 4`` and ``cross_attention_dim == 1024``)
    and raises ``ValueError`` otherwise. Captures the U-Net's base
    dtype before LoRA injection, so subsequent casts route correctly
    even after LoRA params are flipped to FP32.

    Parameters
    ----------
    pipe : ``StableDiffusionPipeline``
        A non-inpainting SD-2.x text-to-image pipe. Training a LoRA on
        the 9-channel inpainting U-Net (with synthesised
        ``mask=ones``/``masked_image=full`` conditioning) gives the
        model a trivial copy shortcut and the adapter learns nothing.
    lora_config : ``peft.LoraConfig``
        The student-built configuration. It is passed to
        ``unet.add_adapter(lora_config, adapter_name="default")``.
    lr : float
        AdamW learning rate for the trainable LoRA parameters.

    Returns
    -------
    TrainingState
        See the dataclass docstring.
    """
    from diffusers import DDPMScheduler

    device = pipe.unet.device
    vae = pipe.vae
    tokenizer = pipe.tokenizer
    text_encoder = pipe.text_encoder
    unet = pipe.unet

    # Hard guard: this loop is only correct on a 4-channel
    # text-to-image UNet. The 9-channel inpainting UNet expects
    # additional mask + masked-image conditioning channels and would
    # need a different training scheme.
    if unet.config.in_channels != 4:
        raise ValueError(
            f"setup_lora_training expects a non-inpainting SD UNet "
            f"(in_channels=4); got in_channels={unet.config.in_channels}. "
            f"Load 'sd2-community/stable-diffusion-2-1-base' (or another "
            f"4-channel SD variant) for training, then move the adapter "
            f"to whichever pipeline you use at inference."
        )
    # Strict guard: the LoRA hyperparameters and the lab's caption
    # template assume SD-2.x (cross_attention_dim 1024). SD-1.5 (768)
    # and SDXL (2048) would also pass the in_channels check but produce
    # sub-optimal adapters with these defaults, so we refuse them.
    if unet.config.cross_attention_dim != 1024:
        raise ValueError(
            f"setup_lora_training expects an SD-2.x UNet "
            f"(cross_attention_dim=1024); got "
            f"cross_attention_dim={unet.config.cross_attention_dim}. "
            f"Did you load SD-1.5 or SDXL by mistake?"
        )

    # Capture the base UNet dtype BEFORE injecting / casting the LoRA
    # params. Once cast_training_params(unet, fp32) runs, querying
    # unet.dtype becomes ambiguous (the first parameter encountered
    # may now be a fp32 LoRA weight) and using it to cast inputs would
    # silently upcast the frozen fp16 base to fp32.
    base_dtype = next(unet.parameters()).dtype

    # Freeze base weights. The LoRA adapter injected below is the
    # only part that will receive gradients. ``eval()`` makes the
    # frozen-but-active modules' contract explicit (no dropout / no
    # BatchNorm running stats updates).
    for p in vae.parameters(): p.requires_grad_(False)
    for p in text_encoder.parameters(): p.requires_grad_(False)
    for p in unet.parameters(): p.requires_grad_(False)
    vae.eval()
    text_encoder.eval()

    # Noise schedule — reuse the pipe's scheduler config instead of
    # re-downloading from a hardcoded repo id. DDPMScheduler is used at
    # training time even when the pipe uses a fast deterministic
    # sampler for inference.
    noise_scheduler = DDPMScheduler.from_config(pipe.scheduler.config)

    # Inject the LoRA adapter into the U-Net via the diffusers public
    # API. ``add_adapter`` modifies the U-Net's module tree in place,
    # adds the adapter under the name "default", and marks the LoRA
    # parameters as trainable.
    unet.add_adapter(lora_config, adapter_name="default")
    unet.train()
    unet.enable_gradient_checkpointing()

    # Cast the (freshly added) LoRA parameters to fp32. ``add_adapter``
    # initialises them in the U-Net's dtype (fp16 here), which underflows
    # during backprop and produces NaN losses. Diffusers' own DreamBooth
    # LoRA example applies the same cast immediately after the adapter
    # is added.
    from diffusers.training_utils import cast_training_params
    cast_training_params(unet, dtype=torch.float32)

    trainable_params = [p for p in unet.parameters() if p.requires_grad]
    trainable = sum(p.numel() for p in trainable_params)
    print(f"[LoRA] trainable params: {trainable/1e6:.2f} M")

    optimiser = torch.optim.AdamW(trainable_params, lr=lr)

    return TrainingState(
        pipe=pipe,
        vae=vae,
        text_encoder=text_encoder,
        unet=unet,
        tokenizer=tokenizer,
        optimizer=optimiser,
        noise_scheduler=noise_scheduler,
        base_dtype=base_dtype,
        device=device,
        trainable_params=trainable_params,
    )


def sample_batch(
    dataset: list[tuple[Image.Image, str]],
    state: TrainingState,
    rng: np.random.Generator,
    augment: bool = True,
) -> tuple[torch.Tensor, str]:
    """Sample one (image_tensor, prompt) batch with optional flip.

    Returns ``image_tensor`` of shape ``(1, 3, 512, 512)`` ready for
    ``state.vae.encode``: on ``state.device``, in the VAE's dtype, with
    pixel values normalised to ``[-1, +1]``. The optional 50/50
    horizontal flip is a cheap augmentation that helps the small
    personalisation set.

    Sampling is *with replacement* — the helper does not maintain epoch
    state. Pass a list, not a single-pass generator.
    """
    idx = int(rng.integers(0, len(dataset)))
    img, prompt = dataset[idx]
    do_flip = bool(augment and rng.integers(0, 2))
    pixel = (
        _prepare_image_for_vae(img, flip=do_flip)
        .unsqueeze(0)
        .to(state.device, dtype=state.vae.dtype)
    )
    return pixel, prompt


def save_lora_adapter(state: TrainingState, output_dir: str | Path) -> Path:
    """Save the trained adapter in diffusers native format.

    Writes ``pytorch_lora_weights.safetensors`` under ``output_dir``
    (the directory is created if it does not already exist).

    The save is **crash-safe**: writes to a sibling ``_pending``
    directory first and renames the safetensors file into place
    atomically. A ctrl-C between the two operations leaves the previous
    adapter intact rather than producing a half-written safetensors
    that the next-run skip-guard would mistake for a finished training.
    """
    from peft.utils import get_peft_model_state_dict
    from diffusers.utils import convert_state_dict_to_diffusers

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    lora_state = convert_state_dict_to_diffusers(
        get_peft_model_state_dict(state.unet)
    )
    pending = output_dir / "_pending"
    pending.mkdir(parents=True, exist_ok=True)
    state.pipe.save_lora_weights(
        save_directory=str(pending),
        unet_lora_layers=lora_state,
        safe_serialization=True,
    )
    src = pending / "pytorch_lora_weights.safetensors"
    dst = output_dir / "pytorch_lora_weights.safetensors"
    os.replace(src, dst)            # atomic on POSIX & NTFS
    try:
        pending.rmdir()
    except OSError:
        pass
    print(f"[LoRA] adapter saved to {output_dir}")
    return output_dir


def delete_trained_adapter(state: TrainingState) -> None:
    """Detach the freshly-trained adapter from the U-Net and free the
    optimiser's references to its (now-orphaned) parameters.

    Subsequent ``pipe.load_lora_weights(..., adapter_name="default")``
    on the same pipe must start from a pristine adapter slot;
    otherwise some peft/diffusers version combinations raise
    "Adapter name default already in use".

    Also drops ``state.optimizer`` and ``state.trainable_params`` so
    the AdamW state (~50 MB for a typical SD-2 LoRA) is reclaimed
    when ``state`` later goes out of scope.
    """
    state.unet.delete_adapters(["default"])
    state.optimizer = None
    state.trainable_params = []


# ---------------------------------------------------------------------
# Text-to-image generation wrapper (Task B inference)
# ---------------------------------------------------------------------

def generate_with_lora(
    pipe,
    prompts: list[str],
    *,
    negative_prompt: str = DEFAULT_NEGATIVE,
    num_inference_steps: int = 30,
    guidance_scale: float = 7.5,
    seed: int = 42,
) -> list[Image.Image]:
    """Run a text-to-image SD pipeline on a list of prompts.

    Thin wrapper used by Task B. Each prompt at index ``i`` uses a
    seeded ``torch.Generator`` with seed ``seed + i``, so re-running
    the cell reproduces the exact images bit-for-bit.

    **Reproducibility note.** Because the per-prompt seed is keyed on
    the *index* in ``prompts`` (not on the prompt text), running this
    helper twice with the **same prompt list in the same order** is
    the only configuration that gives a fair A/B comparison — Task C
    in the lab notebook depends on this. Do not reorder ``prompts``
    between the LoRA-on and LoRA-off runs.

    Caller is responsible for any LoRA loading / weight setting on
    ``pipe`` *before* calling this helper — the wrapper does not touch
    adapter state.

    Returns the list of PIL outputs in the same order as ``prompts``.
    Visualisation is the caller's job (use ``show_image_grid``).
    """
    device = pipe.unet.device
    device_type = device.type if hasattr(device, "type") else "cuda"
    outputs: list[Image.Image] = []
    for i, prompt in enumerate(prompts):
        generator = torch.Generator(device).manual_seed(seed + i)
        with torch.autocast(
            device_type=device_type,
            dtype=torch.float16,
            enabled=(device != torch.device("cpu")),
        ):
            out = pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator=generator,
            ).images[0]
        outputs.append(out)
    return outputs
