"""
Generate images for one class using a fine-tuned LoRA adapter.

Steps:
  1. Load SiT-XL/2 base checkpoint (last.pt).
  2. Inject LoRA structure and load per-class .safetensors weights.
  3. Sanity-check: generate n_sanity images → save preview grid.
     Warn if possible mode collapse is detected.
  4. Generate n_samples total images → save to output_dir/<class>/.

Example:
  # Run A (cfg=2.0, no REPA loss was used during training)
  python generate_lora.py --class-idx 0 \
      --lora-path lora_A/AnnualCrop.safetensors \
      --cfg-scale 2.0 --output-dir synthetic_A

  # Run B (cfg=1.4, REPA loss was used during training)
  python generate_lora.py --class-idx 0 \
      --lora-path lora_B/AnnualCrop.safetensors \
      --cfg-scale 1.4 --output-dir synthetic_B

Must be run from the REPA repo root directory.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch
import torchvision.transforms.functional as TF
from torchvision.utils import make_grid
from PIL import Image
from tqdm import tqdm

from diffusers.models import AutoencoderKL
from models.sit import SiT_models
from models.lora_sit import inject_lora, load_lora
from samplers import euler_sampler, euler_maruyama_sampler
from utils import download_model

# EuroSAT class order (alphabetical, matches torchvision)
EUROSAT_CLASSES: list[str] = [
    "AnnualCrop", "Forest", "HerbaceousVegetation", "Highway", "Industrial",
    "Pasture", "PermanentCrop", "Residential", "River", "SeaLake",
]

LATENT_SCALE = 0.18215


# ---------------------------------------------------------------------------- #
# Model construction
# ---------------------------------------------------------------------------- #

def build_model(args: argparse.Namespace, device: str) -> torch.nn.Module:
    """
    Build SiT-XL/2, load the REPA pretrained checkpoint,
    inject LoRA adapter structure, and restore per-class LoRA weights.
    """
    block_kwargs = {"fused_attn": False, "qk_norm": False}
    model = SiT_models["SiT-XL/2"](
        input_size=32,
        num_classes=args.num_classes,
        use_cfg=True,                          # always True at inference
        z_dims=[args.encoder_z_dim],
        encoder_depth=args.encoder_depth,
        **block_kwargs,
    ).to(device)

    print("Loading base checkpoint (last.pt) …")
    state_dict = download_model("last.pt")
    model.load_state_dict(state_dict, strict=False)

    # Inject the same LoRA structure used during fine-tuning
    inject_lora(model, r=args.lora_rank, alpha=float(args.lora_rank))

    # Restore per-class LoRA (and projector) weights
    load_lora(model, args.lora_path)

    model.eval()
    return model


# ---------------------------------------------------------------------------- #
# Sampling helpers
# ---------------------------------------------------------------------------- #

def _sample_batch(
    model: torch.nn.Module,
    class_idx: int,
    batch_size: int,
    latent_size: int,
    cfg_scale: float,
    num_steps: int,
    mode: str,
    guidance_high: float,
    device: str,
    dtype: torch.dtype = torch.float16,
) -> torch.Tensor:
    """Draw one batch of latents.  Returns float32 tensor [B, 4, H, W]."""
    z = torch.randn(batch_size, 4, latent_size, latent_size,
                    device=device, dtype=dtype)
    y = torch.full((batch_size,), class_idx, dtype=torch.long, device=device)

    sampler_kwargs = dict(
        model=model,
        latents=z,
        y=y,
        num_steps=num_steps,
        cfg_scale=cfg_scale,
        guidance_low=0.0,
        guidance_high=guidance_high,
        path_type="linear",
    )
    with torch.no_grad():
        if mode == "sde":
            out = euler_maruyama_sampler(**sampler_kwargs)
        else:
            out = euler_sampler(**sampler_kwargs)
    return out.float()


def _decode_latents(
    vae: AutoencoderKL,
    latents: torch.Tensor,
    device: str,
) -> list[Image.Image]:
    """Decode a batch of VAE latents to a list of PIL images."""
    latents = latents.to(device, dtype=torch.float32)
    with torch.no_grad():
        pixels = vae.decode(latents / LATENT_SCALE).sample   # [-1, 1]
    pixels = (pixels.clamp(-1, 1) + 1) / 2.0               # [0, 1]
    imgs = []
    for px in pixels:
        arr = (px.permute(1, 2, 0).cpu().float().numpy() * 255).astype("uint8")
        imgs.append(Image.fromarray(arr))
    return imgs


def _check_collapse(latents: torch.Tensor, threshold: float = 0.05) -> bool:
    """Return True if per-sample variance suggests mode collapse."""
    return latents.var(dim=[1, 2, 3]).mean().item() < threshold


# ---------------------------------------------------------------------------- #
# Main
# ---------------------------------------------------------------------------- #

def main(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    device = "cuda" if torch.cuda.is_available() else "cpu"

    class_name = args.class_name or EUROSAT_CLASSES[args.class_idx]
    run_tag    = Path(args.output_dir).name
    print(f"\n{'='*60}")
    print(f"Generate | class={class_name} | cfg={args.cfg_scale} | "
          f"mode={args.mode} | steps={args.num_steps}")
    print(f"{'='*60}\n")

    model = build_model(args, device)
    model = model.half()   # fp16 for faster inference

    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)
    vae.eval()

    latent_size = 32   # 256 // 8

    # ------------------------------------------------------------------ #
    # Sanity check — generate n_sanity images and inspect
    # ------------------------------------------------------------------ #
    print(f"Sanity check: generating {args.n_sanity} images …")
    n_sanity_batches = math.ceil(args.n_sanity / args.batch_size)
    sanity_latents_list = []
    for _ in range(n_sanity_batches):
        bs = min(args.batch_size, args.n_sanity - sum(x.shape[0] for x in sanity_latents_list))
        if bs <= 0:
            break
        sanity_latents_list.append(
            _sample_batch(model, args.class_idx, bs, latent_size,
                          args.cfg_scale, args.num_steps, args.mode,
                          args.guidance_high, device)
        )
    sanity_latents = torch.cat(sanity_latents_list, dim=0)[: args.n_sanity]

    if _check_collapse(sanity_latents):
        print(
            f"  WARNING: possible mode collapse (low variance). "
            f"Consider reducing --cfg-scale from {args.cfg_scale}."
        )

    sanity_imgs = _decode_latents(vae, sanity_latents, device)

    # Save sanity preview grid
    preview_dir = Path(args.preview_dir)
    preview_dir.mkdir(parents=True, exist_ok=True)
    grid_tensors = [TF.to_tensor(img) for img in sanity_imgs]
    nrow = min(8, len(grid_tensors))
    grid = make_grid(torch.stack(grid_tensors), nrow=nrow, padding=2)
    grid_path = preview_dir / f"{class_name}_{run_tag}_sanity.png"
    TF.to_pil_image(grid).save(str(grid_path))
    print(f"  Sanity grid → {grid_path}")

    # ------------------------------------------------------------------ #
    # Full generation — n_samples images
    # ------------------------------------------------------------------ #
    out_dir = Path(args.output_dir) / class_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # Resume from however many images already exist
    existing    = sorted(out_dir.glob("*.png"))
    start_idx   = len(existing)
    remaining   = args.n_samples - start_idx

    if remaining <= 0:
        print(f"Already have {start_idx} / {args.n_samples} images in {out_dir} — skipping.")
        return

    print(f"Generating {remaining} images (index {start_idx} → {start_idx + remaining - 1}) …")
    generated = 0
    pbar = tqdm(total=remaining, desc=class_name, unit="img")

    while generated < remaining:
        bs      = min(args.batch_size, remaining - generated)
        latents = _sample_batch(
            model, args.class_idx, bs, latent_size,
            args.cfg_scale, args.num_steps, args.mode,
            args.guidance_high, device,
        )
        imgs = _decode_latents(vae, latents, device)
        for img in imgs:
            img.save(out_dir / f"{start_idx + generated:06d}.png")
            generated += 1
        pbar.update(len(imgs))

    pbar.close()

    # Save a final full grid (first 64 images)
    all_imgs = [
        TF.to_tensor(Image.open(p))
        for p in sorted(out_dir.glob("*.png"))[:64]
    ]
    if all_imgs:
        full_grid = make_grid(torch.stack(all_imgs), nrow=8, padding=2)
        full_grid_path = preview_dir / f"{class_name}_{run_tag}_grid.png"
        TF.to_pil_image(full_grid).save(str(full_grid_path))
        print(f"  Full grid  → {full_grid_path}")

    print(f"\nDone! {args.n_samples} images → {out_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate images with a per-class LoRA adapter"
    )

    # Class
    parser.add_argument("--class-idx",  type=int, required=True,
                        help="EuroSAT class index 0-9")
    parser.add_argument("--class-name", type=str, default=None,
                        help="Override class name (auto-inferred from --class-idx)")

    # LoRA checkpoint
    parser.add_argument("--lora-path",  type=str, required=True,
                        help="Path to .safetensors LoRA weights file")

    # Output
    parser.add_argument("--output-dir",  type=str, required=True,
                        help="Root output directory (e.g. synthetic_A or synthetic_B)")
    parser.add_argument("--preview-dir", type=str, default="previews",
                        help="Directory for sanity and summary grids")

    # Model architecture (must match fine-tuning settings)
    parser.add_argument("--encoder-depth",  type=int, default=8)
    parser.add_argument("--encoder-z-dim",  type=int, default=768)
    parser.add_argument("--lora-rank",      type=int, default=16)
    parser.add_argument("--num-classes",    type=int, default=1000)

    # Sampling
    parser.add_argument("--n-samples",    type=int,   default=500)
    parser.add_argument("--n-sanity",     type=int,   default=32)
    parser.add_argument("--batch-size",   type=int,   default=32)
    parser.add_argument("--cfg-scale",    type=float, default=2.0)
    parser.add_argument("--num-steps",    type=int,   default=50)
    parser.add_argument("--mode",         type=str,   default="sde",
                        choices=["sde", "ode"])
    parser.add_argument("--guidance-high", type=float, default=0.7)
    parser.add_argument("--vae",          type=str,   default="ema",
                        choices=["ema", "mse"])
    parser.add_argument("--seed",         type=int,   default=0)

    args = parser.parse_args()
    main(args)
