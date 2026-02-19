"""
Per-class LoRA fine-tuning of SiT-XL/2.

Both runs start from the REPA pretrained checkpoint (last.pt).
The difference is whether the REPA alignment loss is added during fine-tuning.

Run A — v-pred only (no REPA loss):
  python finetune_lora.py --class-idx 0 --output-dir lora_A

Run B — v-pred + REPA alignment loss (λ=0.1):
  python finetune_lora.py --class-idx 0 --output-dir lora_B \
      --use-repa --repa-coeff 0.1

Must be run from the REPA repo root directory.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T
from PIL import Image
from tqdm import tqdm

from diffusers.models import AutoencoderKL
from models.sit import SiT_models
from models.lora_sit import inject_lora, inject_lora_with_projectors, save_lora
from utils import download_model, load_encoders

# EuroSAT class order produced by torchvision (alphabetical folder sort)
EUROSAT_CLASSES: list[str] = [
    "AnnualCrop", "Forest", "HerbaceousVegetation", "Highway", "Industrial",
    "Pasture", "PermanentCrop", "Residential", "River", "SeaLake",
]

# ImageNet normalisation (DINOv2 / ViT encoders)
_IMGNET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_IMGNET_STD  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

# CLIP normalisation
_CLIP_MEAN = torch.tensor([0.48145466, 0.4578275,  0.40821073]).view(1, 3, 1, 1)
_CLIP_STD  = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1)


# ---------------------------------------------------------------------------- #
# Dataset
# ---------------------------------------------------------------------------- #

class FewShotLatentDataset(Dataset):
    """
    Returns (raw_uint8 [3,256,256], latent [4,32,32]) pairs.

    raw_uint8  — uint8 pixel tensor [0, 255], used for REPA encoder in Run B.
    latent     — pre-encoded VAE latent (already scaled by 0.18215).
    """

    def __init__(self, img_dir: str, latent_dir: str) -> None:
        img_paths    = sorted(Path(img_dir).glob("*.jpg"))
        img_paths   += sorted(Path(img_dir).glob("*.png"))
        latent_paths = sorted(Path(latent_dir).glob("*.pt"))

        if len(img_paths) != len(latent_paths):
            raise RuntimeError(
                f"Image/latent count mismatch: {len(img_paths)} vs {len(latent_paths)}"
            )
        self.img_paths    = img_paths
        self.latent_paths = latent_paths
        self._to_tensor   = T.ToTensor()

    def __len__(self) -> int:
        return len(self.img_paths)

    def __getitem__(self, idx: int):
        img    = Image.open(self.img_paths[idx]).convert("RGB")
        raw    = (self._to_tensor(img) * 255).byte()      # uint8 [3, 256, 256]
        latent = torch.load(self.latent_paths[idx])       # float [4, 32, 32]
        return raw, latent


# ---------------------------------------------------------------------------- #
# Encoder preprocessing
# ---------------------------------------------------------------------------- #

def preprocess_raw_image(x: torch.Tensor, enc_type: str) -> torch.Tensor:
    """
    Convert a uint8 [0,255] batch to the normalisation expected by the encoder.
    x : (B, 3, H, W) uint8
    """
    x = x.float() / 255.0
    if "dinov2" in enc_type or "dinov1" in enc_type:
        mean = _IMGNET_MEAN.to(x.device)
        std  = _IMGNET_STD.to(x.device)
    elif "clip" in enc_type:
        mean = _CLIP_MEAN.to(x.device)
        std  = _CLIP_STD.to(x.device)
    else:
        return x
    return (x - mean) / std


# ---------------------------------------------------------------------------- #
# Training
# ---------------------------------------------------------------------------- #

def main(args: argparse.Namespace) -> None:
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    class_name = args.class_name or EUROSAT_CLASSES[args.class_idx]
    run_tag    = "B (REPA)" if args.use_repa else "A (vanilla)"
    print(f"\n{'='*60}")
    print(f"Run {run_tag}  |  class={class_name}  |  idx={args.class_idx}")
    print(f"{'='*60}\n")

    # ------------------------------------------------------------------ #
    # Dataset & DataLoader
    # ------------------------------------------------------------------ #
    img_dir    = Path(args.data_dir) / "real_train_fewshot" / f"seed{args.seed}" / class_name
    latent_dir = Path(args.data_dir) / "latents" / class_name
    dataset    = FewShotLatentDataset(str(img_dir), str(latent_dir))

    loader = DataLoader(
        dataset,
        batch_size=min(args.batch_size, len(dataset)),
        shuffle=True,
        drop_last=True,
        num_workers=0,
        pin_memory=device == "cuda",
    )
    steps_per_epoch = len(loader)
    total_steps     = args.epochs * steps_per_epoch
    print(f"Dataset  : {len(dataset)} images | batch={args.batch_size} | "
          f"{steps_per_epoch} steps/epoch | {total_steps} total steps")

    # ------------------------------------------------------------------ #
    # Model — SiT-XL/2 with REPA projectors
    # ------------------------------------------------------------------ #
    block_kwargs = {"fused_attn": False, "qk_norm": False}
    model = SiT_models["SiT-XL/2"](
        input_size=32,
        num_classes=args.num_classes,
        use_cfg=(args.cfg_prob > 0),
        z_dims=[args.encoder_z_dim],
        encoder_depth=args.encoder_depth,
        **block_kwargs,
    ).to(device)

    # Load REPA pretrained checkpoint
    print("Loading REPA checkpoint (last.pt) …")
    state_dict = download_model("last.pt")
    missing, _ = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"  {len(missing)} missing keys (projectors may need random init — OK for Run A)")

    # ------------------------------------------------------------------ #
    # Inject LoRA
    # ------------------------------------------------------------------ #
    if args.use_repa:
        inject_lora_with_projectors(model, r=args.lora_rank, alpha=float(args.lora_rank))
    else:
        inject_lora(model, r=args.lora_rank, alpha=float(args.lora_rank))

    # ------------------------------------------------------------------ #
    # Load visual encoder for REPA loss (Run B only)
    # ------------------------------------------------------------------ #
    encoders: list     = []
    encoder_types: list = []
    if args.use_repa:
        print(f"Loading encoder: {args.enc_type} …")
        encoders, encoder_types, _ = load_encoders(args.enc_type, device, resolution=256)
        for enc in encoders:
            enc.eval()
            for p in enc.parameters():
                p.requires_grad_(False)

    # ------------------------------------------------------------------ #
    # Optimiser + LR schedule
    # ------------------------------------------------------------------ #
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(total_steps, 1), eta_min=1e-6
    )
    scaler = GradScaler(enabled=args.fp16)

    # ------------------------------------------------------------------ #
    # Training loop
    # ------------------------------------------------------------------ #
    model.train()
    step = 0
    loss_log: list[dict] = []

    pbar = tqdm(range(args.epochs), desc=f"{class_name} [{run_tag}]")
    for epoch in pbar:
        for raw_imgs, latents in loader:
            raw_imgs = raw_imgs.to(device)          # [B, 3, 256, 256]  uint8
            latents  = latents.to(device)           # [B, 4, 32, 32]    float32
            bsz      = latents.shape[0]

            # Class label: fixed index per class; drop with cfg_prob for CFG
            y = torch.full((bsz,), args.class_idx, dtype=torch.long, device=device)
            if args.cfg_prob > 0:
                drop_mask = torch.rand(bsz, device=device) < args.cfg_prob
                y = torch.where(drop_mask, torch.full_like(y, args.num_classes), y)

            with autocast(enabled=args.fp16):
                # v-prediction: α(t)=1-t, σ(t)=t → target = -x₀ + ε
                t      = torch.rand(bsz, 1, 1, 1, device=device, dtype=latents.dtype)
                noise  = torch.randn_like(latents)
                x_noisy = (1 - t) * latents + t * noise
                target  = -latents + noise          # d_alpha=-1, d_sigma=1

                model_out, zs_tilde = model(x_noisy, t.flatten(), y=y)
                denoising_loss = F.mse_loss(model_out, target)

                # REPA alignment loss — Run B only
                proj_loss = torch.tensor(0.0, device=device)
                if args.use_repa and encoders:
                    with torch.no_grad():
                        raw_f = raw_imgs.float()
                        zs_enc = []
                        for enc, enc_type in zip(encoders, encoder_types):
                            raw_pre = preprocess_raw_image(raw_f, enc_type)
                            z = enc.forward_features(raw_pre)
                            if "dinov2" in enc_type:
                                z = z["x_norm_patchtokens"]
                            elif "mocov3" in enc_type:
                                z = z[:, 1:]
                            zs_enc.append(z)

                    # Cosine-similarity alignment across all encoder/projector pairs
                    for z, z_tilde in zip(zs_enc, zs_tilde):
                        z_flat  = F.normalize(z.reshape(-1, z.shape[-1]),         dim=-1)
                        zt_flat = F.normalize(z_tilde.reshape(-1, z_tilde.shape[-1]), dim=-1)
                        proj_loss += -(z_flat * zt_flat).sum(dim=-1).mean()
                    proj_loss = proj_loss / len(zs_enc)

                total_loss = denoising_loss + args.repa_coeff * proj_loss

            optimizer.zero_grad()
            scaler.scale(total_loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            step += 1

        # Logging every 10 epochs
        if (epoch + 1) % 10 == 0 or epoch == 0:
            entry = {
                "epoch":          epoch + 1,
                "step":           step,
                "denoising_loss": denoising_loss.item(),
                "proj_loss":      proj_loss.item(),
                "lr":             scheduler.get_last_lr()[0],
            }
            loss_log.append(entry)
            pbar.set_postfix(
                denoise=f"{entry['denoising_loss']:.4f}",
                proj=f"{entry['proj_loss']:.4f}",
                lr=f"{entry['lr']:.1e}",
            )

    # ------------------------------------------------------------------ #
    # Save LoRA weights + training log
    # ------------------------------------------------------------------ #
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    lora_path = out_dir / f"{class_name}.safetensors"
    save_lora(model, str(lora_path))

    log_path = out_dir / f"{class_name}_log.json"
    with open(log_path, "w") as f:
        json.dump({
            "class_name": class_name,
            "class_idx":  args.class_idx,
            "use_repa":   args.use_repa,
            "repa_coeff": args.repa_coeff,
            "epochs":     args.epochs,
            "total_steps": step,
            "losses":     loss_log,
        }, f, indent=2)

    print(f"\nSaved LoRA : {lora_path}")
    print(f"Saved log  : {log_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Per-class LoRA fine-tuning of SiT-XL/2")

    # Data
    parser.add_argument("--data-dir",   type=str, default="data/eurosat",
                        help="Root of prepared EuroSAT data")
    parser.add_argument("--class-idx",  type=int, required=True,
                        help="EuroSAT class index 0-9")
    parser.add_argument("--class-name", type=str, default=None,
                        help="Override class name (inferred from --class-idx by default)")
    parser.add_argument("--seed",       type=int, default=0)

    # Output
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Directory to save LoRA .safetensors (e.g. lora_A or lora_B)")

    # LoRA
    parser.add_argument("--lora-rank",  type=int, default=16)

    # REPA alignment (Run B flags)
    parser.add_argument("--use-repa",   action="store_true",
                        help="Enable REPA alignment loss during fine-tuning (Run B)")
    parser.add_argument("--repa-coeff", type=float, default=0.1,
                        help="λ coefficient for REPA projection loss")
    parser.add_argument("--enc-type",   type=str, default="dinov2-vit-b",
                        help="Encoder type for REPA loss (must match REPA checkpoint)")
    parser.add_argument("--encoder-depth", type=int, default=8,
                        help="Encoder depth — must match the REPA checkpoint (default 8)")
    parser.add_argument("--encoder-z-dim", type=int, default=768,
                        help="Encoder embedding dim — 768 for DINOv2-ViT-B")

    # Training hyper-parameters
    parser.add_argument("--epochs",     type=int,   default=100)
    parser.add_argument("--batch-size", type=int,   default=8)
    parser.add_argument("--lr",         type=float, default=1e-4)
    parser.add_argument("--fp16",       action=argparse.BooleanOptionalAction, default=True,
                        help="Mixed-precision training (default: on)")
    parser.add_argument("--cfg-prob",   type=float, default=0.1,
                        help="Label-drop probability for CFG training")
    parser.add_argument("--num-classes", type=int,  default=1000,
                        help="Number of ImageNet classes (fixed — do not change)")

    args = parser.parse_args()
    main(args)
