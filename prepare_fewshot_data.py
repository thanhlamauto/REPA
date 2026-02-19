"""
Prepare EuroSAT few-shot data for REPA LoRA fine-tuning experiment.

Downloads EuroSAT (RGB), samples n_shot images per class,
encodes them to VAE latents, and caches everything to disk.

Output structure:
  data/eurosat/real_train_fewshot/seed<N>/<class>/XXXX.jpg
  data/eurosat/latents/<class>/XXXX.pt

Usage:
  python prepare_fewshot_data.py --n-shot 16 --seed 0
"""

import argparse
import random
from pathlib import Path

import torch
import torchvision.transforms as T
from torchvision.datasets import EuroSAT
from PIL import Image
from diffusers.models import AutoencoderKL
from tqdm import tqdm

LATENT_SCALE = 0.18215


def build_class_index(dataset) -> dict[int, list[int]]:
    """Group dataset indices by class label using dataset.targets."""
    class_to_indices: dict[int, list[int]] = {}
    for idx, label in enumerate(dataset.targets):
        class_to_indices.setdefault(label, []).append(idx)
    return class_to_indices


def main(args: argparse.Namespace) -> None:
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    data_root = Path(args.data_dir)

    # ------------------------------------------------------------------ #
    # 1.  Download / load EuroSAT
    # ------------------------------------------------------------------ #
    print("Downloading / loading EuroSAT (RGB)...")
    resize_crop = T.Compose([
        T.Resize(256, interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(256),
    ])
    dataset = EuroSAT(root=args.raw_dir, download=True)
    class_names: list[str] = dataset.classes          # ordered by folder name
    class_to_idx: dict[str, int] = dataset.class_to_idx
    print(f"  Classes ({len(class_names)}): {class_names}")

    class_to_indices = build_class_index(dataset)

    # ------------------------------------------------------------------ #
    # 2.  Sample n_shot images per class and save JPEG copies
    # ------------------------------------------------------------------ #
    print(f"\nSampling {args.n_shot} images per class (seed={args.seed})...")
    to_tensor = T.ToTensor()

    for class_name in class_names:
        class_idx = class_to_idx[class_name]
        indices = class_to_indices[class_idx].copy()
        random.shuffle(indices)
        selected = indices[: args.n_shot]

        img_out_dir = data_root / "real_train_fewshot" / f"seed{args.seed}" / class_name
        img_out_dir.mkdir(parents=True, exist_ok=True)

        for i, ds_idx in enumerate(selected):
            img_pil, _ = dataset[ds_idx]           # PIL Image
            img_pil = resize_crop(img_pil.convert("RGB"))
            save_path = img_out_dir / f"{i:04d}.jpg"
            if not save_path.exists():
                img_pil.save(save_path, quality=95)

        print(f"  {class_name:25s}: {len(selected)} images → {img_out_dir}")

    # ------------------------------------------------------------------ #
    # 3.  Encode latents with the SD VAE and cache to disk
    # ------------------------------------------------------------------ #
    print("\nEncoding latents with VAE (stabilityai/sd-vae-ft-mse)...")
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(device)
    vae.eval()

    for class_name in tqdm(class_names, desc="Encoding classes"):
        img_dir = data_root / "real_train_fewshot" / f"seed{args.seed}" / class_name
        latent_dir = data_root / "latents" / class_name
        latent_dir.mkdir(parents=True, exist_ok=True)

        for img_path in sorted(img_dir.glob("*.jpg")):
            latent_path = latent_dir / f"{img_path.stem}.pt"
            if latent_path.exists():
                continue

            img = Image.open(img_path).convert("RGB")
            x = to_tensor(img).unsqueeze(0).to(device)   # [1, 3, 256, 256]  [0, 1]
            x = x * 2.0 - 1.0                             # → [-1, 1]

            with torch.no_grad():
                posterior = vae.encode(x).latent_dist
                latent = posterior.sample() * LATENT_SCALE   # [1, 4, 32, 32]

            torch.save(latent.squeeze(0).cpu().float(), latent_path)

    # ------------------------------------------------------------------ #
    # 4.  Summary
    # ------------------------------------------------------------------ #
    print(f"\nAll done!")
    print(f"  Images  : {data_root}/real_train_fewshot/seed{args.seed}/<class>/")
    print(f"  Latents : {data_root}/latents/<class>/")
    print(f"  Classes : {class_names}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare EuroSAT few-shot data")
    parser.add_argument("--raw-dir", type=str, default="data/eurosat_raw",
                        help="Directory to download raw EuroSAT dataset")
    parser.add_argument("--data-dir", type=str, default="data/eurosat",
                        help="Output directory for processed data")
    parser.add_argument("--n-shot", type=int, default=16,
                        help="Number of images per class to use")
    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed for reproducible sampling")
    args = parser.parse_args()
    main(args)
