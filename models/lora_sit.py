"""
LoRA adapters for SiT attention layers.

Injects low-rank adapters into attn.qkv and attn.proj of every SiTBlock:
    out = W x + (B @ A) x * (alpha / r)

No third-party LoRA library needed — adapters are plain nn.Parameters.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import save_file, load_file


# ---------------------------------------------------------------------------- #
# LoRA Linear layer
# ---------------------------------------------------------------------------- #

class LoRALinear(nn.Module):
    """
    Wraps a frozen nn.Linear with trainable low-rank adapters.

    Forward:
        out = W x + B @ (A @ x) * scaling
    where W is frozen, A and B are trainable.
    """

    def __init__(self, linear: nn.Linear, r: int = 16, alpha: float = 16.0) -> None:
        super().__init__()
        in_f = linear.in_features
        out_f = linear.out_features
        self.r = r
        self.scaling = alpha / r

        # Frozen backbone weights (copied from the original layer)
        self.weight = nn.Parameter(linear.weight.data.clone(), requires_grad=False)
        if linear.bias is not None:
            self.bias: nn.Parameter | None = nn.Parameter(
                linear.bias.data.clone(), requires_grad=False
            )
        else:
            self.bias = None

        # Trainable LoRA matrices — A uses kaiming init, B is zero-init
        self.lora_A = nn.Parameter(torch.empty(r, in_f))
        self.lora_B = nn.Parameter(torch.zeros(out_f, r))
        nn.init.kaiming_uniform_(self.lora_A, a=5 ** 0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = F.linear(x, self.weight, self.bias)
        lora = F.linear(F.linear(x, self.lora_A), self.lora_B) * self.scaling
        return base + lora

    def extra_repr(self) -> str:
        return (
            f"in={self.weight.shape[1]}, out={self.weight.shape[0]}, "
            f"r={self.r}, scaling={self.scaling:.3f}"
        )


# ---------------------------------------------------------------------------- #
# Injection helpers
# ---------------------------------------------------------------------------- #

def _replace_attn_linears(model: nn.Module, r: int, alpha: float) -> None:
    """Replace qkv and proj Linear layers inside every SiTBlock with LoRALinear."""
    for module in model.modules():
        if module.__class__.__name__ == "SiTBlock":
            module.attn.qkv = LoRALinear(module.attn.qkv, r=r, alpha=alpha)
            module.attn.proj = LoRALinear(module.attn.proj, r=r, alpha=alpha)


def _freeze_all(model: nn.Module) -> None:
    for param in model.parameters():
        param.requires_grad_(False)


def _unfreeze_lora(model: nn.Module) -> None:
    for name, param in model.named_parameters():
        if "lora_A" in name or "lora_B" in name:
            param.requires_grad_(True)


def _print_trainable(model: nn.Module, tag: str = "") -> None:
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(
        f"LoRA{(' ' + tag) if tag else ''} | "
        f"trainable: {n_trainable:,} / {n_total:,} "
        f"({100 * n_trainable / n_total:.2f}%)"
    )


def inject_lora(model: nn.Module, r: int = 16, alpha: float = 16.0) -> nn.Module:
    """
    Inject LoRA into attn.qkv and attn.proj of every SiTBlock.
    Freezes the entire model then marks only lora_A / lora_B as trainable.

    Use for Run A (v-pred fine-tuning only, no REPA loss).
    """
    _replace_attn_linears(model, r=r, alpha=alpha)
    _freeze_all(model)
    _unfreeze_lora(model)
    _print_trainable(model, "only")
    return model


def inject_lora_with_projectors(
    model: nn.Module, r: int = 16, alpha: float = 16.0
) -> nn.Module:
    """
    Like inject_lora, but also keeps the REPA projector MLPs trainable.

    Use for Run B (v-pred + REPA alignment loss).
    The projectors are lightweight (~few hundred K params) and need to
    adapt so the alignment loss is effective with the new LoRA features.
    """
    _replace_attn_linears(model, r=r, alpha=alpha)
    _freeze_all(model)
    _unfreeze_lora(model)

    for name, param in model.named_parameters():
        if "projectors" in name:
            param.requires_grad_(True)

    _print_trainable(model, "+ projectors")
    return model


# ---------------------------------------------------------------------------- #
# Save / load
# ---------------------------------------------------------------------------- #

def get_lora_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    """Return all trainable parameters (LoRA adapters + optional projectors)."""
    return {
        name: param.data.cpu()
        for name, param in model.named_parameters()
        if param.requires_grad
    }


def save_lora(model: nn.Module, path: str) -> None:
    """Persist trainable parameters as a safetensors file."""
    state = get_lora_state_dict(model)
    save_file(state, path)
    print(f"Saved LoRA weights ({len(state)} tensors) → {path}")


def load_lora(model: nn.Module, path: str) -> nn.Module:
    """
    Load LoRA (and optional projector) weights from a safetensors file.
    Uses strict=False so non-LoRA keys that are absent from the file
    are left untouched (their frozen pretrained values are preserved).
    """
    state = load_file(path)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if unexpected:
        print(f"  Unexpected keys ({len(unexpected)}): {unexpected[:3]} ...")
    print(f"Loaded LoRA weights ({len(state)} tensors) ← {path}")
    return model
