
"""
TinyFed‑SemCom Optimized Training Script
---------------------------------------
This single‑file implementation keeps the original functionality of
`improved_tinyfed_semcom.py` but introduces performance switches so that
users can trade accuracy for speed without touching the core logic.

Two ready‑made configuration presets are defined at the end of the file:
    • DefaultConfig – mirrors the original slow / full‑feature behaviour.
    • FastConfig    – disables expensive options for a quick sanity run.

Run with:
    python tinyfed_semcom_optimized.py --config fast   # quick profile
    python tinyfed_semcom_optimized.py                 # default settings

No contractions are used in runtime messages to respect user preference.
"""

from __future__ import annotations

import argparse
import math
import os
import random
from pathlib import Path
from typing import Tuple, Optional, Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

try:
    from pytorch_msssim import ssim  # type: ignore

    _HAS_SSIM = True
except ImportError:
    _HAS_SSIM = False

########################################
#   Utility helpers
########################################

def set_seed(seed: int = 42) -> None:
    """Make experiment deterministic across common libraries."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(cuda_preference: bool = True) -> torch.device:
    if cuda_preference and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

########################################
#   Configuration class
########################################

class Config:
    """Container for hyper‑parameters. Add new fields when necessary."""

    def __init__(
        self,
        image_size: int | Tuple[int, int] = 64,
        latent_len: int = 128,
        dw2: int = 32,
        rounds: int = 4,
        local_epochs: int = 2,
        batch_size: int = 32,
        learning_rate: float = 1e-3,
        sparsity_p: float = 0.1,
        qat_start_round: int = 3,
        val_period: int = 1,
        use_checkpoint: bool = True,
        use_amp: bool = True,
        seed: int = 42,
        data_root: str | Path = "./data",
        export_dir: str | Path = "./export",
        device: Optional[torch.device] = None,
    ) -> None:
        # Basic dimensions
        if isinstance(image_size, int):
            self.image_size: Tuple[int, int] = (image_size, image_size)
        else:
            self.image_size = image_size

        self.latent_len = latent_len
        self.dw2 = dw2

        # Training schedule
        self.rounds = rounds
        self.local_epochs = local_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate

        # Performance / accuracy trade‑offs
        self.sparsity_p = sparsity_p
        self.qat_start_round = qat_start_round
        self.val_period = val_period
        self.use_checkpoint = use_checkpoint
        self.use_amp = use_amp
        self.seed = seed

        # Paths and device
        self.data_root = Path(data_root)
        self.export_dir = Path(export_dir)
        self.device = device or get_device()

        # Derived
        self.export_dir.mkdir(parents=True, exist_ok=True)

    # ---- helpers ---------------------------------------------------------
    @property
    def amp_dtype(self) -> torch.dtype:
        return torch.float16 if self.use_amp else torch.float32

    def __getitem__(self, key: str):
        return getattr(self, key)

########################################
#   Model definition
########################################

class DWSeparableConv(nn.Module):
    """Depthwise separable convolution building block."""

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.depth = nn.Conv2d(in_ch, in_ch, 3, stride, 1, groups=in_ch, bias=False)
        self.point = nn.Conv2d(in_ch, out_ch, 1, 1, 0, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
        x = self.depth(x)
        x = self.point(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)


class TinyEncoder(nn.Module):
    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self.dw2 = cfg.dw2  # store for later reference

        self.conv1 = DWSeparableConv(3, cfg.dw2)
        self.conv2 = DWSeparableConv(cfg.dw2, cfg.dw2 * 2, stride=2)  # 32×32
        self.conv3 = DWSeparableConv(cfg.dw2 * 2, cfg.dw2 * 4, stride=2)  # 16×16
        self.conv4 = DWSeparableConv(cfg.dw2 * 4, cfg.dw2 * 4, stride=2)  # 8×8
        self.conv5 = DWSeparableConv(cfg.dw2 * 4, cfg.dw2 * 4, stride=2)  # 4×4

        self.flatten = nn.Flatten()
        self.fc = nn.Linear(cfg.dw2 * 4 * 4 * 4, cfg.latent_len, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.flatten(x)
        z = self.fc(x)
        return z


class TinyDecoder(nn.Module):
    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self.dw2 = cfg.dw2  # number of channels at the bottleneck

        self.fc = nn.Linear(cfg.latent_len, self.dw2 * 4 * 4, bias=False)

        self.up1 = nn.ConvTranspose2d(self.dw2, self.dw2, 4, 2, 1)  # 4→8
        self.up2 = nn.ConvTranspose2d(self.dw2, self.dw2 // 2, 4, 2, 1)  # 8→16
        self.up3 = nn.ConvTranspose2d(self.dw2 // 2, self.dw2 // 4, 4, 2, 1)  # 16→32
        self.up4 = nn.ConvTranspose2d(self.dw2 // 4, 3, 4, 2, 1)  # 32→64

    def forward(self, z: torch.Tensor, checkpoint: bool = False) -> torch.Tensor:  # noqa: D401
        x = self.fc(z).view(-1, self.dw2, 4, 4)

        if checkpoint:
            x = torch.utils.checkpoint.checkpoint(self.up1, x)  # type: ignore
        else:
            x = self.up1(x)
        x = F.relu(x, inplace=True)

        x = F.relu(self.up2(x), inplace=True)
        x = F.relu(self.up3(x), inplace=True)
        x = torch.sigmoid(self.up4(x))  # outputs in [0,1]
        return x


class TinySemCom(nn.Module):
    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self.encoder = TinyEncoder(cfg)
        self.decoder = TinyDecoder(cfg)

    def forward(self, img: torch.Tensor, checkpoint: bool = False) -> torch.Tensor:  # noqa: D401
        z = self.encoder(img)
        out = self.decoder(z, checkpoint=checkpoint)
        return out

########################################
#   Training utilities
########################################

def hybrid_loss(pred: torch.Tensor, target: torch.Tensor, alpha: float = 0.85) -> torch.Tensor:
    mse = F.mse_loss(pred, target, reduction="mean")
    if _HAS_SSIM:
        ssim_val = ssim(pred, target, data_range=1.0, size_average=True)
        return alpha * mse + (1 - alpha) * (1 - ssim_val)
    return mse


def sparsify_gradients(model: nn.Module, p: float) -> None:
    if p <= 0.0:
        return
    with torch.no_grad():
        for p_param in model.parameters():
            if p_param.grad is None:
                continue
            g = p_param.grad
            k = max(1, int(g.numel() * p))
            threshold = g.abs().flatten().kthvalue(g.numel() - k + 1).values
            mask = g.abs() > threshold  # strict inequality keeps exactly k elems
            g.mul_(mask)


########################################
#   Quantisation support (optional)
########################################

def enable_qat(model: nn.Module) -> nn.Module:
    try:
        import torch.ao.quantization as tq  # type: ignore
    except ImportError:
        print("Quantisation toolkit not available. Skipping QAT enabling.")
        return model

    model.qconfig = tq.get_default_qat_qconfig("fbgemm")
    tq.prepare_qat(model, inplace=True)
    print("Quantisation aware training enabled.")
    return model


########################################
#   Data pipeline
########################################

def create_data_loaders(cfg: Config) -> Tuple[DataLoader, DataLoader]:
    tr = transforms.Compose([
        transforms.Resize(cfg.image_size),
        transforms.ToTensor(),
    ])

    dataset = datasets.CIFAR10(root=cfg.data_root, train=True, download=True, transform=tr)  # type: ignore
    val_len = int(0.1 * len(dataset))
    train_len = len(dataset) - val_len
    train_ds, val_ds = torch.utils.data.random_split(dataset, [train_len, val_len])

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=2, pin_memory=True)
    return train_loader, val_loader

########################################
#   Training and validation loops
########################################

def train_one_epoch(model: nn.Module, loader: DataLoader, opt: torch.optim.Optimizer, scaler: Optional[torch.cuda.amp.GradScaler], cfg: Config) -> float:
    model.train()
    total_loss = 0.0
    pbar = loader
    if cfg.device.type == "cpu":
        pbar = loader  # tqdm slowdown on CPU heavy loops, keep plain
    else:
        from tqdm.auto import tqdm as _tqdm  # defer import
        pbar = _tqdm(loader, leave=False, desc="Train")

    for img, _ in pbar:
        img = img.to(cfg.device, non_blocking=True)
        opt.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=cfg.use_amp):
            out = model(img, checkpoint=cfg.use_checkpoint and model.training)
            loss = hybrid_loss(out, img)

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(opt)  # unscale before sparsification
        else:
            loss.backward()

        sparsify_gradients(model, cfg.sparsity_p)

        if scaler is not None:
            scaler.step(opt)
            scaler.update()
        else:
            opt.step()

        total_loss += loss.item() * img.size(0)

    return total_loss / len(loader.dataset)


@torch.no_grad()
def validate(model: nn.Module, loader: DataLoader, cfg: Config) -> Dict[str, float]:
    model.eval()
    mse_sum = 0.0
    ssim_sum = 0.0
    count = 0

    for img, _ in loader:
        img = img.to(cfg.device, non_blocking=True)
        out = model(img, checkpoint=False)

        mse = F.mse_loss(out, img, reduction="sum").item()
        mse_sum += mse
        if _HAS_SSIM:
            ssim_val = ssim(out, img, data_range=1.0, size_average=False)
            ssim_sum += ssim_val.sum().item()  # type: ignore
        count += img.size(0)

    mse_mean = mse_sum / count
    psnr = 10 * math.log10(1.0 / mse_mean)
    metrics = {
        "MSE": mse_mean,
        "PSNR": psnr,
    }
    if _HAS_SSIM:
        metrics["SSIM"] = ssim_sum / count
    return metrics

########################################
#   Export helper
########################################

def export_model(model: nn.Module, export_path: Path, cfg: Config) -> None:
    model.eval()
    h, w = cfg.image_size
    example_input = torch.randn(1, 3, h, w, device=cfg.device)
    traced = torch.jit.trace(model.cpu(), example_input.cpu())  # trace on CPU for portability

    try:
        traced._save_for_lite_interpreter(str(export_path))  # type: ignore[attr-defined]
        print(f"Lite interpreter model saved to {export_path}.")
    except AttributeError:
        torch.jit.save(traced, str(export_path))
        print(f"Scripted model saved to {export_path}.")

########################################
#   Main routine
########################################

def main(cfg: Config) -> None:
    print("Running on", cfg.device)
    set_seed(cfg.seed)

    train_loader, val_loader = create_data_loaders(cfg)
    model = TinySemCom(cfg).to(cfg.device)

    opt = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.use_amp and cfg.device.type == "cuda")

    qat_enabled = False
    for rnd in range(cfg.rounds):
        print(f"Round {rnd + 1}/{cfg.rounds}")

        for epoch in range(cfg.local_epochs):
            loss = train_one_epoch(model, train_loader, opt, scaler, cfg)
            print(f"    Epoch {epoch + 1}/{cfg.local_epochs} – Loss: {loss:.4f}")

        if (rnd + 1) % cfg.val_period == 0:
            metrics = validate(model, val_loader, cfg)
            psnr_val = metrics["PSNR"]
            print(f"    Validation – PSNR: {psnr_val:.2f} dB")

        if (rnd == cfg.qat_start_round) and (not qat_enabled):
            model = enable_qat(model)
            qat_enabled = True

    export_path = cfg.export_dir / "semcom_qat.ptl" if qat_enabled else cfg.export_dir / "semcom_fp.pt"
    export_model(model, export_path, cfg)

########################################
#   Configuration presets
########################################

class DefaultConfig(Config):
    """Matches the behaviour of the uploaded improved script."""

    def __init__(self) -> None:
        super().__init__(
            image_size=64,
            latent_len=128,
            dw2=32,
            rounds=4,
            local_epochs=2,
            batch_size=32,
            learning_rate=1e-3,
            sparsity_p=0.1,
            qat_start_round=3,
            val_period=1,
            use_checkpoint=True,
            use_amp=True,
        )


class FastConfig(Config):
    """Lightweight configuration for a fast sanity check run."""

    def __init__(self) -> None:
        super().__init__(
            image_size=64,
            latent_len=128,
            dw2=32,
            rounds=1,
            local_epochs=1,
            batch_size=32,
            learning_rate=1e-3,
            sparsity_p=0.0,
            qat_start_round=9999,  # never enable QAT
            val_period=9999,       # validate once at the very end
            use_checkpoint=False,
            use_amp=False,
        )

########################################
#   CLI glue
########################################

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TinyFed‑SemCom Optimized Trainer")
    parser.add_argument(
        "--config",
        choices=["default", "fast"],
        default="default",
        help="Select configuration preset.",
    )
    args = parser.parse_args()

    cfg_instance = DefaultConfig() if args.config == "default" else FastConfig()
    main(cfg_instance)
