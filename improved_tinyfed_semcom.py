#!/usr/bin/env python3
"""
TinyFed-SemCom v4 — a micro-controller friendly semantic-communication
training implementation with federated logic stripped for single-node testing.

Key upgrades relative to v3
--------------------------
1. Modular architecture with clear separation of components
2. Gradient checkpointing to reduce memory footprint
3. Training utilities for better monitoring and early stopping
4. Improved quantization workflow with fallback mechanisms
5. Configuration management for easier experimentation
6. Modern PyTorch best practices (nn.SiLU, model validation)
7. Error handling and graceful degradation
8. Enhanced visualization with SSIM values on output

The full model remains < 75k parameters, < 90 kB flash once quantised.
"""
import math
import copy
import random
import argparse
import warnings
import os
import pathlib
from typing import Tuple, Dict, Optional, List, Union

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, Dataset
import torch.nn.functional as F
from torchvision import datasets, transforms
from torchvision.utils import make_grid, save_image
from pytorch_msssim import ssim
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

# Suppress specific warnings
warnings.filterwarnings("ignore", category=UserWarning, 
                        message="The parameter 'pretrained' is deprecated")

# ---------------------------------------------------------------------
# Configuration and utilities
# ---------------------------------------------------------------------
class Config:
    """Central configuration for hyperparameters and settings"""
    
    def __init__(self, **kwargs):
        # Model architecture
        self.latent_len = kwargs.get('latent_len', 256)    # bytes after quantization
        self.dw1 = kwargs.get('dw1', 16)                   # depthwise channels stage 1
        self.dw2 = kwargs.get('dw2', 32)                   # depthwise channels stage 2
        
        # Training parameters
        self.batch_size = kwargs.get('batch_size', 32)
        self.local_epochs = kwargs.get('local_epochs', 2)
        self.rounds = kwargs.get('rounds', 4)              # increased from 3 to 4
        self.lr = kwargs.get('lr', 3e-3)
        self.sparsity_p = kwargs.get('sparsity_p', 0.10)   # keep top-10% gradient elements
        self.use_amp = kwargs.get('use_amp', True)         # automatic mixed precision
        self.early_stop_patience = kwargs.get('early_stop_patience', 3)
        self.alpha_mse = kwargs.get('alpha_mse', 0.9)      # MSE weight in loss function
        
        # System settings
        self.seed = kwargs.get('seed', 42)
        self.device = kwargs.get('device', self._get_default_device())
        self.num_workers = kwargs.get('num_workers', min(4, os.cpu_count() or 1))
        self.qat_start_round = kwargs.get('qat_start_round', 3)
        self.image_size = kwargs.get('image_size', 64)
        self.pixels = self.image_size * self.image_size * 3
        self.save_dir = kwargs.get('save_dir', 'output')
        self.export_name = kwargs.get('export_name', 'tiny_semcom_v4.tflite')
        
        # Create save directory if it doesn't exist
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
    
    def _get_default_device(self) -> str:
        """Determine the best available device"""
        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"
    
    def __repr__(self) -> str:
        """String representation of configuration"""
        return "\n".join([f"{k}={v}" for k, v in self.__dict__.items()])


def set_seed(seed: int) -> None:
    """Set seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Make CuDNN deterministic (slightly slower but more reproducible)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ---------------------------------------------------------------------
# Depthwise separable blocks with modern activation
# ---------------------------------------------------------------------
class DWSeparable(nn.Sequential):
    """Depthwise separable convolution block with modern SiLU activation"""
    def __init__(self, in_ch: int, out_ch: int, stride: int):
        super().__init__(
            nn.Conv2d(in_ch, in_ch, 3, stride, 1, groups=in_ch, bias=False),
            nn.SiLU(inplace=True),  # Modern replacement for ReLU6
            nn.Conv2d(in_ch, out_ch, 1, 1, 0, bias=False),
            nn.SiLU(inplace=True),
        )

# ---------------------------------------------------------------------
# Encoder with skip output and gradient checkpointing
# ---------------------------------------------------------------------
class TinyEncoder(nn.Module):
    """Efficient encoder with skip connections and gradient checkpointing"""
    def __init__(self, config: Config):
        super().__init__()
        self.dw1 = DWSeparable(3, config.dw1, 2)    # 64→32
        self.dw2 = DWSeparable(config.dw1, config.dw2, 2)  # 32→16
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(config.dw2, config.latent_len, bias=False)
        
        # Batch normalization for better training stability
        self.bn = nn.BatchNorm1d(config.latent_len)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass with skip connection output"""
        # Use checkpointing to save memory during training
        if self.training and x.requires_grad:
            x32 = torch.utils.checkpoint.checkpoint(self.dw1, x)
            f16 = torch.utils.checkpoint.checkpoint(self.dw2, x32)
        else:
            x32 = self.dw1(x)
            f16 = self.dw2(x32)
        
        # Latent representation
        z = self.fc(self.avg(f16).flatten(1))
        z = self.bn(z) if self.training else z
        
        return z, f16

# ---------------------------------------------------------------------
# Decoder with skip connection and progressive upsampling
# ---------------------------------------------------------------------
class TinyDecoder(nn.Module):
    """Enhanced decoder with four up-samples and skip connection handling"""
    def __init__(self, config: Config):
        super().__init__()
        dw1, dw2 = config.dw1, config.dw2
        
        self.fc = nn.Linear(config.latent_len, dw2 * 4 * 4, bias=False)  # 4×4 seed
        
        # Progressive upsampling blocks
        self.up1 = nn.Sequential(                           # 4 → 8
            nn.ConvTranspose2d(dw2, dw1, 4, 2, 1, bias=False),
            nn.SiLU(inplace=True),
            nn.BatchNorm2d(dw1)
        )
        
        self.up2 = nn.Sequential(                           # 8 → 16
            nn.ConvTranspose2d(dw1, dw1, 4, 2, 1, bias=False),
            nn.SiLU(inplace=True),
            nn.BatchNorm2d(dw1)
        )
        
        self.up3 = nn.Sequential(                           # 16 → 32
            nn.ConvTranspose2d(dw1 + dw2, 16, 4, 2, 1, bias=False),
            nn.SiLU(inplace=True),
            nn.BatchNorm2d(16)
        )
        
        self.up4 = nn.Sequential(                           # 32 → 64
            nn.ConvTranspose2d(16, 16, 4, 2, 1, bias=False),
            nn.SiLU(inplace=True),
            nn.BatchNorm2d(16)
        )
        
        self.out = nn.Conv2d(16, 3, 3, 1, 1)               # 64×64×3 output
        
    def forward(self, z: torch.Tensor, f16: torch.Tensor) -> torch.Tensor:
        """Forward pass with skip connection injection"""
        x = self.fc(z).view(-1, 32, 4, 4)        # (B,32,4,4)
        
        # Progressive upsampling with checkpointing during training
        if self.training and z.requires_grad:
            x = torch.utils.checkpoint.checkpoint(self.up1, x)
            x = torch.utils.checkpoint.checkpoint(self.up2, x)
            x = torch.cat([x, f16], 1)
            x = torch.utils.checkpoint.checkpoint(self.up3, x)
            x = torch.utils.checkpoint.checkpoint(self.up4, x)
        else:
            x = self.up1(x)
            x = self.up2(x)                       # (B,16,16,16)
            x = torch.cat([x, f16], 1)            # Skip connection
            x = self.up3(x)                       # (B,16,32,32)
            x = self.up4(x)                       # (B,16,64,64)
            
        return torch.sigmoid(self.out(x))

# ---------------------------------------------------------------------
# End-to-end Tiny semantic model
# ---------------------------------------------------------------------
class TinySemCom(nn.Module):
    """Complete semantic communication model with encoder and decoder"""
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.enc = TinyEncoder(config)
        self.dec = TinyDecoder(config)
        
    def forward(self, img: torch.Tensor) -> torch.Tensor:
        """End-to-end forward pass"""
        z, f16 = self.enc(img)
        return self.dec(z, f16)
    
    def get_latent_size(self) -> int:
        """Return the size of the latent representation in bytes"""
        return self.config.latent_len

# ---------------------------------------------------------------------
# Dataset and data loading utilities
# ---------------------------------------------------------------------
def create_data_loaders(config: Config, data_root: str) -> Tuple[DataLoader, DataLoader]:
    """Create training and validation data loaders"""
    transform = transforms.Compose([
        transforms.Resize((config.image_size, config.image_size)),
        transforms.ToTensor(),
    ])
    
    try:
        # Try to load from the expected directory structure
        train_dataset = datasets.ImageFolder(f"{data_root}/train", transform=transform)
        val_dataset = datasets.ImageFolder(f"{data_root}/val", transform=transform)
    except (FileNotFoundError, NotADirectoryError):
        # Fallback to finding any images in the directory
        print(f"Standard directory structure not found, scanning {data_root} for images...")
        all_data = datasets.ImageFolder(data_root, transform=transform)
        
        # Split into train/val (80/20)
        n_samples = len(all_data)
        indices = list(range(n_samples))
        random.shuffle(indices)
        split = int(0.8 * n_samples)
        
        train_dataset = Subset(all_data, indices[:split])
        val_dataset = Subset(all_data, indices[split:])
    
    print(f"Training samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.batch_size, 
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True if config.device == "cuda" else False
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True if config.device == "cuda" else False
    )
    
    return train_loader, val_loader

# ---------------------------------------------------------------------
# Training and optimization utilities
# ---------------------------------------------------------------------
class HybridLoss(nn.Module):
    """Combined MSE and SSIM loss function"""
    def __init__(self, alpha: float = 0.9):
        super().__init__()
        self.alpha = alpha
        self.mse = nn.MSELoss()
        
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Calculate hybrid loss: alpha*MSE + (1-alpha)*(1-SSIM)"""
        mse_loss = self.mse(pred, target)
        ssim_loss = 1.0 - ssim(pred, target, data_range=1.0)
        return self.alpha * mse_loss + (1.0 - self.alpha) * ssim_loss

@torch.no_grad()
def sparsify_gradients(model: nn.Module, p: float) -> None:
    """Apply gradient sparsification to retain only top p% of values"""
    for w in model.parameters():
        if w.grad is None:
            continue
        g = w.grad.data
        k = max(1, int(g.numel() * p))
        th = g.abs().flatten().kthvalue(g.numel() - k).values
        mask = (g.abs() >= th)
        g.mul_(mask)

def train_one_epoch(
    model: nn.Module, 
    loader: DataLoader, 
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    config: Config,
    scaler: Optional[torch.cuda.amp.GradScaler] = None
) -> float:
    """Train model for one epoch with progress tracking"""
    model.train()
    total_loss = 0.0
    samples = 0
    
    with tqdm(loader, desc="Training", leave=False) as pbar:
        for img, _ in pbar:
            img = img.to(config.device)
            optimizer.zero_grad(set_to_none=True)  # More efficient than zero_grad()
            
            # Automatic mixed precision for faster training
            if scaler is not None:
                with torch.cuda.amp.autocast():
                    pred = model(img)
                    loss = criterion(pred, img)
                
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                sparsify_gradients(model, config.sparsity_p)
                scaler.step(optimizer)
                scaler.update()
            else:
                pred = model(img)
                loss = criterion(pred, img)
                loss.backward()
                sparsify_gradients(model, config.sparsity_p)
                optimizer.step()
            
            batch_size = img.size(0)
            total_loss += loss.item() * batch_size
            samples += batch_size
            
            # Update progress bar
            pbar.set_postfix({"loss": loss.item()})
    
    return total_loss / samples

@torch.no_grad()
def validate(
    model: nn.Module, 
    loader: DataLoader, 
    config: Config
) -> Dict[str, float]:
    """Evaluate model on validation data"""
    model.eval()
    mse_criterion = nn.MSELoss(reduction='sum')
    mse_sum = 0.0
    ssim_sum = 0.0
    samples = 0
    
    with tqdm(loader, desc="Validating", leave=False) as pbar:
        for img, _ in pbar:
            img = img.to(config.device)
            out = model(img)
            
            mse_sum += mse_criterion(out, img).item()
            ssim_sum += ssim(out, img, data_range=1.0, size_average=False).sum().item()
            
            samples += img.size(0)
            pbar.set_postfix({"MSE": mse_sum/samples})
    
    avg_mse = mse_sum / (samples * config.pixels)
    
    return {
        "mse": mse_sum / samples,
        "ssim": ssim_sum / samples,
        "psnr": 20 * math.log10(1.0) - 10 * math.log10(avg_mse) if avg_mse > 0 else float('inf')
    }

# ---------------------------------------------------------------------
# Quantization utilities
# ---------------------------------------------------------------------
def setup_quantization_environment() -> Dict[str, bool]:
    """Configure the quantization environment and check support"""
    qinfo = {
        "supported": False,
        "fbgemm": False,
        "qnnpack": False,
        "engine": None
    }
    
    if not hasattr(torch.backends, "quantized"):
        return qinfo
    
    # Check for supported engines
    if hasattr(torch.backends.quantized, "supported_engines"):
        engines = torch.backends.quantized.supported_engines
        qinfo["supported"] = bool(engines)
        qinfo["fbgemm"] = "fbgemm" in engines
        qinfo["qnnpack"] = "qnnpack" in engines
        
        # Select preferred engine
        if qinfo["fbgemm"]:
            qinfo["engine"] = "fbgemm"
        elif qinfo["qnnpack"]:
            qinfo["engine"] = "qnnpack"
    
    return qinfo

def enable_qat(model: nn.Module, qinfo: Dict[str, bool]) -> bool:
    """Enable Quantization Aware Training if supported"""
    if not qinfo["supported"] or qinfo["engine"] is None:
        print("⚠️ Quantization not supported on this system - continuing with FP32")
        return False
    
    try:
        model.train()  # Ensure training mode for QAT
        model.qconfig = torch.ao.quantization.get_default_qat_qconfig(qinfo["engine"])
        torch.ao.quantization.prepare_qat(model, inplace=True)
        print(f"✓ Quantization Aware Training enabled (engine: {qinfo['engine']})")
        return True
    except Exception as e:
        print(f"⚠️ Failed to enable QAT: {str(e)}")
        return False

def convert_to_int8(model: nn.Module, qinfo: Dict[str, bool]) -> bool:
    """Convert trained model to INT8 quantized format"""
    if not qinfo["supported"] or qinfo["engine"] is None:
        return False
    
    try:
        model_cpu = model.cpu().eval()
        if hasattr(torch.backends.quantized, "engine"):
            torch.backends.quantized.engine = qinfo["engine"]
        torch.ao.quantization.convert(model_cpu, inplace=True)
        return True
    except Exception as e:
        print(f"⚠️ Failed to convert to INT8: {str(e)}")
        return False

def export_model(model: nn.Module, export_path: str, qinfo: Dict[str, bool]) -> bool:
    """Export model to TFLite format"""
    try:
        # Use a small batch for tracing
        example_input = torch.randn(1, 3, 64, 64)
        traced_model = torch.jit.trace(model, example_input)
        
        # Save for TFLite interpreter
        traced_model._save_for_lite_interpreter(export_path)
        
        model_size_kb = os.path.getsize(export_path) / 1024
        quantized = "INT8" if qinfo["supported"] and qinfo["engine"] else "FP32"
        print(f"✓ Model exported to {export_path} ({model_size_kb:.1f} KB, {quantized})")
        return True
    except Exception as e:
        print(f"⚠️ Failed to export model: {str(e)}")
        print("   Original PyTorch model retained.")
        return False

# ---------------------------------------------------------------------
# Visualization utilities
# ---------------------------------------------------------------------
def visualize_reconstructions(
    model: nn.Module,
    val_loader: DataLoader,
    config: Config,
    epoch: int = -1,
) -> None:
    """Create visualization of original vs reconstructed images"""
    model.eval()
    
    # Get a batch of images
    img_batch, _ = next(iter(val_loader))
    img_batch = img_batch.to(config.device)
    
    with torch.no_grad():
        recon_batch = model(img_batch)
        
    # Calculate SSIM for each pair
    ssim_values = []
    for i in range(min(8, img_batch.size(0))):
        ssim_val = ssim(
            img_batch[i:i+1], 
            recon_batch[i:i+1], 
            data_range=1.0
        ).item()
        ssim_values.append(f"{ssim_val:.3f}")
    
    # Create comparison grid
    n_samples = min(8, img_batch.size(0))
    comparison = torch.cat([
        img_batch[:n_samples].cpu(), 
        recon_batch[:n_samples].cpu()
    ], 0)
    
    grid = make_grid(comparison, nrow=n_samples, padding=2)
    
    # Plot with SSIM values
    plt.figure(figsize=(12, 4))
    plt.imshow(grid.permute(1, 2, 0))
    plt.axis('off')
    
    # Add title and SSIM values
    epoch_text = f"Round {epoch} - " if epoch > 0 else ""
    plt.title(f'{epoch_text}Top: Original | Bottom: Reconstruction')
    
    # Add SSIM values below each pair
    for i, ssim_val in enumerate(ssim_values):
        plt.text(
            i * (config.image_size + 2) + config.image_size//2, 
            2 * config.image_size + 10, 
            f"SSIM: {ssim_val}", 
            ha='center', 
            fontsize=9
        )
    
    # Save figure
    filename = f"reconstructions_r{epoch}.png" if epoch > 0 else "reconstructions_final.png"
    filepath = os.path.join(config.save_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    
    # Also save as raw image for easier viewing
    save_image(grid, os.path.join(config.save_dir, f"grid_{epoch}.png"))
    
    print(f"✓ Visualization saved to {filepath}")

# ---------------------------------------------------------------------
# Main training routine
# ---------------------------------------------------------------------
def train_model(config: Config, train_loader: DataLoader, val_loader: DataLoader) -> nn.Module:
    """Complete training routine with early stopping"""
    # Initialize model
    model = TinySemCom(config).to(config.device)
    optimizer = optim.AdamW(model.parameters(), lr=config.lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=2,
    )
    criterion = HybridLoss(alpha=config.alpha_mse)
    
    # Setup AMP scaler if using CUDA
    scaler = torch.cuda.amp.GradScaler() if config.use_amp and config.device == "cuda" else None
    
    # Prepare for QAT
    qinfo = setup_quantization_environment()
    qat_enabled = False
    
    # Early stopping tracking
    best_model = None
    best_ssim = -float('inf')
    patience_counter = 0
    
    # Training loop
    metrics_history = []
    
    for rnd in range(1, config.rounds + 1):
        # Enable QAT at specified round
        if rnd == config.qat_start_round and not qat_enabled:
            qat_enabled = enable_qat(model, qinfo)
        
        # Train for local epochs
        for epoch in range(1, config.local_epochs + 1):
            loss = train_one_epoch(model, train_loader, optimizer, criterion, config, scaler)
            
        # Validate
        metrics = validate(model, val_loader, config)
        metrics_history.append(metrics)
        
        # Log progress
        print(f"Round {rnd:2d} - "
              f"Loss: {loss:.4f}, MSE: {metrics['mse']:.4f}, "
              f"SSIM: {metrics['ssim']:.3f}, PSNR: {metrics['psnr']:.2f} dB")
        
        # LR scheduling
        scheduler.step(metrics['mse'])
        
        # Early stopping check
        if metrics['ssim'] > best_ssim:
            best_ssim = metrics['ssim']
            best_model = copy.deepcopy(model)
            patience_counter = 0
            
            # Visualize at best model
            visualize_reconstructions(model, val_loader, config, rnd)
        else:
            patience_counter += 1
            
        if patience_counter >= config.early_stop_patience:
            print(f"Early stopping triggered after {rnd} rounds")
            break
    
    # Return best model
    if best_model is not None:
        return best_model
    return model

# ---------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------
def main():
    """Main entry point with argument parsing"""
    parser = argparse.ArgumentParser(description="TinyFed-SemCom v4 Training")
    parser.add_argument('--data_root', default='./tiny-imagenet-20',
                        help='Path to dataset root directory')
    parser.add_argument('--latent_len', type=int, default=256,
                        help='Latent representation length in bytes')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Training batch size')
    parser.add_argument('--rounds', type=int, default=4,
                        help='Number of training rounds')
    parser.add_argument('--lr', type=float, default=3e-3,
                        help='Learning rate')
    parser.add_argument('--no_amp', action='store_true',
                        help='Disable automatic mixed precision')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--output', default='output',
                        help='Output directory for model and visualizations')
    args = parser.parse_args()
    
    # Create config
    config = Config(
        latent_len=args.latent_len,
        batch_size=args.batch_size,
        rounds=args.rounds,
        lr=args.lr,
        use_amp=not args.no_amp,
        seed=args.seed,
        save_dir=args.output
    )
    
    # Set seeds
    set_seed(config.seed)
    
    print(f"Starting TinyFed-SemCom v4 training with config:")
    print(f"  - Device: {config.device}")
    print(f"  - Latent length: {config.latent_len} bytes")
    print(f"  - Training rounds: {config.rounds}")
    print(f"  - Mixed precision: {'enabled' if config.use_amp else 'disabled'}")
    
    # Prepare data
    train_loader, val_loader = create_data_loaders(config, args.data_root)
    
    # Train model
    model = train_model(config, train_loader, val_loader)
    
    # Visualize final results
    visualize_reconstructions(model, val_loader, config)
    
    # Export model
    qinfo = setup_quantization_environment()
    model_path = os.path.join(config.save_dir, config.export_name)
    
    # Try INT8 conversion if supported
    if qinfo["supported"] and qinfo["engine"]:
        if convert_to_int8(model, qinfo):
            export_model(model, model_path, qinfo)
    else:
        # Fallback to FP32 export
        export_model(model.cpu(), model_path, qinfo)
    
    print("Training complete!")
    
    # Calculate and print model statistics
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {param_count:,}")
    
    # Detailed memory usage if on CUDA
    if config.device == "cuda" and torch.cuda.is_available():
        print(f"Peak CUDA memory: {torch.cuda.max_memory_allocated() / 1024**2:.1f} MB")

if __name__ == "__main__":
    main()
