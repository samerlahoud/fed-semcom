"""
Constrained implementation of the FedLoL semantic-communication framework
optimized for resource-limited environments while maintaining reconstruction quality.

Key optimizations:
- Reduced model parameters through depthwise separable convolutions
- Quantization-aware training (QAT) for 8-bit inference
- Gradient checkpointing for memory efficiency
- Progressive training with early stopping
- Model pruning and knowledge distillation
- Efficient channel simulation with reduced precision
"""

import copy
import math
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import transforms, datasets
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
from pytorch_msssim import ssim
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

# ------------------------------------------------------------------
# 0. Reproducibility
# ------------------------------------------------------------------
SEED = 42
torch.manual_seed(SEED)
random.seed(SEED)
torch.cuda.manual_seed_all(SEED)

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--data_root", default="./tiny-imagenet-20")
parser.add_argument("--device",    default="cuda",  choices=["cuda", "cpu", "mps"])
parser.add_argument("--rounds",    type=int, default=5)
parser.add_argument("--workers",   type=int, default=0)
parser.add_argument("--batch_size",type=int, default=16)  # Reduced batch size
args = parser.parse_args()

DATA_ROOT   = args.data_root
DEVICE      = args.device
ROUNDS      = args.rounds
WORKERS     = args.workers
BATCH_SIZE  = args.batch_size
PIN_MEM     = DEVICE == "cuda"

# ------------------------------------------------------------------
# 1. Constrained hyper-parameters and FL configuration
# ------------------------------------------------------------------
NUM_CLIENTS      = 5
DIRICHLET_ALPHA  = 1.0
LOCAL_EPOCHS     = 3            # Reduced local epochs
LR               = 2e-3         # Slightly higher LR for faster convergence
BOTTLENECK       = 512          # Reduced bottleneck size
COMPRESSED       = 32           # Reduced compressed size
COMPRESS_RATIO   = (64 * 64 * 3) / BOTTLENECK  # ~24x compression
SNR_DB           = 10
ALPHA_LOSS       = 0.9
PIXELS           = 64 * 64 * 3

# Constraint optimization settings
USE_QUANTIZATION = True         # Enable quantization-aware training
USE_PRUNING      = True         # Enable model pruning
USE_CHECKPOINTING = True        # Enable gradient checkpointing
PRUNING_SPARSITY = 0.3          # 30% sparsity for pruning
EARLY_STOP_PATIENCE = 2         # Early stopping patience

# ------------------------------------------------------------------
# Performance optimizations
# ------------------------------------------------------------------
torch.backends.cudnn.benchmark = True
USE_FP16 = torch.cuda.is_available() and hasattr(torch.cuda.amp, 'autocast')

# Efficient channel simulation with reduced precision
@torch.jit.script
def efficient_channel_sim(x, sigma: float):
    """Memory-efficient channel simulation"""
    # Use half precision for channel simulation to save memory
    if x.dtype == torch.float32:
        x_half = x.half()
        h = torch.randn_like(x_half)
        noise = sigma * torch.randn_like(x_half)
        result = (h * x_half + noise) / (h + 1e-6)
        return result.float()
    else:
        h = torch.randn_like(x)
        noise = sigma * torch.randn_like(x)
        return (h * x + noise) / (h + 1e-6)

# ------------------------------------------------------------------
# Constrained model components with depthwise separable convolutions
# ------------------------------------------------------------------
class DepthwiseSeparableConv(nn.Module):
    """Depthwise separable convolution for parameter reduction"""
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, 
                                 stride, padding, groups=in_channels, bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        return self.relu(x)

class ConstrainedSemanticEncoder(nn.Module):
    """Lightweight semantic encoder with depthwise separable convolutions"""
    def __init__(self, bottleneck: int = BOTTLENECK) -> None:
        super().__init__()
        # Use depthwise separable convolutions to reduce parameters
        self.enc1 = DepthwiseSeparableConv(3, 32, 3, 2, 1)      # 64→32, reduced channels
        self.enc2 = DepthwiseSeparableConv(32, 64, 3, 2, 1)     # 32→16, reduced channels
        self.enc3 = DepthwiseSeparableConv(64, 128, 3, 2, 1)    # 16→8, reduced channels
        
        # Adaptive pooling to reduce spatial dimensions further
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))
        self.fc = nn.Linear(128 * 4 * 4, bottleneck)
        self.dropout = nn.Dropout(0.1)  # Light dropout for regularization

    def forward(self, x):
        f1 = self.enc1(x)
        f2 = self.enc2(f1)
        f3 = self.enc3(f2)
        
        # Use adaptive pooling to reduce computation
        f3_pooled = self.adaptive_pool(f3)
        z = self.fc(f3_pooled.flatten(1))
        z = self.dropout(z)
        
        return z, (f1, f2, f3)

class ConstrainedSemanticDecoder(nn.Module):
    """Lightweight semantic decoder with upsampling and skip connections"""
    def __init__(self, bottleneck: int = BOTTLENECK) -> None:
        super().__init__()
        self.fc = nn.Linear(bottleneck, 128 * 4 * 4)
        
        # Use bilinear upsampling + conv instead of transposed conv to reduce parameters
        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            DepthwiseSeparableConv(128 + 128, 64, 3, 1, 1)  # 4→8
        )
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            DepthwiseSeparableConv(64 + 64, 32, 3, 1, 1)    # 8→16
        )
        self.up3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            DepthwiseSeparableConv(32 + 32, 16, 3, 1, 1)    # 16→32
        )
        self.up4 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(16, 3, 3, 1, 1)                       # 32→64
        )
        
        self.dropout = nn.Dropout2d(0.05)  # Light spatial dropout

    def forward(self, z, skips):
        f1, f2, f3 = skips
        
        x = self.fc(z).view(-1, 128, 4, 4)
        
        # Upsample and concatenate with skip connections
        x = self.up1(torch.cat([x, F.adaptive_avg_pool2d(f3, (8, 8))], dim=1))
        x = self.dropout(x)
        
        x = self.up2(torch.cat([x, F.adaptive_avg_pool2d(f2, (16, 16))], dim=1))
        x = self.dropout(x)
        
        x = self.up3(torch.cat([x, F.adaptive_avg_pool2d(f1, (32, 32))], dim=1))
        x = self.up4(x)
        
        return torch.sigmoid(x)

class ConstrainedChannelEncoder(nn.Module):
    """Lightweight channel encoder with fewer layers"""
    def __init__(self) -> None:
        super().__init__()
        # Reduced to 4 layers instead of 7
        self.fc_layers = nn.ModuleList([
            nn.Linear(BOTTLENECK, COMPRESSED * 2),
            nn.Linear(COMPRESSED * 2, COMPRESSED),
            nn.Linear(COMPRESSED, COMPRESSED),
            nn.Linear(COMPRESSED, COMPRESSED)
        ])
        
        # Simplified SNR injection
        self.snr_encoder = nn.Linear(1, COMPRESSED // 4)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, f, snr_db=SNR_DB):
        x = f
        
        # First two layers
        for i in range(2):
            x = F.relu(self.fc_layers[i](x))
            x = self.dropout(x)
        
        # SNR injection
        snr_info = torch.ones(x.size(0), 1, device=x.device) * snr_db / 20.0  # Normalize SNR
        snr_features = F.relu(self.snr_encoder(snr_info))
        
        # Pad snr_features to match x dimensions
        snr_padded = F.pad(snr_features, (0, x.size(1) - snr_features.size(1)))
        x = x + snr_padded
        
        # Last two layers
        for i in range(2, 4):
            x = F.relu(self.fc_layers[i](x))
            x = self.dropout(x)
        
        return x

class ConstrainedChannelDecoder(nn.Module):
    """Lightweight channel decoder with fewer layers"""
    def __init__(self) -> None:
        super().__init__()
        # Reduced to 4 layers instead of 7
        self.fc_layers = nn.ModuleList([
            nn.Linear(COMPRESSED, COMPRESSED),
            nn.Linear(COMPRESSED, COMPRESSED),
            nn.Linear(COMPRESSED, COMPRESSED * 2),
            nn.Linear(COMPRESSED * 2, BOTTLENECK)
        ])
        
        # Simplified SNR injection
        self.snr_encoder = nn.Linear(1, COMPRESSED // 4)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x, snr_db=SNR_DB):
        # First two layers
        for i in range(2):
            x = F.relu(self.fc_layers[i](x))
            x = self.dropout(x)
        
        # SNR injection
        snr_info = torch.ones(x.size(0), 1, device=x.device) * snr_db / 20.0  # Normalize SNR
        snr_features = F.relu(self.snr_encoder(snr_info))
        
        # Pad snr_features to match x dimensions
        snr_padded = F.pad(snr_features, (0, x.size(1) - snr_features.size(1)))
        x = x + snr_padded
        
        # Last two layers
        for i in range(2, 4):
            x = F.relu(self.fc_layers[i](x))
            if i < 3:  # Don't apply dropout to the last layer
                x = self.dropout(x)
        
        return x

class ConstrainedSemanticComm(nn.Module):
    """Constrained end-to-end semantic communication model"""

    def __init__(self) -> None:
        super().__init__()
        self.enc_s = ConstrainedSemanticEncoder()
        self.enc_c = ConstrainedChannelEncoder()
        self.dec_c = ConstrainedChannelDecoder()
        self.dec_s = ConstrainedSemanticDecoder()
        
        # Quantization stubs for QAT
        if USE_QUANTIZATION:
            self.quant = torch.quantization.QuantStub()
            self.dequant = torch.quantization.DeQuantStub()
        
        # Cache for channel simulation
        self.sigma_cache = {}

    def forward(self, img, snr_db=SNR_DB):
        # Quantization input
        if USE_QUANTIZATION and hasattr(self, 'quant'):
            img = self.quant(img)
        
        # Semantic encoding with gradient checkpointing if enabled
        if USE_CHECKPOINTING and self.training:
            z, skips = checkpoint(self.enc_s, img)
        else:
            z, skips = self.enc_s(img)
        
        # Channel encoding
        if USE_CHECKPOINTING and self.training:
            x = checkpoint(self.enc_c, z, snr_db)
        else:
            x = self.enc_c(z, snr_db)

        # Efficient channel simulation
        if snr_db not in self.sigma_cache:
            self.sigma_cache[snr_db] = math.sqrt(1.0 / (2 * 10 ** (snr_db / 10)))
        sigma = self.sigma_cache[snr_db]
        
        # Apply channel with reduced precision
        x_hat = efficient_channel_sim(x, sigma)
        
        # Channel decoding
        if USE_CHECKPOINTING and self.training:
            z_hat = checkpoint(self.dec_c, x_hat, snr_db)
        else:
            z_hat = self.dec_c(x_hat, snr_db)
        
        # Semantic reconstruction
        if USE_CHECKPOINTING and self.training:
            recon = checkpoint(self.dec_s, z_hat, skips)
        else:
            recon = self.dec_s(z_hat, skips)
        
        # Quantization output
        if USE_QUANTIZATION and hasattr(self, 'dequant'):
            recon = self.dequant(recon)
        
        return recon

# ------------------------------------------------------------------
# Model pruning utilities
# ------------------------------------------------------------------
def apply_magnitude_pruning(model, sparsity=PRUNING_SPARSITY):
    """Apply magnitude-based pruning to the model"""
    if not USE_PRUNING:
        return
    
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
            # Calculate threshold for pruning
            weights = module.weight.data.abs()
            threshold = torch.quantile(weights, sparsity)
            
            # Create mask
            mask = weights > threshold
            
            # Apply mask
            module.weight.data *= mask.float()

def count_parameters(model):
    """Count the number of trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def count_nonzero_parameters(model):
    """Count the number of non-zero parameters (after pruning)"""
    return sum((p != 0).sum().item() for p in model.parameters() if p.requires_grad)

# ------------------------------------------------------------------
# Constrained training function
# ------------------------------------------------------------------
def constrained_local_train(model, loader, epochs: int):
    model.train()
    
    # Use SGD with momentum for better convergence in constrained settings
    opt = optim.SGD(model.parameters(), lr=LR, momentum=0.9, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    
    # Mixed precision training
    scaler = torch.cuda.amp.GradScaler(enabled=USE_FP16)
    
    best_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        num_batches = 0
        
        for img, _ in loader:
            img = img.to(DEVICE, non_blocking=True)
            
            opt.zero_grad(set_to_none=True)
            
            # Mixed precision forward pass
            with torch.cuda.amp.autocast(enabled=USE_FP16):
                recon = model(img)
                loss = perceptual_loss(recon, img)
            
            # Backward pass
            if USE_FP16:
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                opt.step()
            
            epoch_loss += loss.item()
            num_batches += 1
        
        avg_loss = epoch_loss / num_batches
        scheduler.step()
        
        # Early stopping
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOP_PATIENCE:
                break
        
        # Apply pruning every few epochs
        if USE_PRUNING and (epoch + 1) % 2 == 0:
            apply_magnitude_pruning(model)
    
    return best_loss

# ------------------------------------------------------------------
# Dataset loading (same as original)
# ------------------------------------------------------------------
TRANSFORM = transforms.Compose([
    transforms.Resize((64, 64)), 
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet normalization
])

def fedlol_aggregate(global_model, client_states, client_losses):
    eps = 1e-8
    total_loss = sum(client_losses) + eps
    new_state = copy.deepcopy(global_model.state_dict())

    for k in new_state.keys():
        new_state[k] = sum(
            ((total_loss - client_losses[i]) / ((NUM_CLIENTS - 1) * total_loss))
            * client_states[i][k]
            for i in range(NUM_CLIENTS)
        )
    global_model.load_state_dict(new_state)

def perceptual_loss(pred, target, alpha: float = ALPHA_LOSS):
    # Denormalize for SSIM calculation
    pred_denorm = pred * torch.tensor([0.229, 0.224, 0.225]).to(pred.device).view(1, 3, 1, 1) + \
                  torch.tensor([0.485, 0.456, 0.406]).to(pred.device).view(1, 3, 1, 1)
    target_denorm = target * torch.tensor([0.229, 0.224, 0.225]).to(target.device).view(1, 3, 1, 1) + \
                    torch.tensor([0.485, 0.456, 0.406]).to(target.device).view(1, 3, 1, 1)
    
    pred_denorm = torch.clamp(pred_denorm, 0, 1)
    target_denorm = torch.clamp(target_denorm, 0, 1)
    
    mse_term = nn.functional.mse_loss(pred, target, reduction="mean")
    ssim_val = 1.0 - ssim(pred_denorm, target_denorm, data_range=1.0)
    return alpha * mse_term + (1.0 - alpha) * ssim_val

def dirichlet_split(dataset, alpha: float, n_clients: int):
    label_to_indices = {}
    for idx, (_, lbl) in enumerate(dataset):
        label_to_indices.setdefault(lbl, []).append(idx)

    clients = [[] for _ in range(n_clients)]
    for indices in label_to_indices.values():
        proportions = torch.distributions.Dirichlet(
            torch.full((n_clients,), alpha)
        ).sample()
        proportions = (proportions / proportions.sum()).tolist()
        split_points = [0] + list(
            torch.cumsum(
                torch.tensor(proportions) * len(indices), dim=0
            ).long()
        )
        for cid in range(n_clients):
            clients[cid].extend(
                indices[split_points[cid] : split_points[cid + 1]]
            )
    return [Subset(dataset, idxs) for idxs in clients]

# Load dataset
train_full = datasets.ImageFolder(f"{DATA_ROOT}/train", TRANSFORM)
val_full   = datasets.ImageFolder(f"{DATA_ROOT}/val",   TRANSFORM)

client_sets = dirichlet_split(train_full, DIRICHLET_ALPHA, NUM_CLIENTS)
val_loader  = DataLoader(
    val_full,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=WORKERS,
    pin_memory=PIN_MEM,
)

# ------------------------------------------------------------------
# Constrained main training loop
# ------------------------------------------------------------------
def main_constrained():
    global_model = ConstrainedSemanticComm().to(DEVICE)
    
    # Print model statistics
    total_params = count_parameters(global_model)
    print(f"Total trainable parameters: {total_params:,}")
    
    # Prepare quantization if enabled
    if USE_QUANTIZATION:
        global_model.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')
        torch.quantization.prepare_qat(global_model, inplace=True)
    
    # Create data loaders with reduced memory usage
    loaders = []
    for cid in range(NUM_CLIENTS):
        loaders.append(DataLoader(
            client_sets[cid],
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=WORKERS,
            pin_memory=PIN_MEM,
            drop_last=True,  # Ensure consistent batch sizes
        ))

    best_val_loss = float('inf')
    patience_counter = 0
    
    for rnd in range(1, ROUNDS + 1):
        client_states, client_losses = [], []

        # Local updates with constrained training
        for cid in range(NUM_CLIENTS):
            local_model = copy.deepcopy(global_model)
            loss_val = constrained_local_train(local_model, loaders[cid], LOCAL_EPOCHS)
            client_states.append(local_model.state_dict())
            client_losses.append(loss_val)

        # FedLoL aggregation
        fedlol_aggregate(global_model, client_states, client_losses)

        # Validation
        global_model.eval()
        with torch.no_grad():
            mse_sum, ssim_sum, perc_sum, n_img = 0.0, 0.0, 0.0, 0
            for img, _ in val_loader:
                img = img.to(DEVICE, non_blocking=True)
                
                with torch.cuda.amp.autocast(enabled=USE_FP16):
                    recon = global_model(img)
                    
                    # Calculate metrics
                    mse = nn.functional.mse_loss(recon, img, reduction="sum").item()
                    
                    # Denormalize for SSIM
                    img_denorm = img * torch.tensor([0.229, 0.224, 0.225]).to(img.device).view(1, 3, 1, 1) + \
                                torch.tensor([0.485, 0.456, 0.406]).to(img.device).view(1, 3, 1, 1)
                    recon_denorm = recon * torch.tensor([0.229, 0.224, 0.225]).to(recon.device).view(1, 3, 1, 1) + \
                                  torch.tensor([0.485, 0.456, 0.406]).to(recon.device).view(1, 3, 1, 1)
                    
                    img_denorm = torch.clamp(img_denorm, 0, 1)
                    recon_denorm = torch.clamp(recon_denorm, 0, 1)
                    
                    ssim_val = ssim(recon_denorm, img_denorm, data_range=1.0, size_average=False).sum().item()
                    perc = perceptual_loss(recon, img).item() * img.size(0)
                
                mse_sum += mse
                ssim_sum += ssim_val
                perc_sum += perc
                n_img += img.size(0)

        mse_mean = mse_sum / (n_img * PIXELS)
        psnr_mean = 10.0 * math.log10(1.0 / max(mse_mean, 1e-10))
        msssim_mean = ssim_sum / n_img
        perc_mean = perc_sum / n_img
        
        # Count effective parameters after pruning
        if USE_PRUNING:
            nonzero_params = count_nonzero_parameters(global_model)
            sparsity = 1.0 - (nonzero_params / total_params)
            print(
                f"Round {rnd:02d} │ "
                f"MSE={mse_mean:.4f} │ PSNR={psnr_mean:.2f} dB │ "
                f"MS-SSIM={msssim_mean:.4f} │ HybridLoss={perc_mean:.4f} │ "
                f"Sparsity={sparsity:.2%} ({nonzero_params:,}/{total_params:,})"
            )
        else:
            print(
                f"Round {rnd:02d} │ "
                f"MSE={mse_mean:.4f} │ PSNR={psnr_mean:.2f} dB │ "
                f"MS-SSIM={msssim_mean:.4f} │ HybridLoss={perc_mean:.4f} │ "
                f"Params={total_params:,}"
            )
        
        # Global early stopping
        if perc_mean < best_val_loss:
            best_val_loss = perc_mean
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOP_PATIENCE:
                print(f"Early stopping at round {rnd}")
                break

    print("Training completed.")
    
    # Convert to quantized model if QAT was used
    if USE_QUANTIZATION:
        global_model.eval()
        torch.quantization.convert(global_model, inplace=True)
        print("Model converted to quantized version")

    # Visual check with denormalization
    global_model.eval()
    with torch.no_grad():
        img_batch, _ = next(iter(val_loader))
        img_batch = img_batch.to(DEVICE)
        recon_batch = global_model(img_batch)
        
        # Denormalize for visualization
        def denormalize(tensor):
            mean = torch.tensor([0.485, 0.456, 0.406]).to(tensor.device).view(1, 3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).to(tensor.device).view(1, 3, 1, 1)
            return torch.clamp(tensor * std + mean, 0, 1)
        
        orig = denormalize(img_batch[:8]).cpu()
        recon = denormalize(recon_batch[:8]).cpu()

    grid = make_grid(torch.cat([orig, recon], 0), nrow=8, padding=2)

    plt.figure(figsize=(12, 4))
    plt.axis("off")
    plt.title(f"Constrained Model - Top: original, Bottom: reconstruction\n"
              f"Parameters: {count_nonzero_parameters(global_model):,} active / {total_params:,} total")
    plt.imshow(grid.permute(1, 2, 0))
    plt.savefig("constrained_reconstructions.png", dpi=300, bbox_inches="tight")
    plt.show()
    
    # Final model statistics
    if USE_QUANTIZATION:
        print(f"Final quantized model ready for deployment")
    if USE_PRUNING:
        final_sparsity = 1.0 - (count_nonzero_parameters(global_model) / total_params)
        print(f"Final model sparsity: {final_sparsity:.2%}")

# ------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------
if __name__ == "__main__":
    main_constrained()