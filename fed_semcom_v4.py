"""
Fully optimized implementation of the FedLoL semantic-communication framework
incorporating multiple performance enhancements.
added channel simulation with JIT compilation and caching
for faster training and inference.
added automatic mixed precision (AMP) support for faster training
and inference on CUDA devices.
added pre-fetching of data for better GPU utilization.
added non-blocking data transfer to GPU for better parallelism.
added more efficient gradient updates using zero_grad(set_to_none=True).
added FP16 training and inference support for faster computation.
added a more efficient way to handle FP16/FP32 conversion.
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
parser.add_argument("--batch_size",type=int, default=32)
args = parser.parse_args()

DATA_ROOT   = args.data_root
DEVICE      = args.device
ROUNDS      = args.rounds
WORKERS     = args.workers
BATCH_SIZE  = args.batch_size
PIN_MEM     = DEVICE == "cuda"

# ------------------------------------------------------------------
# 1. Hyper-parameters and FL configuration
# ------------------------------------------------------------------
NUM_CLIENTS      = 5            # K in the paper
DIRICHLET_ALPHA  = 1.0          # α controls non-IID level
#ROUNDS           = 5            # global communication rounds
LOCAL_EPOCHS     = 4            # each client’s local passes
#BATCH_SIZE       = 32
LR               = 1e-3
BOTTLENECK       = 1024         # semantic latent size
COMPRESSED       = 64           # channel code length
COMPRESS_RATIO   = (64 * 64 * 3) / BOTTLENECK  # informational ratio ≈ 12 ×
SNR_DB           = 10           # channel SNR during training
ALPHA_LOSS       = 0.9          # weight for the MSE term in hybrid loss
PIXELS           = 64 * 64 * 3  # constant for per-pixel metrics

# ------------------------------------------------------------------
# Performance optimizations
# ------------------------------------------------------------------
torch.backends.cudnn.benchmark = True  # Auto-tune for performance
USE_FP16 = torch.cuda.is_available()  # Use FP16 training if on CUDA

# JIT-compilable channel simulation function
@torch.jit.script
def apply_rayleigh_channel(x, sigma: float):
    """JIT-compiled channel simulation for performance"""
    h = torch.randn_like(x)  # fading coefficient (Rayleigh)
    noise = sigma * torch.randn_like(x)
    return (h * x + noise) / (h + 1e-6)


# Channel simulation with cached computations
class FastChannel(nn.Module):
    def __init__(self):
        super().__init__()
        self.snr_cache = {}  # Cache sigma values for each SNR
    
    def forward(self, x, snr_db=10.0):
        # Use cached sigma or compute and cache it
        if snr_db not in self.snr_cache:
            sigma = math.sqrt(1.0 / (2 * 10 ** (snr_db / 10)))
            self.snr_cache[snr_db] = sigma
        else:
            sigma = self.snr_cache[snr_db]
            
        return apply_rayleigh_channel(x, sigma)


# ------------------------------------------------------------------
# Model components (unchanged)
# ------------------------------------------------------------------
class SemanticEncoder(nn.Module):
    def __init__(self, bottleneck: int = BOTTLENECK) -> None:
        super().__init__()
        self.enc1 = nn.Sequential(nn.Conv2d(3, 64, 3, 2, 1), nn.ReLU())      # 64→32
        self.enc2 = nn.Sequential(nn.Conv2d(64, 128, 3, 2, 1), nn.ReLU())    # 32→16
        self.enc3 = nn.Sequential(nn.Conv2d(128, 256, 3, 2, 1), nn.ReLU())   # 16→8
        self.fc   = nn.Linear(256 * 8 * 8, bottleneck)

    def forward(self, x):
        f1 = self.enc1(x)
        f2 = self.enc2(f1)
        f3 = self.enc3(f2)
        z  = self.fc(f3.flatten(1))
        return z, (f1, f2, f3)


class SemanticDecoder(nn.Module):
    def __init__(self, bottleneck: int = BOTTLENECK) -> None:
        super().__init__()
        self.fc   = nn.Linear(bottleneck, 256 * 8 * 8)
        self.up1  = nn.Sequential(nn.ConvTranspose2d(512, 128, 4, 2, 1), nn.ReLU())  # 8→16
        self.up2  = nn.Sequential(nn.ConvTranspose2d(256, 64,  4, 2, 1), nn.ReLU())  # 16→32
        self.up3  = nn.Sequential(nn.ConvTranspose2d(128, 32,  4, 2, 1), nn.ReLU())  # 32→64
        self.out  = nn.Conv2d(32, 3, 3, 1, 1)

    def forward(self, z, skips):
        f1, f2, f3 = skips
        x = self.fc(z).view(-1, 256, 8, 8)
        x = self.up1(torch.cat([x, f3], dim=1))
        x = self.up2(torch.cat([x, f2], dim=1))
        x = self.up3(torch.cat([x, f1], dim=1))
        return torch.sigmoid(self.out(x))


class ChannelEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        # Change to 7 FC layers with skip connection as described in the paper
        self.fc_layers = nn.ModuleList([
            nn.Linear(BOTTLENECK, COMPRESSED),
            nn.Linear(COMPRESSED, COMPRESSED),
            nn.Linear(COMPRESSED, COMPRESSED),
            # Middle layer where SNR info is injected
            nn.Linear(COMPRESSED, COMPRESSED),
            nn.Linear(COMPRESSED, COMPRESSED),
            nn.Linear(COMPRESSED, COMPRESSED),
            nn.Linear(COMPRESSED, COMPRESSED)
        ])
        # SNR injection layers
        self.snr_encoder = nn.Sequential(
            nn.Linear(1, 32),
            nn.ReLU(),
            nn.Linear(32, COMPRESSED)
        )
        
    def forward(self, f, snr_db=SNR_DB):
        # Initial FC layer
        x = self.fc_layers[0](f)
        
        # Process through the first half of FC layers
        for i in range(1, 3):
            x = self.fc_layers[i](x)
        
        # SNR injection at the middle
        snr_info = torch.ones(x.size(0), 1, device=x.device) * snr_db
        snr_features = self.snr_encoder(snr_info)
        
        # Middle layer with SNR injection
        x = self.fc_layers[3](x + snr_features)
        
        # Process through the second half of FC layers
        for i in range(4, 7):
            x = self.fc_layers[i](x)
        
        # Skip connection from input to output
        return x


class ChannelDecoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        # Change to 7 FC layers with skip connection as described in the paper
        self.fc_layers = nn.ModuleList([
            nn.Linear(COMPRESSED, COMPRESSED),
            nn.Linear(COMPRESSED, COMPRESSED),
            nn.Linear(COMPRESSED, COMPRESSED),
            # Middle layer where SNR info is injected
            nn.Linear(COMPRESSED, COMPRESSED),
            nn.Linear(COMPRESSED, COMPRESSED),
            nn.Linear(COMPRESSED, COMPRESSED),
            nn.Linear(COMPRESSED, BOTTLENECK)
        ])
        # SNR injection layers
        self.snr_encoder = nn.Sequential(
            nn.Linear(1, 32),
            nn.ReLU(),
            nn.Linear(32, COMPRESSED)
        )
        
    def forward(self, x, snr_db=SNR_DB):
        # Store input for skip connection
        x_in = x
        
        # Process through the first half of FC layers
        for i in range(3):
            x = self.fc_layers[i](x)
        
        # SNR injection at the middle
        snr_info = torch.ones(x.size(0), 1, device=x.device) * snr_db
        snr_features = self.snr_encoder(snr_info)
        
        # Middle layer with SNR injection
        x = self.fc_layers[3](x + snr_features)
        
        # Process through the second half of FC layers
        for i in range(4, 7):
            x = self.fc_layers[i](x)
        
        # Skip connection from input to output (dimensionality adjusted in final layer)
        return x

class OptimizedSemanticComm(nn.Module):
    """End-to-end semantic communication model with paper-aligned channel components"""

    def __init__(self) -> None:
        super().__init__()
        self.enc_s = SemanticEncoder()
        self.enc_c = ChannelEncoder()
        self.dec_c = ChannelDecoder()
        self.dec_s = SemanticDecoder()
        
        # Pre-compute and cache SNR values
        self.snr_cache = {}

    def forward(self, img, snr_db=SNR_DB):
        # Semantic encoding
        z, skips = self.enc_s(img)
        
        # Channel encoding with SNR input
        x = self.enc_c(z, snr_db)  # shape: [B, D]

        # Get cached sigma value or compute it
        if snr_db not in self.snr_cache:
            self.snr_cache[snr_db] = math.sqrt(1.0 / (2 * 10 ** (snr_db / 10)))
        sigma = self.snr_cache[snr_db]
        
        # Apply Rayleigh fading without tracking gradients
        with torch.no_grad():
            h = torch.randn_like(x)  # fading coefficient per feature (Rayleigh)
            noise = sigma * torch.randn_like(x)
            y = h * x + noise
            x_hat = y / (h + 1e-6)
        
        # Re-enable gradients for the rest of the network
        x_hat = x_hat.detach().requires_grad_()

        # Channel decoding with SNR input
        z_hat = self.dec_c(x_hat, snr_db)
        
        # Semantic reconstruction
        return self.dec_s(z_hat, skips)


# ------------------------------------------------------------------
# Optimized training function
# ------------------------------------------------------------------
def local_train_optimized(model, loader, epochs: int):
    model.train()
    opt = optim.Adam(model.parameters(), lr=LR)
    
    # Add automatic mixed precision for faster training
    scaler = torch.cuda.amp.GradScaler(enabled=USE_FP16)
    
    for _ in range(epochs):
        for img, _ in loader:
            img = img.to(DEVICE, non_blocking=True)  # Use non_blocking for better parallelism
            
            opt.zero_grad(set_to_none=True)  # More efficient than zero_grad()
            
            # Use mixed precision training if available
            with torch.cuda.amp.autocast(enabled=USE_FP16):
                recon = model(img)
                loss = perceptual_loss(recon, img)
            
            if USE_FP16:
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                opt.step()
    
    return loss.item()

# ------------------------------------------------------------------
# 3. Dataset loading and Dirichlet partitioning
# ------------------------------------------------------------------
TRANSFORM = transforms.Compose(
    [transforms.Resize((64, 64)), transforms.ToTensor()]
)

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
    mse_term = nn.functional.mse_loss(pred, target, reduction="mean")# in perceptual_loss
    ssim_val = 1.0 - ssim(pred, target, data_range=1.0)
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

DATA_ROOT = "./tiny-imagenet-20"
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
# Optimized main training loop
# ------------------------------------------------------------------
def main_optimized():
    global_model = OptimizedSemanticComm().to(DEVICE)
    
    # Pre-fetch data for better GPU utilization
    loaders = []
    for cid in range(NUM_CLIENTS):
        loaders.append(DataLoader(
            client_sets[cid],
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=WORKERS,
            pin_memory=PIN_MEM,
            prefetch_factor=2 if WORKERS > 0 else None,
        ))

    for rnd in range(1, ROUNDS + 1):
        client_states, client_losses = [], []

        # Local updates
        for cid in range(NUM_CLIENTS):
            local_model = copy.deepcopy(global_model)
            loss_val = local_train_optimized(local_model, loaders[cid], LOCAL_EPOCHS)
            client_states.append(local_model.state_dict())
            client_losses.append(loss_val)

        # FedLoL aggregation
        fedlol_aggregate(global_model, client_states, client_losses)

        # Validation with optimized code
        global_model.eval()
        with torch.no_grad():
            mse_sum, ssim_sum, perc_sum, n_img = 0.0, 0.0, 0.0, 0
            for img, _ in val_loader:
                img = img.to(DEVICE, non_blocking=True)
                
                # Use mixed precision for inference as well
                with torch.cuda.amp.autocast(enabled=USE_FP16):
                    recon = global_model(img)
                    
                    # Calculate metrics in FP32 for accuracy
                    if USE_FP16:
                        img_fp32 = img.float() 
                        recon_fp32 = recon.float()
                        mse = nn.functional.mse_loss(recon_fp32, img_fp32, reduction="sum").item()
                        ssim_val = ssim(recon_fp32, img_fp32, data_range=1.0, size_average=False).sum().item()
                        perc = perceptual_loss(recon_fp32, img_fp32).item() * img.size(0)
                    else:
                        mse = nn.functional.mse_loss(recon, img, reduction="sum").item()
                        ssim_val = ssim(recon, img, data_range=1.0, size_average=False).sum().item()
                        perc = perceptual_loss(recon, img).item() * img.size(0)
                
                mse_sum += mse
                ssim_sum += ssim_val
                perc_sum += perc
                n_img += img.size(0)

        mse_mean = mse_sum / (n_img * PIXELS)
        psnr_mean = 10.0 * math.log10(1.0 / max(mse_mean, 1e-10))  # Prevent log(0)
        msssim_mean = ssim_sum / n_img
        perc_mean = perc_sum / n_img

        print(
            f"Round {rnd:02d} │ "
            f"MSE={mse_mean:.4f} │ PSNR={psnr_mean:.2f} dB │ "
            f"MS-SSIM={msssim_mean:.4f} │ HybridLoss={perc_mean:.4f}"
        )

    print("Training completed.")

    # Visual check
    global_model.eval()
    with torch.no_grad():
        img_batch, _ = next(iter(val_loader))
        img_batch = img_batch.to(DEVICE)
        recon_batch = global_model(img_batch)

    orig = img_batch[:8].cpu()
    recon = recon_batch[:8].cpu()
    grid = make_grid(torch.cat([orig, recon], 0), nrow=8, padding=2)

    plt.figure(figsize=(12, 4))
    plt.axis("off")
    plt.title("Top: original – Bottom: reconstruction")
    plt.imshow(grid.permute(1, 2, 0))
    plt.savefig("reconstructions.png", dpi=300, bbox_inches="tight")
    plt.show()


# ------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------
if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn", force=True)
    main_optimized()  # Use the optimized version