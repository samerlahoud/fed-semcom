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
# 1. Hyper-parameters and FL configuration (with reduced complexity)
# ------------------------------------------------------------------
NUM_CLIENTS      = 5            # K in the paper
DIRICHLET_ALPHA  = 1.0          # α controls non-IID level
#ROUNDS           = 5            # global communication rounds (from args)
LOCAL_EPOCHS     = 4            # each client’s local passes
#BATCH_SIZE       = 32           # (from args)
LR               = 1e-3
# Reduced complexity for IoT
BOTTLENECK       = 256          # semantic latent size (reduced from 1024)
COMPRESSED       = 32           # channel code length (reduced from 64)
COMPRESS_RATIO   = (64 * 64 * 3) / BOTTLENECK  # informational ratio (will be higher)
SNR_DB           = 10           # channel SNR during training
ALPHA_LOSS       = 0.9          # weight for the MSE term in hybrid loss
PIXELS           = 64 * 64 * 3  # constant for per-pixel metrics

# ------------------------------------------------------------------
# Performance optimizations (kept as is)
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
# Model components (modified for reduced complexity)
# ------------------------------------------------------------------
class SemanticEncoder(nn.Module):
    def __init__(self, bottleneck: int = BOTTLENECK) -> None:
        super().__init__()
        # Reduced channel sizes
        self.enc1 = nn.Sequential(nn.Conv2d(3, 32, 3, 2, 1), nn.ReLU())      # 64→32 (was 64)
        self.enc2 = nn.Sequential(nn.Conv2d(32, 64, 3, 2, 1), nn.ReLU())    # 32→16 (was 128)
        self.enc3 = nn.Sequential(nn.Conv2d(64, 128, 3, 2, 1), nn.ReLU())   # 16→8  (was 256)
        self.fc   = nn.Linear(128 * 8 * 8, bottleneck) # Input features reduced

    def forward(self, x):
        f1 = self.enc1(x)
        f2 = self.enc2(f1)
        f3 = self.enc3(f2)
        z  = self.fc(f3.flatten(1))
        return z, (f1, f2, f3)


class SemanticDecoder(nn.Module):
    def __init__(self, bottleneck: int = BOTTLENECK) -> None:
        super().__init__()
        self.fc   = nn.Linear(bottleneck, 128 * 8 * 8) # Output features reduced
        # Adjusted channel sizes for skip connections and upsampling
        self.up1  = nn.Sequential(nn.ConvTranspose2d(128 + 128, 64, 4, 2, 1), nn.ReLU()) # 8→16 (Input from f3 (128) and fc (128))
        self.up2  = nn.Sequential(nn.ConvTranspose2d(64 + 64, 32,  4, 2, 1), nn.ReLU())  # 16→32 (Input from f2 (64) and up1 (64))
        self.up3  = nn.Sequential(nn.ConvTranspose2d(32 + 32, 16,  4, 2, 1), nn.ReLU())  # 32→64 (Input from f1 (32) and up2 (32))
        self.out  = nn.Conv2d(16, 3, 3, 1, 1) # Input channels reduced

    def forward(self, z, skips):
        f1, f2, f3 = skips # f1: [B,32,32,32], f2: [B,64,16,16], f3: [B,128,8,8]
        x = self.fc(z).view(-1, 128, 8, 8)
        x = self.up1(torch.cat([x, f3], dim=1))
        x = self.up2(torch.cat([x, f2], dim=1))
        x = self.up3(torch.cat([x, f1], dim=1))
        return torch.sigmoid(self.out(x))


class ChannelEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc_layers = nn.ModuleList([
            nn.Linear(BOTTLENECK, COMPRESSED),
            nn.Linear(COMPRESSED, COMPRESSED),
            nn.Linear(COMPRESSED, COMPRESSED),
            nn.Linear(COMPRESSED, COMPRESSED), # Middle layer
            nn.Linear(COMPRESSED, COMPRESSED),
            nn.Linear(COMPRESSED, COMPRESSED),
            nn.Linear(COMPRESSED, COMPRESSED)
        ])
        self.snr_encoder = nn.Sequential(
            nn.Linear(1, 32), # Hidden layer size for SNR encoding
            nn.ReLU(),
            nn.Linear(32, COMPRESSED)
        )
        
    def forward(self, f, snr_db=SNR_DB):
        x = self.fc_layers[0](f)
        for i in range(1, 3):
            x = self.fc_layers[i](x)
        
        snr_info = torch.ones(x.size(0), 1, device=x.device) * snr_db
        snr_features = self.snr_encoder(snr_info)
        
        x = self.fc_layers[3](x + snr_features) # Middle layer with SNR injection
        
        for i in range(4, 7):
            x = self.fc_layers[i](x)
        return x


class ChannelDecoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc_layers = nn.ModuleList([
            nn.Linear(COMPRESSED, COMPRESSED),
            nn.Linear(COMPRESSED, COMPRESSED),
            nn.Linear(COMPRESSED, COMPRESSED),
            nn.Linear(COMPRESSED, COMPRESSED), # Middle layer
            nn.Linear(COMPRESSED, COMPRESSED),
            nn.Linear(COMPRESSED, COMPRESSED),
            nn.Linear(COMPRESSED, BOTTLENECK) # Output to BOTTLENECK size
        ])
        self.snr_encoder = nn.Sequential(
            nn.Linear(1, 32), # Hidden layer size for SNR encoding
            nn.ReLU(),
            nn.Linear(32, COMPRESSED)
        )
        
    def forward(self, x, snr_db=SNR_DB):
        # Store input for skip connection (original paper does not explicitly show skip here, but common practice)
        # The provided code for ChannelEncoder/Decoder implies a feed-forward structure with SNR injection.
        # For simplicity and adherence to the original structure provided in the question's code,
        # we'll maintain the 7-layer FC structure with SNR injection.
        
        # Process through the first half of FC layers
        for i in range(3):
            x = self.fc_layers[i](x)
        
        # SNR injection at the middle
        snr_info = torch.ones(x.size(0), 1, device=x.device) * snr_db
        snr_features = self.snr_encoder(snr_info)
        
        # Middle layer with SNR injection
        x = self.fc_layers[3](x + snr_features)
        
        # Process through the second half of FC layers
        for i in range(4, 7): # up to the final layer that outputs BOTTLENECK size
            x = self.fc_layers[i](x)
        
        return x

class OptimizedSemanticComm(nn.Module):
    """End-to-end semantic communication model with paper-aligned channel components"""

    def __init__(self) -> None:
        super().__init__()
        self.enc_s = SemanticEncoder()
        self.enc_c = ChannelEncoder()
        self.dec_c = ChannelDecoder()
        self.dec_s = SemanticDecoder()
        
        self.snr_cache = {}

    def forward(self, img, snr_db=SNR_DB):
        z, skips = self.enc_s(img)
        x = self.enc_c(z, snr_db)

        if snr_db not in self.snr_cache:
            self.snr_cache[snr_db] = math.sqrt(1.0 / (2 * 10 ** (snr_db / 10)))
        sigma = self.snr_cache[snr_db]
        
        with torch.no_grad():
            h = torch.randn_like(x)
            noise = sigma * torch.randn_like(x)
            y = h * x + noise
            x_hat = y / (h + 1e-6)
        
        x_hat = x_hat.detach().requires_grad_()
        z_hat = self.dec_c(x_hat, snr_db)
        return self.dec_s(z_hat, skips)


# ------------------------------------------------------------------
# Optimized training function (unchanged logic, uses new model)
# ------------------------------------------------------------------
def local_train_optimized(model, loader, epochs: int):
    model.train()
    opt = optim.Adam(model.parameters(), lr=LR)
    scaler = torch.cuda.amp.GradScaler(enabled=USE_FP16)
    
    cumulative_loss = 0.0
    num_batches = 0
    for _ in range(epochs):
        for img, _ in loader:
            img = img.to(DEVICE, non_blocking=True)
            opt.zero_grad(set_to_none=True)
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
            cumulative_loss += loss.item()
            num_batches += 1
    
    return cumulative_loss / num_batches if num_batches > 0 else 0.0

# ------------------------------------------------------------------
# 3. Dataset loading and Dirichlet partitioning (unchanged)
# ------------------------------------------------------------------
TRANSFORM = transforms.Compose(
    [transforms.Resize((64, 64)), transforms.ToTensor()]
)

def fedlol_aggregate(global_model, client_states, client_losses):
    eps = 1e-8
    total_loss = sum(client_losses) + eps
    # Ensure client_losses are positive; if local_train returns average loss, it should be.
    # If any loss is 0 or negative, this weighting can be problematic.
    # Consider using 1/loss or a fixed weight if losses can be non-positive or unstable.
    # For now, assuming perceptual_loss is always > 0.
    
    # Normalize weights to prevent instability from very small losses
    weights = [(total_loss - loss) / ((NUM_CLIENTS - 1) * total_loss + eps) if (NUM_CLIENTS > 1) else 1.0 for loss in client_losses]
    
    # If all losses are identical and total_loss becomes very small, weights could become NaN or Inf due to (total_loss - loss) being 0
    # and denominator also close to 0. Let's add a check for uniform losses.
    if len(set(client_losses)) == 1 and NUM_CLIENTS > 1: # All losses are the same
        weights = [1.0 / NUM_CLIENTS] * NUM_CLIENTS


    new_state = copy.deepcopy(global_model.state_dict())
    for k in new_state.keys():
        if new_state[k].is_floating_point(): # Aggregate only floating point tensors
            new_state[k].zero_() # Zero out the global model's parameter
            for i in range(NUM_CLIENTS):
                new_state[k] += weights[i] * client_states[i][k]
        else: # For non-floating point (e.g. num_batches_tracked in BN), just copy from first client or keep global
            new_state[k] = client_states[0][k] 

    global_model.load_state_dict(new_state)


def perceptual_loss(pred, target, alpha: float = ALPHA_LOSS):
    mse_term = nn.functional.mse_loss(pred, target, reduction="mean")
    ssim_val = 1.0 - ssim(pred, target, data_range=1.0, nonnegative_ssim=True) # ensure ssim_val >= 0 for loss
    return alpha * mse_term + (1.0 - alpha) * ssim_val

def dirichlet_split(dataset, alpha: float, n_clients: int):
    label_to_indices = {}
    # Assuming dataset.targets or a similar attribute exists for labels
    # If using ImageFolder, dataset.samples gives (filepath, class_index)
    # Or iterate through dataset to get labels if .targets is not available
    
    try: # Try to access dataset.targets (common for MNIST, CIFAR)
        labels = [dataset.targets[i] for i in range(len(dataset))]
    except AttributeError: # Fallback for ImageFolder etc.
        labels = [sample[1] for sample in dataset.samples]


    for idx, lbl in enumerate(labels):
        label_to_indices.setdefault(lbl, []).append(idx)

    clients_indices = [[] for _ in range(n_clients)]
    for label_indices in label_to_indices.values(): # Iterate over indices for each class
        proportions = torch.distributions.Dirichlet(
            torch.full((n_clients,), alpha)
        ).sample()
        
        # Ensure proportions sum to 1 and handle potential floating point inaccuracies for splitting
        proportions = proportions / proportions.sum() 
        
        current_idx = 0
        for cid in range(n_clients):
            num_samples_for_client_class = int(round(proportions[cid].item() * len(label_indices)))
            if cid == n_clients -1: # Assign remaining to last client to ensure all samples are used
                 clients_indices[cid].extend(label_indices[current_idx:])
            else:
                 clients_indices[cid].extend(label_indices[current_idx : current_idx + num_samples_for_client_class])
            current_idx += num_samples_for_client_class
            
    return [Subset(dataset, idxs) for idxs in clients_indices]

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
# Optimized main training loop (unchanged logic)
# ------------------------------------------------------------------
def main_optimized():
    global_model = OptimizedSemanticComm().to(DEVICE)
    
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

        for cid in range(NUM_CLIENTS):
            local_model = copy.deepcopy(global_model).to(DEVICE) # Ensure model is on correct device
            loss_val = local_train_optimized(local_model, loaders[cid], LOCAL_EPOCHS)
            client_states.append(copy.deepcopy(local_model.cpu().state_dict())) # Move to CPU before storing
            client_losses.append(loss_val)
            print(f"Round {rnd} | Client {cid+1}/{NUM_CLIENTS} | Local Loss: {loss_val:.4f}")


        fedlol_aggregate(global_model, client_states, client_losses)

        global_model.eval()
        with torch.no_grad():
            mse_sum, ssim_sum, perc_sum, n_img = 0.0, 0.0, 0.0, 0
            for img, _ in val_loader:
                img = img.to(DEVICE, non_blocking=True)
                
                with torch.cuda.amp.autocast(enabled=USE_FP16):
                    recon = global_model(img)
                    
                    img_fp32 = img.float() 
                    recon_fp32 = recon.float()
                    
                    mse = nn.functional.mse_loss(recon_fp32, img_fp32, reduction="sum").item()
                    # Ensure ssim is calculated correctly when reduction is per image
                    # pytorch_msssim returns a tensor of ssim values per image if size_average=False
                    ssim_val_tensor = ssim(recon_fp32, img_fp32, data_range=1.0, size_average=False, nonnegative_ssim=True)
                    ssim_sum_batch = ssim_val_tensor.sum().item()
                    
                    # Perceptual loss for each image, then sum
                    batch_perc_loss = 0.0
                    for i in range(img_fp32.size(0)):
                        batch_perc_loss += perceptual_loss(recon_fp32[i].unsqueeze(0), img_fp32[i].unsqueeze(0)).item()

                mse_sum += mse
                ssim_sum += ssim_sum_batch
                perc_sum += batch_perc_loss # Already a sum for the batch
                n_img += img.size(0)

        mse_mean = mse_sum / (n_img * PIXELS)
        psnr_mean = 10.0 * math.log10(1.0 / max(mse_mean, 1e-12)) 
        msssim_mean = ssim_sum / n_img
        perc_mean = perc_sum / n_img

        print(
            f"Round {rnd:02d} │ Val MSE={mse_mean:.6f} │ PSNR={psnr_mean:.2f} dB │ "
            f"MS-SSIM={msssim_mean:.4f} │ HybridLoss={perc_mean:.4f}"
        )

    print("Training completed.")

    global_model.eval()
    with torch.no_grad():
        img_batch, _ = next(iter(val_loader))
        img_batch = img_batch.to(DEVICE)
        recon_batch = global_model(img_batch)

    orig = img_batch[:8].cpu()
    recon = recon_batch[:8].cpu()
    # Clamp reconstructions to [0,1] for visualization if necessary, sigmoid should handle this
    recon = torch.clamp(recon, 0, 1) 
    
    grid_img = torch.cat([orig, recon], 0)
    # Ensure grid is suitable for make_grid (e.g. BxCxHxW)
    if grid_img.ndim == 3: # If single image, add batch dimension
        grid_img = grid_img.unsqueeze(0)
        
    grid = make_grid(grid_img, nrow=8, padding=2)

    plt.figure(figsize=(12, 4))
    plt.axis("off")
    plt.title("Top: original – Bottom: reconstruction")
    plt.imshow(grid.permute(1, 2, 0))
    plt.savefig("reconstructions_iot_simplified.png", dpi=300, bbox_inches="tight")
    print("Saved reconstruction image to reconstructions_iot_simplified.png")
    # plt.show() # Comment out if running in a non-GUI environment


# ------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------
if __name__ == "__main__":
    # Attempt to set start method, useful for multiprocessing with CUDA
    try:
        if torch.cuda.is_available() or args.workers > 0 : # only set if relevant
            torch.multiprocessing.set_start_method("spawn", force=True)
            print("Set multiprocessing start method to spawn.")
    except RuntimeError as e:
        print(f"Could not set multiprocessing start method: {e}")

    main_optimized()