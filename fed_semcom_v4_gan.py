"""
Fully optimized implementation of the FedLoL semantic-communication framework
with adversarial channel modeling (GAN-style)
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

# ------------------------------------------------------------------
# 1. Hyper-parameters and FL configuration
# ------------------------------------------------------------------
NUM_CLIENTS      = 5            # K in the paper
DIRICHLET_ALPHA  = 1.0          # α controls non-IID level
ROUNDS           = 5            # global communication rounds
LOCAL_EPOCHS     = 4            # each client’s local passes
BATCH_SIZE       = 32
LR               = 1e-3
BOTTLENECK       = 1024         # semantic latent size
COMPRESSED       = 64           # channel code length
SNR_DB           = 10           # channel SNR during training
ALPHA_LOSS       = 0.9          # weight for the MSE term in hybrid loss
PIXELS           = 64 * 64 * 3  # constant for per-pixel metrics
USE_FP16         = True         # Use FP16 training if on CUDA

# Performance optimizations
torch.backends.cudnn.benchmark = True  # Auto-tune for performance

# Device setup
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PIN_MEM = DEVICE == "cuda"

# Argument parser stub (can be extended for CLI args)
class Args:
    def __init__(self):
        self.data_root = "./tiny-imagenet-20"
        self.device = DEVICE
        self.rounds = ROUNDS
        self.workers = 0
        self.batch_size = BATCH_SIZE

args = Args()
DATA_ROOT = args.data_root
DEVICE = args.device
ROUNDS = args.rounds
WORKERS = args.workers
BATCH_SIZE = args.batch_size


# ------------------------------------------------------------------
# Channel Simulation with Learnable Distortion
# ------------------------------------------------------------------

@torch.jit.script
def apply_rayleigh_channel(x, sigma: float):
    """Fixed noise injection (Rayleigh fading + AWGN)"""
    h = torch.randn_like(x).abs()  # |h| ~ Rayleigh
    noise = sigma * torch.randn_like(x)
    return h * x + noise

class AdversarialChannel(nn.Module):
    """Learnable channel that learns to distort the signal"""
    def __init__(self, init_sigma=0.1):
        super().__init__()
        self.sigma = nn.Parameter(torch.tensor(init_sigma))

    def forward(self, x):
        # Apply adversarial distortion
        h = torch.randn_like(x).abs()
        noise = self.sigma * torch.randn_like(x)
        distorted = h * x + noise
        return distorted.detach()  # Detach to avoid backprop through adversary


# ------------------------------------------------------------------
# Model components
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
        self.fc_layers = nn.ModuleList([
            nn.Linear(BOTTLENECK, COMPRESSED),
            nn.Linear(COMPRESSED, COMPRESSED),
            nn.Linear(COMPRESSED, COMPRESSED),
            nn.Linear(COMPRESSED, COMPRESSED),
            nn.Linear(COMPRESSED, COMPRESSED),
            nn.Linear(COMPRESSED, COMPRESSED),
            nn.Linear(COMPRESSED, COMPRESSED)
        ])
        self.snr_encoder = nn.Sequential(
            nn.Linear(1, 32),
            nn.ReLU(),
            nn.Linear(32, COMPRESSED)
        )

    def forward(self, f, snr_db=SNR_DB):
        x = self.fc_layers[0](f)
        for i in range(1, 3):
            x = self.fc_layers[i](x)
        snr_info = torch.ones(x.size(0), 1, device=x.device) * snr_db
        snr_features = self.snr_encoder(snr_info)
        x = self.fc_layers[3](x + snr_features)
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
            nn.Linear(COMPRESSED, COMPRESSED),
            nn.Linear(COMPRESSED, COMPRESSED),
            nn.Linear(COMPRESSED, COMPRESSED),
            nn.Linear(COMPRESSED, BOTTLENECK)
        ])
        self.snr_encoder = nn.Sequential(
            nn.Linear(1, 32),
            nn.ReLU(),
            nn.Linear(32, COMPRESSED)
        )

    def forward(self, x, snr_db=SNR_DB):
        x_in = x
        for i in range(3):
            x = self.fc_layers[i](x)
        snr_info = torch.ones(x.size(0), 1, device=x.device) * snr_db
        snr_features = self.snr_encoder(snr_info)
        x = self.fc_layers[3](x + snr_features)
        for i in range(4, 7):
            x = self.fc_layers[i](x)
        return x


class OptimizedSemanticComm(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.enc_s = SemanticEncoder()
        self.enc_c = ChannelEncoder()
        self.dec_c = ChannelDecoder()
        self.dec_s = SemanticDecoder()

    def forward(self, img, snr_db=SNR_DB):
        z, skips = self.enc_s(img)
        x = self.enc_c(z, snr_db)
        x_hat = adversarial_channel(x)
        z_hat = self.dec_c(x_hat, snr_db)
        return self.dec_s(z_hat, skips)


# Instantiate adversarial channel
adversarial_channel = AdversarialChannel().to(DEVICE)


# ------------------------------------------------------------------
# Loss Functions
# ------------------------------------------------------------------

def perceptual_loss(pred, target, alpha: float = ALPHA_LOSS):
    mse_term = nn.functional.mse_loss(pred, target, reduction="mean")
    ssim_val = 1.0 - ssim(pred, target, data_range=1.0)
    return alpha * mse_term + (1.0 - alpha) * ssim_val


# ------------------------------------------------------------------
# Optimized training function with adversarial updates
# ------------------------------------------------------------------

def local_train_optimized(model, loader, epochs: int):
    model.train()
    opt_gen = optim.Adam(model.parameters(), lr=LR)
    opt_adv = optim.Adam(adversarial_channel.parameters(), lr=LR / 10)
    scaler = torch.cuda.amp.GradScaler(enabled=USE_FP16)

    for _ in range(epochs):
        for img, _ in loader:
            img = img.to(DEVICE, non_blocking=True)

            # --- Train Generator ---
            opt_gen.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=USE_FP16):
                recon = model(img)
                loss_gen = perceptual_loss(recon, img)
            if USE_FP16:
                scaler.scale(loss_gen).backward()
                scaler.step(opt_gen)
            else:
                loss_gen.backward()
                opt_gen.step()
            if USE_FP16:
                scaler.update()

            # --- Train Adversary ---
            opt_adv.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=USE_FP16):
                recon_adv = model(img)
                loss_adv = -perceptual_loss(recon_adv, img)  # Maximize reconstruction loss
            if USE_FP16:
                scaler.scale(loss_adv).backward()
                scaler.step(opt_adv)
            else:
                loss_adv.backward()
                opt_adv.step()
            if USE_FP16:
                scaler.update()

    return loss_gen.item(), loss_adv.item()


# ------------------------------------------------------------------
# Dataset loading and Dirichlet partitioning
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
    global_model.load_dict(new_state)

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

train_full = datasets.ImageFolder(f"{DATA_ROOT}/train", TRANSFORM)
val_full = datasets.ImageFolder(f"{DATA_ROOT}/val", TRANSFORM)
client_sets = dirichlet_split(train_full, DIRICHLET_ALPHA, NUM_CLIENTS)
val_loader = DataLoader(val_full, batch_size=BATCH_SIZE, shuffle=False, num_workers=WORKERS, pin_memory=PIN_MEM)


# ------------------------------------------------------------------
# Optimized main training loop
# ------------------------------------------------------------------

def main_optimized():
    global_model = OptimizedSemanticComm().to(DEVICE)
    loaders = [
        DataLoader(
            client_sets[cid],
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=WORKERS,
            pin_memory=PIN_MEM,
            prefetch_factor=2 if WORKERS > 0 else None,
        ) for cid in range(NUM_CLIENTS)
    ]

    for rnd in range(1, ROUNDS + 1):
        client_states, client_losses = [], []
        for cid in range(NUM_CLIENTS):
            local_model = copy.deepcopy(global_model)
            loss_gen, loss_adv = local_train_optimized(local_model, loaders[cid], LOCAL_EPOCHS)
            client_states.append(local_model.state_dict())
            client_losses.append(loss_gen)
            print(f"Client {cid} | Gen Loss: {loss_gen:.4f} | Adv Loss: {loss_adv:.4f}")

        # Aggregate models using FedLoL
        fedlol_aggregate(global_model, client_states, client_losses)

        # Validation
        global_model.eval()
        with torch.no_grad():
            mse_sum = psnr_sum = ssim_sum = n_img = 0
            for img, _ in val_loader:
                img = img.to(DEVICE, non_blocking=True)
                with torch.cuda.amp.autocast(enabled=USE_FP16):
                    recon = global_model(img)
                    recon = recon.float()
                    img = img.float()
                    mse = nn.functional.mse_loss(recon, img, reduction="mean").item()
                    msssim = ssim(recon, img, data_range=1.0, size_average=True).item()
                    psnr = 10 * math.log10(1.0 / max(mse, 1e-10))

                mse_sum += mse
                psnr_sum += psnr
                ssim_sum += msssim
                n_img += 1

        print(f"Round {rnd} │ Avg MSE: {mse_sum/n_img:.4f} │ PSNR: {psnr_sum/n_img:.2f} dB │ SSIM: {ssim_sum/n_img:.4f}")

    # Final visualization
    global_model.eval()
    with torch.no_grad():
        img_batch, _ = next(iter(val_loader))
        img_batch = img_batch.to(DEVICE)
        recon_batch = global_model(img_batch)

    orig = img_batch[:8].cpu()
    recon = recon_batch[:8].cpu()
    grid = make_grid(torch.cat([orig, recon], 0), nrow=8, padding=2)
    plt.figure(figsize=(12, 4))
    plt.imshow(grid.permute(1, 2, 0))
    plt.axis("off")
    plt.title("Top: Original | Bottom: Reconstruction")
    plt.savefig("reconstructions_adversarial.png")
    plt.show()


if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn", force=True)
    main_optimized()