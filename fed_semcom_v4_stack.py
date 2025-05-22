import copy
import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import transforms, datasets
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
from pytorch_msssim import ssim

# Reproducibility
SEED = 42
torch.manual_seed(SEED)
random.seed(SEED)
torch.cuda.manual_seed_all(SEED)

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--data_root", default="./tiny-imagenet-20")
parser.add_argument("--device", default="cuda", choices=["cuda", "cpu", "mps"])
parser.add_argument("--rounds", type=int, default=5)
parser.add_argument("--workers", type=int, default=0)
parser.add_argument("--batch_size", type=int, default=32)
args = parser.parse_args()

DATA_ROOT = args.data_root
DEVICE = args.device
ROUNDS = args.rounds
WORKERS = args.workers
BATCH_SIZE = args.batch_size
PIN_MEM = DEVICE == "cuda"

# Hyper-parameters
NUM_CLIENTS = 5
DIRICHLET_ALPHA = 1.0
LOCAL_EPOCHS = 4
LR = 1e-3
BOTTLENECK = 1024
COMPRESSED = 64
COMPRESS_RATIO = (64 * 64 * 3) / BOTTLENECK
SNR_DB = 10
ALPHA_LOSS = 0.9
PIXELS = 64 * 64 * 3

# Performance optimizations
torch.backends.cudnn.benchmark = True
USE_FP16 = torch.cuda.is_available()

# JIT-compiled channel simulation
@torch.jit.script
def apply_rayleigh_channel(x, sigma: float):
    h = torch.randn_like(x)
    noise = sigma * torch.randn_like(x)
    return (h * x + noise) / (h + 1e-6)

class FastChannel(nn.Module):
    def __init__(self):
        super().__init__()
        self.snr_cache = {}
    
    def forward(self, x, snr_db=10.0):
        if snr_db not in self.snr_cache:
            sigma = math.sqrt(1.0 / (2 * 10 ** (snr_db / 10)))
            self.snr_cache[snr_db] = sigma
        else:
            sigma = self.snr_cache[snr_db]
        return apply_rayleigh_channel(x, sigma)

class SemanticEncoder(nn.Module):
    def __init__(self, bottleneck: int = BOTTLENECK) -> None:
        super().__init__()
        self.enc1 = nn.Sequential(nn.Conv2d(3, 64, 3, 2, 1), nn.ReLU())
        self.enc2 = nn.Sequential(nn.Conv2d(64, 128, 3, 2, 1), nn.ReLU())
        self.enc3 = nn.Sequential(nn.Conv2d(128, 256, 3, 2, 1), nn.ReLU())
        self.fc = nn.Linear(256 * 8 * 8, bottleneck)
        self.bottleneck = bottleneck

    def forward(self, x, gamma=1.0):
        f1 = self.enc1(x)
        f2 = self.enc2(f1)
        f3 = self.enc3(f2)
        z = self.fc(f3.flatten(1))
        if gamma < 1.0:
            z_abs = torch.abs(z)
            threshold = torch.quantile(z_abs, 1.0 - gamma, dim=1, keepdim=True)
            mask = (z_abs >= threshold).float()
            z = z * mask
        return z, (f1, f2, f3)

class SemanticDecoder(nn.Module):
    def __init__(self, bottleneck: int = BOTTLENECK) -> None:
        super().__init__()
        self.fc = nn.Linear(bottleneck, 256 * 8 * 8)
        self.up1 = nn.Sequential(nn.ConvTranspose2d(512, 128, 4, 2, 1), nn.ReLU())
        self.up2 = nn.Sequential(nn.ConvTranspose2d(256, 64, 4, 2, 1), nn.ReLU())
        self.up3 = nn.Sequential(nn.ConvTranspose2d(128, 32, 4, 2, 1), nn.ReLU())
        self.out = nn.Conv2d(32, 3, 3, 1, 1)

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

    def forward(self, f, snr_db=SNR_DB, gamma=1.0):
        x = self.fc_layers[0](f)
        for i in range(1, 3):
            x = self.fc_layers[i](x)
        snr_info = torch.ones(x.size(0), 1, device=x.device) * snr_db
        snr_features = self.snr_encoder(snr_info)
        x = self.fc_layers[3](x + snr_features)
        for i in range(4, 7):
            x = self.fc_layers[i](x)
        if gamma < 1.0:
            x_abs = torch.abs(x)
            threshold = torch.quantile(x_abs, 1.0 - gamma, dim=1, keepdim=True)
            mask = (x_abs >= threshold).float()
            x = x * mask
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
        self.snr_cache = {}

    def forward(self, img, snr_db=SNR_DB, gamma=1.0):
        z, skips = self.enc_s(img, gamma=gamma)
        x = self.enc_c(z, snr_db=snr_db, gamma=gamma)
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

class Client:
    def __init__(self, cid, dataset, local_epochs=LOCAL_EPOCHS, batch_size=BATCH_SIZE):
        self.cid = cid
        self.dataset = dataset
        self.local_epochs = local_epochs
        self.batch_size = batch_size
        self.loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=WORKERS,
            pin_memory=PIN_MEM,
            prefetch_factor=2 if WORKERS > 0 else None,
        )
        self.model = OptimizedSemanticComm()
        self.prev_loss = float('inf')

    def choose_encoding_strategy(self, global_model, snr_db=SNR_DB, alpha=1.0, beta=0.1, delta=0.1):
        gamma_values = [0.5, 0.75, 1.0]
        best_gamma, best_utility = 1.0, float('-inf')
        
        for gamma in gamma_values:
            self.model.eval()
            local_loss = 0.0
            n_img = 0
            with torch.no_grad():
                for img, _ in self.loader:
                    img = img.to(DEVICE, non_blocking=True)
                    with torch.cuda.amp.autocast(enabled=USE_FP16):
                        recon = self.model(img, snr_db=snr_db, gamma=gamma)
                        local_loss += perceptual_loss(recon.float(), img.float()).item() * img.size(0)
                    n_img += img.size(0)
            local_loss /= n_img
            loss_reduction = self.prev_loss - local_loss if self.prev_loss < float('inf') else 0.0
            
            comm_cost = gamma * COMPRESSED * (1.0 / (10 ** (snr_db / 10)))
            comp_cost = self.local_epochs * self.batch_size
            utility = alpha * loss_reduction - beta * comm_cost - delta * comp_cost
            
            if utility > best_utility:
                best_utility = utility
                best_gamma = gamma
        
        return best_gamma

def local_train_optimized(model, loader, epochs: int, gamma=1.0):
    model.train()
    opt = optim.Adam(model.parameters(), lr=LR)
    scaler = torch.cuda.amp.GradScaler(enabled=USE_FP16)
    
    for _ in range(epochs):
        for img, _ in loader:
            img = img.to(DEVICE, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=USE_FP16):
                recon = model(img, gamma=gamma)
                loss = perceptual_loss(recon, img)
            if USE_FP16:
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                opt.step()
    return loss.item()

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
    mse_term = nn.functional.mse_loss(pred, target, reduction="mean")
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

TRANSFORM = transforms.Compose(
    [transforms.Resize((64, 64)), transforms.ToTensor()]
)

train_full = datasets.ImageFolder(f"{DATA_ROOT}/train", TRANSFORM)
val_full = datasets.ImageFolder(f"{DATA_ROOT}/val", TRANSFORM)
client_sets = dirichlet_split(train_full, DIRICHLET_ALPHA, NUM_CLIENTS)
val_loader = DataLoader(
    val_full,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=WORKERS,
    pin_memory=PIN_MEM,
)

def main_optimized_stackelberg():
    global_model = OptimizedSemanticComm().to(DEVICE)
    clients = [Client(cid, client_sets[cid]) for cid in range(NUM_CLIENTS)]
    R_total = 1000.0
    
    for rnd in range(1, ROUNDS + 1):
        client_states, client_losses, client_gammas = [], [], []
        snr_dbs = [SNR_DB + np.random.uniform(-2, 2) for _ in range(NUM_CLIENTS)]
        
        for client in clients:
            client.model = copy.deepcopy(global_model)
            gamma = client.choose_encoding_strategy(global_model, snr_db=snr_dbs[client.cid])
            loss_val = local_train_optimized(client.model, client.loader, client.local_epochs, gamma=gamma)
            client.prev_loss = loss_val
            client_states.append(client.model.state_dict())
            client_losses.append(loss_val)
            client_gammas.append(gamma)
        
        fedlol_aggregate(global_model, client_states, client_losses)
        
        val_losses = []
        for client in clients:
            client.model.eval()
            val_loss = 0.0
            n_img = 0
            with torch.no_grad():
                for img, _ in val_loader:
                    img = img.to(DEVICE, non_blocking=True)
                    with torch.cuda.amp.autocast(enabled=USE_FP16):
                        recon = client.model(img, snr_db=snr_dbs[client.cid], gamma=client_gammas[client.cid])
                        val_loss += perceptual_loss(recon.float(), img.float()).item() * img.size(0)
                    n_img += img.size(0)
            val_losses.append(val_loss / n_img)
        
        contribution_scores = [math.exp(-val_loss) for val_loss in val_losses]
        total_score = sum(contribution_scores) + 1e-8
        rewards = [score / total_score * R_total for score in contribution_scores]
        
        global_model.eval()
        with torch.no_grad():
            mse_sum, ssim_sum, perc_sum, n_img = 0.0, 0.0, 0.0, 0
            for img, _ in val_loader:
                img = img.to(DEVICE, non_blocking=True)
                with torch.cuda.amp.autocast(enabled=USE_FP16):
                    recon = global_model(img)
                    mse = nn.functional.mse_loss(recon.float(), img.float(), reduction="sum").item()
                    ssim_val = ssim(recon.float(), img.float(), data_range=1.0, size_average=False).sum().item()
                    perc = perceptual_loss(recon.float(), img.float()).item() * img.size(0)
                mse_sum += mse
                ssim_sum += ssim_val
                perc_sum += perc
                n_img += img.size(0)
        
        mse_mean = mse_sum / (n_img * PIXELS)
        psnr_mean = 10.0 * math.log10(1.0 / max(mse_mean, 1e-10))
        msssim_mean = ssim_sum / n_img
        perc_mean = perc_sum / n_img
        
        print(f"Round {rnd:02d} │ MSE={mse_mean:.4f} │ PSNR={psnr_mean:.2f} dB │ MS-SSIM={msssim_mean:.4f} │ HybridLoss={perc_mean:.4f}")
        print(f"Client Gammas: {client_gammas} │ Rewards: {rewards}")
    
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

if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn", force=True)
    main_optimized_stackelberg()