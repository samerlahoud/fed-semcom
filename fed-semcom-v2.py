"""
Toy implementation of the FedLoL semantic-communication framework
on the TinyImageNet-20 subset.

Key simplifications compared with the original paper
----------------------------------------------------
1. Semantic encoder / decoder  → lightweight CNN with U-Net skip connections
   (replaces the Swin Transformer for compactness).
2. Channel encoder / decoder   → single linear layer pair.
3. Channel model               → AWGN only (no fading or burst errors).
4. FedLoL aggregation          → Equation (10) with an epsilon guard.

Assumptions
-----------
* Images are rescaled to 64 × 64 and mapped to [0, 1] by `ToTensor()`.
* Clients are sampled once per round; all participate (K = NUM_CLIENTS).
* The hybrid perceptual loss  L = α·MSE + (1−α)·(1 − MS-SSIM)
  is used both for local optimisation and for global aggregation.
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

# Device selection with proper MPS fallback
#DEVICE = (
#    "cuda"
#    if torch.cuda.is_available()
#    else ("mps" if torch.backends.mps.is_available() else "cpu")
#)

# Worker settings (spawn issues on macOS); set > 0 on Linux / Colab
#WORKERS = 0
#PIN_MEM = DEVICE == "cuda"

# ------------------------------------------------------------------
# 2. Model components
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
        self.fc = nn.Linear(BOTTLENECK, COMPRESSED)

    def forward(self, f):
        return self.fc(f)


class ChannelDecoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(COMPRESSED, BOTTLENECK)

    def forward(self, x):
        return self.fc(x)


class SemanticComm(nn.Module):
    """End-to-end semantic communication model"""

    def __init__(self) -> None:
        super().__init__()
        self.enc_s = SemanticEncoder()
        self.enc_c = ChannelEncoder()
        self.dec_c = ChannelDecoder()
        self.dec_s = SemanticDecoder()

    def forward(self, img, snr_db: float = SNR_DB):
        z, skips = self.enc_s(img)
        x = self.enc_c(z)
        sigma = math.sqrt(1.0 / (2 * 10 ** (snr_db / 10)))
        y = x + sigma * torch.randn_like(x)
        z_hat = self.dec_c(y)
        return self.dec_s(z_hat, skips)


# ------------------------------------------------------------------
# 3. Dataset loading and Dirichlet partitioning
# ------------------------------------------------------------------
TRANSFORM = transforms.Compose(
    [transforms.Resize((64, 64)), transforms.ToTensor()]
)

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
# 4. Loss, aggregation, and local training
# ------------------------------------------------------------------
def perceptual_loss(pred, target, alpha: float = ALPHA_LOSS):
    mse_term = nn.functional.mse_loss(pred, target, reduction="mean")# in perceptual_loss
    ssim_val = 1.0 - ssim(pred, target, data_range=1.0)
    return alpha * mse_term + (1.0 - alpha) * ssim_val


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


def local_train(model, loader, epochs: int):
    model.train()
    opt = optim.Adam(model.parameters(), lr=LR)
    for _ in range(epochs):
        for img, _ in loader:
            img = img.to(DEVICE)
            opt.zero_grad()
            recon = model(img)
            loss = perceptual_loss(recon, img)
            loss.backward()
            opt.step()
    return loss.item()


# ------------------------------------------------------------------
# 5. Federated learning main loop
# ------------------------------------------------------------------
def main():
    global_model = SemanticComm().to(DEVICE)

    for rnd in range(1, ROUNDS + 1):
        client_states, client_losses = [], []

        # local updates
        for cid in range(NUM_CLIENTS):
            loader = DataLoader(
                client_sets[cid],
                batch_size=BATCH_SIZE,
                shuffle=True,
                num_workers=WORKERS,
                pin_memory=PIN_MEM,
            )
            local_model = copy.deepcopy(global_model)
            local_model.to(DEVICE)
            loss_val = local_train(local_model, loader, LOCAL_EPOCHS)
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
                recon = global_model(img)

                mse_sum += nn.functional.mse_loss(
                    recon, img, reduction="sum"
                ).item()
                ssim_sum += ssim(recon, img, data_range=1.0,
                 size_average=False).sum().item()
                perc_sum += perceptual_loss(recon, img).item() * img.size(0)
                n_img += img.size(0)

        mse_mean   = mse_sum / (n_img * PIXELS)
        psnr_mean  = 10.0 * math.log10(1.0 / mse_mean)
        msssim_mean = ssim_sum / n_img
        perc_mean  = perc_sum / n_img

        print(
            f"Round {rnd:02d} │ "
            f"MSE={mse_mean:.4f} │ PSNR={psnr_mean:.2f} dB │ "
            f"MS-SSIM={msssim_mean:.4f} │ HybridLoss={perc_mean:.4f}"
        )

    print("Training completed.")

    # ----------------------------------------------------------------
    # Visual sanity check
    # ----------------------------------------------------------------
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
# 6. Entry point
# ------------------------------------------------------------------
if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn", force=True)
    main()
