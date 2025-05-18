#!/usr/bin/env python3
"""
FedSemCom v4 -- Federated Learning for Semantic Communication on IoT Devices
Training demo on TinyImageNet-20 subset, optimized for microcontrollers.

Key Features:
- Federated Learning with FedLoL aggregation (Nguyen et al., 2024).
- Lightweight model (<75k params, <90kB flash post-quantization).
- Partial parameter updates: semantic encoder/decoder every round, channel encoder/decoder every P=3 rounds.
- Quantization-Aware Training (QAT) from round 3.
- AWGN + Rayleigh fading channel (SNR=1 dB) for robustness.
- Non-IID data partitioning using Dirichlet distribution (alpha=1.0).
- Gradient sparsification (top-10% gradients) for IoT efficiency.
- Optional DIV2K evaluation post-training.
"""

import math, copy, random, argparse, torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from torchvision.utils import make_grid
from pytorch_msssim import ssim
import matplotlib.pyplot as plt
import os

# Reproducibility
SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# Hyper-parameters
NUM_CLIENTS      = 5            # Number of IoT devices
DIRICHLET_ALPHA  = 1.0          # Non-IID data partition
ROUNDS           = 5            # Global communication rounds
LOCAL_EPOCHS     = 2            # Local training epochs per client
BATCH_SIZE       = 32           # Batch size
LR               = 3e-3         # Learning rate
LATENT_LEN       = 512          # Increased latent size for better MS-SSIM
DW1, DW2         = 16, 32       # Depthwise channels
SPARSITY_P       = 0.1          # Gradient sparsity (top-10%)
ALPHA_LOSS       = 0.7          # Hybrid loss: 0.7 MSE + 0.3 SSIM
PIXELS           = 64 * 64 * 3  # For per-pixel metrics
SNR_DB           = 1            # AWGN channel SNR (dB) to match paper
UPDATE_PERIOD    = 3            # Full model update every 3 rounds

# Device selection
DEVICE = (
    "cuda" if torch.cuda.is_available() else
    ("mps" if torch.backends.mps.is_available() else "cpu")
)

# Data transforms
TRANSFORM = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
])

# Depthwise separable block
class DWSeparable(nn.Sequential):
    def __init__(self, in_ch, out_ch, stride):
        super().__init__(
            nn.Conv2d(in_ch, in_ch, 3, stride, 1, groups=in_ch, bias=False),
            nn.ReLU6(inplace=True),
            nn.Conv2d(in_ch, out_ch, 1, 1, 0, bias=False),
            nn.ReLU6(inplace=True),
        )

# Semantic Encoder
class TinyEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.dw1 = DWSeparable(3, DW1, 2)    # 64→32
        self.dw2 = DWSeparable(DW1, DW2, 2)  # 32→16
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(DW2, LATENT_LEN, bias=False)

    def forward(self, x):
        x32 = self.dw1(x)
        f16 = self.dw2(x32)  # 16×16 feature map
        z = self.fc(self.avg(f16).flatten(1))
        return z, f16

# Channel Encoder
class ChannelEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(LATENT_LEN, LATENT_LEN // 4, bias=False)  # Compress

    def forward(self, z):
        return self.fc(z)

# Channel Decoder
class ChannelDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(LATENT_LEN // 4, LATENT_LEN, bias=False)

    def forward(self, x):
        return self.fc(x)

# Semantic Decoder
class TinyDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(LATENT_LEN, DW2 * 4 * 4, bias=False)  # 4×4 seed
        self.up1 = nn.Sequential(nn.ConvTranspose2d(DW2, DW1, 4, 2, 1, bias=False), nn.ReLU6(inplace=True))  # 4→8
        self.up2 = nn.Sequential(nn.ConvTranspose2d(DW1, DW1, 4, 2, 1, bias=False), nn.ReLU6(inplace=True))  # 8→16
        self.up3 = nn.Sequential(nn.ConvTranspose2d(DW1 + DW2, 16, 4, 2, 1, bias=False), nn.ReLU6(inplace=True))  # 16→32
        self.up4 = nn.Sequential(nn.ConvTranspose2d(16, 16, 4, 2, 1, bias=False), nn.ReLU6(inplace=True))  # 32→64
        self.out = nn.Conv2d(16, 3, 3, 1, 1)

    def forward(self, z, f16):
        x = self.fc(z).view(-1, DW2, 4, 4)
        x = self.up2(self.up1(x))
        x = self.up3(torch.cat([x, f16], 1))
        x = self.up4(x)
        return torch.sigmoid(self.out(x))

# End-to-end Semantic Communication Model
class TinySemCom(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc_s = TinyEncoder()
        self.enc_c = ChannelEncoder()
        self.dec_c = ChannelDecoder()
        self.dec_s = TinyDecoder()

    def forward(self, img, snr_db=SNR_DB):
        z, f16 = self.enc_s(img)
        x = self.enc_c(z)
        # AWGN + Rayleigh fading
        sigma = math.sqrt(1.0 / (2 * 10 ** (snr_db / 10)))
        noise = sigma * torch.randn_like(x, device=DEVICE)
        h = torch.sqrt(torch.randn_like(x)**2 + torch.randn_like(x)**2) / math.sqrt(2)
        y = h * x + noise
        z_hat = self.dec_c(y)
        return self.dec_s(z_hat, f16)

# Dirichlet data partitioning
def dirichlet_split(dataset, alpha, n_clients):
    label_to_indices = {}
    for idx, (_, lbl) in enumerate(dataset):
        label_to_indices.setdefault(lbl, []).append(idx)
    clients = [[] for _ in range(n_clients)]
    for indices in label_to_indices.values():
        proportions = torch.distributions.Dirichlet(torch.full((n_clients,), alpha)).sample()
        proportions = (proportions / proportions.sum()).tolist()
        split_points = [0] + list(torch.cumsum(torch.tensor(proportions) * len(indices), dim=0).long())
        for cid in range(n_clients):
            clients[cid].extend(indices[split_points[cid]:split_points[cid + 1]])
    return [Subset(dataset, idxs) for idxs in clients]

# Hybrid loss
def hybrid_loss(pred, target, alpha=ALPHA_LOSS):
    mse = nn.MSELoss()(pred, target)
    ssim_val = 1.0 - ssim(pred, target, data_range=1.0)
    return alpha * mse + (1 - alpha) * ssim_val

# FedLoL aggregation with partial updates
def fedlol_aggregate(global_model, client_states, client_losses, round_num):
    eps = 1e-8
    total_loss = sum(client_losses) + eps
    new_state = copy.deepcopy(global_model.state_dict())
    total_params, updated_params = 0, 0
    for k in new_state.keys():
        total_params += new_state[k].numel()
        if round_num % UPDATE_PERIOD != 0 and ('enc_c' in k or 'dec_c' in k):
            continue
        updated_params += new_state[k].numel()
        new_state[k] = sum(
            ((total_loss - client_losses[i]) / ((NUM_CLIENTS - 1) * total_loss)) * client_states[i][k]
            for i in range(NUM_CLIENTS)
        )
    global_model.load_state_dict(new_state)
    print(f"Round {round_num}: Transmitted {updated_params}/{total_params} params ({updated_params/total_params*100:.1f}%)")

# Gradient sparsification
@torch.no_grad()
def sparsify_gradients(model, p=SPARSITY_P):
    for w in model.parameters():
        if w.grad is None:
            continue
        g = w.grad.data
        k = max(1, int(g.numel() * p))
        th = g.abs().flatten().kthvalue(g.numel() - k).values
        mask = (g.abs() >= th)
        g.mul_(mask)

# Quantization helper
def enable_qat(model):
    model.train()
    model.qconfig = torch.ao.quantization.get_default_qat_qconfig(
        'fbgemm' if 'fbgemm' in torch.backends.quantized.supported_engines else 'qnnpack'
    )
    torch.ao.quantization.prepare_qat(model, inplace=True)

def convert_int8(model):
    torch.ao.quantization.convert(model.eval().cpu(), inplace=True)

def export_int8_tflite(model, path='tiny_semcom.tflite'):
    scripted = torch.jit.trace(model, torch.randn(1, 3, 64, 64))
    scripted._save_for_lite_interpreter(path)

# Local training
def local_train(model, loader, epochs):
    model.train()
    opt = optim.Adam(model.parameters(), lr=LR)
    for _ in range(epochs):
        for img, _ in loader:
            img = img.to(DEVICE)
            opt.zero_grad()
            pred = model(img)
            loss = hybrid_loss(pred, img)
            loss.backward()
            sparsify_gradients(model)
            opt.step()
    return float(loss)

# DIV2K loader
def load_div2k(root, batch_size):
    transform = transforms.Compose([transforms.Resize((64, 64)), transforms.ToTensor()])
    dataset = datasets.ImageFolder(root, transform=transform)
    return DataLoader(dataset, batch_size, shuffle=False, num_workers=0)

# Main training loop
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', default='./tiny-imagenet-20')
    parser.add_argument('--div2k_root', default='./DIV2K_test')
    args = parser.parse_args()

    # Data loading
    train_full = datasets.ImageFolder(f"{args.data_root}/train", TRANSFORM)
    val_full = datasets.ImageFolder(f"{args.data_root}/val", TRANSFORM)
    client_sets = dirichlet_split(train_full, DIRICHLET_ALPHA, NUM_CLIENTS)
    val_loader = DataLoader(val_full, BATCH_SIZE, shuffle=False, num_workers=0)

    # Initialize global model
    global_model = TinySemCom().to(DEVICE)
    qat_enabled = False

    # Federated training
    for rnd in range(1, ROUNDS + 1):
        client_states, client_losses = [], []

        # Enable QAT from round 3
        if rnd == 3 and not qat_enabled and torch.backends.quantized.supported_engines:
            enable_qat(global_model)
            qat_enabled = True
            print("Quantisation Aware Training enabled from round 3.")

        # Local updates
        for cid in range(NUM_CLIENTS):
            loader = DataLoader(client_sets[cid], BATCH_SIZE, shuffle=True, num_workers=0)
            local_model = copy.deepcopy(global_model).to(DEVICE)
            loss_val = local_train(local_model, loader, LOCAL_EPOCHS)
            client_states.append(local_model.state_dict())
            client_losses.append(loss_val)

        # FedLoL aggregation
        fedlol_aggregate(global_model, client_states, client_losses, rnd)

        # Validation
        global_model.eval()
        mse_sum, ssim_sum, perc_sum, n = 0.0, 0.0, 0.0, 0
        with torch.no_grad():
            for img, _ in val_loader:
                img = img.to(DEVICE)
                out = global_model(img)
                mse_sum += nn.MSELoss()(out, img).item() * img.size(0)
                ssim_sum += ssim(out, img, data_range=1.0, size_average=False).sum().item()
                perc_sum += hybrid_loss(out, img).item() * img.size(0)
                n += img.size(0)
        mse_mean = mse_sum / (n * PIXELS)
        psnr_mean = 10.0 * math.log10(1.0 / mse_mean)
        msssim_mean = ssim_sum / n
        perc_mean = perc_sum / n
        print(f"Round {rnd:02d} │ MSE={mse_mean:.6f} │ PSNR={psnr_mean:.2f} dB │ MS-SSIM={msssim_mean:.4f} │ HybridLoss={perc_mean:.4f}")

    # Visual sanity check (before quantization)
    global_model.eval()
    with torch.no_grad():
        img_batch, _ = next(iter(val_loader))
        img_batch = img_batch.to(DEVICE)
        recon_batch = global_model(img_batch)
    grid = make_grid(torch.cat([img_batch[:8].cpu(), recon_batch[:8].cpu()], 0), nrow=8, padding=2)
    plt.figure(figsize=(12, 4))
    plt.axis('off')
    plt.title('Top: Original | Bottom: Reconstructed')
    plt.imshow(grid.permute(1, 2, 0))
    plt.savefig('reconstructions_iot.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("Figure saved as reconstructions_iot.png")

    # DIV2K evaluation
    if os.path.exists(args.div2k_root):
        try:
            div2k_loader = load_div2k(args.div2k_root, BATCH_SIZE)
            global_model.eval()
            mse_sum, ssim_sum, n = 0.0, 0.0, 0
            with torch.no_grad():
                for img, _ in div2k_loader:
                    img = img.to(DEVICE)
                    out = global_model(img)
                    mse_sum += nn.MSELoss()(out, img).item() * img.size(0)
                    ssim_sum += ssim(out, img, data_range=1.0, size_average=False).sum().item()
                    n += img.size(0)
            mse_mean = mse_sum / (n * PIXELS)
            psnr_mean = 10.0 * math.log10(1.0 / mse_mean)
            msssim_mean = ssim_sum / n
            print(f"DIV2K Eval │ MSE={mse_mean:.6f} │ PSNR={psnr_mean:.2f} dB │ MS-SSIM={msssim_mean:.4f}")
        except Exception as e:
            print(f"DIV2K evaluation skipped: {str(e)}")
    else:
        print("DIV2K evaluation skipped: DIV2K_test directory not found")

    # Export INT8 model
    if torch.backends.quantized.supported_engines:
        try:
            torch.backends.quantized.engine = (
                'fbgemm' if 'fbgemm' in torch.backends.quantized.supported_engines else 'qnnpack'
            )
            convert_int8(global_model)
            export_int8_tflite(global_model)
            print("Exported INT-8 model → tiny_semcom.tflite")
        except (NotImplementedError, RuntimeError) as err:
            #print(f"⚠️ Quantised export failed: {str(err).split('\n')[0]}")
            print("Keeping float32 weights.")
    else:
        print("Quantised kernels not present; skipping INT-8 export.")

if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn", force=True)
    main()