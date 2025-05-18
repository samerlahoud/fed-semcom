"""
Toy implementation of the FedLol semantic-communication framework
on the TinyImageNet dataset.

Main simplifications vs. the paper:
• Semantic encoder / decoder → lightweight CNN auto-encoder
  (Swin-Transformer is replaced to keep the demo compact).
• Channel encoder / decoder → single linear layer pair.
• AWGN channel simulation only.
• FedLol aggregation exactly as in Equation (10) of the paper.
"""

import copy, math, random, torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms, datasets
import torchvision.utils as vutils
import matplotlib.pyplot as plt
from pytorch_msssim import ms_ssim

# ---------------------------------------------------------------------
# 1. Hyper-parameters and FL configuration
# ---------------------------------------------------------------------
NUM_CLIENTS        = 5          # K in the paper
DIRICHLET_ALPHA    = 1.0        # α controls non-IID level
ROUNDS             = 5         # global communication rounds
LOCAL_EPOCHS       = 4          # R in the paper
BATCH_SIZE         = 32
LR                 = 1e-3
COMPRESS_RATIO     = 16         # 1/16 as in the paper
DEVICE             = 'mps' if torch.cuda.is_available() else 'cpu'
PIXELS = 64 * 64 * 3
BOTTLENECK = 1024      # instead of 256
COMPRESSED = 64        # instead of 16

# ---------------------------------------------------------------------
# 2. Semantic + channel encoder / decoder
# ---------------------------------------------------------------------
class SemanticEncoder(nn.Module):
    def __init__(self, bottleneck=BOTTLENECK):
        super().__init__()
        self.feat1 = nn.Sequential(nn.Conv2d(3, 64, 3, 2, 1),  nn.ReLU())  # 64→32
        self.feat2 = nn.Sequential(nn.Conv2d(64, 128, 3, 2, 1), nn.ReLU()) # 32→16
        self.feat3 = nn.Sequential(nn.Conv2d(128, 256, 3, 2, 1), nn.ReLU())# 16→8
        self.flat  = nn.Flatten()
        self.fc    = nn.Linear(256*8*8, bottleneck)

    def forward(self, x):
        f1 = self.feat1(x)
        f2 = self.feat2(f1)
        f3 = self.feat3(f2)
        z  = self.fc(self.flat(f3))
        return z, (f1, f2, f3)                 # return intermediate maps

class SemanticDecoder(nn.Module):
    def __init__(self, bottleneck=BOTTLENECK):
        super().__init__()
        self.fc  = nn.Linear(bottleneck, 256*8*8)
        self.up1 = nn.Sequential(nn.ConvTranspose2d(512, 128, 4, 2, 1), nn.ReLU()) # 8→16
        self.up2 = nn.Sequential(nn.ConvTranspose2d(256, 64, 4, 2, 1),  nn.ReLU()) # 16→32
        self.up3 = nn.Sequential(nn.ConvTranspose2d(128, 32, 4, 2, 1),  nn.ReLU()) # 32→64
        self.out = nn.Conv2d(32, 3, 3, 1, 1)                                       # 64→64

    def forward(self, z, skips):
        f1, f2, f3 = skips                          # encoder feature maps
        x = self.fc(z).view(-1, 256, 8, 8)
        x = self.up1(torch.cat([x, f3], dim=1))
        x = self.up2(torch.cat([x, f2], dim=1))
        x = self.up3(torch.cat([x, f1], dim=1))
        return torch.sigmoid(self.out(x))

class ChannelEncoder(nn.Module):
    def __init__(self): super().__init__(); self.fc = nn.Linear(BOTTLENECK, COMPRESSED)
    def forward(self, f): return self.fc(f)

class ChannelDecoder(nn.Module):
    def __init__(self): super().__init__(); self.fc = nn.Linear(COMPRESSED, BOTTLENECK)
    def forward(self, x): return self.fc(x)

class SemanticComm(nn.Module):
    def __init__(self, bottleneck=256, compressed=16):
        super().__init__()
        self.enc_s = SemanticEncoder()
        self.enc_c = ChannelEncoder()
        self.dec_c = ChannelDecoder()
        self.dec_s = SemanticDecoder()
    def forward(self, img, snr_db=10):
        z, skips = self.enc_s(img)           # get latent and skip features
        x = self.enc_c(z)
        sigma = math.sqrt(1/(2*10**(snr_db/10)))
        y = x + sigma*torch.randn_like(x)
        z_hat = self.dec_c(y)
        img_hat = self.dec_s(z_hat, skips)   # pass skips to decoder
        return img_hat


# ---------------------------------------------------------------------
# 3. TinyImageNet loading and Dirichlet partitioning
# ---------------------------------------------------------------------
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])
train_full = datasets.ImageFolder('./tiny-imagenet-20/train', transform=transform)
val_full   = datasets.ImageFolder('./tiny-imagenet-20/val',   transform=transform)

def dirichlet_split(dataset, alpha, n_clients):
    label_indices = {}
    for idx, (_, label) in enumerate(dataset):
        label_indices.setdefault(label, []).append(idx)

    client_indices = [[] for _ in range(n_clients)]
    for label, idxs in label_indices.items():
        proportions = torch.distributions.Dirichlet(torch.tensor([alpha] * n_clients)).sample()
        proportions = (proportions / proportions.sum()).tolist()
        split_points = [0] + list(torch.cumsum(torch.tensor(proportions) * len(idxs), 0).int())
        for cid in range(n_clients):
            client_indices[cid].extend(idxs[split_points[cid]: split_points[cid + 1]])
    return [Subset(dataset, ids) for ids in client_indices]

client_datasets = dirichlet_split(train_full, DIRICHLET_ALPHA, NUM_CLIENTS)
val_loader      = DataLoader(val_full, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

# ---------------------------------------------------------------------
# 4. FedLol aggregation (Equation (10))
# ---------------------------------------------------------------------
def fedlol_aggregate(global_model, client_states, client_losses):
    K = len(client_states)
    total_loss = sum(client_losses)
    new_state = copy.deepcopy(global_model.state_dict())
    for key in new_state.keys():
        new_state[key] = sum(
            ((total_loss - client_losses[k]) / ((K - 1) * total_loss)) * client_states[k][key]
            for k in range(K)
        )
    global_model.load_state_dict(new_state)

def reconstruction_loss(pred, target, alpha=0.9):
    mse      = nn.functional.mse_loss(pred, target)
    msssim   = 1 - ms_ssim(pred, target, data_range=1.0)
    return alpha*mse + (1-alpha)*msssim

# ---------------------------------------------------------------------
# 5. Local training loop
# ---------------------------------------------------------------------
def local_train(model, loader, epochs):
    model.train()
    opt = optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.MSELoss()
    for _ in range(epochs):
        for img, _ in loader:
            img = img.to(DEVICE)
            opt.zero_grad()
            img_hat = model(img)
            loss = loss_fn(img_hat, img)
            loss.backward()
            opt.step()
    return loss.item()

# ------------------------------------------------------------------
# 3. Main federated-learning loop
# ------------------------------------------------------------------
def main():
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor()
    ])
    root = './tiny-imagenet-20'          # ← adjust
    train_full = datasets.ImageFolder(f'{root}/train', transform=transform)
    val_full   = datasets.ImageFolder(f'{root}/val',   transform=transform)

    client_datasets = dirichlet_split(train_full, DIRICHLET_ALPHA, NUM_CLIENTS)
    val_loader      = DataLoader(val_full, batch_size=BATCH_SIZE,
                                 shuffle=False, num_workers=0, pin_memory=True)

    global_model = SemanticComm().to(DEVICE)

    for rnd in range(1, ROUNDS + 1):
        client_states, client_losses = [], []
        for cid in range(NUM_CLIENTS):
            loader = DataLoader(client_datasets[cid], batch_size=BATCH_SIZE,
                                shuffle=True, num_workers=0, pin_memory=True)
            client_model = copy.deepcopy(global_model).to(DEVICE)
            loss = local_train(client_model, loader, LOCAL_EPOCHS)
            client_states.append(client_model.state_dict())
            client_losses.append(loss)

        fedlol_aggregate(global_model, client_states, client_losses)

        # validation
        global_model.eval()
        mse_sum, perc_sum, msssim_sum, img_count = 0.0, 0.0, 0.0, 0

        with torch.no_grad():
            for img, _ in val_loader:
                img = img.to(DEVICE, non_blocking=True)
                recon = global_model(img)

                # 1. MSE for PSNR
                mse_sum += nn.functional.mse_loss(recon, img, reduction='sum').item()

                # 2. MS-SSIM for perceptual similarity
                batch_msssim = ms_ssim(recon, img, data_range=1.0, size_average=False)
                msssim_sum += batch_msssim.sum().item()

                # 3. Hybrid perceptual loss
                perc_sum += reconstruction_loss(recon, img, alpha=0.9).item() * img.size(0)

                img_count += img.size(0)

        # Mean metrics
        mse_mean   = mse_sum  / (img_count * PIXELS)
        psnr_mean  = 10 * math.log10(1.0 / mse_mean)
        msssim_mean = msssim_sum / img_count
        perc_mean  = perc_sum  / img_count

        print(f"Round {rnd:02d} Val metrics ─ MSE={mse_mean:.4f}  PSNR={psnr_mean:.2f} dB  "
            f"MS-SSIM={msssim_mean:.4f}  HybridLoss={perc_mean:.4f}")

    print('Training completed.')
    global_model.eval()
    with torch.no_grad():
        img_batch, _ = next(iter(val_loader))      # one validation mini-batch
        img_batch = img_batch.to(DEVICE)
        recon_batch = global_model(img_batch)

    # prepare first eight originals and reconstructions
    orig  = img_batch[:8].cpu()
    recon = recon_batch[:8].cpu()

    grid = torch.cat([orig, recon], dim=0)        # 16 images total
    grid = vutils.make_grid(grid, nrow=8, padding=2)

    plt.figure(figsize=(12, 4))
    plt.axis("off")
    plt.title("Top: original – Bottom: reconstruction")
    plt.imshow(grid.permute(1, 2, 0))
    plt.show()
    plt.savefig('reconstructions.png', dpi=300)

# ------------------------------------------------------------------
# 4. Standard multiprocessing entry-point
# ------------------------------------------------------------------
if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn', force=True)  # optional
    main()