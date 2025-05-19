#!/usr/bin/env python3
"""
no channel model enc/dec
single node no FL
TinyFed‑SemCom v3  — a micro‑controller friendly semantic‑communication
training demo with federated logic stripped for single‑node testing.
Key upgrades relative to v2
--------------------------
1. Latent length doubled to 256 bytes (INT‑8) for higher fidelity.
2. Depthwise encoder keeps two stages but returns a 16 × 16 feature map
   used as a skip connection in the decoder.
3. Decoder seed enlarged to 8 × 8 and now performs four stride‑2
   up‑samples, last one concatenating the skip. Output size 64 × 64 × 3.
4. Two local epochs per round and Quantisation Aware Training (QAT)
   starts only from round 3 to avoid early saturation.
5. Quantised export guarded by backend check; script falls back to
   float32 when FBGEMM / QNNPACK kernels are absent.
6. Figure with originals vs. reconstructions saved as
    reconstructions_tiny.png.
The full model remains < 75 k parameters, < 90 kB flash once quantised.
"""
# ---------------------------------------------------------------------
# Imports and deterministic seed
# ---------------------------------------------------------------------
import math, copy, random, argparse, warnings, os, pathlib
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from torchvision.utils import make_grid
from pytorch_msssim import ssim        # single‑scale, MCU‑friendly
import matplotlib.pyplot as plt

SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)

# ---------------------------------------------------------------------
# Hyper‑parameters and TinyML knobs
# ---------------------------------------------------------------------
LATENT_LEN   = 256          # bytes after quantisation
DW1, DW2     = 16, 32       # depthwise channels
SPARSITY_P   = 0.10         # keep top‑10 % gradient elements
BATCH_SIZE   = 32
LOCAL_EPOCHS = 2
ROUNDS       = 5
LR           = 3e-3
PIXELS       = 64 * 64 * 3

DEVICE = (
    "cuda" if torch.cuda.is_available() else
    ("mps" if torch.backends.mps.is_available() else "cpu")
)

# ---------------------------------------------------------------------
# Depthwise separable blocks
# ---------------------------------------------------------------------
class DWSeparable(nn.Sequential):
    def __init__(self, in_ch, out_ch, stride):
        super().__init__(
            nn.Conv2d(in_ch, in_ch, 3, stride, 1, groups=in_ch, bias=False),
            nn.ReLU6(inplace=True),
            nn.Conv2d(in_ch, out_ch, 1, 1, 0, bias=False),
            nn.ReLU6(inplace=True),
        )

# ---------------------------------------------------------------------
# Encoder with skip output
# ---------------------------------------------------------------------
class TinyEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.dw1 = DWSeparable(3, DW1, 2)    # 64→32
        self.dw2 = DWSeparable(DW1, DW2, 2)  # 32→16
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.fc  = nn.Linear(DW2, LATENT_LEN, bias=False)

    def forward(self, x):
        x32  = self.dw1(x)
        f16  = self.dw2(x32)                 # 16×16 feature map
        z    = self.fc(self.avg(f16).flatten(1))
        return z, f16

# ---------------------------------------------------------------------
# Decoder with skip connection and four up‑samples
# ---------------------------------------------------------------------
# ---------------- TinyDecoder ------------------------------------
class TinyDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc  = nn.Linear(LATENT_LEN, DW2 * 4 * 4, bias=False)  # 4×4 seed

        self.up1 = nn.Sequential(                                  # 4 → 8
            nn.ConvTranspose2d(DW2, DW1, 4, 2, 1, bias=False),
            nn.ReLU6(inplace=True)
        )
        self.up2 = nn.Sequential(                                  # 8 → 16
            nn.ConvTranspose2d(DW1, DW1, 4, 2, 1, bias=False),
            nn.ReLU6(inplace=True)
        )

        # ✱ correct input‐channel count: DW1(16) + DW2(32) = 48 ✱
        self.up3 = nn.Sequential(                                  # 16 → 32
            nn.ConvTranspose2d(DW1 + DW2, 16, 4, 2, 1, bias=False),
            nn.ReLU6(inplace=True)
        )

        self.up4 = nn.Sequential(                                  # 32 → 64
            nn.ConvTranspose2d(16, 16, 4, 2, 1, bias=False),
            nn.ReLU6(inplace=True)
        )

        self.out = nn.Conv2d(16, 3, 3, 1, 1)                       # 64×64

    def forward(self, z, f16):
        x = self.fc(z).view(-1, DW2, 4, 4)  # (B,32,4,4)
        x = self.up2(self.up1(x))           # (B,16,16,16)
        x = self.up3(torch.cat([x, f16], 1))# (B,16,32,32)
        x = self.up4(x)                     # (B,16,64,64)
        return torch.sigmoid(self.out(x))

# ---------------------------------------------------------------------
# End‑to‑end Tiny semantic model
# ---------------------------------------------------------------------
class TinySemCom(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = TinyEncoder()
        self.dec = TinyDecoder()

    def forward(self, img):
        z, f16 = self.enc(img)
        return self.dec(z, f16)

# ---------------------------------------------------------------------
# Dataset helper – Tiny‑ImageNet‑20 expected
# ---------------------------------------------------------------------
TRANSFORM = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
])

def tiny_loader(root, train):
    folder = "train" if train else "val"
    ds = datasets.ImageFolder(f"{root}/{folder}", transform=TRANSFORM)
    return DataLoader(ds, BATCH_SIZE, shuffle=train, num_workers=0)

# ---------------------------------------------------------------------
# Gradient sparsification
# ---------------------------------------------------------------------
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

# ---------------------------------------------------------------------
# Loss: 0.9 MSE + 0.1 SSIM
# ---------------------------------------------------------------------
MSE = nn.MSELoss()

def hybrid_loss(pred, target, alpha=0.9):
    return alpha * MSE(pred, target) + (1 - alpha) * (1 - ssim(pred, target, data_range=1.0))

# ---------------------------------------------------------------------
# Quantisation helper – guarded
# ---------------------------------------------------------------------
def enable_qat(model):
    model.train()                                   # ← ensure training mode
    model.qconfig = torch.ao.quantization.get_default_qat_qconfig(
        'fbgemm' if 'fbgemm' in torch.backends.quantized.supported_engines else 'qnnpack'
    )
    torch.ao.quantization.prepare_qat(model, inplace=True)

def convert_int8(model):
    torch.ao.quantization.convert(model.eval().cpu(), inplace=True)

def export_int8_tflite(model, path='tiny_semcom.tflite'):
    scripted = torch.jit.trace(model, torch.randn(1, 3, 64, 64))
    scripted._save_for_lite_interpreter(path)

# ---------------------------------------------------------------------
# Training utilities
# ---------------------------------------------------------------------

def train_one_epoch(model, loader, opt):
    model.train()
    for img, _ in loader:
        img = img.to(DEVICE)
        opt.zero_grad()
        pred = model(img)
        loss = hybrid_loss(pred, img)
        loss.backward()
        sparsify_gradients(model)
        opt.step()
    return float(loss)

# ---------------------------------------------------------------------
# Main entry
# ---------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', default='./tiny-imagenet-20')
    args = parser.parse_args()

    tr_loader = tiny_loader(args.data_root, True)
    val_loader = tiny_loader(args.data_root, False)

    net = TinySemCom().to(DEVICE)
    opt = optim.Adam(net.parameters(), lr=LR)

    qat_enabled = False

    for rnd in range(1, ROUNDS + 1):
        if rnd == 5 and not qat_enabled and torch.backends.quantized.supported_engines:
            enable_qat(net)
            qat_enabled = True
            print("Quantisation Aware Training enabled from round 3.")

        loss = train_one_epoch(net, tr_loader, opt)

        net.eval()
        mse_sum, ssim_sum, n = 0.0, 0.0, 0
        with torch.no_grad():
            for img, _ in val_loader:
                img = img.to(DEVICE)
                out = net(img)
                mse_sum += MSE(out, img).item() * img.size(0)
                ssim_sum += ssim(out, img, data_range=1.0, size_average=False).sum().item()
                n += img.size(0)
        print(f"Round {rnd}  loss={loss:.4f}  MSE={mse_sum/n:.4f}  SSIM={ssim_sum/n:.3f}")

    # ------------------------------------------------------------------
    # Export INT‑8 if kernels are available
    # ------------------------------------------------------------------
    # has_backend = bool(torch.backends.quantized.supported_engines)
    # if has_backend:
    #     try:
    #         torch.backends.quantized.engine = (
    #             'fbgemm' if 'fbgemm' in torch.backends.quantized.supported_engines else 'qnnpack')
    #         convert_int8(net)
    #         export_int8_tflite(net)
    #         print("Exported INT‑8 model → tiny_semcom.tflite")
    #     except (NotImplementedError, RuntimeError) as err:
    #         print("⚠️  Quantised export failed: " + str(err).split('\n')[0])
    #         print("   Keeping float32 weights.")
    # else:
    #     print("Quantised kernels not present; skipping INT‑8 export.")

    # ------------------------------------------------------------------
    # Visual sanity check
    # ------------------------------------------------------------------
    net.eval()
    with torch.no_grad():
        img_batch, _ = next(iter(val_loader))
        img_batch = img_batch.to(DEVICE)
        recon_batch = net(img_batch)

    grid = make_grid(torch.cat([img_batch[:8].cpu(), recon_batch[:8].cpu()], 0),
                     nrow=8, padding=2)

    plt.figure(figsize=(12, 4))
    plt.axis('off')
    plt.title('Top row = original  |  Bottom row = TinyML reconstruction')
    plt.imshow(grid.permute(1, 2, 0))
    plt.savefig('reconstructions_tiny.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("Figure saved as reconstructions_tiny.png")
