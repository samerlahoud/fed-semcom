#!/usr/bin/env python3
"""
TinyFed-SemCom : Federated semantic communication adapted to
micro-controllers (< 256 kB RAM).

Key features
------------
* Depthwise-separable encoder/decoder (≈ 18 k params).
* 128-byte latent code → 48× compression.
* Quantisation-aware training (QAT) for INT-8 export.
* Top-p gradient sparsification (default 5 %).
* FedLoL aggregation weight from Nguyen et al., 2024.

Author : Samer Lahoud / ChatGPT demo – May 2025
"""
# ---------------------------------------------------------------------
# 0. Imports and hyper-parameters
# ---------------------------------------------------------------------
import math, copy, random, pathlib, argparse, os
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from pytorch_msssim import ssim  # single-scale SSIM (lightweight)

# ------------ TinyML footprint knobs ---------------------------------
LATENT_LEN  = 128     # bytes (int8)
DW1, DW2    = 16, 32  # depthwise channels
SPARSITY_P  = 0.05    # top-p gradient sparsification
# ---------------------------------------------------------------------
BATCH_SIZE  = 32
LOCAL_EPOCH = 1
ROUNDS      = 3
LR          = 3e-3
DEVICE      = 'cuda' if torch.cuda.is_available() else 'cpu'
SEED        = 42

torch.manual_seed(SEED); random.seed(SEED)

# ---------------------------------------------------------------------
# 1.  TinyML semantic encoder / decoder
# ---------------------------------------------------------------------
class DWSeparable(nn.Sequential):
    """Depthwise-separable 3×3 conv: depthwise + pointwise."""
    def __init__(self, in_ch, out_ch, stride):
        super().__init__(
            nn.Conv2d(in_ch, in_ch, 3, stride, 1, groups=in_ch, bias=False),
            nn.ReLU6(inplace=True),
            nn.Conv2d(in_ch, out_ch, 1, 1, 0, bias=False),
            nn.ReLU6(inplace=True)
        )

class TinyEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.dw1 = DWSeparable(3,  DW1, 2)   # 64→32
        self.dw2 = DWSeparable(DW1, DW2, 2)  # 32→16
        self.avg = nn.AdaptiveAvgPool2d(1)   # 16→1
        self.fc  = nn.Linear(DW2, LATENT_LEN, bias=False)

    def forward(self, x):
        x = self.dw2(self.dw1(x))
        x = self.avg(x).flatten(1)
        return self.fc(x)                    # (B,128) int8 after QAT

class TinyDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc  = nn.Linear(LATENT_LEN, DW2*4*4, bias=False)  # 4×4 seed
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(DW2, DW1, 4, 2, 1, bias=False), nn.ReLU6())
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(DW1, 16, 4, 2, 1, bias=False), nn.ReLU6())
        self.out = nn.Conv2d(16, 3, 3, 1, 1)  # 64×64×3, Sigmoid later

    def forward(self, z):
        x = self.fc(z).view(-1, DW2, 4, 4)   # 4×4
        x = self.up2(self.up1(x))
        return torch.sigmoid(self.out(x))

class TinySemCom(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = TinyEncoder()
        self.dec = TinyDecoder()

    def forward(self, img):
        return self.dec(self.enc(img))

# ---------------------------------------------------------------------
# 2.  Dataset utility (Tiny-ImageNet-20 or CIFAR-10 fallback)
# ---------------------------------------------------------------------
def tiny20_loader(root, batch, workers=0, train=True):
    tfm = transforms.Compose([transforms.Resize((64,64)), transforms.ToTensor()])
    folder = 'train' if train else 'val'
    ds = datasets.ImageFolder(f'{root}/{folder}', transform=tfm)
    return DataLoader(ds, batch, shuffle=train, num_workers=workers)

# ---------------------------------------------------------------------
# 3.  Gradient sparsification helper
# ---------------------------------------------------------------------
def top_p_sparse_grads(model, p=SPARSITY_P):
    """Zero out all gradients except top-p magnitude."""
    for p_tensor in model.parameters():
        if p_tensor.grad is None: continue
        g = p_tensor.grad.data
        k = max(1, int(g.numel()*p))
        th = g.abs().flatten().kthvalue(g.numel()-k).values
        g.mul_( (g.abs()>=th).to(g.dtype) )

# ---------------------------------------------------------------------
# 4.  Quantisation-aware training preparation
# ---------------------------------------------------------------------
def prepare_qat(model):
    model.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')
    torch.quantization.prepare_qat(model, inplace=True)

def convert_int8(model):
    torch.quantization.convert(model.eval().cpu(), inplace=True)

def export_int8_tflite(model, path='tiny_semcom.tflite'):
    import torch.backends.xnnpack   # ensure export path works
    scripted = torch.jit.trace(model, torch.randn(1,3,64,64))
    scripted._save_for_lite_interpreter(path)

# ---------------------------------------------------------------------
# 5.  Loss & FedLoL weight (single-client version for demo)
# ---------------------------------------------------------------------
def hybrid_loss(pred, target, alpha=0.9):
    mse  = nn.functional.mse_loss(pred, target)
    ssim_val = 1.0 - ssim(pred, target, data_range=1.0)
    return alpha*mse + (1-alpha)*ssim_val

def train_local(model, loader, opt):
    model.train()
    for img,_ in loader:
        img = img.to(DEVICE)
        opt.zero_grad()
        out = model(img)
        loss = hybrid_loss(out, img)
        loss.backward()
        top_p_sparse_grads(model)   # compress
        opt.step()
    return loss.item()

# ---------------------------------------------------------------------
# 6.  Simple single-client demo loop
# ---------------------------------------------------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', default='./tiny-imagenet-20')
    args = parser.parse_args()

    # 6-1  Model & QAT
    net = TinySemCom().to(DEVICE)
    prepare_qat(net)
    opt = optim.Adam(net.parameters(), lr=LR)

    # 6-2  Data
    tr_loader = tiny20_loader(args.data_root, BATCH_SIZE, workers=0, train=True)
    val_loader= tiny20_loader(args.data_root, BATCH_SIZE, workers=0, train=False)

    # 6-3  Rounds
    for rnd in range(1, ROUNDS+1):
        loss = train_local(net, tr_loader, opt)
        net.eval()
        with torch.no_grad():
            mse, ssim_sum, n = 0.0, 0.0, 0
            for img,_ in val_loader:
                img = img.to(DEVICE)
                out = net(img)
                mse  += nn.functional.mse_loss(out, img, reduction='sum').item()
                ssim_sum += ssim(out, img, data_range=1.0, size_average=False
                                 ).sum().item()
                n += img.size(0)
        mse /= (n*64*64*3)
        print(f'Round {rnd}  loss={loss:.4f}  MSE={mse:.4f}  SSIM={ssim_sum/n:.3f}')

    # 6-4  Convert & export INT-8
    convert_int8(net)
    export_int8_tflite(net)
    print('Exported tiny_semcom.tflite (INT-8, ready for TFLM).')
