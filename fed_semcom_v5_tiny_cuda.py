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
NUM_CLIENTS      = 5
DIRICHLET_ALPHA  = 1.0
LOCAL_EPOCHS     = 4
LR               = 1e-3
BOTTLENECK       = 256
COMPRESSED       = 32
COMPRESS_RATIO   = (64 * 64 * 3) / BOTTLENECK
SNR_DB           = 10
ALPHA_LOSS       = 0.9
PIXELS           = 64 * 64 * 3
GRAD_CLIP_VALUE  = 1.0 # Added for gradient clipping

# ------------------------------------------------------------------
# Performance optimizations
# ------------------------------------------------------------------
torch.backends.cudnn.benchmark = True
# USE_FP16 should only be true if CUDA is available and selected
USE_FP16 = (DEVICE == "cuda" and torch.cuda.is_available())

# JIT-compilable channel simulation function
@torch.jit.script
def apply_rayleigh_channel(x, sigma: float):
    h = torch.randn_like(x)
    noise = sigma * torch.randn_like(x)
    return (h * x + noise) / (h + 1e-6) # Added epsilon to denominator

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

# ------------------------------------------------------------------
# Model components (modified for reduced complexity)
# ------------------------------------------------------------------
class SemanticEncoder(nn.Module):
    def __init__(self, bottleneck: int = BOTTLENECK) -> None:
        super().__init__()
        self.enc1 = nn.Sequential(nn.Conv2d(3, 32, 3, 2, 1), nn.ReLU())
        self.enc2 = nn.Sequential(nn.Conv2d(32, 64, 3, 2, 1), nn.ReLU())
        self.enc3 = nn.Sequential(nn.Conv2d(64, 128, 3, 2, 1), nn.ReLU())
        self.fc   = nn.Linear(128 * 8 * 8, bottleneck)

    def forward(self, x):
        f1 = self.enc1(x)
        f2 = self.enc2(f1)
        f3 = self.enc3(f2)
        z  = self.fc(f3.flatten(1))
        return z, (f1, f2, f3)

class SemanticDecoder(nn.Module):
    def __init__(self, bottleneck: int = BOTTLENECK) -> None:
        super().__init__()
        self.fc   = nn.Linear(bottleneck, 128 * 8 * 8)
        self.up1  = nn.Sequential(nn.ConvTranspose2d(128 + 128, 64, 4, 2, 1), nn.ReLU())
        self.up2  = nn.Sequential(nn.ConvTranspose2d(64 + 64, 32,  4, 2, 1), nn.ReLU())
        self.up3  = nn.Sequential(nn.ConvTranspose2d(32 + 32, 16,  4, 2, 1), nn.ReLU())
        self.out  = nn.Conv2d(16, 3, 3, 1, 1)

    def forward(self, z, skips):
        f1, f2, f3 = skips
        x = self.fc(z).view(-1, 128, 8, 8)
        x = self.up1(torch.cat([x, f3], dim=1))
        x = self.up2(torch.cat([x, f2], dim=1))
        x = self.up3(torch.cat([x, f1], dim=1))
        return torch.sigmoid(self.out(x))

class ChannelEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc_layers = nn.ModuleList([
            nn.Linear(BOTTLENECK, COMPRESSED), nn.Linear(COMPRESSED, COMPRESSED),
            nn.Linear(COMPRESSED, COMPRESSED), nn.Linear(COMPRESSED, COMPRESSED),
            nn.Linear(COMPRESSED, COMPRESSED), nn.Linear(COMPRESSED, COMPRESSED),
            nn.Linear(COMPRESSED, COMPRESSED)
        ])
        self.snr_encoder = nn.Sequential(
            nn.Linear(1, 32), nn.ReLU(), nn.Linear(32, COMPRESSED)
        )
        
    def forward(self, f, snr_db=SNR_DB):
        x = self.fc_layers[0](f)
        for i in range(1, 3): x = self.fc_layers[i](x)
        snr_info = torch.ones(x.size(0), 1, device=x.device) * snr_db
        snr_features = self.snr_encoder(snr_info)
        x = self.fc_layers[3](x + snr_features)
        for i in range(4, 7): x = self.fc_layers[i](x)
        return x

class ChannelDecoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc_layers = nn.ModuleList([
            nn.Linear(COMPRESSED, COMPRESSED), nn.Linear(COMPRESSED, COMPRESSED),
            nn.Linear(COMPRESSED, COMPRESSED), nn.Linear(COMPRESSED, COMPRESSED),
            nn.Linear(COMPRESSED, COMPRESSED), nn.Linear(COMPRESSED, COMPRESSED),
            nn.Linear(COMPRESSED, BOTTLENECK)
        ])
        self.snr_encoder = nn.Sequential(
            nn.Linear(1, 32), nn.ReLU(), nn.Linear(32, COMPRESSED)
        )
        
    def forward(self, x, snr_db=SNR_DB):
        for i in range(3): x = self.fc_layers[i](x)
        snr_info = torch.ones(x.size(0), 1, device=x.device) * snr_db
        snr_features = self.snr_encoder(snr_info)
        x = self.fc_layers[3](x + snr_features)
        for i in range(4, 7): x = self.fc_layers[i](x)
        return x

class OptimizedSemanticComm(nn.Module):
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
            h = torch.randn_like(x); noise = sigma * torch.randn_like(x)
            y = h * x + noise; x_hat = y / (h + 1e-6) # Added epsilon here too
        x_hat = x_hat.detach().requires_grad_()
        z_hat = self.dec_c(x_hat, snr_db)
        return self.dec_s(z_hat, skips)

# ------------------------------------------------------------------
# Optimized training function
# ------------------------------------------------------------------
def local_train_optimized(model, loader, epochs: int):
    model.train()
    opt = optim.Adam(model.parameters(), lr=LR)
    # Updated GradScaler call
    scaler = torch.amp.GradScaler(device_type=DEVICE, enabled=(USE_FP16 and DEVICE=="cuda"))
    
    cumulative_loss = 0.0
    num_batches = 0
    for epoch_num in range(epochs):
        for img, _ in loader:
            img = img.to(DEVICE, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            
            # Updated autocast call
            with torch.amp.autocast(device_type=DEVICE, dtype=torch.float16 if (USE_FP16 and DEVICE=="cuda") else torch.float32, enabled=(USE_FP16 and DEVICE=="cuda")):
                recon = model(img)
                loss = perceptual_loss(recon, img)
            
            if torch.isnan(loss):
                print(f"Warning: NaN loss encountered in local_train_optimized at epoch {epoch_num+1}, batch. Skipping update.")
                # Optionally, one could try to recover or log more info here
                continue # Skip backprop and step for this batch

            if (USE_FP16 and DEVICE=="cuda"):
                scaler.scale(loss).backward()
                # Gradient Clipping
                scaler.unscale_(opt) # Unscale gradients before clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_VALUE)
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                # Gradient Clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_VALUE)
                opt.step()
            
            if not torch.isnan(loss): # only accumulate valid losses
                cumulative_loss += loss.item()
                num_batches += 1
    
    return cumulative_loss / num_batches if num_batches > 0 else float('nan') # Return NaN if no valid batches

# ------------------------------------------------------------------
# Data loading, FL aggregation, loss (Dirichlet, perceptual_loss are from user's previous version)
# ------------------------------------------------------------------
TRANSFORM = transforms.Compose(
    [transforms.Resize((64, 64)), transforms.ToTensor()]
)

def fedlol_aggregate(global_model, client_states, client_losses):
    eps = 1e-8
    
    valid_clients_indices = [i for i, loss in enumerate(client_losses) if not (math.isnan(loss) or math.isinf(loss))]
    
    if not valid_clients_indices:
        print("Warning: All clients reported NaN/inf loss. Global model not updated.")
        return

    valid_client_losses = [client_losses[i] for i in valid_clients_indices]
    valid_client_states = [client_states[i] for i in valid_clients_indices]
    
    num_valid_clients = len(valid_clients_indices)

    if num_valid_clients == 0: # Should be caught by above 'if not valid_clients_indices'
        return

    total_loss = sum(valid_client_losses) + eps

    weights = []
    if num_valid_clients == 1: # If only one client is valid
        weights = [1.0]
    elif len(set(valid_client_losses)) == 1: # All valid losses are the same
         weights = [1.0 / num_valid_clients] * num_valid_clients
    else:
        weights = [(total_loss - loss) / ((num_valid_clients - 1) * total_loss + eps) 
                   for loss in valid_client_losses]
    
    # Normalize weights to sum to 1
    sum_weights = sum(weights) + eps
    weights = [w / sum_weights for w in weights]

    global_model_device = next(global_model.parameters()).device
    new_state = copy.deepcopy(global_model.state_dict()) # new_state is on global_model_device

    for k in new_state.keys():
        if new_state[k].is_floating_point():
            new_state[k].zero_()
            for i in range(num_valid_clients):
                # Move client_states[k] to the device of new_state[k] (global model's device)
                param_data = valid_client_states[i][k].to(global_model_device)
                new_state[k].add_(param_data * weights[i]) # Use add_ for in-place addition
        else: # For non-floating point (e.g. num_batches_tracked in BN)
             if num_valid_clients > 0: # copy from the first valid client
                new_state[k] = valid_client_states[0][k].to(global_model_device)


    global_model.load_state_dict(new_state)


def perceptual_loss(pred, target, alpha: float = ALPHA_LOSS):
    mse_term = nn.functional.mse_loss(pred, target, reduction="mean")
    # Ensure pred and target are in expected range [0,1] for SSIM if data_range=1.0
    pred_clamped = torch.clamp(pred, 0.0, 1.0)
    target_clamped = torch.clamp(target, 0.0, 1.0)
    ssim_val = 1.0 - ssim(pred_clamped, target_clamped, data_range=1.0, nonnegative_ssim=True, size_average=True)
    
    if torch.isnan(mse_term) or torch.isinf(mse_term) or torch.isnan(ssim_val) or torch.isinf(ssim_val):
        return torch.tensor(float('nan'), device=pred.device) # Propagate NaN if parts are NaN

    return alpha * mse_term + (1.0 - alpha) * ssim_val

def dirichlet_split(dataset, alpha: float, n_clients: int):
    # Using the original user's dirichlet split logic for robustness
    label_to_indices = {}
    # Assuming dataset.samples for ImageFolder structure: (filepath, class_index)
    for idx, (_, lbl) in enumerate(dataset.samples): # Iterate through dataset.samples
        label_to_indices.setdefault(lbl, []).append(idx)

    clients_indices = [[] for _ in range(n_clients)]
    for class_indices in label_to_indices.values(): # Iterate over indices for each class
        proportions = torch.distributions.Dirichlet(
            torch.full((n_clients,), alpha)
        ).sample()
        
        proportions = proportions / proportions.sum() # Normalize
        
        current_idx = 0
        for cid in range(n_clients):
            # Number of samples for this client from this class
            count = int(round(proportions[cid].item() * len(class_indices)))
            
            # Ensure last client gets all remaining samples of this class to avoid loss from rounding
            if cid == n_clients - 1:
                clients_indices[cid].extend(class_indices[current_idx:])
            else:
                clients_indices[cid].extend(class_indices[current_idx : current_idx + count])
            current_idx += count
            
    # Filter out clients that might have ended up with no data, though unlikely with this method
    final_client_subsets = [Subset(dataset, idxs) for idxs in clients_indices if idxs]
    if len(final_client_subsets) < n_clients:
        print(f"Warning: Only {len(final_client_subsets)} clients have data after split, expected {n_clients}.")
        # This could happen if alpha is very small and some clients get 0 proportion consistently
        # For simplicity, we proceed with clients that have data.
    return final_client_subsets


DATA_ROOT = "./tiny-imagenet-20"
train_full = datasets.ImageFolder(f"{DATA_ROOT}/train", TRANSFORM)
val_full   = datasets.ImageFolder(f"{DATA_ROOT}/val",   TRANSFORM)

client_sets = dirichlet_split(train_full, DIRICHLET_ALPHA, NUM_CLIENTS)
# Update NUM_CLIENTS if dirichlet_split resulted in fewer clients with data
# However, FedLoL expects NUM_CLIENTS, so it's better if dirichlet_split guarantees data for all
# or the main loop handles fewer effective clients. For now, assume it works for NUM_CLIENTS.
if len(client_sets) != NUM_CLIENTS:
    print(f"Warning: dirichlet_split created {len(client_sets)} subsets, but NUM_CLIENTS is {NUM_CLIENTS}. FedLoL might behave unexpectedly.")
    # Adjust NUM_CLIENTS to actual number of clients with data if necessary,
    # but this changes the FL setup. Better to ensure dirichlet split provides for all.
    # For now, we'll proceed, but this could be a source of issues if some client_sets are empty.

val_loader  = DataLoader(
    val_full, batch_size=BATCH_SIZE, shuffle=False,
    num_workers=WORKERS, pin_memory=PIN_MEM,
)

# ------------------------------------------------------------------
# Optimized main training loop
# ------------------------------------------------------------------
def main_optimized():
    global_model = OptimizedSemanticComm().to(DEVICE)
    
    loaders = []
    actual_num_clients = 0
    for cid in range(NUM_CLIENTS): # Iterate up to original NUM_CLIENTS
        if cid < len(client_sets) and len(client_sets[cid]) > 0 : # Check if client set exists and is not empty
            loaders.append(DataLoader(
                client_sets[cid], batch_size=BATCH_SIZE, shuffle=True,
                num_workers=WORKERS, pin_memory=PIN_MEM,
                prefetch_factor=2 if WORKERS > 0 else None,
            ))
            actual_num_clients +=1
        else:
            print(f"Warning: Client {cid} has no data. Will be skipped.")
            loaders.append(None) # Placeholder for clients with no data
    
    if actual_num_clients == 0:
        print("Error: No clients have any data. Aborting training.")
        return

    for rnd in range(1, ROUNDS + 1):
        client_states, client_losses = [], []
        active_client_this_round = 0

        for cid in range(NUM_CLIENTS):
            if loaders[cid] is None: # Skip clients with no data
                # Append dummy values or handle carefully in aggregation
                # For FedLoL, we need a loss. Appending NaN or inf will have it ignored by the modified aggregate
                client_losses.append(float('nan')) 
                client_states.append(None) # No state to append
                continue

            local_model = copy.deepcopy(global_model).to(DEVICE)
            loss_val = local_train_optimized(local_model, loaders[cid], LOCAL_EPOCHS)
            
            if not (math.isnan(loss_val) or math.isinf(loss_val)):
                client_states.append(copy.deepcopy(local_model.cpu().state_dict()))
                active_client_this_round +=1
            else:
                client_states.append(None) # Placeholder for failed client
            client_losses.append(loss_val) # Append actual loss (or NaN)
            
            print(f"Round {rnd} | Client {cid+1}/{NUM_CLIENTS} | Local Loss: {loss_val:.4f}")

        if active_client_this_round > 0:
             # Filter out None states before aggregation
            valid_states_for_agg = [st for st in client_states if st is not None]
            # Note: client_losses already contains NaNs for failed/skipped clients
            # fedlol_aggregate is now robust to NaNs in client_losses and filters states accordingly
            if valid_states_for_agg: # only aggregate if there are valid states
                 fedlol_aggregate(global_model, valid_states_for_agg, client_losses)
            else:
                print(f"Round {rnd}: No clients successfully trained. Global model not updated.")
        else:
            print(f"Round {rnd}: No clients successfully trained. Global model not updated.")


        global_model.eval()
        with torch.no_grad():
            mse_sum, ssim_sum, perc_sum, n_img = 0.0, 0.0, 0.0, 0
            for img, _ in val_loader:
                img = img.to(DEVICE, non_blocking=True)
                
                with torch.amp.autocast(device_type=DEVICE, dtype=torch.float16 if (USE_FP16 and DEVICE=="cuda") else torch.float32, enabled=(USE_FP16 and DEVICE=="cuda")):
                    recon = global_model(img)
                    
                    img_fp32 = img.float() 
                    recon_fp32 = recon.float()
                    
                    current_mse = nn.functional.mse_loss(recon_fp32, img_fp32, reduction="sum").item()
                    if math.isnan(current_mse): continue # Skip if recon is bad

                    current_ssim_tensor = ssim(recon_fp32, img_fp32, data_range=1.0, size_average=False, nonnegative_ssim=True)
                    current_ssim_sum_batch = current_ssim_tensor.sum().item()
                    
                    batch_perc_loss = 0.0
                    for i in range(img_fp32.size(0)):
                        single_perc_loss = perceptual_loss(recon_fp32[i].unsqueeze(0), img_fp32[i].unsqueeze(0)).item()
                        if not math.isnan(single_perc_loss):
                             batch_perc_loss += single_perc_loss
                        else: # if perceptual loss is nan for an image, we might want to exclude this image's metrics
                            # For simplicity, we'll sum what we have, but this could skew if NaNs are frequent
                            pass


                mse_sum += current_mse
                ssim_sum += current_ssim_sum_batch
                perc_sum += batch_perc_loss
                n_img += img.size(0)

        if n_img == 0:
            print(f"Round {rnd:02d} │ Validation resulted in no valid images processed.")
            continue

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
        try:
            img_batch, _ = next(iter(val_loader))
            img_batch = img_batch.to(DEVICE)
            recon_batch = global_model(img_batch)

            orig = img_batch[:8].cpu()
            recon = recon_batch[:8].cpu()
            recon = torch.clamp(recon, 0, 1) 
            
            grid_img = torch.cat([orig, recon], 0)
            if grid_img.ndim == 3: grid_img = grid_img.unsqueeze(0)
                
            grid = make_grid(grid_img, nrow=8, padding=2)

            plt.figure(figsize=(12, 4)); plt.axis("off")
            plt.title("Top: original – Bottom: reconstruction")
            plt.imshow(grid.permute(1, 2, 0))
            plt.savefig("reconstructions_iot_simplified_v2.png", dpi=300, bbox_inches="tight")
            print("Saved reconstruction image to reconstructions_iot_simplified_v2.png")
        except Exception as e:
            print(f"Could not generate reconstruction image: {e}")
    # plt.show()

if __name__ == "__main__":
    try:
        if torch.cuda.is_available() or args.workers > 0 :
            torch.multiprocessing.set_start_method("spawn", force=True)
            # print("Set multiprocessing start method to spawn.") # Already printed by user
    except RuntimeError as e:
        if "context has already been set" not in str(e): # ignore if already set
             print(f"Could not set multiprocessing start method: {e}")

    main_optimized()