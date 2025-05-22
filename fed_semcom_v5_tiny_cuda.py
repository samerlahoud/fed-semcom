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
GRAD_CLIP_VALUE  = 1.0
CHANNEL_EPSILON  = 1e-5 # Increased epsilon for channel simulation stability with FP16

# Debug flag to force FP32 on CUDA if FP16 causes issues
DEBUG_FORCE_FP32_ON_CUDA = True # Set to True to disable FP16 for debugging

# ------------------------------------------------------------------
# Performance optimizations
# ------------------------------------------------------------------
torch.backends.cudnn.benchmark = True
# USE_FP16 should only be true if CUDA is available, selected, AND not forced to FP32
USE_FP16 = (DEVICE == "cuda" and torch.cuda.is_available() and not DEBUG_FORCE_FP32_ON_CUDA)

if DEVICE == "cuda":
    if torch.cuda.is_available():
        print(f"CUDA device selected. FP16 training {'Enabled' if USE_FP16 else 'Disabled (Forced FP32 or FP16 not supported/enabled)'}.")
    else:
        print("CUDA device selected, but torch.cuda.is_available() is False. Will run on CPU if PyTorch falls back.")
elif DEVICE == "mps":
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        print("MPS device selected. FP16 training is not explicitly managed by this script for MPS (uses PyTorch defaults).")
    else:
        print("MPS device selected, but not available. Will run on CPU if PyTorch falls back.")
else: # CPU
    print("CPU device selected. FP16 training not applicable.")


# JIT-compilable channel simulation function
@torch.jit.script
def apply_rayleigh_channel(x, sigma: float, epsilon: float):
    h = torch.randn_like(x)
    noise = sigma * torch.randn_like(x)
    # Using the new CHANNEL_EPSILON
    return (h * x + noise) / (h + epsilon)

class FastChannel(nn.Module):
    def __init__(self, channel_epsilon: float = CHANNEL_EPSILON):
        super().__init__()
        self.snr_cache = {}
        self.channel_epsilon = channel_epsilon
    
    def forward(self, x, snr_db=10.0):
        if snr_db not in self.snr_cache:
            sigma = math.sqrt(1.0 / (2 * 10 ** (snr_db / 10)))
            self.snr_cache[snr_db] = sigma
        else:
            sigma = self.snr_cache[snr_db]
        return apply_rayleigh_channel(x, sigma, self.channel_epsilon)

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
    def __init__(self, channel_epsilon: float = CHANNEL_EPSILON) -> None:
        super().__init__()
        self.enc_s = SemanticEncoder()
        self.enc_c = ChannelEncoder()
        self.dec_c = ChannelDecoder()
        self.dec_s = SemanticDecoder()
        self.snr_cache = {}
        self.channel_epsilon = channel_epsilon

    def forward(self, img, snr_db=SNR_DB):
        z, skips = self.enc_s(img)
        x = self.enc_c(z, snr_db)
        if snr_db not in self.snr_cache:
            self.snr_cache[snr_db] = math.sqrt(1.0 / (2 * 10 ** (snr_db / 10)))
        sigma = self.snr_cache[snr_db]
        with torch.no_grad():
            h = torch.randn_like(x); noise = sigma * torch.randn_like(x)
            y = h * x + noise; 
            # Using the new CHANNEL_EPSILON
            x_hat = y / (h + self.channel_epsilon) 
        x_hat = x_hat.detach().requires_grad_()
        z_hat = self.dec_c(x_hat, snr_db)
        return self.dec_s(z_hat, skips)

# ------------------------------------------------------------------
# Optimized training function
# ------------------------------------------------------------------
def local_train_optimized(model, loader, epochs: int):
    model.train()
    opt = optim.Adam(model.parameters(), lr=LR)
    scaler_enabled = USE_FP16 and DEVICE=="cuda" # Explicitly check DEVICE too
    scaler = torch.amp.GradScaler(enabled=scaler_enabled)
    
    cumulative_loss = 0.0
    num_batches = 0
    for epoch_num in range(epochs):
        for img, _ in loader:
            img = img.to(DEVICE, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            
            autocast_enabled = USE_FP16 and DEVICE=="cuda" # Explicitly check DEVICE
            autocast_dtype = torch.float16 if autocast_enabled else torch.float32

            with torch.amp.autocast(device_type=DEVICE, dtype=autocast_dtype, enabled=autocast_enabled):
                recon = model(img)
                loss = perceptual_loss(recon, img)
            
            if torch.isnan(loss) or torch.isinf(loss): # Check for inf too
                print(f"Warning: NaN/Inf loss encountered in local_train_optimized at epoch {epoch_num+1}, batch. Skipping update. Loss: {loss.item()}")
                continue 

            if scaler_enabled:
                scaler.scale(loss).backward()
                scaler.unscale_(opt) 
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_VALUE)
                scaler.step(opt)
                scaler.update()
            else: # Not using scaler (either CPU, MPS, or FP16 disabled on CUDA)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_VALUE)
                opt.step()
            
            cumulative_loss += loss.item() # loss.item() should be safe now
            num_batches += 1
    
    return cumulative_loss / num_batches if num_batches > 0 else float('nan')

# ------------------------------------------------------------------
# Data loading, FL aggregation, loss
# ------------------------------------------------------------------
TRANSFORM = transforms.Compose(
    [transforms.Resize((64, 64)), transforms.ToTensor()]
)

def fedlol_aggregate(global_model, client_states, client_losses):
    eps = 1e-12 # Small epsilon for numerical stability in weights
    
    valid_clients_indices = [i for i, loss in enumerate(client_losses) if loss is not None and not (math.isnan(loss) or math.isinf(loss))]
    
    if not valid_clients_indices:
        print("Warning: All clients reported NaN/inf loss or had no state. Global model not updated.")
        return

    valid_client_losses = [client_losses[i] for i in valid_clients_indices]
    # Ensure client_states correspond to valid_clients_indices
    valid_client_states = [cs for i, cs in enumerate(client_states) if i in valid_clients_indices and cs is not None]

    if not valid_client_states: # Should not happen if valid_clients_indices is not empty and states were stored
        print("Warning: No valid client states for aggregation despite valid losses. Global model not updated.")
        return

    num_valid_clients = len(valid_client_states) # Use count of actual states being aggregated

    if num_valid_clients == 0:
        return

    total_loss = sum(valid_client_losses) + eps * num_valid_clients # adjust eps scaling

    weights = []
    if num_valid_clients == 1:
        weights = [1.0]
    elif len(set(valid_client_losses)) == 1: # All valid losses are the same
         weights = [1.0 / num_valid_clients] * num_valid_clients
    else:
        # Ensure denominator is not zero if (num_valid_clients - 1) is 0
        denominator_factor = (num_valid_clients - 1) if num_valid_clients > 1 else 1
        weights = [(total_loss - loss) / (denominator_factor * total_loss + eps) 
                   for loss in valid_client_losses]
    
    sum_weights = sum(weights) + eps
    if abs(sum_weights) < eps : # Avoid division by zero if all weights are ~0
        print("Warning: Sum of FedLoL weights is close to zero. Using equal weights for valid clients.")
        weights = [1.0 / num_valid_clients] * num_valid_clients
    else:
        weights = [w / sum_weights for w in weights]

    global_model_device = next(global_model.parameters()).device
    new_state = copy.deepcopy(global_model.state_dict()) 

    for k in new_state.keys():
        if new_state[k].is_floating_point():
            new_state[k].zero_()
            for i in range(num_valid_clients):
                param_data = valid_client_states[i][k].to(global_model_device)
                new_state[k].add_(param_data * weights[i]) 
        else: 
             if num_valid_clients > 0:
                new_state[k] = valid_client_states[0][k].to(global_model_device)
    global_model.load_state_dict(new_state)


def perceptual_loss(pred, target, alpha: float = ALPHA_LOSS):
    mse_term = nn.functional.mse_loss(pred, target, reduction="mean")
    pred_clamped = torch.clamp(pred, 0.0, 1.0)
    target_clamped = torch.clamp(target, 0.0, 1.0)
    # Use size_average=True for a scalar ssim value if ssim function expects it for 1-ssim
    # pytorch_msssim ssim returns a scalar if size_average=True (default)
    ssim_metric = ssim(pred_clamped, target_clamped, data_range=1.0, nonnegative_ssim=True) 
    ssim_loss_term = 1.0 - ssim_metric
    
    # Check for NaNs / Infs from components
    if torch.isnan(mse_term) or torch.isinf(mse_term) or \
       torch.isnan(ssim_loss_term) or torch.isinf(ssim_loss_term):
        # Return a NaN tensor on the same device as pred to propagate the issue if it occurs
        return torch.tensor(float('nan'), device=pred.device, dtype=pred.dtype)

    return alpha * mse_term + (1.0 - alpha) * ssim_loss_term

def dirichlet_split(dataset, alpha: float, n_clients: int):
    label_to_indices = {}
    try:
        # ImageFolder stores samples as list of (image_path, class_index) tuples
        samples = dataset.samples
    except AttributeError:
        # Fallback if .samples is not available (e.g. for a Subset of ImageFolder)
        # This part might need adjustment if dataset is not a direct ImageFolder
        print("Warning: dataset.samples not found in dirichlet_split. Assuming dataset provides (data, label) tuples.")
        samples = [(None, dataset[i][1]) for i in range(len(dataset))] # Less efficient

    for idx, (_, lbl) in enumerate(samples):
        label_to_indices.setdefault(lbl, []).append(idx)

    clients_indices = [[] for _ in range(n_clients)]
    for class_indices in label_to_indices.values():
        if not class_indices: continue # Skip if a class has no samples (should not happen with ImageFolder)
        proportions = torch.distributions.Dirichlet(
            torch.full((n_clients,), alpha)
        ).sample()
        proportions = proportions / proportions.sum()
        
        current_idx_in_class = 0
        for cid in range(n_clients):
            num_samples_for_client_class = int(round(proportions[cid].item() * len(class_indices)))
            
            # Assign remaining samples to the last client to ensure all are used
            if cid == n_clients - 1:
                assigned_indices = class_indices[current_idx_in_class:]
            else:
                assigned_indices = class_indices[current_idx_in_class : current_idx_in_class + num_samples_for_client_class]
            
            clients_indices[cid].extend(assigned_indices)
            current_idx_in_class += len(assigned_indices)
            
    final_client_subsets = []
    for idxs in clients_indices:
        if idxs: # Only create subset if there are indices
            final_client_subsets.append(Subset(dataset, idxs))
        else: # Add an empty subset placeholder to maintain NUM_CLIENTS length for loader list
            final_client_subsets.append(None) 
            
    return final_client_subsets


DATA_ROOT = "./tiny-imagenet-20"
train_full = datasets.ImageFolder(f"{DATA_ROOT}/train", TRANSFORM)
val_full   = datasets.ImageFolder(f"{DATA_ROOT}/val",   TRANSFORM)

client_sets = dirichlet_split(train_full, DIRICHLET_ALPHA, NUM_CLIENTS)

val_loader  = DataLoader(
    val_full, batch_size=BATCH_SIZE, shuffle=False,
    num_workers=WORKERS, pin_memory=PIN_MEM,
)

# ------------------------------------------------------------------
# Optimized main training loop
# ------------------------------------------------------------------
def main_optimized():
    # Pass channel epsilon to model constructor
    global_model = OptimizedSemanticComm(channel_epsilon=CHANNEL_EPSILON).to(DEVICE)
    
    loaders = []
    active_clients_setup = 0
    for cid in range(NUM_CLIENTS): 
        if client_sets[cid] is not None and len(client_sets[cid]) > 0 :
            loaders.append(DataLoader(
                client_sets[cid], batch_size=BATCH_SIZE, shuffle=True,
                num_workers=WORKERS, pin_memory=PIN_MEM,
                prefetch_factor=2 if WORKERS > 0 else None, # None if workers is 0
            ))
            active_clients_setup +=1
        else:
            print(f"Info: Client {cid} has no data from dirichlet_split. Will be skipped in training rounds.")
            loaders.append(None) 
    
    if active_clients_setup == 0:
        print("Error: No clients have any data after dataset splitting. Aborting training.")
        return
    print(f"Initialized {active_clients_setup}/{NUM_CLIENTS} clients with data.")


    for rnd in range(1, ROUNDS + 1):
        client_states, client_losses = [], []
        active_clients_this_round = 0

        for cid in range(NUM_CLIENTS):
            if loaders[cid] is None: 
                client_losses.append(float('nan')) 
                client_states.append(None) 
                continue

            print(f"Round {rnd} | Starting training for Client {cid+1}/{NUM_CLIENTS}...")
            local_model = copy.deepcopy(global_model).to(DEVICE)
            loss_val = local_train_optimized(local_model, loaders[cid], LOCAL_EPOCHS)
            
            if not (math.isnan(loss_val) or math.isinf(loss_val)):
                client_states.append(copy.deepcopy(local_model.cpu().state_dict()))
                active_clients_this_round +=1
                print(f"Round {rnd} | Client {cid+1}/{NUM_CLIENTS} | Local Loss: {loss_val:.4f}")
            else:
                client_states.append(None) 
                print(f"Round {rnd} | Client {cid+1}/{NUM_CLIENTS} | Local Loss: {loss_val} (NaN/Inf, update skipped)")
            client_losses.append(loss_val) 
            
        if active_clients_this_round > 0:
            print(f"Round {rnd} | Aggregating models from {active_clients_this_round} active clients.")
            fedlol_aggregate(global_model, client_states, client_losses) # client_states already filtered implicitly by fedlol_aggregate
        else:
            print(f"Round {rnd}: No clients successfully trained. Global model not updated.")

        # Validation
        global_model.eval()
        with torch.no_grad():
            mse_sum, ssim_sum, perc_sum, n_img_val = 0.0, 0.0, 0.0, 0
            for img_val, _ in val_loader:
                img_val = img_val.to(DEVICE, non_blocking=True)
                
                autocast_enabled_val = USE_FP16 and DEVICE=="cuda"
                autocast_dtype_val = torch.float16 if autocast_enabled_val else torch.float32

                with torch.amp.autocast(device_type=DEVICE, dtype=autocast_dtype_val, enabled=autocast_enabled_val):
                    recon_val = global_model(img_val)
                
                img_fp32_val = img_val.float() 
                recon_fp32_val = recon_val.float()
                
                current_mse_val = nn.functional.mse_loss(recon_fp32_val, img_fp32_val, reduction="sum").item()
                if math.isnan(current_mse_val) or math.isinf(current_mse_val): continue 

                # For MS-SSIM, ssim function from pytorch_msssim usually returns a scalar per batch item if size_average=False
                # and then sum them.
                current_ssim_tensor_val = ssim(recon_fp32_val, img_fp32_val, data_range=1.0, size_average=False, nonnegative_ssim=True)
                current_ssim_sum_batch_val = current_ssim_tensor_val.sum().item()
                
                batch_perc_loss_val = 0.0
                for i in range(img_fp32_val.size(0)):
                    single_img_recon = recon_fp32_val[i].unsqueeze(0)
                    single_img_orig = img_fp32_val[i].unsqueeze(0)
                    single_perc_loss_val = perceptual_loss(single_img_recon, single_img_orig).item()
                    if not (math.isnan(single_perc_loss_val) or math.isinf(single_perc_loss_val)):
                         batch_perc_loss_val += single_perc_loss_val
                
                mse_sum += current_mse_val
                ssim_sum += current_ssim_sum_batch_val
                perc_sum += batch_perc_loss_val
                n_img_val += img_val.size(0)

        if n_img_val == 0:
            print(f"Round {rnd:02d} │ Validation resulted in no valid images processed.")
            continue

        mse_mean = mse_sum / (n_img_val * PIXELS)
        psnr_mean = 10.0 * math.log10(1.0 / max(mse_mean, 1e-12)) 
        msssim_mean = ssim_sum / n_img_val
        perc_mean = perc_sum / n_img_val

        print(
            f"Round {rnd:02d} │ Val MSE={mse_mean:.6f} │ PSNR={psnr_mean:.2f} dB │ "
            f"MS-SSIM={msssim_mean:.4f} │ HybridLoss={perc_mean:.4f}"
        )

    print("Training completed.")

    # Visual check
    if n_img_val > 0 : # only plot if validation happened
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
                plt.savefig("reconstructions_iot_simplified_final.png", dpi=300, bbox_inches="tight")
                print("Saved reconstruction image to reconstructions_iot_simplified_final.png")
            except Exception as e:
                print(f"Could not generate reconstruction image: {e}")
    # plt.show()

if __name__ == "__main__":
    try:
        # Set start method only if using CUDA or multiple workers, and not already set.
        # 'spawn' is generally safer for CUDA.
        if (DEVICE == "cuda" or WORKERS > 0):
             if torch.multiprocessing.get_start_method(allow_none=True) is None:
                torch.multiprocessing.set_start_method("spawn")
                print("Set multiprocessing start method to 'spawn'.")
             elif torch.multiprocessing.get_start_method() != "spawn":
                print(f"Warning: Multiprocessing start method already set to '{torch.multiprocessing.get_start_method()}'. Required 'spawn' for CUDA/workers, attempting to force.")
                torch.multiprocessing.set_start_method("spawn", force=True)
                print("Forced multiprocessing start method to 'spawn'.")


    except RuntimeError as e:
        if "context has already been set" not in str(e).lower(): 
             print(f"Could not set multiprocessing start method: {e}")
        # else: already set, which is fine.
    
    main_optimized()