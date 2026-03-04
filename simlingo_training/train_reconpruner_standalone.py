"""
Standalone training script for ReconPruner on driving images.

Trains only the ReconPruner module (scorer + reconstruction decoders) while
keeping the InternVL2-1B visual encoder completely frozen.  Any directory
of driving images can be used; by default it scans the Bench2Drive dataset.

Reference: FastDriveVLA (arxiv 2507.23318)

Usage:
    python simlingo_training/train_reconpruner_standalone.py \
        --data_dir /mnt/SSD7/dow904/dataset_B2D/Bench2Drive \
        --output_dir outputs/reconpruner \
        --pruning_ratio 0.5 \
        --epochs 5 \
        --batch_size 16 \
        --lr 1e-4 \
        --wandb_project simlingo-token-pruning \
        --wandb_name reconpruner_b2d_p50 \
        --gpus 0,1,2,3
"""
from __future__ import annotations

import argparse
import glob
import os
import random
import time
from pathlib import Path
from typing import List

import torch
import torch.nn as nn
import torchvision.transforms as T
import wandb
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel

# Make sure project root is on path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from simlingo_training.models.token_pruner import ReconPruner


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)
INPUT_SIZE    = 448  # InternVL2 default tile size


class DrivingImageDataset(Dataset):
    """Loads front-camera JPEG frames from a Bench2Drive-style directory tree."""

    def __init__(self, data_dir: str, cam_subdir: str = "camera/rgb_front"):
        self.paths: List[str] = []
        data_dir = Path(data_dir)
        # Collect all .jpg files under */cam_subdir/
        for route_dir in sorted(data_dir.iterdir()):
            cam_path = route_dir / cam_subdir
            if cam_path.is_dir():
                self.paths.extend(sorted(cam_path.glob("*.jpg")))

        if len(self.paths) == 0:
            raise RuntimeError(
                f"No images found under {data_dir} / */{cam_subdir}/*.jpg"
            )

        self.transform = T.Compose([
            T.Resize((INPUT_SIZE, INPUT_SIZE), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
        print(f"[Dataset] Found {len(self.paths)} images in {data_dir}")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        return self.transform(img)


# ---------------------------------------------------------------------------
# Feature extractor wrapper (frozen InternVL2-1B)
# ---------------------------------------------------------------------------

class FrozenVisualEncoder(nn.Module):
    """Wraps InternVL2-1B and exposes only extract_feature(), fully frozen."""

    def __init__(self, variant: str = "OpenGVLab/InternVL2-1B"):
        super().__init__()
        self.model = AutoModel.from_pretrained(variant, trust_remote_code=True)
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.hidden_size: int = self.model.language_model.config.hidden_size
        self.num_tokens: int  = self.model.num_image_token

    @torch.no_grad()
    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Returns visual tokens [B, num_tokens, hidden_size]."""
        return self.model.extract_feature(pixel_values)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(args):
    # ---- W&B ---------------------------------------------------------------
    wandb.init(
        project=args.wandb_project,
        name=args.wandb_name,
        config=vars(args),
    )

    # ---- Devices -----------------------------------------------------------
    gpu_ids = [int(g) for g in args.gpus.split(",")]
    primary = torch.device(f"cuda:{gpu_ids[0]}")

    # ---- Model -------------------------------------------------------------
    print("[Setup] Loading frozen visual encoder ...")
    encoder = FrozenVisualEncoder(args.model_variant).to(primary)
    if len(gpu_ids) > 1:
        encoder = nn.DataParallel(encoder, device_ids=gpu_ids)
    encoder.eval()

    hidden_size = encoder.module.hidden_size if isinstance(encoder, nn.DataParallel) else encoder.hidden_size

    print(f"[Setup] Building ReconPruner (hidden={hidden_size}, ratio={args.pruning_ratio}) ...")
    pruner = ReconPruner(
        hidden_size=hidden_size,
        pruning_ratio=args.pruning_ratio,
        num_pruner_heads=args.num_heads,
        num_decoder_layers=args.decoder_layers,
        adversarial_margin=args.adversarial_margin,
        adversarial_alpha=args.adversarial_alpha,
    ).to(primary)
    if len(gpu_ids) > 1:
        pruner = nn.DataParallel(pruner, device_ids=gpu_ids)

    # ---- Optimizer ---------------------------------------------------------
    optimizer = torch.optim.AdamW(pruner.parameters(), lr=args.lr, weight_decay=1e-2)

    num_train = int(0.95 * len(DrivingImageDataset(args.data_dir).__class__(args.data_dir)))  # placeholder
    dataset   = DrivingImageDataset(args.data_dir)
    n_total   = len(dataset)
    n_train   = int(0.95 * n_total)
    n_val     = n_total - n_train
    train_set, val_set = torch.utils.data.random_split(
        dataset, [n_train, n_val], generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_set, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )

    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * args.epochs
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.lr, total_steps=total_steps, pct_start=0.05
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    scaler = torch.cuda.amp.GradScaler()

    # ---- Training ----------------------------------------------------------
    global_step = 0
    for epoch in range(args.epochs):
        pruner.train()
        epoch_loss = 0.0
        t0 = time.time()

        for batch_idx, pixel_values in enumerate(train_loader):
            pixel_values = pixel_values.to(primary, dtype=torch.float16)

            # Extract frozen features [B, N, D]
            with torch.no_grad():
                visual_tokens = encoder(pixel_values)
            visual_tokens = visual_tokens.float()  # pruner trains in fp32

            # Forward pruner (training=True computes reconstruction loss)
            with torch.cuda.amp.autocast(enabled=False):
                _pruned, loss = pruner(visual_tokens, training=True)
                # DataParallel returns per-GPU losses; take mean
                if isinstance(loss, torch.Tensor) and loss.dim() > 0:
                    loss = loss.mean()

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(pruner.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            epoch_loss += loss.item()
            global_step += 1

            if batch_idx % args.log_every == 0:
                lr_now = scheduler.get_last_lr()[0]
                wandb.log({
                    "train/loss": loss.item(),
                    "train/lr": lr_now,
                    "epoch": epoch + batch_idx / steps_per_epoch,
                }, step=global_step)
                print(
                    f"Epoch {epoch+1}/{args.epochs} "
                    f"Step {batch_idx}/{steps_per_epoch} "
                    f"loss={loss.item():.4f} lr={lr_now:.2e}"
                )

        avg_train_loss = epoch_loss / steps_per_epoch
        elapsed = time.time() - t0

        # ---- Validation ----------------------------------------------------
        pruner.eval()
        val_loss = 0.0
        with torch.no_grad():
            for pixel_values in val_loader:
                pixel_values = pixel_values.to(primary, dtype=torch.float16)
                visual_tokens = encoder(pixel_values).float()
                _pruned, loss = pruner(visual_tokens, training=True)
                if isinstance(loss, torch.Tensor) and loss.dim() > 0:
                    loss = loss.mean()
                val_loss += loss.item()
        val_loss /= max(len(val_loader), 1)

        print(
            f"\n=== Epoch {epoch+1}/{args.epochs} | "
            f"train_loss={avg_train_loss:.4f} val_loss={val_loss:.4f} "
            f"time={elapsed:.0f}s ===\n"
        )
        wandb.log({
            "val/loss": val_loss,
            "train/epoch_loss": avg_train_loss,
            "epoch": epoch + 1,
        }, step=global_step)

        # ---- Checkpoint ----------------------------------------------------
        ckpt_path = output_dir / f"reconpruner_epoch{epoch+1:02d}.pt"
        state = pruner.module.state_dict() if isinstance(pruner, nn.DataParallel) else pruner.state_dict()
        torch.save({
            "epoch": epoch + 1,
            "state_dict": state,
            "val_loss": val_loss,
            "args": vars(args),
        }, ckpt_path)
        print(f"[Checkpoint] Saved to {ckpt_path}")
        wandb.save(str(ckpt_path))

    print("[Done] Training complete.")
    wandb.finish()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Train ReconPruner on driving images")
    p.add_argument("--data_dir", type=str,
                   default="/mnt/SSD7/dow904/dataset_B2D/Bench2Drive",
                   help="Root directory of Bench2Drive (or any dir with */camera/rgb_front/*.jpg)")
    p.add_argument("--output_dir", type=str, default="outputs/reconpruner")
    p.add_argument("--model_variant", type=str, default="OpenGVLab/InternVL2-1B")
    p.add_argument("--pruning_ratio", type=float, default=0.5,
                   help="Fraction of visual tokens to prune (0.5 = keep 50%%)")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--num_heads", type=int, default=8)
    p.add_argument("--decoder_layers", type=int, default=2)
    p.add_argument("--adversarial_margin", type=float, default=0.1)
    p.add_argument("--adversarial_alpha", type=float, default=0.5)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--log_every", type=int, default=50)
    p.add_argument("--gpus", type=str, default="0",
                   help="Comma-separated GPU ids, e.g. 0,1,2,3")
    p.add_argument("--wandb_project", type=str, default="simlingo-token-pruning")
    p.add_argument("--wandb_name", type=str, default="reconpruner_b2d_p50")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
