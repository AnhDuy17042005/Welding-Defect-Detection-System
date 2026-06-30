"""
    Training script for U-Net ripple segmentation.

    Run:
        python train.py

    Colab usage:
        !python train.py --data /content/drive/MyDrive/ripple_data \
                        --epochs 50 --batch_size 8 --img_size 256
"""

import argparse
import os
import time
from pathlib import Path

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm.auto import tqdm

from .unet import build
from .dataset import build_dataloaders
from .losses import BCEDiceLoss, compute_iou, compute_dice

def parse_args():
    p = argparse.ArgumentParser(
        description="Train U-Net for ripple segmentation"
    )
    p.add_argument("--data",        type=str,   default="dataset/ripple_split", help="Root data folder")
    p.add_argument("--epochs",      type=int,   default=100)
    p.add_argument("--batch_size",  type=int,   default=8)
    p.add_argument("--img_size",    type=int,   default=256)
    p.add_argument("--lr",          type=float, default=0.001)
    p.add_argument("--base_channel",type=int,   default=64,        help="U-Net width")
    p.add_argument("--save_dir",    type=str,   default="models")
    p.add_argument("--resume",      type=str,   default=None,      help="Path to checkpoint")
    p.add_argument(
        "--pretrained",
        type=str,
        default=None,
        help="Load model weights only and start fine-tuning from epoch 0",
    )
    p.add_argument("--num_workers", type=int,   default=4)
    args = p.parse_args()

    if args.resume and args.pretrained:
        p.error("--resume and --pretrained cannot be used together")
    if args.epochs < 1:
        p.error("--epochs must be at least 1")

    return args

def train_one_epoch(
    model,
    loader,
    optimizer,
    criterion,
    device,
    epoch,
    total_epochs,
):
    """
        Training function per each epoch
    """
    model.train()
    total_loss, total_iou, total_dice = 0.0, 0.0, 0.0

    """Print progress during training"""
    progress = tqdm(
        loader,
        desc=f"=== [Progress training] Epoch {epoch + 1:03d}/{total_epochs:03d}",
        unit="batch",
        dynamic_ncols=True,
    )

    for batch_index, (images, masks) in enumerate(progress, start=1):
        images = images.to(device, non_blocking=True)
        masks  = masks.to(device, non_blocking=True)

        """Delete old gradient"""
        optimizer.zero_grad(set_to_none=True)
        
        logits = model(images)
        loss   = criterion(logits, masks)

        """Compute gradient"""
        loss.backward()

        """Normalize gradient value"""
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        """Update parameters"""
        optimizer.step()

        """Computer metrics"""
        total_loss += loss.item()
        total_iou  += compute_iou(logits, masks)
        total_dice += compute_dice(logits, masks)

        progress.set_postfix(
            loss=f"{total_loss / batch_index:.4f}",
            iou=f"{total_iou / batch_index:.4f}",
            dice=f"{total_dice / batch_index:.4f}",
        )

    """Average on all batch"""
    n = len(loader)
    return total_loss / n, total_iou / n, total_dice / n        


@torch.no_grad()
def validate(model, loader, criterion, device):
    """
        Validation function
    """
    model.eval()
    total_loss, total_iou, total_dice = 0.0, 0.0, 0.0

    for images, masks in loader:
        images = images.to(device, non_blocking=True)
        masks  = masks.to(device, non_blocking=True)

        logits = model(images)
        loss   = criterion(logits, masks)

        total_loss += loss.item()
        total_iou  += compute_iou(logits, masks)
        total_dice += compute_dice(logits, masks)

    n = len(loader)
    return total_loss / n, total_iou / n, total_dice / n

def save_checkpoint(state: dict, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state, path)
    print(f"  ✓ Checkpoint saved → {path}")


def load_pretrained_weights(model, checkpoint_path: str, device) -> None:
    path = Path(checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError(f"Pretrained checkpoint not found: {path}")

    checkpoint = torch.load(path, map_location=device)
    state_dict = checkpoint.get("model", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    model.load_state_dict(state_dict)

    source_epoch = checkpoint.get("epoch") if isinstance(checkpoint, dict) else None
    if source_epoch is None:
        print(f"Loaded pretrained weights: {path}")
    else:
        print(f"Loaded pretrained weights from epoch {source_epoch + 1}: {path}")
    print("Optimizer and scheduler will start from a new state.")


def format_duration(seconds: float) -> str:
    total_seconds = int(round(seconds))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

"""
    Main function
"""
def main():
    args = parse_args()

    """Device setup"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    """Data"""
    train_loader, valid_loader, test_loader = build_dataloaders(
        data_root   = args.data,
        img_size    = args.img_size,
        batch_size  = args.batch_size,
        num_workers = args.num_workers,
    )

    """Model"""
    model = build(
        in_channels     = 3, 
        num_classes     = 1, 
        base_channels   = args.base_channel
    ).to(device)

    if args.pretrained:
        load_pretrained_weights(model, args.pretrained, device)

    """Loss"""
    criterion = BCEDiceLoss(alpha=0.5)

    """Optimizer: Using AdamW"""
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    """
        Scheduler: change learning rate during training
    """
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    """Resume training from checkpoint"""
    start_epoch   = 0
    best_val_iou  = -1.0

    if args.resume:
        if not os.path.isfile(args.resume):
            raise FileNotFoundError(f"Resume checkpoint not found: {args.resume}")

        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch  = ckpt["epoch"] + 1
        best_val_iou = ckpt.get("best_val_iou", 0.0)
        print(f"Resumed from epoch {start_epoch}, best IoU: {best_val_iou:.4f}")    

    if start_epoch >= args.epochs:
        raise ValueError(
            f"Checkpoint already reached epoch {start_epoch}; "
            f"--epochs must be greater than {start_epoch} to resume."
        )

    """Training Loop"""
    print(f"\nTraining epochs {start_epoch + 1} to {args.epochs}...\n")
    history = {"train_loss": [], "val_loss": [], "train_iou": [], "val_iou": []}
    training_started = time.perf_counter()

    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()

        train_loss, train_iou, train_dice = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            epoch=epoch,
            total_epochs=args.epochs,
        )

        val_loss, val_iou, val_dice = validate(
            model=model,
            loader=valid_loader,
            criterion=criterion,
            device=device
        )
        
        """Compute time per each epoch"""
        scheduler.step()
        elapsed = time.time() - t0

        """Logger"""
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_iou"].append(train_iou)
        history["val_iou"].append(val_iou)

        lr_now = scheduler.get_last_lr()[0]
        print(f"Epoch [{epoch+1:03d}/{args.epochs}] "
              f"| train_loss={train_loss:.4f} train_iou={train_iou:.4f} train_dice={train_dice:.4f} "
              f"| val_loss={val_loss:.4f} val_iou={val_iou:.4f} val_dice={val_dice:.4f} "
              f"| lr={lr_now:.2e} | {elapsed:.1f}s")
        
        """ave best model (highest val IoU)"""
        if val_iou > best_val_iou:
            best_val_iou = val_iou
            save_checkpoint({
                "epoch"        : epoch,
                "model"        : model.state_dict(),
                "optimizer"    : optimizer.state_dict(),
                "scheduler"    : scheduler.state_dict(),
                "best_val_iou" : best_val_iou,
                "args"         : vars(args),
            }, path=os.path.join(args.save_dir, "best.pth"))
        
        # """Periodic save every 5 epochs"""
        # if (epoch + 1) % 5 == 0:
        #     save_checkpoint({
        #         "epoch"        : epoch,
        #         "model"        : model.state_dict(),
        #         "optimizer"    : optimizer.state_dict(),
        #         "scheduler"    : scheduler.state_dict(),
        #         "best_val_iou" : best_val_iou,
        #         "args"         : vars(args),
        #     }, path=os.path.join(args.save_dir, f"epoch_{epoch+1}.pth"))
        
    total_training_time = time.perf_counter() - training_started
    print(f"\nTraining complete. Best val IoU: {best_val_iou:.4f}")
    print(f"Total training time: {format_duration(total_training_time)}")
    print(f"Best model saved at: {args.save_dir}/best.pth")
    return history


if __name__ == "__main__":
    main()
