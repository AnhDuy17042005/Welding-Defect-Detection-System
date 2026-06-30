"""
Dataset for U-Net ripple segmentation.

Expected folder structure:

ripple_unet/
    train/
        images/
        masks/
    valid/
        images/
        masks/
    test/
        images/
        masks/

Each image must have a corresponding mask with the same stem name.

Example:
    train/images/img001.jpg
    train/masks/img001.png

Mask:
    0   = background
    > 0 = ripple
"""

from pathlib import Path

import cv2
import numpy as np
from torch.utils.data import Dataset, DataLoader
from .augment import (
    get_train_transforms,
    get_val_transforms
)

class RippleDataset(Dataset):
    """
    Dataset for binary ripple segmentation.

    Input:
        image: RGB welding ROI

    Output:
        image tensor: 3 x H x W
        mask tensor : 1 x H x W
    """
    def __init__(self, image_dir: str, mask_dir: str,
                 transform = None):
        
        self.image_dir = Path(image_dir)
        self.mask_dir  = Path(mask_dir)
        self.transform = transform

        """Match images to masks by stem name"""
        exts = {".jpg", ".jpeg", ".png", ".bmp"}
        self.images = sorted([
            p for p in self.image_dir.iterdir()
            if p.suffix.lower() in exts
        ])

        """Verify all masks exist"""
        missing = []

        for img_path in self.images:
            mask_path = self.mask_dir / (img_path.stem + ".png")
            if not mask_path.exists():
                missing.append(img_path.name)
        
        if missing:
            raise FileNotFoundError(
                f"Missing masks for: {missing[:5]}{'...' if len(missing)>5 else ''}"
            )

        print(f"Dataset: {len(self.images)} samples from {image_dir}")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path  = self.images[idx]
        mask_path = self.mask_dir / (img_path.stem + ".png")

        """Load image as RGB"""
        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        """Load mask as grayscale, binarize"""
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        mask = (mask > 0).astype(np.float32)

        if self.transform:
            result = self.transform(image=image, mask=mask)
            image  = result["image"]   # Tensor (3, H, W)
            mask   = result["mask"]    # Tensor (H, W)

        """Add channel dim: (H, W) → (1, H, W) for BCEWithLogitsLoss"""
        return image, mask.unsqueeze(0)
    
def build_dataloaders(
        data_root: str,
        img_size: int = 256,
        batch_size: int = 8,
        num_workers: int = 2,
        pin_memory: bool = True,
    ):
    """
        Build train, validation, and test dataloaders.

        Expected:
            data_root/train/images
            data_root/train/masks
            data_root/valid/images
            data_root/valid/masks
            data_root/test/images
            data_root/test/masks
    """
    data_root = Path(data_root)

    train_image_dir = data_root / "train" / "images"
    train_mask_dir = data_root / "train" / "masks"

    valid_image_dir = data_root / "valid" / "images"
    valid_mask_dir = data_root / "valid" / "masks"

    test_image_dir = data_root / "test" / "images"
    test_mask_dir = data_root / "test" / "masks"
    
    train_dataset = RippleDataset(
        image_dir=train_image_dir,
        mask_dir=train_mask_dir,
        transform=get_train_transforms(img_size)
    )

    valid_dataset = RippleDataset(
        image_dir=valid_image_dir,
        mask_dir=valid_mask_dir,
        transform=get_val_transforms(img_size)
    )

    test_dataset = RippleDataset(
        image_dir=test_image_dir,
        mask_dir=test_mask_dir,
        transform=get_val_transforms(img_size)
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    print(
        f"Train: {len(train_dataset)} | "
        f"Valid: {len(valid_dataset)} | "
        f"Test: {len(test_dataset)}"
    )

    return train_loader, valid_loader, test_loader


if __name__ == "__main__":
    train_loader, valid_loader, test_loader = build_dataloaders(
        data_root="ripple_unet",
        img_size=256,
        batch_size=4,
        num_workers=2
    )

    images, masks = next(iter(train_loader))

    print("Image batch:", images.shape)
    print("Mask batch :", masks.shape)