"""
    Augmentation script for ripple dataset
"""

import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2

def get_train_transforms(img_size: int = 256):
    """
        Augmentation pipeline for training.
        Strategy: geometric transforms that preserve ripple structure +
                  photometric transforms to handle lighting variation.

        We do NOT use heavy distortions (ElasticTransform, GridDistortion) because
        they would deform the periodic ripple pattern and confuse the model.
    """
    return A.Compose([
        A.Resize(img_size, img_size),

        # Geometric transforms
        A.HorizontalFlip(p=0.3),
        A.VerticalFlip(p=0.5),
        A.Rotate(limit=15, border_mode=cv2.BORDER_REFLECT_101, p=0.5),
        A.RandomCrop(height=int(img_size*0.9), width=int(img_size*0.9), p=0.5),
        A.Resize(img_size, img_size),

        # Photometric transforms
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.6),
        A.GaussNoise(std_range=(0.04, 0.12), p=0.4),
        A.GaussianBlur(blur_limit=(3, 5), p=0.3),
        A.CLAHE(clip_limit=2.0, p=0.3),

        # Normalize to ImageNet stats
        A.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])


def get_val_transforms(img_size: int = 256):
    """
        Transform for validation and test set.
        No random augmentation.
    """
    return A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])