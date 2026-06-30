"""
    U-Net for welding ripple binary segmentation.

    Architecture:
    Encoder    : 4 conv blocks, each block = Conv→BN→ReLU→Conv→BN→ReLU, followed by MaxPool
    Bottleneck : deepest conv block (no pooling after)
    Decoder    : 4 up-blocks, each = Upsample→Cat(skip)→Conv→BN→ReLU→Conv→BN→ReLU
    Head       : 1x1 Conv → num_classes (1 for binary)

    Key insight: skip connections concatenate encoder feature maps to the decoder at the
    same resolution. This is what lets U-Net recover fine spatial detail (like thin ripple
    boundaries) that would be lost during downsampling.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    """
        Two conv layers, each followed by BatchNorm + ReLU.
        This is the fundamental building block repeated throughout U-Net.
    """
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()

        """Sequential layer"""
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels, 
                out_channels=out_channels, 
                kernel_size=3, 
                padding=1, 
                bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                in_channels=out_channels, 
                out_channels=out_channels, 
                kernel_size=3, 
                padding=1, 
                bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),    
        )

    def forward(self, x):
        return self.block(x)
        
   
class UpBlock(nn.Module):
    """
        Upsample → concatenate skip → two conv layers.

        We use bilinear upsample + conv instead of ConvTranspose2d because:
        - ConvTranspose2d can produce 'checkerboard artifacts' on thin structures
        - Bilinear upsampling is smoother and more stable for ripple-like patterns
    """
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()

        """Scaling width and height"""
        self.up   = nn.Upsample(
            scale_factor=2, 
            mode="bilinear", 
            align_corners=True
        )
        self.conv = ConvBlock(in_channels, out_channels)

    def forward(self, x, skip):
        x = self.up(x)

        """Handle input size"""
        if x.shape[2:] != skip.shape[2:]:
            x = F.pad(x, [0, skip.shape[3] - x.shape[3],
                           0, skip.shape[2] - x.shape[2]]) 
            
        """Concatunate on channel dim"""
        x = torch.cat([skip, x], dim=1) 
        return self.conv(x)


class UNet(nn.Module):
    """
        U-Net for binary segmentation (ripple / no-ripple).

        Args:
            in_channels  : 1 for grayscale, 3 for RGB
            num_classes  : 1 for binary (use BCEWithLogitsLoss)
                        N for multi-class (use CrossEntropyLoss)
            base_channels: controls model width. Default 64.
                        Reduce to 32 for lighter model on Colab T4.
    """
    def __init__(self, in_channels: int=3, num_classes: int=1, base_channels: int=64):
        super().__init__()
        c = base_channels

        """Encoder Block"""
        self.enc1 = ConvBlock(in_channels, c)       # 256×256, 64ch
        self.enc2 = ConvBlock(c, 2 * c)             # 128×128, 128ch
        self.enc3 = ConvBlock(2 * c, 4 * c)         # 64×64,   256ch
        self.enc4 = ConvBlock(4 * c, 8 * c)         # 32×32,   512ch

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        """Bottleneck: enc5"""
        self.bottleneck = ConvBlock(8 * c, 16 * c)  # 16×16,   1024ch

        """Decoder Block"""
        self.dec4 = UpBlock(16 * c + 8 * c, 8 * c)
        self.dec3 = UpBlock(8  * c + 4 * c, 4 * c)
        self.dec2 = UpBlock(4  * c + 2 * c, 2 * c)
        self.dec1 = UpBlock(2  * c + c, c)

        """Head: conv 1x1"""
        self.head = nn.Conv2d(
            in_channels=c, 
            out_channels=num_classes, 
            kernel_size=1
        )

    def forward(self, x):
        """Encoder"""
        s1 = self.enc1(x)               # skip 1
        s2 = self.enc2(self.pool(s1))   # skip 2
        s3 = self.enc3(self.pool(s2))   # skip 3
        s4 = self.enc4(self.pool(s3))   # skip 4

        """Bottleneck"""
        b  = self.bottleneck(self.pool(s4))

        """Decoder"""
        x  = self.dec4(b, s4)
        x  = self.dec3(x, s3)
        x  = self.dec2(x, s2)
        x  = self.dec1(x, s1)

        """Head"""
        x  = self.head(x)
        return x
    

def build(in_channels=3, num_classes=1, base_channels=64):
    model = UNet(
        in_channels=in_channels,
        num_classes=num_classes,
        base_channels=base_channels 
    )   
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"U-Net built — trainable params: {total_params:,}")
    
    return model

if __name__ == "__main__":
    model = build()

    """Test model with random tensor"""
    x = torch.randn(2, 3, 256, 256)
    out = model(x)

    print(f"Input:  {x.shape}")   # (2, 3, 256, 256)
    print(f"Output: {out.shape}") # (2, 1, 256, 256)
