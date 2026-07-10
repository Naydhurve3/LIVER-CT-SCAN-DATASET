"""S10: Debug head — visualize final conv layer outputs and test alternative upsampling strategies."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from src.utils import setup_logging, logger
from src.config import DEVICE
from src.models import create_model, MobileNetV2UNet

OUT = Path(__file__).resolve().parent.parent / "outputs" / "debug_head"
OUT.mkdir(parents=True, exist_ok=True)


class PixelShuffleHead(nn.Module):
    """Replaces final interpolate+conv with pixel shuffle for artifact-free upsampling."""
    def __init__(self, in_ch=16, out_ch=1, up_factor=2):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch * up_factor * up_factor, kernel_size=3, padding=1)
        self.shuffle = nn.PixelShuffle(up_factor)

    def forward(self, x):
        return self.shuffle(self.conv(x))


def compare_heads():
    device = DEVICE
    base_model = create_model("mobilenetv2_unet", in_channels=1, out_channels=1)
    ckpt = torch.load("models/best_model.pth", map_location='cpu', weights_only=True)
    if isinstance(ckpt, dict) and 'model_state' in ckpt:
        ckpt = ckpt['model_state']
    base_model.load_state_dict(ckpt, strict=False)
    base_model.to(device)
    base_model.eval()

    # Build alternate model with pixel-shuffle head
    alt_model = create_model("mobilenetv2_unet", in_channels=1, out_channels=1)
    alt_model.load_state_dict(ckpt, strict=False)
    # Override forward to skip the extra F.interpolate before final
    # (the model has a built-in scale_factor=2 bilinear upsampling before .final)
    def _alt_forward(self, x):
        f0 = self.enc_0(x); f1 = self.enc_1(f0)
        f2 = self.enc_2(f1); f3 = self.enc_3(f2)
        f4 = self.enc_4(f3); f5 = self.enc_5(f4)
        dec = self.decoder([f0, f1, f2, f3, f4, f5])
        return self.final(dec)
    alt_model.forward = _alt_forward.__get__(alt_model)
    alt_model.final = PixelShuffleHead(16, 1, 2)
    alt_model.to(device)
    alt_model.eval()

    # Test image
    candidates = list(Path("D:/DATA SCIENCE AND ANALYTICS/Dataset/Liver Img Dataset").glob("Volume-001-*.png"))
    img_pil = Image.open(str(candidates[0])).convert('L')
    img = np.array(img_pil, dtype=np.float32) / 255.0
    img_t = torch.from_numpy(img).unsqueeze(0).unsqueeze(0).to(device)

    with torch.no_grad():
        out_base = torch.sigmoid(base_model(img_t)).cpu().numpy()[0, 0]
        out_alt = torch.sigmoid(alt_model(img_t)).cpu().numpy()[0, 0]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(out_base, cmap='gray')
    axes[0].set_title(f"Bilinear + 1x1 Conv ({out_base.shape})")
    axes[0].axis('off')
    axes[1].imshow(out_alt, cmap='gray')
    axes[1].set_title(f"PixelShuffle Head ({out_alt.shape})")
    axes[1].axis('off')
    axes[2].imshow(np.abs(out_base - out_alt), cmap='hot')
    axes[2].set_title("Absolute difference")
    axes[2].axis('off')
    plt.tight_layout()
    plt.savefig(OUT / "head_comparison.png", dpi=150)
    logger.info(f"Head comparison saved to {OUT / 'head_comparison.png'}")

    print(f"\nDebug head results saved to {OUT}/")
    print("Check for spatial artifacts in the bilinear output vs pixel-shuffle output.")
    logger.info("Debug head complete")


if __name__ == "__main__":
    setup_logging()
    compare_heads()
