"""S5: LoD (Level of Detail) debug pipeline — visualize decoder stages to detect checkerboard artifacts."""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from src.utils import setup_logging, logger
from src.config import DEVICE
from src.models import create_model

OUT = Path(__file__).resolve().parent.parent / "outputs" / "lod_debug"
OUT.mkdir(parents=True, exist_ok=True)


def hook_fn(name, storage):
    def hook(module, inp, out):
        storage[name] = out.detach().cpu()
    return hook


def run():
    device = DEVICE
    model = create_model("mobilenetv2_unet", in_channels=1, out_channels=1)
    ckpt = torch.load("models/best_model.pth", map_location='cpu', weights_only=True)
    if isinstance(ckpt, dict) and 'model_state' in ckpt:
        ckpt = ckpt['model_state']
    model.load_state_dict(ckpt)
    model.to(device)
    model.eval()

    # Register hooks on all decoder blocks
    activations = {}
    handles = []
    for name, mod in model.named_modules():
        if 'decoder' in name and 'conv' in name and isinstance(mod, torch.nn.Sequential):
            handles.append(mod.register_forward_hook(hook_fn(name, activations)))
    handles.append(model.final.register_forward_hook(hook_fn('final', activations)))

    # Load a test slice
    import glob as gb
    candidates = list(Path("D:/DATA SCIENCE AND ANALYTICS/Dataset/Liver Img Dataset").glob("Volume-001-*.png"))
    if not candidates:
        logger.error("No test images found")
        return
    img_pil = Image.open(str(candidates[0])).convert('L')
    img = np.array(img_pil, dtype=np.float32) / 255.0
    img_t = torch.from_numpy(img).unsqueeze(0).unsqueeze(0).to(device)

    with torch.no_grad():
        _ = model(img_t)

    for h in handles:
        h.remove()

    # Visualize each decoder stage
    n = len(activations)
    fig, axes = plt.subplots(2, (n + 1) // 2, figsize=(4 * ((n + 1) // 2), 8))
    axes = axes.flatten()
    for i, (name, feat) in enumerate(activations.items()):
        fm = feat[0].abs().mean(dim=0).numpy()
        axes[i].imshow(fm, cmap='viridis')
        axes[i].set_title(f"{name}\n{tuple(feat.shape)}")
        axes[i].axis('off')
        # Checkerboard detection: horizontal/vertical gradient
        grad_h = np.abs(np.diff(fm, axis=1)).mean()
        grad_v = np.abs(np.diff(fm, axis=0)).mean()
        ratio = grad_h / max(grad_v, 1e-8)
        axes[i].text(0.02, 0.98, f"gH/gV={ratio:.2f}", transform=axes[i].transAxes,
                     fontsize=8, va='top', color='white',
                     bbox=dict(boxstyle='round', facecolor='black', alpha=0.5))
        if ratio > 1.5 or ratio < 0.67:
            logger.warning(f"  {name}: anisotropic gradients (gH/gV={ratio:.2f}) — possible artifact")
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')
    plt.tight_layout()
    plt.savefig(OUT / "decoder_stages.png", dpi=150)
    logger.info(f"Decoder stage visualizations saved to {OUT / 'decoder_stages.png'}")

    # Fourier analysis on final output
    final_map = activations.get('final', activations.get(list(activations.keys())[-1]))
    fm = final_map[0, 0].numpy()
    fft = np.fft.fft2(fm)
    fft_shift = np.fft.fftshift(fft)
    mag = np.log(np.abs(fft_shift) + 1e-8)
    fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.imshow(fm, cmap='gray')
    ax1.set_title("Final output")
    ax1.axis('off')
    ax2.imshow(mag, cmap='inferno')
    ax2.set_title("FFT magnitude (log)")
    ax2.axis('off')
    # Check for high-frequency grid patterns (checkerboard = peaks at Nyquist)
    h, w = mag.shape
    corner = mag[max(0, h // 2 - 5):min(h, h // 2 + 6), max(0, w // 2 - 5):min(w, w // 2 + 6)]
    center_val = mag[h // 2, w // 2]
    off_center_max = corner.max() if corner.size > 1 else 0
    if off_center_max > center_val * 0.3:
        logger.warning("  High off-center FFT peaks detected — possible checkerboard artifact")
    plt.tight_layout()
    plt.savefig(OUT / "fft_analysis.png", dpi=150)
    logger.info(f"FFT analysis saved to {OUT / 'fft_analysis.png'}")

    # Summary
    print(f"\nLoD debug results saved to {OUT}/")
    print("Check for anisotropic gradient ratios (gH/gV != 1.0) and off-center FFT peaks.")
    logger.info("LoD debug complete")


if __name__ == "__main__":
    setup_logging()
    run()
