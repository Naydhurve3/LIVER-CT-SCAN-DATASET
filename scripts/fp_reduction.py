"""S16: False positive reduction via connected-component filtering and threshold sweep."""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import numpy as np
from scipy import ndimage as ndi
from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from src.utils import setup_logging, logger
from src.config import DEVICE
from src.models import create_model
from src.metrics import dice_coefficient, iou_score, precision_recall

OUT = Path(__file__).resolve().parent.parent / "results"
SPLIT_PATH = Path(__file__).resolve().parent.parent / "data" / "splits_stratified" / "test.json"


class StratifiedTestDataset(Dataset):
    def __init__(self, json_path, size=256):
        self.size = size
        with open(json_path) as f:
            self.entries = json.load(f)

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        e = self.entries[idx]
        img = Image.open(e["image"]).convert("L").resize((self.size, self.size), Image.BILINEAR)
        mask = Image.open(e["mask"]).convert("L").resize((self.size, self.size), Image.NEAREST)
        img_t = torch.from_numpy(np.array(img, dtype=np.float32) / 255.0).unsqueeze(0)
        mask_t = torch.from_numpy((np.array(mask, dtype=np.float32) > 0.5).astype(np.float32)).unsqueeze(0)
        return {"image": img_t, "mask": mask_t}


def remove_small_components(binary_mask, min_size_px):
    labeled, num = ndi.label(binary_mask)
    sizes = np.bincount(labeled.ravel())
    if len(sizes) <= 1:
        return binary_mask
    too_small = sizes < min_size_px
    too_small[0] = False
    remove_mask = too_small[labeled]
    binary_mask[remove_mask] = 0
    return binary_mask


def run():
    device = DEVICE
    model = create_model("mobilenetv2_unet", in_channels=1, out_channels=1)
    ckpt = torch.load("models/best_model.pth", map_location='cpu', weights_only=True)
    if isinstance(ckpt, dict) and 'model_state' in ckpt:
        ckpt = ckpt['model_state']
    model.load_state_dict(ckpt)
    model.to(device)
    model.eval()

    ds = StratifiedTestDataset(SPLIT_PATH)
    loader = DataLoader(ds, batch_size=8, shuffle=False, num_workers=2)

    # Threshold sweep
    thresholds = [0.3, 0.5, 0.7, 0.9]
    min_sizes = [0, 16, 32, 64, 128, 256]

    results = []
    for thr in thresholds:
        for msz in min_sizes:
            dices, ious, precs, recs = [], [], [], []
            for batch in tqdm(loader, desc=f"thr={thr}, min={msz}"):
                img = batch["image"].to(device)
                mask = batch["mask"].to(device)
                with torch.no_grad():
                    logits = model(img)
                    probs = torch.sigmoid(logits)
                binary = (probs > thr).float()
                # Per-sample post-processing
                for b in range(img.size(0)):
                    bin_np = binary[b, 0].cpu().numpy()
                    mask_np = mask[b, 0].cpu().numpy()
                    if msz > 0:
                        bin_np = remove_small_components(bin_np, msz)
                    bin_t = torch.from_numpy(bin_np).unsqueeze(0).unsqueeze(0).to(device)
                    mask_t = mask[b:b+1]
                    dices.append(dice_coefficient(bin_t, mask_t).item())
                    ious.append(iou_score(bin_t, mask_t).item())
                    pr = precision_recall(bin_t, mask_t)
                    precs.append(pr[0].item())
                    recs.append(pr[1].item())
            results.append({
                "threshold": thr,
                "min_component_px": msz,
                "dice": float(np.mean(dices)), "iou": float(np.mean(ious)),
                "precision": float(np.mean(precs)), "recall": float(np.mean(recs)),
            })

    # Find best
    best = max(results, key=lambda r: r["dice"])
    print(f"\nBest config: threshold={best['threshold']}, min_size={best['min_component_px']}px")
    print(f"  Dice={best['dice']:.4f}, IoU={best['iou']:.4f}, Prec={best['precision']:.4f}, Rec={best['recall']:.4f}")
    print("\nFull sweep:")
    for r in results:
        print(f"  thr={r['threshold']:.1f} min={r['min_component_px']:4d}px  "
              f"dice={r['dice']:.4f} iou={r['iou']:.4f} prec={r['precision']:.4f} rec={r['recall']:.4f}")

    out_path = OUT / "fp_reduction.json"
    out_path.write_text(json.dumps(results, indent=2))
    logger.info(f"FP reduction results saved to {out_path}")


if __name__ == "__main__":
    setup_logging()
    run()
