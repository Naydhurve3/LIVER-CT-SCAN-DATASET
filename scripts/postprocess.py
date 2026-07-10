"""S17: Post-processing refinement — CRF, morphological cleanup, and comparison."""
import sys, json, importlib
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
from src.metrics import dice_coefficient, nsd

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
        return {"image": img_t, "mask": mask_t, "img_path": e["image"]}


def crf_refine(probs, image_uint8, n_iter=5):
    """Dense CRF refinement using pydensecrf if available."""
    spec = importlib.util.find_spec("pydensecrf")
    if spec is None:
        return probs
    import pydensecrf.densecrf as dcrf
    from pydensecrf.utils import unary_from_softmax
    H, W = probs.shape
    n_labels = 2
    prob = np.stack([1 - probs, probs], axis=0)
    unary = unary_from_softmax(prob)
    unary = np.ascontiguousarray(unary)
    d = dcrf.DenseCRF2D(W, H, n_labels)
    d.setUnaryEnergy(unary)
    d.addPairwiseGaussian(sxy=3, compat=3)
    d.addPairwiseBilateral(sxy=20, srgb=3, rgbim=image_uint8, compat=10)
    Q = d.inference(n_iter)
    refined = np.array(Q).reshape(2, H, W)[1]
    return refined


def morphological_cleanup(binary):
    """Closing small holes + removing small islands."""
    closed = ndi.binary_closing(binary, structure=np.ones((3, 3)), iterations=2)
    cleaned = ndi.binary_opening(closed, structure=np.ones((3, 3)), iterations=1)
    labeled, num = ndi.label(cleaned)
    if num > 0:
        sizes = np.bincount(labeled.ravel())
        if len(sizes) > 1:
            largest = sizes[1:].argmax() + 1
            cleaned = (labeled == largest)
    return cleaned.astype(np.float32)


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
    loader = DataLoader(ds, batch_size=4, shuffle=False, num_workers=2)

    # Subset for speed (first 200 slices)
    baseline_dices, morph_dices, crf_dices = [], [], []
    crf_available = importlib.util.find_spec("pydensecrf") is not None

    count = 0
    for batch in tqdm(loader, desc="Post-processing"):
        if count >= 200:
            break
        img = batch["image"].to(device)
        mask = batch["mask"].to(device)
        with torch.no_grad():
            logits = model(img)
            probs = torch.sigmoid(logits)
        binary = (probs > 0.5).float()

        for b in range(img.size(0)):
            mask_np = mask[b, 0]
            bin_np = binary[b, 0].cpu().numpy()
            prob_np = probs[b, 0].cpu().numpy()

            # Baseline
            bin_t = binary[b:b+1]
            baseline_dices.append(dice_coefficient(bin_t, mask[b:b+1]).item())

            # Morphological cleanup
            morph_np = morphological_cleanup(bin_np)
            morph_t = torch.from_numpy(morph_np).unsqueeze(0).unsqueeze(0)
            morph_dices.append(dice_coefficient(morph_t.to(device), mask[b:b+1]).item())

            # CRF (if available)
            if crf_available:
                img_uint8 = (img[b, 0].cpu().numpy() * 255).astype(np.uint8)
                crf_prob = crf_refine(prob_np, np.stack([img_uint8] * 3, axis=-1))
                crf_bin = (crf_prob > 0.5).astype(np.float32)
                crf_t = torch.from_numpy(crf_bin).unsqueeze(0).unsqueeze(0)
                crf_dices.append(dice_coefficient(crf_t.to(device), mask[b:b+1]).item())
            count += 1

    print(f"\nBaseline Dice: {np.mean(baseline_dices):.4f} ± {np.std(baseline_dices):.4f}")
    print(f"Morph. Dice:  {np.mean(morph_dices):.4f} ± {np.std(morph_dices):.4f}")
    if crf_dices:
        print(f"CRF Dice:     {np.mean(crf_dices):.4f} ± {np.std(crf_dices):.4f}")
    if not crf_available:
        print("\n[NOTE] pydensecrf not installed. Install with: pip install pydensecrf")
        print("  CRF refinement skipped. Morphological cleanup results shown above.")

    results = {
        "n_slices_sampled": count,
        "baseline": {"dice_mean": float(np.mean(baseline_dices)), "dice_std": float(np.std(baseline_dices))},
        "morphological": {"dice_mean": float(np.mean(morph_dices)), "dice_std": float(np.std(morph_dices))},
    }
    if crf_dices:
        results["crf"] = {"dice_mean": float(np.mean(crf_dices)), "dice_std": float(np.std(crf_dices))}
    out_path = OUT / "postprocess_results.json"
    out_path.write_text(json.dumps(results, indent=2))
    logger.info(f"Post-processing results saved to {out_path}")


if __name__ == "__main__":
    setup_logging()
    run()
