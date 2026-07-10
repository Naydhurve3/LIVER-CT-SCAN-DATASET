"""S14: Ensemble evaluation with full corrected metrics on 3 UP³RE members."""
import sys, json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
from tqdm import tqdm
from src.utils import setup_logging, logger
from src.config import DEVICE
from src.models import create_model, EnsembleWrapper
from src.trainer import evaluate_model
from src.metrics import (
    dice_coefficient, iou_score, hd95, asd, nsd,
    precision_recall, sensitivity_specificity,
    compute_volume_metrics,
)


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
        img_t = torch.from_numpy(np.array(img, dtype=np.float32) / 255.0).unsqueeze(0).float()
        mask_t = torch.from_numpy((np.array(mask, dtype=np.float32) > 0.5).astype(np.float32)).unsqueeze(0).float()
        return {"image": img_t, "mask": mask_t}


def evaluate_ensemble():
    device = DEVICE
    logger.info(f"Device: {device}")

    # Load 3 members
    members = []
    for i in range(3):
        m = create_model("mobilenetv2_unet", in_channels=1, out_channels=1)
        state = torch.load(f"models/upre/member_{i}.pth", map_location='cpu', weights_only=True)
        if isinstance(state, dict) and 'model_state' in state:
            state = state['model_state']
        m.load_state_dict(state)
        m.to(device)
        m.eval()
        members.append(m)
        logger.info(f"Member {i} loaded")

    # Also load best single model for comparison
    best_model = create_model("mobilenetv2_unet", in_channels=1, out_channels=1)
    best_ckpt = torch.load("models/best_model.pth", map_location='cpu', weights_only=True)
    best_model.load_state_dict(best_ckpt['model_state'])
    best_model.to(device)
    best_model.eval()
    logger.info(f"Best model: val_dice={best_ckpt.get('best_val_dice', 'N/A')}")

    # Get test dataloader from stratified split
    split_path = Path(__file__).resolve().parent.parent / "data" / "splits_stratified" / "test.json"
    test_ds = StratifiedTestDataset(split_path)
    test_loader = DataLoader(test_ds, batch_size=8, shuffle=False, num_workers=2)
    logger.info(f"Test set: {len(test_ds):,} slices")

    # Per-sample accumulators
    member_metrics = {i: {"dice": [], "iou": [], "hd95": [], "asd": [], "nsd": [],
                          "precision": [], "recall": [], "sensitivity": [], "specificity": []}
                      for i in range(3)}
    best_metrics = {k: [] for k in member_metrics[0]}
    ensemble_metrics = {k: [] for k in member_metrics[0]}

    ensemble = EnsembleWrapper(members)

    all_preds = {"best": [], "ensemble": [], "members": [[] for _ in range(3)]}
    all_targets = []

    for batch in tqdm(test_loader, desc="Evaluating"):
        images = batch['image'].to(device)
        masks = batch['mask'].to(device)
        masks_np = masks.cpu().numpy()

        # Best single model
        best_logits = best_model(images)
        best_probs = torch.sigmoid(best_logits)
        best_binary = (best_probs > 0.5).float()
        best_binary_np = best_binary.cpu().numpy()

        # Per-member predictions (one at a time to avoid OOM)
        member_binaries = []
        for m in members:
            m.to(device)
            with torch.no_grad():
                logits = m(images)
                probs = torch.sigmoid(logits)
                binary = (probs > 0.5).float()
            member_binaries.append(binary.cpu().numpy())
            m.to('cpu')
            torch.cuda.empty_cache()

        # Ensemble prediction (sequential mode moves models to/from GPU)
        mean_pred, variance = ensemble.predict_with_uncertainty(images, device, sequential=True)
        ens_binary = (mean_pred > 0.5).float().to(device)
        ens_binary_np = ens_binary.cpu().numpy()

        for b in range(images.size(0)):
            t_np = masks_np[b, 0]
            all_targets.append(t_np)
            all_preds["best"].append(best_binary_np[b, 0])
            all_preds["ensemble"].append(ens_binary_np[b, 0])
            for i in range(3):
                all_preds["members"][i].append(member_binaries[i][b, 0])

            # Compute surface metrics if tumor present
            has_tumor = t_np.sum() > 0
            spacing = (1.0, 1.0, 1.0)

            # Best model
            best_metrics["dice"].append(dice_coefficient(best_binary[b:b+1], masks[b:b+1]).item())
            best_metrics["iou"].append(iou_score(best_binary[b:b+1], masks[b:b+1]).item())
            pr = precision_recall(best_binary[b:b+1], masks[b:b+1])
            best_metrics["precision"].append(pr[0].item())
            best_metrics["recall"].append(pr[1].item())
            ss = sensitivity_specificity(best_binary[b:b+1], masks[b:b+1])
            best_metrics["sensitivity"].append(ss[0].item())
            best_metrics["specificity"].append(ss[1].item())
            if has_tumor:
                best_metrics["hd95"].append(hd95(best_binary_np[b, 0], t_np, spacing))
                best_metrics["asd"].append(asd(best_binary_np[b, 0], t_np, spacing))
                best_metrics["nsd"].append(nsd(best_binary_np[b, 0], t_np, spacing))

            # Ensemble
            ensemble_metrics["dice"].append(dice_coefficient(ens_binary[b:b+1], masks[b:b+1]).item())
            ensemble_metrics["iou"].append(iou_score(ens_binary[b:b+1], masks[b:b+1]).item())
            pr = precision_recall(ens_binary[b:b+1], masks[b:b+1])
            ensemble_metrics["precision"].append(pr[0].item())
            ensemble_metrics["recall"].append(pr[1].item())
            ss = sensitivity_specificity(ens_binary[b:b+1], masks[b:b+1])
            ensemble_metrics["sensitivity"].append(ss[0].item())
            ensemble_metrics["specificity"].append(ss[1].item())
            if has_tumor:
                ensemble_metrics["hd95"].append(hd95(ens_binary_np[b, 0], t_np, spacing))
                ensemble_metrics["asd"].append(asd(ens_binary_np[b, 0], t_np, spacing))
                ensemble_metrics["nsd"].append(nsd(ens_binary_np[b, 0], t_np, spacing))

            # Per-member
            for i in range(3):
                mb = torch.from_numpy(member_binaries[i][b:b+1, 0:1]).float().to(device)
                member_metrics[i]["dice"].append(dice_coefficient(mb, masks[b:b+1]).item())
                member_metrics[i]["iou"].append(iou_score(mb, masks[b:b+1]).item())

    def summarize(metrics_dict, name):
        out = {}
        for k, v in metrics_dict.items():
            if v:
                out[k] = {"mean": float(np.mean(v)), "std": float(np.std(v))}
            else:
                out[k] = {"mean": None, "std": None}
        out["name"] = name
        return out

    results = {
        "best_model": summarize(best_metrics, "MobileNetV2-UNet (Single)"),
        "ensemble": summarize(ensemble_metrics, "UP³RE Ensemble (3 members)"),
    }
    for i in range(3):
        results[f"member_{i}"] = summarize(member_metrics[i], f"Member {i}")

    # Save per-sample Dice for bootstrap testing
    per_sample = {
        "best_model": best_metrics["dice"],
        "ensemble": ensemble_metrics["dice"],
    }
    for i in range(3):
        per_sample[f"member_{i}"] = member_metrics[i]["dice"]

    per_sample_path = Path(__file__).resolve().parent.parent / "results" / "per_sample_dice.json"
    per_sample_path.write_text(json.dumps(per_sample))
    logger.info(f"Per-sample Dice saved to {per_sample_path}")

    out_path = Path(__file__).resolve().parent.parent / "results" / "ensemble_evaluation.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))

    print("\n" + "=" * 60)
    print("ENSEMBLE EVALUATION RESULTS")
    print("=" * 60)
    for name, res in results.items():
        if name == "name":
            continue
        print(f"\n--- {res.get('name', name)} ---")
        for k, v in res.items():
            if k == "name":
                continue
            if v["mean"] is not None:
                print(f"  {k}: {v['mean']:.4f} ± {v['std']:.4f}")
            else:
                print(f"  {k}: N/A")

    logger.info(f"Ensemble evaluation saved to {out_path}")
    return results


if __name__ == "__main__":
    setup_logging()
    evaluate_ensemble()
