from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from src.framework.evaluation.checkpoints import (
    environment_snapshot, load_checkpoint_into_model, split_hashes, write_json,
)
from src.framework.evaluation.research_metrics import (
    evaluate_research_model, model_probabilities,
)
from src.framework.experiment import (
    build_experiment_loaders, build_experiment_model, load_experiment,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate one locked research checkpoint")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--threshold", type=float)
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--images-dir")
    parser.add_argument("--masks-dir")
    parser.add_argument("--output-dir")
    return parser.parse_args()


def _selected_threshold(cfg, explicit):
    if explicit is not None:
        return explicit
    path = Path(cfg["outputs"]["run_dir"]) / "threshold_selection.json"
    if not path.exists():
        raise FileNotFoundError(
            f"Threshold selection not found: {path}. Tune on validation or pass --threshold."
        )
    return float(json.loads(path.read_text(encoding="utf-8"))["selected_threshold"])


def save_qualitative_examples(model, loader, device, threshold, slice_rows, output_dir):
    positive = sorted((row for row in slice_rows if row["has_tumor"]), key=lambda row: row["dice"])
    if not positive:
        return []
    choices = {"worst": positive[0], "median": positive[len(positive) // 2], "best": positive[-1]}
    targets = {(row["volume_id"], row["slice_id"]): label for label, row in choices.items()}
    saved = []
    output_dir.mkdir(parents=True, exist_ok=True)
    model.eval()
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            probs = model_probabilities(model, images).cpu().numpy()
            masks = batch["mask"].cpu().numpy()
            vids = batch["volume_id"].cpu().tolist()
            sids = batch["slice_id"].cpu().tolist()
            for index, key in enumerate(zip(vids, sids)):
                if key not in targets:
                    continue
                label = targets.pop(key)
                image = images[index, 0].cpu().numpy()
                truth = masks[index, 0] > 0
                pred = probs[index, 0] >= threshold
                error = np.zeros((*truth.shape, 3), dtype=np.float32)
                error[np.logical_and(pred, ~truth)] = (1.0, 0.0, 0.0)
                error[np.logical_and(~pred, truth)] = (0.0, 0.4, 1.0)
                fig, axes = plt.subplots(1, 3, figsize=(12, 4))
                axes[0].imshow(image, cmap="gray")
                axes[0].contour(truth, colors="lime", linewidths=0.8)
                axes[0].set_title("Ground truth")
                axes[1].imshow(image, cmap="gray")
                axes[1].contour(pred, colors="red", linewidths=0.8)
                axes[1].set_title(f"Prediction @ {threshold:.2f}")
                axes[2].imshow(image, cmap="gray")
                axes[2].imshow(error, alpha=np.max(error, axis=2) * 0.7)
                axes[2].set_title("FP red / FN blue")
                for axis in axes:
                    axis.axis("off")
                fig.suptitle(f"{label.title()} positive slice: volume {key[0]}, slice {key[1]}")
                fig.tight_layout()
                path = output_dir / f"{label}_v{key[0]:03d}_s{key[1]:03d}.png"
                fig.savefig(path, dpi=160, bbox_inches="tight")
                plt.close(fig)
                saved.append(str(path))
            if not targets:
                break
    return saved


def main() -> int:
    args = parse_args()
    cfg = load_experiment(args.config, args.images_dir, args.masks_dir)
    threshold = _selected_threshold(cfg, args.threshold)
    loaders, _, _, split_dir = build_experiment_loaders(cfg, include_sampler=False)
    loader = loaders[1] if args.split == "val" else loaders[2]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_experiment_model(cfg, pretrained=False)
    checkpoint_info = load_checkpoint_into_model(model, args.checkpoint, device)
    output_dir = Path(args.output_dir or Path(cfg["outputs"]["run_dir"]) / f"{args.split}_evaluation")
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    result = evaluate_research_model(model, loader, device, threshold)
    pd.DataFrame(result["slices"]).to_csv(output_dir / "per_slice.csv", index=False)
    pd.DataFrame(result["volumes"]).to_csv(output_dir / "per_volume.csv", index=False)
    qualitative = save_qualitative_examples(
        model, loader, device, threshold, result["slices"], output_dir / "qualitative"
    )
    payload = {
        "experiment": cfg["experiment"]["name"], "split": args.split,
        "aggregate": result["aggregate"], "plot_data": result["plot_data"],
        "threshold": threshold, "checkpoint": checkpoint_info["checkpoint"],
        "checkpoint_sha256": checkpoint_info["sha256"],
        "environment": environment_snapshot(), "split_hashes": split_hashes(split_dir),
        "preprocessing_profile": "png_8bit_no_hu_no_clahe_v1",
        "runtime_seconds": time.perf_counter() - started,
        "qualitative_files": qualitative,
        "test_set_accessed": args.split == "test",
    }
    write_json(output_dir / "evaluation.json", payload)
    print(json.dumps(result["aggregate"], indent=2))
    print(f"output_dir={output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
