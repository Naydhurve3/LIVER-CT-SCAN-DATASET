from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from src.framework.core.reproducibility import set_seed
from src.framework.evaluation.checkpoints import (
    environment_snapshot, load_checkpoint_into_model, sha256_file, split_hashes, write_json,
)
from src.framework.evaluation.research_metrics import select_threshold
from src.framework.experiment import (
    build_experiment_loaders, build_experiment_loss, build_experiment_model, load_experiment,
)
from src.framework.training.research_trainer import ResearchTrainer


def parse_args():
    parser = argparse.ArgumentParser(description="Train a locked research-validation experiment")
    parser.add_argument("--config", default="configs/experiments/research_baseline.yaml")
    parser.add_argument("--images-dir")
    parser.add_argument("--masks-dir")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--output-dir")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cfg = load_experiment(args.config, args.images_dir, args.masks_dir)
    set_seed(cfg["experiment"].get("seed", 42))
    training = cfg["training"]
    epochs = 1 if args.dry_run else (args.epochs or training["epochs"])
    model_dir = Path(args.output_dir or cfg["outputs"]["model_dir"])
    run_dir = Path(cfg["outputs"]["run_dir"])
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"experiment={cfg['experiment']['name']} device={'cuda' if torch.cuda.is_available() else 'cpu'}")
    loaders, index, splits, split_dir = build_experiment_loaders(
        cfg, include_sampler=True, limit_volumes=1 if args.dry_run else None
    )
    train_loader, val_loader, _ = loaders
    print(f"volumes={len(index['volumes'])} train_slices={len(train_loader.dataset)} val_slices={len(val_loader.dataset)}")
    model = build_experiment_model(cfg, pretrained=True)
    loss_fn = build_experiment_loss(cfg)
    trainer = ResearchTrainer(
        model, loss_fn, train_loader, val_loader, model_dir,
        epochs=epochs, learning_rate=training["learning_rate"],
        weight_decay=training["weight_decay"],
        mixed_precision=training["mixed_precision"],
    )
    started = time.perf_counter()
    result = trainer.fit()
    checkpoint = Path(result["best_checkpoint"])
    load_checkpoint_into_model(model, checkpoint, trainer.device)
    threshold_result = select_threshold(
        model, val_loader, trainer.device, cfg["evaluation"]["thresholds"]
    )
    write_json(run_dir / "threshold_selection.json", threshold_result)
    manifest = {
        "experiment": cfg["experiment"]["name"], "config_path": str(Path(args.config).resolve()),
        "resolved_config": cfg, "environment": environment_snapshot(),
        "split_hashes": split_hashes(split_dir), "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": sha256_file(checkpoint),
        "selected_threshold": threshold_result["selected_threshold"],
        "preprocessing_profile": "png_8bit_no_hu_no_clahe_v1",
        "runtime_seconds": time.perf_counter() - started,
        "training_result": result,
        "test_set_accessed": False,
    }
    write_json(run_dir / "training_manifest.json", manifest)
    print(f"best_checkpoint={checkpoint}")
    print(f"selected_threshold={threshold_result['selected_threshold']:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
