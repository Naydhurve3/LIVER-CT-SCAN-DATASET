from __future__ import annotations

import argparse
import json
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from src.framework.core.reproducibility import set_seed
from src.framework.evaluation.checkpoints import (
    environment_snapshot, sha256_file, split_hashes, write_json,
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
    parser.add_argument("--epochs", type=int, help="Total target epochs, including resumed epochs")
    parser.add_argument("--output-dir", help="Model/checkpoint directory override")
    parser.add_argument("--run-dir", help="Manifest/result directory override")
    parser.add_argument("--resume", help="Checkpoint to resume from")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _resolve_directories(cfg, args):
    model_dir = Path(args.output_dir or cfg["outputs"]["model_dir"])
    configured_run_dir = Path(args.run_dir or cfg["outputs"]["run_dir"])
    if args.dry_run:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_dir = model_dir / "dry_run" / stamp
        run_dir = configured_run_dir / "dry_run" / stamp
    else:
        run_dir = configured_run_dir
    return model_dir, run_dir


def _prepare_manifest(run_dir: Path, cfg, args, model_dir: Path) -> tuple[Path, dict]:
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / "training_manifest.json"
    existing = None
    resolved_model_dir = model_dir.resolve()
    resume_path = Path(args.resume).resolve() if args.resume else None
    if not args.resume and any(model_dir.glob("*_checkpoint.pth")):
        raise FileExistsError(
            f"Checkpoint files already exist in {model_dir}. Resume them or choose --output-dir."
        )
    if resume_path is not None and resume_path.parent != resolved_model_dir:
        raise ValueError(
            f"Resume checkpoint must belong to the selected model directory: {resolved_model_dir}"
        )
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if not args.resume:
            raise FileExistsError(
                f"Run manifest already exists: {path}. Resume it or choose --run-dir."
            )
        if existing.get("experiment") != cfg["experiment"]["name"]:
            raise ValueError("Existing run manifest belongs to a different experiment")
        if Path(existing["model_dir"]).resolve() != resolved_model_dir:
            raise ValueError("Existing run manifest points to a different model directory")
    manifest = existing or {
        "manifest_id": str(uuid.uuid4()),
        "experiment": cfg["experiment"]["name"],
        "created_at": _utc_now(),
    }
    manifest.update({
        "status": "running", "updated_at": _utc_now(),
        "config_path": str(Path(args.config).resolve()), "resolved_config": cfg,
        "model_dir": str(resolved_model_dir), "run_dir": str(run_dir.resolve()),
        "resume_from": str(resume_path) if resume_path else None,
        "test_set_accessed": False,
    })
    write_json(path, manifest)
    return path, manifest


def _update_manifest(path: Path, manifest: dict, **updates) -> None:
    manifest.update(updates)
    manifest["updated_at"] = _utc_now()
    write_json(path, manifest)


def main() -> int:
    args = parse_args()
    cfg = load_experiment(args.config, args.images_dir, args.masks_dir)
    set_seed(cfg["experiment"].get("seed", 42))
    training = cfg["training"]
    target_epochs = 1 if args.dry_run else (args.epochs or training["epochs"])
    schedule_epochs = training["epochs"]
    model_dir, run_dir = _resolve_directories(cfg, args)
    manifest_path, manifest = _prepare_manifest(run_dir, cfg, args, model_dir)
    started = time.perf_counter()
    _update_manifest(
        manifest_path, manifest, target_epochs=target_epochs,
        schedule_epochs=schedule_epochs, environment=environment_snapshot(),
        preprocessing_profile="png_8bit_no_hu_no_clahe_v1",
    )
    print(f"experiment={cfg['experiment']['name']} device={'cuda' if torch.cuda.is_available() else 'cpu'}")
    try:
        loaders, index, splits, split_dir = build_experiment_loaders(
            cfg, include_sampler=True,
            limit_volumes=1 if args.dry_run else None,
            limit_slices=1 if args.dry_run else None,
        )
        train_loader, val_loader, _ = loaders
        _update_manifest(manifest_path, manifest, split_hashes=split_hashes(split_dir))
        print(
            f"volumes={len(index['volumes'])} train_slices={len(train_loader.dataset)} "
            f"val_slices={len(val_loader.dataset)}"
        )
        model = build_experiment_model(cfg, pretrained=not bool(args.resume))
        loss_fn = build_experiment_loss(cfg)
        trainer = ResearchTrainer(
            model, loss_fn, train_loader, val_loader, model_dir,
            epochs=target_epochs, schedule_epochs=schedule_epochs,
            learning_rate=training["learning_rate"],
            weight_decay=training["weight_decay"],
            mixed_precision=training["mixed_precision"],
        )
        resume_info = trainer.resume(args.resume) if args.resume else None
        if resume_info and resume_info["start_epoch"] > target_epochs:
            raise ValueError(
                f"Checkpoint already completed {resume_info['start_epoch']} epochs, "
                f"beyond target {target_epochs}"
            )
        _update_manifest(manifest_path, manifest, resume=resume_info)
        result = trainer.fit()
        if result["status"] == "interrupted":
            checkpoint = Path(result["resume_checkpoint"])
            _update_manifest(
                manifest_path, manifest, status="interrupted",
                runtime_seconds=time.perf_counter() - started,
                interrupted_checkpoint=str(checkpoint.resolve()),
                interrupted_checkpoint_sha256=sha256_file(checkpoint),
                training_result=result,
            )
            print(f"Training interrupted safely. Resume from: {checkpoint}")
            return 130

        checkpoint = Path(result["best_checkpoint"])
        if not checkpoint.exists():
            checkpoint = Path(result.get("last_checkpoint") or checkpoint)
        threshold_result = select_threshold(
            model, val_loader, trainer.device, cfg["evaluation"]["thresholds"]
        )
        write_json(run_dir / "threshold_selection.json", threshold_result)
        _update_manifest(
            manifest_path, manifest, status="completed", completed_at=_utc_now(),
            runtime_seconds=time.perf_counter() - started,
            checkpoint=str(checkpoint.resolve()),
            checkpoint_sha256=sha256_file(checkpoint),
            selected_threshold=threshold_result["selected_threshold"],
            training_result=result,
        )
        print(f"best_checkpoint={checkpoint}")
        print(f"selected_threshold={threshold_result['selected_threshold']:.2f}")
        return 0
    except KeyboardInterrupt:
        _update_manifest(
            manifest_path, manifest, status="interrupted",
            runtime_seconds=time.perf_counter() - started,
            interruption="before trainer checkpoint lifecycle started",
        )
        print("Interrupted before a resumable trainer checkpoint was available.")
        return 130
    except Exception as exc:
        _update_manifest(
            manifest_path, manifest, status="failed",
            runtime_seconds=time.perf_counter() - started,
            error=f"{type(exc).__name__}: {exc}",
        )
        raise


if __name__ == "__main__":
    raise SystemExit(main())
