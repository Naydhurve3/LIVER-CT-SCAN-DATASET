"""S19: Ablation runner — vary one hyperparameter at a time, compare metrics.

Usage:
    python scripts/ablation.py --dry-run          # Print commands without running
    python scripts/ablation.py --quick             # Run only 5-epoch ablations
    python scripts/ablation.py --full              # Run all ablations (takes days)
"""
import sys, json, subprocess, argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from src.utils import setup_logging, logger

OUT = Path(__file__).resolve().parent.parent / "results" / "ablation"
OUT.mkdir(parents=True, exist_ok=True)

TRAIN = [sys.executable, str(Path(__file__).resolve().parent / "train.py")]
BASE_ARGS = ["--epochs", "5", "--batch-size", "8", "--seed", "42"]

ABLATIONS = {
    # (name, extra_args, description)
    "baseline": ([], "Baseline (pretrained MobileNetV2-UNet, BCE+Dice, lr=1e-3)"),
    "no_pretrain": (["--no-pretrained"], "Scratch init (no ImageNet pretraining)"),
    "lr_1e-4": (["--lr", "1e-4"], "Lower learning rate"),
    "lr_1e-2": (["--lr", "1e-2"], "Higher learning rate"),
    "bs_4": (["--batch-size", "4"], "Smaller batch size"),
    "bs_16": (["--batch-size", "16"], "Larger batch size (requires >4GB GPU)"),
    "no_aug": ([], "No augmentation (TODO: add --no-aug flag)"),
    "warmup_5": ([], "5-epoch warmup (TODO: add --warmup-epochs flag)"),
}


def run():
    parser = argparse.ArgumentParser(description="Ablation study runner")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running")
    parser.add_argument("--quick", action="store_true", help="Run only 5-epoch ablations")
    parser.add_argument("--full", action="store_true", help="Run all ablations (days)")
    args = parser.parse_args()

    results = []
    for name, (extra_args, desc) in ABLATIONS.items():
        cmd = TRAIN + BASE_ARGS + extra_args + ["--output-dir", str(OUT / name)]
        log_path = OUT / name / "train.log"

        print(f"\n{'=' * 60}")
        print(f"Ablation: {name}")
        print(f"  {desc}")
        print(f"  Command: {' '.join(cmd)}")
        print(f"  Log: {log_path}")

        if not args.dry_run:
            (OUT / name).mkdir(parents=True, exist_ok=True)
            logger.info(f"Running ablation: {name}")
            result = subprocess.run(cmd, capture_output=True, text=True)
            log_path.write_text(result.stdout + "\n" + result.stderr)
            # Parse final validation dice from log
            for line in result.stdout.split("\n") + result.stderr.split("\n"):
                if "Validation" in line and "dice" in line.lower():
                    results.append({"name": name, "log_line": line.strip()})
            logger.info(f"  Exit code: {result.returncode}")
        else:
            results.append({"name": name, "command": " ".join(cmd)})

    summary_path = OUT / "summary.json"
    summary_path.write_text(json.dumps(results, indent=2))
    print(f"\nAblation summary saved to {summary_path}")
    print("\nTo view results:")
    print("  for d in results/ablation/*/; do echo \"=== $d ===\"; cat \"$d/history.json\" 2>/dev/null | python -c \"import sys,json; h=json.load(sys.stdin); print('Best val_dice:', max(h.get('val_dice', ['N/A'])))\"; done")


if __name__ == "__main__":
    setup_logging()
    run()
