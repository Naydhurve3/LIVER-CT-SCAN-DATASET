"""Mark an obsolete dataset build as superseded without deleting evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--build-dir", type=Path, required=True)
    parser.add_argument("--superseded-by", type=Path, required=True)
    parser.add_argument("--reason", required=True)
    args = parser.parse_args()
    build_dir = args.build_dir.resolve()
    replacement = args.superseded_by.resolve()
    marker = {
        "status": "superseded_do_not_use",
        "superseded_by": str(replacement),
        "reason": args.reason,
        "raw_evidence_preserved": True,
    }
    (build_dir / "SUPERSEDED.json").write_text(json.dumps(marker, indent=2), encoding="utf-8")
    readiness_path = build_dir / "dataset_readiness.json"
    if readiness_path.exists():
        readiness = json.loads(readiness_path.read_text(encoding="utf-8"))
        readiness.update({
            "status": "superseded_do_not_use",
            "nonspatial_eda_ready": False,
            "spatial_eda_ready": False,
            "training_ready": False,
            "superseded_by": str(replacement),
            "superseded_reason": args.reason,
        })
        readiness_path.write_text(json.dumps(readiness, indent=2), encoding="utf-8")
    print(json.dumps(marker, indent=2))


if __name__ == "__main__":
    main()
