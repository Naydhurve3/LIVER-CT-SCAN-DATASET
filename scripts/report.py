"""
CLI for generating formatted analysis reports.

Usage:
    python scripts/report.py --type dataset --format txt
    python scripts/report.py --type model --model-name "MobileNetV2-UNet" --metrics '{"dice":0.85,"iou":0.74}'
    python scripts/report.py --type patient --volume-id 5 --format html
    python scripts/report.py --help
"""
import argparse
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from PIL import Image
from src.utils import setup_logging, logger, load_json
from src.analytics import ReportGenerator
from src.data_loader import DataPathManager, VolumeWiseSplitter


def parse_args():
    parser = argparse.ArgumentParser(description="Liver CT Report Generator")
    parser.add_argument("--type", type=str, required=True,
                        choices=["dataset", "model", "patient", "clinical"],
                        help="Report type")
    parser.add_argument("--model-name", type=str, default="Model",
                        help="Model name for model reports")
    parser.add_argument("--metrics", type=str, default=None,
                        help="JSON string of metrics (e.g. '{\"dice\":0.85,\"iou\":0.74}')")
    parser.add_argument("--metrics-file", type=str, default=None,
                        help="Path to JSON file with metrics (alternative to --metrics)")
    parser.add_argument("--volume-id", type=int, default=None,
                        help="Volume ID for patient reports")
    parser.add_argument("--format", type=str, default="txt",
                        choices=["txt", "json", "html"],
                        help="Output format")
    parser.add_argument("--splits", type=str, default="data/splits",
                        help="Splits directory")
    parser.add_argument("--output", type=str, default="outputs/reports",
                        help="Output directory")
    parser.add_argument("--metadata", type=str, default="data/metadata",
                        help="Metadata directory")
    return parser.parse_args()


def main():
    args = parse_args()
    setup_logging()
    report_gen = ReportGenerator(Path(args.output))

    if args.type == "dataset":
        stats_path = Path(args.metadata) / "phase2_statistics.json"
        if stats_path.exists():
            stats = load_json(stats_path)
        else:
            logger.info("No pre-computed stats found. Computing from data...")
            mgr = DataPathManager()
            index = mgr.build_index()
            total_volumes = len(index['volumes'])
            total_slices = sum(len(v) for v in index['image_paths'].values())
            splitter = VolumeWiseSplitter()
            splits_path = Path(args.splits) if Path(args.splits).exists() else None
            if splits_path:
                splits = splitter.load_splits(splits_path)
            else:
                splits = splitter.split(index['volumes'])
            split_dist = {}
            for name, vids in splits.items():
                n_slices = sum(len(index['image_paths'].get(v, [])) for v in vids)
                split_dist[name] = {"volumes": len(vids), "slices": n_slices}
            stats = {
                "total_volumes": total_volumes,
                "total_slices": total_slices,
                "tumor_slices_pct": "N/A",
                "class_imbalance_ratio": "N/A",
                "splits": split_dist,
                "preprocessing": {
                    "hu_low": -100, "hu_high": 400,
                    "clahe_clip": 2.0, "clahe_grid": "(8, 8)",
                    "target_size": "(256, 256)",
                },
            }
        report = report_gen.generate_dataset_report(stats)
        path = report_gen.save_report(report, "dataset_report", format=args.format)
        logger.info(f"Dataset report saved: {path}")
        print(report)

    elif args.type == "model":
        if args.metrics_file:
            metrics = json.loads(Path(args.metrics_file).read_text())
        elif args.metrics:
            metrics = json.loads(args.metrics)
        else:
            metrics = {}
        report = report_gen.generate_model_report(args.model_name, metrics)
        path = report_gen.save_report(report, f"model_report_{args.model_name.lower().replace(' ', '_')}",
                                      format=args.format)
        logger.info(f"Model report saved: {path}")
        print(report)

    elif args.type == "patient":
        if args.volume_id is None:
            logger.error("--volume-id is required for patient reports")
            return
        vid = args.volume_id
        mgr = DataPathManager()
        index = mgr.build_index()
        mask_paths = index.get('mask_paths', {}).get(vid, [])
        if mask_paths:
            from PIL import Image
            import numpy as np
            tumor_pixels = []
            has_tumor = []
            for p in mask_paths:
                m = np.array(Image.open(p).convert('L'), dtype=np.float32)
                tp = int(np.sum(m > 0.5))
                tumor_pixels.append(tp)
                has_tumor.append(tp > 0)
            from src.analytics import PatientProfileBuilder, ClinicalInsightEngine
            builder = PatientProfileBuilder()
            profile = builder.build_profile(vid, len(mask_paths), tumor_pixels, has_tumor)
            cie = ClinicalInsightEngine()
            summary = {
                "total_tumor_pixels": profile["total_tumor_pixels"],
                "tumor_positive_rate": profile["tumor_positive_rate"],
                "tumor_burden_trend": profile["tumor_burden_trend"],
            }
            risk = cie.assess_risk(summary)
            insights = {"assessment": risk}
        else:
            profile = {
                "volume_id": vid,
                "total_slices": 0, "slices_with_tumor": 0,
                "tumor_positive_rate": 0.0, "mean_coverage_pct": 0.0,
                "tumor_burden_trend": "unknown",
            }
            insights = {"assessment": {"risk_level": "unknown", "risk_factors": ["No mask data available"]}}
        report = report_gen.generate_patient_report(profile, insights)
        path = report_gen.save_report(report, f"patient_report_v{vid}", format=args.format)
        logger.info(f"Patient report saved: {path}")
        print(report)

    elif args.type == "clinical":
        report = (
            "Clinical Summary\n"
            "===============\n\n"
            "This is a placeholder for a comprehensive clinical report.\n"
            "Run with actual data to generate meaningful clinical insights."
        )
        path = report_gen.save_report(report, "clinical_summary", format=args.format)
        logger.info(f"Clinical report saved: {path}")
        print(report)


if __name__ == "__main__":
    main()
