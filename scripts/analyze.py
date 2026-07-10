"""
CLI for running data analysis and generating insights.

Usage:
    python scripts/analyze.py --mode dataset
    python scripts/analyze.py --mode tumor-burden --volume-id 5
    python scripts/analyze.py --mode cohort
    python scripts/analyze.py --help
"""
import argparse
import sys
from pathlib import Path
import json

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from PIL import Image
from tqdm import tqdm
from src.utils import setup_logging, logger, load_json
from src.data_loader import DataPathManager, VolumeWiseSplitter, DatasetConfig
from src.analytics import TumorBurdenAnalyzer, PatientProfileBuilder, ClinicalInsightEngine, ReportGenerator
from src.quality import DataQualityChecker


def parse_args():
    parser = argparse.ArgumentParser(description="Liver CT Data Analysis CLI")
    parser.add_argument("--mode", type=str, required=True,
                        choices=["dataset", "tumor-burden", "cohort", "quality"],
                        help="Analysis mode")
    parser.add_argument("--volume-id", type=int, default=None,
                        help="Volume ID for per-volume analysis")
    parser.add_argument("--splits", type=str, default="data/splits",
                        help="Splits directory")
    parser.add_argument("--metadata", type=str, default="data/metadata",
                        help="Metadata directory")
    parser.add_argument("--output", type=str, default="outputs/reports",
                        help="Output directory")
    parser.add_argument("--format", type=str, default="txt", choices=["txt", "json"],
                        help="Output format")
    parser.add_argument("--sample-size", type=int, default=200,
                        help="Number of files to sample for dataset/quality modes")
    return parser.parse_args()


def compute_dataset_stats(index: dict, args) -> dict:
    """Compute dataset statistics directly from the volume index."""
    total_volumes = len(index['volumes'])
    total_slices = sum(len(v) for v in index['image_paths'].values())

    splitter = VolumeWiseSplitter()
    splits_path = Path(args.splits)
    if splits_path.exists():
        splits = splitter.load_splits(splits_path)
    else:
        splits = splitter.split(index['volumes'])

    split_dist = {}
    for name, vids in splits.items():
        n_slices = sum(len(index['image_paths'].get(v, [])) for v in vids)
        split_dist[name] = {"volumes": len(vids), "slices": n_slices}

    tumor_slices = 0
    total_tumor_px = 0
    total_px = 0
    sampled = 0
    for vid in index['volumes']:
        mask_paths = index['mask_paths'].get(vid, [])
        for p in mask_paths:
            m = np.array(Image.open(p).convert('L'))
            tp = int(np.sum(m > 0.5))
            if tp > 0:
                tumor_slices += 1
            total_tumor_px += tp
            total_px += m.size
            sampled += 1
            if sampled >= args.sample_size:
                break
        if sampled >= args.sample_size:
            break

    im_ratio = (total_px - total_tumor_px) / max(total_tumor_px, 1)

    stats = {
        "total_volumes": total_volumes,
        "total_slices": total_slices,
        "tumor_slices_pct": round(tumor_slices / max(sampled, 1) * 100, 1),
        "class_imbalance_ratio": round(im_ratio, 1),
        "splits": split_dist,
        "preprocessing": {
            "hu_low": -100, "hu_high": 400,
            "clahe_clip": 2.0, "clahe_grid": "(8, 8)",
            "target_size": "(256, 256)",
        },
    }
    return stats


def main():
    args = parse_args()
    setup_logging()
    report_gen = ReportGenerator(Path(args.output))

    if args.mode == "dataset":
        stats_path = Path(args.metadata) / "phase2_statistics.json"
        if stats_path.exists():
            stats = load_json(stats_path)
        else:
            logger.info("No pre-computed stats found. Computing from data...")
            mgr = DataPathManager()
            index = mgr.build_index()
            stats = compute_dataset_stats(index, args)
        report = report_gen.generate_dataset_report(stats)
        path = report_gen.save_report(report, "dataset_analysis", format=args.format)
        logger.info(f"Dataset report saved to {path}")
        print(report)

    elif args.mode == "tumor-burden":
        if args.volume_id is None:
            logger.error("--volume-id is required for tumor-burden mode")
            return
        analyzer = TumorBurdenAnalyzer()
        mgr = DataPathManager()
        index = mgr.build_index()
        vid = args.volume_id
        if vid in index.get('image_paths', {}):
            mask_paths = index.get('mask_paths', {}).get(vid, [])
            if mask_paths:
                masks_3d = []
                for p in mask_paths:
                    m = Image.open(p).convert('L')
                    masks_3d.append(np.array(m, dtype=np.float32))
                full_mask = np.stack(masks_3d)
                summary = analyzer.summary(full_mask, volume_id=vid)
                cie = ClinicalInsightEngine()
                report_txt = cie.generate_report(vid, summary)
                report_gen.save_report(report_txt, f"tumor_burden_v{vid}", format=args.format)
                report_gen.save_report(json.dumps(summary, indent=2),
                                       f"tumor_burden_v{vid}", format="json")
                logger.info(f"Tumor burden analysis for volume {vid} saved")
                print(report_txt)
            else:
                logger.error(f"No mask data found for volume {vid}")
        else:
            logger.error(f"Volume {vid} not found in index")

    elif args.mode == "cohort":
        splits_path = Path(args.splits)
        builder = PatientProfileBuilder()
        logger.info(f"Generating cohort summary from {splits_path}")
        mgr = DataPathManager()
        index = mgr.build_index()
        splitter = VolumeWiseSplitter()
        splits = splitter.load_splits(splits_path)
        profiles = []
        for split_name, vids in splits.items():
            for vid in tqdm(vids, desc=f"Processing {split_name}"):
                mask_paths = index.get('mask_paths', {}).get(vid, [])
                if not mask_paths:
                    continue
                tumor_pixels = []
                has_tumor = []
                for p in mask_paths:
                    m = np.array(Image.open(p).convert('L'), dtype=np.float32)
                    tp = int(np.sum(m > 0.5))
                    tumor_pixels.append(tp)
                    has_tumor.append(tp > 0)
                profile = builder.build_profile(vid, len(mask_paths), tumor_pixels, has_tumor)
                profiles.append(profile)
        report = builder.cohort_summary(profiles)
        report_gen.save_report(json.dumps(report, indent=2),
                               "cohort_summary", format="json")
        logger.info(f"Cohort summary generated from {len(profiles)} patients")
        print(json.dumps(report, indent=2))

    elif args.mode == "quality":
        mgr = DataPathManager()
        index = mgr.build_index()
        qc = DataQualityChecker()

        count = 0
        sample_size = args.sample_size
        for vid in index['volumes']:
            for p in index['image_paths'].get(vid, []):
                qc.check_image_integrity(p)
                count += 1
                if count >= sample_size:
                    break
            if count >= sample_size:
                break
        logger.info(f"Checked {count} images for integrity")

        count = 0
        for vid in index['volumes']:
            for p in index['mask_paths'].get(vid, []):
                qc.check_mask_integrity(p)
                count += 1
                if count >= sample_size:
                    break
            if count >= sample_size:
                break
        logger.info(f"Checked {count} masks for integrity")

        result = qc.summary_report()
        out = json.dumps(result, indent=2)
        report_gen.save_report(out, "quality_report", format="json")
        logger.info(f"Quality check complete. Issues found: {result['total_checks']}")
        print(out)


if __name__ == "__main__":
    main()
