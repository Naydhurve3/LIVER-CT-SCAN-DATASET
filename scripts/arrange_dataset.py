"""
LiTS Dataset Canonical Arrangement Script
Arranges source data into standardized canonical structure at D:\DATA SCIENCE AND ANALYTICS\Dataset\Liver
"""
import os, sys, csv, json, shutil, hashlib, re, time
from pathlib import Path
from collections import defaultdict
from PIL import Image
import numpy as np

SRC_LITS_PNG = Path(r"D:\DATA SCIENCE AND ANALYTICS\Dataset\lits-png\dataset_6\dataset_6")
SRC_MASKS = Path(r"D:\DATA SCIENCE AND ANALYTICS\Dataset\LiTS_masks")
SRC_LEGACY = Path(r"D:\DATA SCIENCE AND ANALYTICS\Dataset\Liver Img Dataset")
SRC_CSV = Path(r"D:\DATA SCIENCE AND ANALYTICS\Dataset\lits-png\lits_df.csv")
DST = Path(r"D:\DATA SCIENCE AND ANALYTICS\Dataset\Liver")

DRY_RUN = "--dry-run" in sys.argv

print("=== LiTS Dataset Canonical Arrangement ===")
print(f"Source lits-png: {SRC_LITS_PNG}")
print(f"Source LiTS_masks: {SRC_MASKS}")
print(f"Destination: {DST}")
if DRY_RUN:
    print("*** DRY RUN - No files will be copied ***")

# ---- Helper functions ----
def safe_copy(src, dst):
    if DRY_RUN:
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(str(src), str(dst))

def file_sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(65536), b''):
            h.update(chunk)
    return h.hexdigest()

def parse_lits_png_name(fname):
    m = re.match(r'volume-(\d+)_(\d+)\.png', fname)
    if m:
        return int(m.group(1)), int(m.group(2)), 'image'
    m = re.match(r'segmentation-(\d+)_livermask_(\d+)\.png', fname)
    if m:
        return int(m.group(1)), int(m.group(2)), 'liver'
    m = re.match(r'segmentation-(\d+)_lesionmask_(\d+)\.png', fname)
    if m:
        return int(m.group(1)), int(m.group(2)), 'lesion'
    return None

def parse_mask_name(fname):
    m = re.match(r'mask-(\d+)-(\d+)\.png', fname)
    if m:
        return int(m.group(1)), int(m.group(2))
    return None

# ---- Phase 0: Create directories ----
dirs = [
    "00_source_registry", "01_raw_authoritative",
    "02_staging/images", "02_staging/liver_masks", "02_staging/tumor_masks",
    "03_derived_256/images", "03_derived_256/liver_masks", "03_derived_256/tumor_masks",
    "04_manifests", "05_splits",
    "06_audits/overlays", "06_audits/positive_cases", "06_audits/negative_cases",
    "07_training_cache", "99_quarantine",
]
for d in dirs:
    (DST / d).mkdir(parents=True, exist_ok=True)
print("\n[Phase 0] Directory structure ready")

# ---- Phase 1: Source inventory ----
print("\n[Phase 1] Building source inventory...")
lits_png_files = list(SRC_LITS_PNG.glob("volume-*.png"))
lits_png_liver = list(SRC_LITS_PNG.glob("*livermask*.png"))
lits_png_lesion = list(SRC_LITS_PNG.glob("*lesionmask*.png"))
mask_files = list(SRC_MASKS.glob("*.png"))
legacy_files = list(SRC_LEGACY.glob("*.png"))

reg = {
    "created": time.strftime("%Y-%m-%d %H:%M:%S"),
    "sources": [
        {"name": "lits-png", "path": str(SRC_LITS_PNG), "images": len(lits_png_files),
         "liver_masks": len(lits_png_liver), "lesion_masks": len(lits_png_lesion)},
        {"name": "LiTS_masks", "path": str(SRC_MASKS), "total_masks": len(mask_files)},
        {"name": "Liver Img Dataset", "path": str(SRC_LEGACY), "total_images": len(legacy_files),
         "note": "Legacy 512x512 - spatial alignment unverified for segmentation"},
    ],
    "lits_png_images": len(lits_png_files),
    "lits_png_liver_masks": len(lits_png_liver),
    "lits_png_lesion_masks": len(lits_png_lesion),
    "lits_masks_total": len(mask_files),
    "legacy_images_total": len(legacy_files),
}
if not DRY_RUN:
    with open(DST / "00_source_registry" / "source_registry.json", "w") as f:
        json.dump(reg, f, indent=2)
print(f"  lits-png images: {len(lits_png_files)}")
print(f"  lits-png liver masks: {len(lits_png_liver)}")
print(f"  lits-png lesion masks: {len(lits_png_lesion)}")
print(f"  LiTS_masks: {len(mask_files)}")
print(f"  Legacy images: {len(legacy_files)}")

# ---- Phase 2: Organize images ----
print("\n[Phase 2] Organizing images into 03_derived_256/images/v{XXX}/...")
img_dir = DST / "03_derived_256" / "images"
vol_images = defaultdict(list)
count_images = 0
for f in lits_png_files:
    parsed = parse_lits_png_name(f.name)
    if parsed and parsed[2] == 'image':
        vol, slc = parsed[0], parsed[1]
        vol_key = f"v{vol:03d}"
        slice_file = f"s{slc:04d}.png"
        dst_file = img_dir / vol_key / slice_file
        safe_copy(f, dst_file)
        vol_images[vol].append(slc)
        count_images += 1
print(f"  Organized {count_images} images across {len(vol_images)} volumes")

# ---- Phase 3: Organize tumor masks from LiTS_masks ----
print("\n[Phase 3] Organizing tumor masks into 03_derived_256/tumor_masks/...")
tumor_dir = DST / "03_derived_256" / "tumor_masks"
vol_tumor_masks = defaultdict(list)
count_tumor = 0
for f in mask_files:
    parsed = parse_mask_name(f.name)
    if parsed:
        vol, slc = parsed
        vol_key = f"v{vol:03d}"
        slice_file = f"s{slc:04d}.png"
        dst_file = tumor_dir / vol_key / slice_file
        safe_copy(f, dst_file)
        vol_tumor_masks[vol].append(slc)
        count_tumor += 1
print(f"  Organized {count_tumor} tumor masks across {len(vol_tumor_masks)} volumes")

# ---- Phase 4: Organize liver masks ----
print("\n[Phase 4] Organizing liver masks into 03_derived_256/liver_masks/...")
liver_dir = DST / "03_derived_256" / "liver_masks"
vol_liver_masks = defaultdict(list)
count_liver = 0
for f in lits_png_liver:
    parsed = parse_lits_png_name(f.name)
    if parsed and parsed[2] == 'liver':
        vol, slc = parsed[0], parsed[1]
        vol_key = f"v{vol:03d}"
        slice_file = f"s{slc:04d}.png"
        dst_file = liver_dir / vol_key / slice_file
        safe_copy(f, dst_file)
        vol_liver_masks[vol].append(slc)
        count_liver += 1
print(f"  Organized {count_liver} liver masks across {len(vol_liver_masks)} volumes")
print(f"  Liver mask volumes: {sorted(vol_liver_masks.keys())}")

# ---- Phase 5: Build slice manifest ----
print("\n[Phase 5] Building slice manifest...")

# Load CSV to determine tumor presence
csv_data = {}
with open(SRC_CSV, newline='') as f:
    reader = csv.DictReader(f)
    for row in reader:
        m = re.search(r'volume-(\d+)_(\d+)\.png', row['filepath'])
        if m:
            vol, slc = int(m.group(1)), int(m.group(2))
            # tumor_mask_empty=False means tumor IS present (misleading name)
            tumor_present = row['tumor_mask_empty'].strip().lower() == 'false'
            csv_data[(vol, slc)] = {
                'tumor_mask_empty': row['tumor_mask_empty'],
                'tumor_present': tumor_present,
            }

print(f"  Loaded {len(csv_data)} rows from CSV")

manifest = []
missing_liver = 0
pos_slices = 0
neg_slices = 0

# Build manifest from image list (complete set)
for vol in sorted(vol_images.keys()):
    for slc in sorted(vol_images[vol]):
        sample_id = f"v{vol:03d}_s{slc:04d}"
        img_path = f"03_derived_256/images/v{vol:03d}/s{slc:04d}.png"
        tumor_path = f"03_derived_256/tumor_masks/v{vol:03d}/s{slc:04d}.png"
        liver_path = f"03_derived_256/liver_masks/v{vol:03d}/s{slc:04d}.png"

        has_tumor = slc in vol_tumor_masks.get(vol, [])
        has_liver = slc in vol_liver_masks.get(vol, [])
        tumor_present = csv_data.get((vol, slc), {}).get('tumor_present', False)

        if not has_liver:
            missing_liver += 1
        if tumor_present:
            pos_slices += 1
        else:
            neg_slices += 1

        manifest.append({
            'sample_id': sample_id,
            'volume_id': vol,
            'slice_index': slc,
            'source_package': 'lits-png',
            'has_image': True,
            'has_tumor_mask': has_tumor,
            'has_liver_mask': has_liver,
            'image_path': img_path,
            'tumor_mask_path': tumor_path,
            'liver_mask_path': liver_path,
            'tumor_present': str(tumor_present),
        })

print(f"  Total slices: {len(manifest)}")
print(f"  Tumor-positive: {pos_slices}, Tumor-negative: {neg_slices}")
print(f"  Missing liver masks: {missing_liver}")

if not DRY_RUN:
    with open(DST / "04_manifests" / "slice_manifest.csv", "w", newline='') as f:
        w = csv.DictWriter(f, fieldnames=manifest[0].keys())
        w.writeheader()
        w.writerows(manifest)
    print("  Written: slice_manifest.csv")

# ---- Phase 6: Build volume manifest ----
print("\n[Phase 6] Building volume manifest...")
volume_manifest = []
total_tumor_pixels = 0

for vol in sorted(vol_images.keys()):
    slices = sorted(vol_images[vol])
    vol_pos = 0
    vol_neg = 0
    vol_tumor_px = 0
    for slc in slices:
        tp = csv_data.get((vol, slc), {}).get('tumor_present', False)
        if tp:
            vol_pos += 1
            # Count tumor pixels from mask
            mask_path = tumor_dir / f"v{vol:03d}" / f"s{slc:04d}.png"
            if mask_path.exists() and not DRY_RUN:
                try:
                    img = Image.open(mask_path)
                    arr = np.array(img)
                    vol_tumor_px += int(np.sum(arr > 0) // 3)  # RGB, all channels same
                except:
                    pass
        else:
            vol_neg += 1
    total_tumor_pixels += vol_tumor_px

    liver_count = len(vol_liver_masks.get(vol, []))
    volume_manifest.append({
        'volume_id': vol,
        'slices': len(slices),
        'tumor_positive_slices': vol_pos,
        'tumor_negative_slices': vol_neg,
        'tumor_pixel_count': vol_tumor_px,
        'liver_masks_available': liver_count,
    })

print(f"  {len(volume_manifest)} volumes recorded")
print(f"  Total tumor pixels: {total_tumor_pixels}")

if not DRY_RUN:
    with open(DST / "04_manifests" / "volume_manifest.csv", "w", newline='') as f:
        w = csv.DictWriter(f, fieldnames=volume_manifest[0].keys())
        w.writeheader()
        w.writerows(volume_manifest)
    print("  Written: volume_manifest.csv")

# ---- Phase 7: Generate splits ----
print("\n[Phase 7] Generating volume-wise splits...")
split_dir = DST / "05_splits"

train_vols = list(range(0, 104))
val_vols = list(range(104, 117))
test_vols = list(range(117, 131))

def slices_for_vols(vol_list):
    result = []
    for m in manifest:
        if m['volume_id'] in vol_list:
            result.append(m)
    return result

train_slices = slices_for_vols(train_vols)
val_slices = slices_for_vols(val_vols)
test_slices = slices_for_vols(test_vols)

print(f"  Train: {len(train_slices)} slices (volumes 0-103)")
print(f"  Val:   {len(val_slices)} slices (volumes 104-116)")
print(f"  Test:  {len(test_slices)} slices (volumes 117-130)")
print(f"  Total: {len(train_slices)+len(val_slices)+len(test_slices)} (expected 58638)")

if not DRY_RUN:
    # Volume lists
    with open(split_dir / "train_volumes.txt", "w") as f:
        f.write("\n".join(f"v{v:03d}" for v in train_vols))
    with open(split_dir / "val_volumes.txt", "w") as f:
        f.write("\n".join(f"v{v:03d}" for v in val_vols))
    with open(split_dir / "test_volumes.txt", "w") as f:
        f.write("\n".join(f"v{v:03d}" for v in test_vols))

    # Slice CSVs
    for name, slices in [("train_slices.csv", train_slices),
                          ("val_slices.csv", val_slices),
                          ("test_slices.csv", test_slices)]:
        with open(split_dir / name, "w", newline='') as f:
            if slices:
                w = csv.DictWriter(f, fieldnames=slices[0].keys())
                w.writeheader()
                w.writerows(slices)
    print("  Written: train/val/test volume and slice files")

# ---- Phase 8: Dataset version info ----
print("\n[Phase 8] Writing dataset version JSON...")
ver = {
    "dataset": "lits-canonical-v1.0.0",
    "created": time.strftime("%Y-%m-%d %H:%M:%S"),
    "total_slices": len(manifest),
    "total_volumes": len(vol_images),
    "positive_slices": pos_slices,
    "negative_slices": neg_slices,
    "total_tumor_pixels": total_tumor_pixels,
    "sources": [
        {"name": "lits-png", "images": len(lits_png_files),
         "liver_masks": len(lits_png_liver), "lesion_masks_available": len(lits_png_lesion)},
        {"name": "LiTS_masks", "masks": len(mask_files)},
    ],
    "split": {
        "train": {"volumes": "0-103", "slices": len(train_slices)},
        "val": {"volumes": "104-116", "slices": len(val_slices)},
        "test": {"volumes": "117-130", "slices": len(test_slices)},
    },
    "warnings": [
        "Spatial alignment proven only for volumes 7,8,9,78-99 (byte-level mask match)",
        "Liver masks only available for 25 volumes (7,8,9,78-99)",
        "Do NOT use Liver Img Dataset as segmentation image source",
        "Training BLOCKED until spatial audit is passed",
    ],
    "recommended_next_action": "Run spatial audit to verify image-mask alignment before training",
}

if not DRY_RUN:
    with open(DST / "04_manifests" / "dataset_version.json", "w") as f:
        json.dump(ver, f, indent=2)
    print("  Written: dataset_version.json")

# ---- Summary ----
print(f"""
{'='*60}
Dataset Arrangement Complete
{'='*60}
Destination: {DST}

Structure:
  Liver/
  +-- 00_source_registry/
  |   +-- source_registry.json
  +-- 03_derived_256/
  |   +-- images/           ({count_images} images, {len(vol_images)} volumes)
  |   +-- tumor_masks/      ({count_tumor} masks, {len(vol_tumor_masks)} volumes)
  |   +-- liver_masks/      ({count_liver} masks, {len(vol_liver_masks)} volumes)
  +-- 04_manifests/
  |   +-- slice_manifest.csv
  |   +-- volume_manifest.csv
  |   +-- dataset_version.json
  +-- 05_splits/
      +-- train_volumes.txt, val_volumes.txt, test_volumes.txt
      +-- train_slices.csv, val_slices.csv, test_slices.csv
{'='*60}
""")

if DRY_RUN:
    print("*** DRY RUN - No files were copied ***")
