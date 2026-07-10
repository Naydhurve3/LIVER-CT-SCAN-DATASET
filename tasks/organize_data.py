"""File operations: organize dataset, move/copy files between directories."""
import shutil, json, re
from pathlib import Path

IMG_SRC = Path("D:/DATA SCIENCE AND ANALYTICS/Dataset/Liver Img Dataset")
MASK_SRC = Path("D:/DATA SCIENCE AND ANALYTICS/Dataset/LiTS_masks")
PROJ = Path(__file__).resolve().parent.parent


def link_to_project(force=False):
    """Create symlinks/junctions to dataset inside project (avoids duplication)."""
    dest_img = PROJ / "data" / "images"
    dest_mask = PROJ / "data" / "masks"
    if not force and (dest_img.exists() or dest_mask.exists()):
        print("Data links already exist. Use force=True to recreate.")
        return
    dest_img.parent.mkdir(parents=True, exist_ok=True)
    try:
        dest_img.symlink_to(IMG_SRC, target_is_directory=True)
        dest_mask.symlink_to(MASK_SRC, target_is_directory=True)
        print(f"Linked {IMG_SRC} -> {dest_img}")
        print(f"Linked {MASK_SRC} -> {dest_mask}")
    except OSError as e:
        print(f"Symlink failed (Windows may need admin): {e}")
        print("Tip: run as Admin or use: mklink /J data\\images \"D:\\...\\Liver Img Dataset\"")


def _parse_id(name):
    """Extract (volume, slice) tuple from Volume-XXX-YYY.png or mask-XXX-YYY.png."""
    m = re.match(r"(?:Volume|mask)-(\d+)-(\d+)\.png", name)
    return (int(m.group(1)), int(m.group(2))) if m else None


def count_files():
    """Count images and masks to verify dataset integrity."""
    imgs = list(IMG_SRC.glob("*.png"))
    masks = list(MASK_SRC.glob("*.png"))
    img_ids = {_parse_id(p.name) for p in imgs}
    mask_ids = {_parse_id(p.name) for p in masks}
    print(f"Images: {len(imgs)}")
    print(f"Masks:  {len(masks)}")
    print(f"Matched: {len(img_ids & mask_ids)}")
    print(f"Missing masks: {len(img_ids - mask_ids)}")
    print(f"Extra masks: {len(mask_ids - img_ids)}")
    return {"images": len(imgs), "masks": len(masks), "matched": len(img_ids & mask_ids)}


def export_file_list(output="data/file_list.json"):
    """Export full list of image/mask paths for fast loading."""
    imgs = sorted(IMG_SRC.glob("*.png"))
    masks = sorted(MASK_SRC.glob("*.png"))
    data = {
        "image_dir": str(IMG_SRC),
        "mask_dir": str(MASK_SRC),
        "images": [p.name for p in imgs],
        "masks": [p.name for p in masks],
    }
    path = PROJ / output
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2))
    print(f"Saved file list ({len(imgs)} images, {len(masks)} masks) to {path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--link", action="store_true", help="Create symlinks to dataset")
    parser.add_argument("--count", action="store_true", help="Count and verify files")
    parser.add_argument("--export", action="store_true", help="Export file list JSON")
    args = parser.parse_args()

    if args.link:
        link_to_project()
    if args.count:
        count_files()
    if args.export:
        export_file_list()
    if not any([args.link, args.count, args.export]):
        parser.print_help()
