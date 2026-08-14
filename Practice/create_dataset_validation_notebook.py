from pathlib import Path

import nbformat as nbf


OUT = Path(__file__).with_name("dataset_validation_and_promotion.ipynb")


def code(source: str):
    return nbf.v4.new_code_cell(source.strip() + "\n")


cells = [
    nbf.v4.new_markdown_cell(
        """# LiTS staged-build validation, spatial correction, and promotion

This notebook validates the existing staged build without rebuilding it. It corrects the invalid spatial-review method by overlaying the **saved 256x256 image and saved 256x256 masks in the same coordinate system**.

Run it manually. The first run is read-only except for creating new files named `*_review_corrected.png`. Promotion and metadata changes remain disabled until you explicitly edit the switches in Cell 1.

The notebook must not be used to approve a visibly misaligned volume. An unresolved or rejected review keeps `DATASET_READY=False` and `TRAINING_BLOCKED=True`."""
    ),
    code(
        r'''
# CELL 1 - Parameters and manual decisions
from pathlib import Path
from collections import Counter
from io import BytesIO
import csv, datetime, hashlib, json, os, shutil

import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt

DATASET_ROOT = Path(r"D:\DATA SCIENCE AND ANALYTICS\Dataset\Liver")
BUILD_DIR = DATASET_ROOT / "02_staging" / "build_20260713_112738"
PROJECT_ROOT = Path(r"D:\DATA SCIENCE AND ANALYTICS\PROJECTS\Liver")

EXPECTED_VOLUME_IDS = set(range(131))
TRAIN_VOLUMES = set(range(0, 104))
VAL_VOLUMES = set(range(104, 117))
TEST_VOLUMES = set(range(117, 131))
EXPECTED_SIZE = (256, 256)

# Read-only checks are enabled by default.
FULL_FILE_AUDIT = True
VERIFY_SOURCE_HASHES = False       # Set True before final approval; this can take several minutes.
GENERATE_REVIEW_FIGURES = True     # Creates *_review_corrected.png; does not overwrite old figures.
GENERATE_DERIVED_HASHES = False    # Set True before final approval; this can take several minutes.

# Destructive/state-changing actions are disabled by default.
WRITE_CORRECTIONS = False          # Set True only after reviewing every required corrected figure.
PROMOTE = False                    # Set True only after DATASET_READY=True.

# After inspecting corrected figures, enter every approved high-risk volume ID here.
# Do not approve a volume whose contour does not follow the liver/tumor anatomy.
MANUAL_APPROVED_VOLUMES = []
MANUAL_REJECTED_VOLUMES = []
MANUAL_REVIEWER = "replace_with_reviewer_name"

print("BUILD_DIR =", BUILD_DIR)
print("FULL_FILE_AUDIT =", FULL_FILE_AUDIT)
print("VERIFY_SOURCE_HASHES =", VERIFY_SOURCE_HASHES)
print("GENERATE_DERIVED_HASHES =", GENERATE_DERIVED_HASHES)
print("WRITE_CORRECTIONS =", WRITE_CORRECTIONS)
print("PROMOTE =", PROMOTE)
'''
    ),
    code(
        r'''
# CELL 2 - Safe helpers
def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_replace_bytes(path: Path, payload: bytes):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(payload)
    os.replace(tmp, path)


def atomic_replace_json(path: Path, value):
    payload = json.dumps(value, indent=2, default=str).encode("utf-8")
    atomic_replace_bytes(path, payload)


def atomic_replace_csv(path: Path, frame: pd.DataFrame):
    payload = frame.to_csv(index=False).encode("utf-8")
    atomic_replace_bytes(path, payload)


def backup_once(path: Path):
    backup = Path(str(path) + ".pre_correction.bak")
    if path.exists() and not backup.exists():
        shutil.copy2(path, backup)


def as_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    return series.astype(str).str.strip().str.lower().eq("true")


def sample_id_from_file(path: Path) -> str:
    volume_id = int(path.parent.name.removeprefix("v"))
    slice_id = int(path.stem.removeprefix("s"))
    return f"vol{volume_id:03d}_sli{slice_id:04d}"


def read_png(path: Path):
    with Image.open(path) as image:
        mode = image.mode
        size = image.size
        array = np.asarray(image).copy()
    return mode, size, array


def aggregate_directory_hash(files):
    digest = hashlib.sha256()
    for path in sorted(files, key=lambda p: p.as_posix()):
        digest.update(path.name.encode("utf-8"))
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


print("Helpers loaded.")
'''
    ),
    code(
        r'''
# CELL 3 - Load and reconcile build metadata
assert BUILD_DIR.is_dir(), f"Build not found: {BUILD_DIR}"

manifest_path = BUILD_DIR / "manifests" / "slice_manifest.csv"
version_path = BUILD_DIR / "dataset_version.json"
split_dir = BUILD_DIR / "splits"
audit_dir = BUILD_DIR / "audits"
review_dir = BUILD_DIR / "spatial_reviews"

required_inputs = [
    manifest_path,
    version_path,
    split_dir / "split_hashes.json",
    audit_dir / "source_inventory.csv",
    audit_dir / "nifti_pair_validation.csv",
]
missing_inputs = [str(path) for path in required_inputs if not path.exists()]
assert not missing_inputs, f"Missing required build inputs: {missing_inputs}"

manifest_df = pd.read_csv(manifest_path)
version_info = json.loads(version_path.read_text(encoding="utf-8"))
source_inventory_df = pd.read_csv(audit_dir / "source_inventory.csv")
nifti_validation_df = pd.read_csv(audit_dir / "nifti_pair_validation.csv")
stored_split_hashes = json.loads((split_dir / "split_hashes.json").read_text(encoding="utf-8"))

manifest_df["volume_id"] = manifest_df["volume_id"].astype(int)
manifest_df["slice_index"] = manifest_df["slice_index"].astype(int)
for column in ["image_exists", "organ_mask_exists", "tumor_mask_exists", "organ_present_256", "tumor_present_256"]:
    manifest_df[column] = as_bool(manifest_df[column])

print("Build ID:", version_info.get("build_id"))
print("Version/status:", version_info.get("version"), version_info.get("status"))
print("Manifest rows:", f"{len(manifest_df):,}")
print("Unique sample IDs:", f"{manifest_df['sample_id'].nunique():,}")
print("Volumes:", manifest_df["volume_id"].nunique())
print("Verification status:", manifest_df["verification_status"].value_counts(dropna=False).to_dict())
print("Tumor-positive slices:", int(manifest_df["tumor_present_256"].sum()))
'''
    ),
    code(
        r'''
# CELL 4 - Validate authoritative-source evidence
inventory_ids = set(source_inventory_df["volume_id"].astype(int))
validation_ids = set(nifti_validation_df["volume_id"].astype(int))

source_inventory_ok = (
    len(source_inventory_df) == 131
    and inventory_ids == EXPECTED_VOLUME_IDS
    and source_inventory_df["volume_path"].map(lambda p: Path(p).is_file()).all()
    and source_inventory_df["segmentation_path"].map(lambda p: Path(p).is_file()).all()
)

nifti_shape_ok = (
    len(nifti_validation_df) == 131
    and validation_ids == EXPECTED_VOLUME_IDS
    and as_bool(nifti_validation_df["shape_match"]).all()
)
nifti_labels_ok = as_bool(nifti_validation_df["labels_valid"]).all()
affine_mismatch_ids = set(
    nifti_validation_df.loc[~as_bool(nifti_validation_df["affine_match"]), "volume_id"].astype(int)
)

source_hash_issues = []
if VERIFY_SOURCE_HASHES:
    source_rows = manifest_df.sort_values("slice_index").drop_duplicates("volume_id")
    for number, row in enumerate(source_rows.itertuples(index=False), start=1):
        volume_path = Path(row.source_volume_path)
        segmentation_path = Path(row.source_segmentation_path)
        if sha256_file(volume_path) != row.source_volume_sha256:
            source_hash_issues.append({"volume_id": row.volume_id, "kind": "volume", "path": str(volume_path)})
        if sha256_file(segmentation_path) != row.source_segmentation_sha256:
            source_hash_issues.append({"volume_id": row.volume_id, "kind": "segmentation", "path": str(segmentation_path)})
        if number % 10 == 0 or number == 131:
            print(f"Source hashes checked: {number}/131 volumes")

source_hashes_ok = VERIFY_SOURCE_HASHES and not source_hash_issues

print("Source inventory complete:", source_inventory_ok)
print("NIfTI shapes valid:", nifti_shape_ok)
print("NIfTI labels valid:", nifti_labels_ok)
print("Affine mismatch count:", len(affine_mismatch_ids))
print("Affine mismatch IDs:", sorted(affine_mismatch_ids))
print("Source hashes verified:", source_hashes_ok)
if not VERIFY_SOURCE_HASHES:
    print("Final promotion remains blocked until VERIFY_SOURCE_HASHES=True is run.")
'''
    ),
    code(
        r'''
# CELL 5 - Full derived-file and pixel-level audit
duplicate_rows_df = manifest_df[manifest_df.duplicated("sample_id", keep=False)].copy()
manifest_ids = set(manifest_df["sample_id"])

file_sets = {}
for category in ["images", "organ_masks", "tumor_masks"]:
    files = list((BUILD_DIR / category).glob("v*/s*.png"))
    file_sets[category] = {sample_id_from_file(path) for path in files}
    print(category, "files=", len(files), "unique_keys=", len(file_sets[category]))

key_sets_ok = (
    file_sets["images"] == file_sets["organ_masks"] == file_sets["tumor_masks"] == manifest_ids
)

missing_rows = []
format_rows = []
label_rows = []
containment_rows = []

if FULL_FILE_AUDIT:
    for number, row in enumerate(manifest_df.itertuples(index=False), start=1):
        image_path = BUILD_DIR / row.image_path
        organ_path = BUILD_DIR / row.organ_mask_path
        tumor_path = BUILD_DIR / row.tumor_mask_path
        paths = {"image": image_path, "organ": organ_path, "tumor": tumor_path}

        if not all(path.is_file() for path in paths.values()):
            for kind, path in paths.items():
                if not path.is_file():
                    missing_rows.append({"sample_id": row.sample_id, "kind": kind, "path": str(path)})
            continue

        try:
            image_mode, image_size, image_array = read_png(image_path)
            organ_mode, organ_size, organ_array = read_png(organ_path)
            tumor_mode, tumor_size, tumor_array = read_png(tumor_path)
        except Exception as exc:
            format_rows.append({"sample_id": row.sample_id, "kind": "read_error", "detail": f"{type(exc).__name__}: {exc}"})
            continue

        for kind, mode, size in [
            ("image", image_mode, image_size),
            ("organ", organ_mode, organ_size),
            ("tumor", tumor_mode, tumor_size),
        ]:
            if mode != "L" or size != EXPECTED_SIZE:
                format_rows.append({"sample_id": row.sample_id, "kind": kind, "detail": f"mode={mode}, size={size}"})

        organ_values = set(np.unique(organ_array).tolist())
        tumor_values = set(np.unique(tumor_array).tolist())
        if not organ_values.issubset({0, 255}):
            format_rows.append({"sample_id": row.sample_id, "kind": "organ_values", "detail": str(sorted(organ_values))})
        if not tumor_values.issubset({0, 255}):
            format_rows.append({"sample_id": row.sample_id, "kind": "tumor_values", "detail": str(sorted(tumor_values))})

        organ_binary = organ_array > 0
        tumor_binary = tumor_array > 0
        organ_pixels = int(organ_binary.sum())
        tumor_pixels = int(tumor_binary.sum())

        if (
            organ_pixels != int(row.organ_pixels_256)
            or tumor_pixels != int(row.tumor_pixels_256)
            or bool(organ_pixels) != bool(row.organ_present_256)
            or bool(tumor_pixels) != bool(row.tumor_present_256)
        ):
            label_rows.append({
                "sample_id": row.sample_id,
                "manifest_organ_pixels": int(row.organ_pixels_256),
                "actual_organ_pixels": organ_pixels,
                "manifest_tumor_pixels": int(row.tumor_pixels_256),
                "actual_tumor_pixels": tumor_pixels,
            })

        outside_pixels = int(np.logical_and(tumor_binary, ~organ_binary).sum())
        if outside_pixels:
            containment_rows.append({"sample_id": row.sample_id, "outside_pixels": outside_pixels})

        if number % 5000 == 0 or number == len(manifest_df):
            print(f"Derived slices checked: {number:,}/{len(manifest_df):,}")

full_file_audit_ok = (
    FULL_FILE_AUDIT
    and key_sets_ok
    and duplicate_rows_df.empty
    and not missing_rows
    and not format_rows
    and not label_rows
    and not containment_rows
)

print("Equal image/organ/tumor/manifest key sets:", key_sets_ok)
print("Duplicate rows:", len(duplicate_rows_df))
print("Missing/unreadable files:", len(missing_rows))
print("Format issues:", len(format_rows))
print("Manifest-label mismatches:", len(label_rows))
print("Tumor-containment failures:", len(containment_rows))
print("FULL_FILE_AUDIT_OK =", full_file_audit_ok)
'''
    ),
    code(
        r'''
# CELL 6 - Split membership and hash audit
split_frames = {
    name: pd.read_csv(split_dir / f"{name}_slices.csv")
    for name in ["train", "val", "test"]
}
split_volume_expectations = {
    "train": TRAIN_VOLUMES,
    "val": VAL_VOLUMES,
    "test": TEST_VOLUMES,
}

split_sets = {name: set(frame["sample_id"]) for name, frame in split_frames.items()}
split_disjoint = (
    not (split_sets["train"] & split_sets["val"])
    and not (split_sets["train"] & split_sets["test"])
    and not (split_sets["val"] & split_sets["test"])
)
split_complete = set().union(*split_sets.values()) == manifest_ids
split_volume_ok = all(
    set(frame["volume_id"].astype(int)) == split_volume_expectations[name]
    for name, frame in split_frames.items()
)

volume_text_ok = True
for name, expected in split_volume_expectations.items():
    values = [int(value.strip()) for value in (split_dir / f"{name}_volumes.txt").read_text().splitlines() if value.strip()]
    volume_text_ok = volume_text_ok and set(values) == expected and len(values) == len(expected)

current_hashes = {
    filename: sha256_file(split_dir / filename)
    for filename in [
        "train_volumes.txt", "val_volumes.txt", "test_volumes.txt",
        "train_slices.csv", "val_slices.csv", "test_slices.csv",
    ]
}
split_hashes_ok = current_hashes == stored_split_hashes
split_audit_ok = split_disjoint and split_complete and split_volume_ok and volume_text_ok and split_hashes_ok

print("Train/val/test rows:", {name: len(frame) for name, frame in split_frames.items()})
print("Split disjoint:", split_disjoint)
print("Split union equals manifest:", split_complete)
print("Split volume membership correct:", split_volume_ok)
print("Integer volume files correct:", volume_text_ok)
print("Stored split hashes match:", split_hashes_ok)
print("SPLIT_AUDIT_OK =", split_audit_ok)
'''
    ),
    code(
        r'''
# CELL 7 - Generate corrected same-grid spatial-review figures
# Old review figures are invalid because they placed 256x256 contours on raw 512x512 images.
known_suspicious = {4, 33, 44, 48, 49, 50, 71, 76, 100, 108, 116}
control_volumes = {0, 8, 22}
REQUIRED_MANUAL_VOLUMES = set(affine_mismatch_ids) | known_suspicious
REVIEW_VOLUMES = sorted(REQUIRED_MANUAL_VOLUMES | control_volumes)

review_dir.mkdir(parents=True, exist_ok=True)
corrected_review_paths = {}

def choose_review_rows(volume_frame: pd.DataFrame):
    ordered = volume_frame.sort_values("slice_index")
    organ = ordered[ordered["organ_present_256"]]
    tumor = ordered[ordered["tumor_present_256"]]
    negative = ordered[~ordered["organ_present_256"]]
    choices = []
    if not negative.empty:
        choices.append(negative.iloc[len(negative) // 2])
    if not organ.empty:
        choices.extend([organ.iloc[0], organ.iloc[len(organ) // 2], organ.iloc[-1]])
    if not tumor.empty:
        choices.append(tumor.sort_values("tumor_pixels_256").iloc[-1])
    result = pd.DataFrame(choices).drop_duplicates("sample_id")
    return result.sort_values("slice_index")


if GENERATE_REVIEW_FIGURES:
    for position, volume_id in enumerate(REVIEW_VOLUMES, start=1):
        output_path = review_dir / f"vol{volume_id:03d}_review_corrected.png"
        corrected_review_paths[volume_id] = output_path
        if output_path.exists():
            continue

        selected = choose_review_rows(manifest_df[manifest_df["volume_id"] == volume_id])
        figure, axes = plt.subplots(1, len(selected), figsize=(4 * len(selected), 4))
        axes = np.atleast_1d(axes)

        for axis, row in zip(axes, selected.itertuples(index=False)):
            _, _, image_array = read_png(BUILD_DIR / row.image_path)
            _, _, organ_array = read_png(BUILD_DIR / row.organ_mask_path)
            _, _, tumor_array = read_png(BUILD_DIR / row.tumor_mask_path)
            axis.imshow(image_array, cmap="gray", vmin=0, vmax=255)
            if np.any(organ_array):
                axis.contour(organ_array > 0, levels=[0.5], colors=["lime"], linewidths=1.0)
            if np.any(tumor_array):
                axis.contour(tumor_array > 0, levels=[0.5], colors=["red"], linewidths=1.2)
            axis.set_title(
                f"v{volume_id:03d} s{int(row.slice_index):04d}\n"
                f"organ={int(row.organ_pixels_256)} tumor={int(row.tumor_pixels_256)}",
                fontsize=8,
            )
            axis.axis("off")

        figure.suptitle("Corrected same-grid review: green=organ, red=tumor", fontsize=10)
        figure.tight_layout()
        figure.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(figure)
        if position % 10 == 0 or position == len(REVIEW_VOLUMES):
            print(f"Corrected reviews generated: {position}/{len(REVIEW_VOLUMES)}")
else:
    for volume_id in REVIEW_VOLUMES:
        corrected_review_paths[volume_id] = review_dir / f"vol{volume_id:03d}_review_corrected.png"

missing_corrected_reviews = [volume_id for volume_id, path in corrected_review_paths.items() if not path.exists()]
print("Required manual-review volumes:", sorted(REQUIRED_MANUAL_VOLUMES))
print("Corrected review figures expected:", len(REVIEW_VOLUMES))
print("Missing corrected figures:", missing_corrected_reviews)
'''
    ),
    code(
        r'''
# CELL 8 - Display corrected high-risk reviews for manual inspection
from IPython.display import Image as NotebookImage, display

print("Inspect every figure below. Green must follow the complete liver/organ region; red must follow tumor anatomy.")
print("If alignment is doubtful, add the volume to MANUAL_REJECTED_VOLUMES, not MANUAL_APPROVED_VOLUMES.")

for volume_id in sorted(REQUIRED_MANUAL_VOLUMES):
    path = corrected_review_paths[volume_id]
    print(f"Volume {volume_id}: {path}")
    if path.exists():
        display(NotebookImage(filename=str(path), width=1100))
'''
    ),
    code(
        r'''
# CELL 9 - Evaluate explicit manual decisions
approved = {int(value) for value in MANUAL_APPROVED_VOLUMES}
rejected = {int(value) for value in MANUAL_REJECTED_VOLUMES}

unknown_approvals = approved - REQUIRED_MANUAL_VOLUMES
missing_approvals = REQUIRED_MANUAL_VOLUMES - approved - rejected
decision_overlap = approved & rejected
reviewer_ok = MANUAL_REVIEWER.strip() not in {"", "replace_with_reviewer_name"}

manual_review_ok = (
    not unknown_approvals
    and not missing_approvals
    and not rejected
    and not decision_overlap
    and reviewer_ok
    and not missing_corrected_reviews
)

print("Approved:", sorted(approved))
print("Rejected:", sorted(rejected))
print("Still requiring a decision:", sorted(missing_approvals))
print("Unknown approvals:", sorted(unknown_approvals))
print("Reviewer name supplied:", reviewer_ok)
print("MANUAL_REVIEW_OK =", manual_review_ok)
'''
    ),
    code(
        r'''
# CELL 10 - Apply verified status and rebuild splits only after all pre-gates pass
pre_correction_gates = {
    "source_inventory_complete": source_inventory_ok,
    "source_hashes_verified": source_hashes_ok,
    "nifti_shapes_valid": nifti_shape_ok,
    "nifti_labels_valid": nifti_labels_ok,
    "full_derived_audit": full_file_audit_ok,
    "split_audit": split_audit_ok,
    "manual_spatial_review": manual_review_ok,
}
PRE_CORRECTION_OK = all(pre_correction_gates.values())

print(pd.DataFrame([{"gate": key, "pass": value} for key, value in pre_correction_gates.items()]).to_string(index=False))
print("PRE_CORRECTION_OK =", PRE_CORRECTION_OK)

corrections_applied = False
if WRITE_CORRECTIONS and PRE_CORRECTION_OK:
    backup_once(manifest_path)
    manifest_df["verification_status"] = "verified"
    manifest_df["exclusion_reason"] = ""
    atomic_replace_csv(manifest_path, manifest_df)

    corrected_split_frames = {
        "train": manifest_df[manifest_df["volume_id"].isin(TRAIN_VOLUMES)].copy(),
        "val": manifest_df[manifest_df["volume_id"].isin(VAL_VOLUMES)].copy(),
        "test": manifest_df[manifest_df["volume_id"].isin(TEST_VOLUMES)].copy(),
    }
    for name, frame in corrected_split_frames.items():
        path = split_dir / f"{name}_slices.csv"
        backup_once(path)
        atomic_replace_csv(path, frame)

    new_split_hashes = {
        filename: sha256_file(split_dir / filename)
        for filename in [
            "train_volumes.txt", "val_volumes.txt", "test_volumes.txt",
            "train_slices.csv", "val_slices.csv", "test_slices.csv",
        ]
    }
    backup_once(split_dir / "split_hashes.json")
    atomic_replace_json(split_dir / "split_hashes.json", new_split_hashes)

    review_records = []
    for volume_id in REVIEW_VOLUMES:
        status = "approved" if volume_id in approved or volume_id in control_volumes else "automated_control"
        review_records.append({
            "volume_id": volume_id,
            "reviewer": MANUAL_REVIEWER if volume_id in REQUIRED_MANUAL_VOLUMES else "corrected_notebook_control",
            "review_timestamp": datetime.datetime.now().isoformat(),
            "status": status,
            "notes": "Reviewed on corrected same-grid overlay" if volume_id in REQUIRED_MANUAL_VOLUMES else "control figure generated",
            "figure_path": str(corrected_review_paths[volume_id]),
            "affine_policy": "voxel_index_pairing_reviewed_on_same_grid",
        })
    backup_once(audit_dir / "spatial_audit.csv")
    atomic_replace_csv(audit_dir / "spatial_audit.csv", pd.DataFrame(review_records))
    corrections_applied = True
    print("Verified manifest, corrected splits, hashes, and spatial audit written.")
else:
    print("No metadata was changed. Set WRITE_CORRECTIONS=True only after every pre-gate passes.")
'''
    ),
    code(
        r'''
# CELL 11 - Write complete audit artifacts and provenance
required_report_names = [
    "integrity_report.json", "missing_data.csv", "duplicate_keys.csv",
    "excluded_samples.csv", "key_reconciliation.csv", "geometry_report.csv",
    "label_distribution.json", "source_checksums.csv", "derived_checksums.csv",
    "spatial_audit.csv",
]

if WRITE_CORRECTIONS and PRE_CORRECTION_OK and corrections_applied:
    atomic_replace_csv(audit_dir / "missing_data.csv", pd.DataFrame(missing_rows, columns=["sample_id", "kind", "path"]))
    atomic_replace_csv(audit_dir / "duplicate_keys.csv", duplicate_rows_df)
    atomic_replace_csv(audit_dir / "excluded_samples.csv", pd.DataFrame(columns=["sample_id", "exclusion_reason"]))
    atomic_replace_csv(audit_dir / "geometry_report.csv", nifti_validation_df)

    key_reconciliation = pd.DataFrame([
        {"category": category, "file_count": len(keys), "unique_keys": len(keys), "matches_manifest": keys == manifest_ids}
        for category, keys in file_sets.items()
    ])
    atomic_replace_csv(audit_dir / "key_reconciliation.csv", key_reconciliation)

    source_checksums = manifest_df[[
        "volume_id", "source_volume_path", "source_segmentation_path",
        "source_volume_sha256", "source_segmentation_sha256",
    ]].drop_duplicates("volume_id").sort_values("volume_id")
    atomic_replace_csv(audit_dir / "source_checksums.csv", source_checksums)

    label_distribution = {
        "total_slices": int(len(manifest_df)),
        "organ_positive": int(manifest_df["organ_present_256"].sum()),
        "tumor_positive": int(manifest_df["tumor_present_256"].sum()),
        "total_organ_pixels": int(manifest_df["organ_pixels_256"].sum()),
        "total_tumor_pixels": int(manifest_df["tumor_pixels_256"].sum()),
    }
    atomic_replace_json(audit_dir / "label_distribution.json", label_distribution)

    if GENERATE_DERIVED_HASHES:
        checksum_rows = []
        for category in ["images", "organ_masks", "tumor_masks"]:
            for volume_id in sorted(EXPECTED_VOLUME_IDS):
                files = list((BUILD_DIR / category / f"v{volume_id:03d}").glob("s*.png"))
                checksum_rows.append({
                    "category": category,
                    "volume_id": volume_id,
                    "file_count": len(files),
                    "aggregate_sha256": aggregate_directory_hash(files),
                })
            print("Derived aggregate hashes complete:", category)
        atomic_replace_csv(audit_dir / "derived_checksums.csv", pd.DataFrame(checksum_rows))
    else:
        print("derived_checksums.csv not written: set GENERATE_DERIVED_HASHES=True before final approval.")

    integrity = {
        "build_id": version_info.get("build_id"),
        "timestamp": datetime.datetime.now().isoformat(),
        "all_pass": PRE_CORRECTION_OK and GENERATE_DERIVED_HASHES,
        "checks": pre_correction_gates,
        "manifest_rows": int(len(manifest_df)),
        "unique_sample_ids": int(manifest_df["sample_id"].nunique()),
        "volumes": int(manifest_df["volume_id"].nunique()),
        "pending_rows": int((manifest_df["verification_status"] != "verified").sum()),
    }
    atomic_replace_json(audit_dir / "integrity_report.json", integrity)

    current_split_hashes = json.loads((split_dir / "split_hashes.json").read_text(encoding="utf-8"))
    version_info.update({
        "version": "lits-v1.0.0",
        "status": "verified" if integrity["all_pass"] else "unverified",
        "manifest_hash": sha256_file(manifest_path),
        "split_hashes": current_split_hashes,
        "manual_review_summary": {
            "reviewer": MANUAL_REVIEWER,
            "approved_high_risk_volumes": sorted(approved),
            "corrected_overlay_method": "saved 256x256 image with saved 256x256 masks",
        },
        "audit_artifact_hashes": {
            path.name: sha256_file(path)
            for path in audit_dir.glob("*") if path.is_file()
        },
    })
    backup_once(version_path)
    atomic_replace_json(version_path, version_info)
    print("Audit artifacts and dataset_version.json updated.")
else:
    print("Audit reports were not changed.")
'''
    ),
    code(
        r'''
# CELL 12 - Defensible promotion gate
manifest_now = pd.read_csv(manifest_path)
pending_now = int((manifest_now["verification_status"].astype(str) != "verified").sum())
report_presence = {name: (audit_dir / name).exists() for name in required_report_names}

final_gates = {
    "131_source_pairs": source_inventory_ok,
    "source_hashes_verified": source_hashes_ok,
    "nifti_shapes_and_labels": nifti_shape_ok and nifti_labels_ok,
    "full_pixel_level_derived_audit": full_file_audit_ok,
    "split_membership_and_hashes": split_audit_ok,
    "all_high_risk_spatial_reviews_approved": manual_review_ok,
    "manifest_status_updated": pending_now == 0,
    "all_required_reports_exist": all(report_presence.values()),
    "derived_hashes_recorded": (audit_dir / "derived_checksums.csv").exists(),
    "version_record_verified": json.loads(version_path.read_text(encoding="utf-8")).get("status") == "verified",
}

DATASET_READY = all(final_gates.values())
print(pd.DataFrame([{"gate": key, "pass": value} for key, value in final_gates.items()]).to_string(index=False))
print("Missing reports:", [name for name, exists in report_presence.items() if not exists])
print("Pending manifest rows:", pending_now)
print("DATASET_READY =", DATASET_READY)
print("TRAINING_BLOCKED =", True)
'''
    ),
    code(
        r'''
# CELL 13 - Non-destructive canonical promotion record
canonical_registry = DATASET_ROOT / "03_canonical"
promotion_record = {
    "dataset": "LiTS Liver Tumor Segmentation",
    "version": "lits-v1.0.0",
    "build_dir": str(BUILD_DIR),
    "build_id": version_info.get("build_id"),
    "manifest_path": str(manifest_path),
    "manifest_sha256": sha256_file(manifest_path),
    "promoted_at": datetime.datetime.now().isoformat(),
    "promotion_mode": "versioned_pointer_no_data_copy",
}

PROMOTED = False
if PROMOTE and DATASET_READY:
    canonical_registry.mkdir(parents=True, exist_ok=True)
    version_record_path = canonical_registry / "lits-v1.0.0.json"
    current_record_path = canonical_registry / "current.json"
    if version_record_path.exists():
        existing = json.loads(version_record_path.read_text(encoding="utf-8"))
        assert existing.get("manifest_sha256") == promotion_record["manifest_sha256"], "Canonical v1.0.0 already points to a different manifest"
    atomic_replace_json(version_record_path, promotion_record)
    atomic_replace_json(current_record_path, promotion_record)
    PROMOTED = True
    print("Canonical promotion record written:", current_record_path)
else:
    print("No promotion performed. PROMOTE must be True and DATASET_READY must be True.")

print("PROMOTED =", PROMOTED)
'''
    ),
    code(
        r'''
# CELL 14 - Check the current training-loader contract
loader_path = PROJECT_ROOT / "src" / "framework" / "data" / "lits_dataset.py"
loader_text = loader_path.read_text(encoding="utf-8") if loader_path.exists() else ""

manifest_loader_detected = "slice_manifest" in loader_text or "manifest" in loader_text.lower()
legacy_flat_glob_detected = "Volume-*.png" in loader_text or "mask-*.png" in loader_text
LOADER_READY = manifest_loader_detected and not legacy_flat_glob_detected

print("Loader:", loader_path)
print("Manifest-driven loader detected:", manifest_loader_detected)
print("Legacy flat glob detected:", legacy_flat_glob_detected)
print("LOADER_READY =", LOADER_READY)
print("The loader must also refuse test access during training/threshold selection and pass the 16-slice overfit test.")
'''
    ),
    code(
        r'''
# CELL 15 - Final handoff and next action
TRAINING_BLOCKED = not (DATASET_READY and PROMOTED and LOADER_READY)

print("=" * 72)
print("FINAL STATUS")
print("=" * 72)
print("DATASET_READY =", DATASET_READY)
print("PROMOTED =", PROMOTED)
print("LOADER_READY =", LOADER_READY)
print("TRAINING_BLOCKED =", TRAINING_BLOCKED)

if not manual_review_ok:
    print("NEXT: inspect corrected review figures, then fill MANUAL_APPROVED_VOLUMES / MANUAL_REJECTED_VOLUMES and MANUAL_REVIEWER in Cell 1.")
elif not WRITE_CORRECTIONS:
    print("NEXT: set VERIFY_SOURCE_HASHES=True, GENERATE_DERIVED_HASHES=True, and WRITE_CORRECTIONS=True; restart and run all cells.")
elif DATASET_READY and not PROMOTE:
    print("NEXT: set PROMOTE=True and rerun Cells 12-15.")
elif PROMOTED and not LOADER_READY:
    print("NEXT: implement the manifest-driven loader, then run the 16-slice overfit test.")
elif not TRAINING_BLOCKED:
    print("Dataset and loader gates pass. Proceed to the 16-slice overfit test before baseline training.")
'''
    ),
]


notebook = nbf.v4.new_notebook(
    cells=cells,
    metadata={
        "kernelspec": {
            "display_name": "Python 3 (.venv)",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "version": "3.11"},
    },
)
nbf.write(notebook, OUT)
print(OUT)
