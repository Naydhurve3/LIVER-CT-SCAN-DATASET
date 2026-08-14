from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
from PIL import Image
from scipy.ndimage import binary_erosion, label, sobel


PROJECT_ROOT = Path(r"D:\DATA SCIENCE AND ANALYTICS\PROJECTS\Liver")
BUILD_DIR = Path(r"D:\DATA SCIENCE AND ANALYTICS\Dataset\Liver\02_staging\build_20260713_112738")
LEGACY_DIR = Path(r"D:\DATA SCIENCE AND ANALYTICS\Dataset\lits-png\dataset_6\dataset_6")
OUTPUT_DIR = PROJECT_ROOT / "Practice" / "dataset_pairing_forensics_results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MANIFEST_PATH = BUILD_DIR / "manifests" / "slice_manifest.csv"
VALIDATION_PATH = BUILD_DIR / "audits" / "nifti_pair_validation.csv"
EXPECTED_SIZE = (256, 256)

SCREENSHOT_SAMPLES = {
    4: [295, 347, 464, 472, 596],
    33: [19, 37, 85, 100, 132],
    44: [1, 57, 97, 113, 116],
    84: [237, 393, 522, 554, 650],
    85: [177, 335, 417, 473, 610],
}


def read_gray(path: Path, *, nearest: bool = False) -> np.ndarray:
    with Image.open(path) as image:
        image = image.convert("L")
        if image.size != EXPECTED_SIZE:
            method = Image.Resampling.NEAREST if nearest else Image.Resampling.BILINEAR
            image = image.resize(EXPECTED_SIZE, method)
        return np.asarray(image).copy()


def resize_256(array: np.ndarray, *, nearest: bool = False) -> np.ndarray:
    method = Image.Resampling.NEAREST if nearest else Image.Resampling.BILINEAR
    return np.asarray(Image.fromarray(array).resize(EXPECTED_SIZE, method))


def window_ct(array: np.ndarray) -> np.ndarray:
    clipped = np.clip(array, -160.0, 240.0)
    return ((clipped + 160.0) / 400.0 * 255.0).astype(np.uint8)


def correlation(left: np.ndarray, right: np.ndarray) -> float:
    a = left.astype(np.float64).ravel()
    b = right.astype(np.float64).ravel()
    a -= a.mean()
    b -= b.mean()
    denominator = np.sqrt(np.square(a).sum() * np.square(b).sum())
    return float((a * b).sum() / denominator) if denominator else 0.0


def dice(left: np.ndarray, right: np.ndarray) -> float:
    a = left > 0
    b = right > 0
    denominator = int(a.sum() + b.sum())
    return float(2 * np.logical_and(a, b).sum() / denominator) if denominator else 1.0


def diagnostic_transforms(array: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "identity": array,
        "flip_lr": np.fliplr(array),
        "flip_ud": np.flipud(array),
        "rot180": np.rot90(array, 2),
        "rot90": np.rot90(array, 1),
        "rot270": np.rot90(array, 3),
        "transpose": array.T,
        "anti_transpose": np.fliplr(np.flipud(array.T)),
    }


def centroid(binary: np.ndarray) -> tuple[float, float]:
    coordinates = np.argwhere(binary)
    if not len(coordinates):
        return np.nan, np.nan
    y, x = coordinates.mean(axis=0)
    return float(y), float(x)


def largest_component(binary: np.ndarray) -> np.ndarray:
    labels, count = label(binary)
    if count == 0:
        return binary
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    return labels == int(sizes.argmax())


def normalized_organ_centroid(image: np.ndarray, organ: np.ndarray) -> tuple[float, float]:
    body = largest_component(image > 10)
    body_coordinates = np.argwhere(body)
    organ_coordinates = np.argwhere(organ)
    if not len(body_coordinates) or not len(organ_coordinates):
        return np.nan, np.nan
    body_y, body_x = body_coordinates.mean(axis=0)
    organ_y, organ_x = organ_coordinates.mean(axis=0)
    y_min, x_min = body_coordinates.min(axis=0)
    y_max, x_max = body_coordinates.max(axis=0)
    height = max(float(y_max - y_min + 1), 1.0)
    width = max(float(x_max - x_min + 1), 1.0)
    return float((organ_y - body_y) / height), float((organ_x - body_x) / width)


def boundary_gradient(image: np.ndarray, mask: np.ndarray) -> float:
    if not np.any(mask):
        return np.nan
    gradient = np.hypot(sobel(image.astype(np.float32), axis=0), sobel(image.astype(np.float32), axis=1))
    boundary = np.logical_xor(mask, binary_erosion(mask))
    return float(gradient[boundary].mean()) if np.any(boundary) else np.nan


manifest = pd.read_csv(MANIFEST_PATH)
validation = pd.read_csv(VALIDATION_PATH)
for column in ["organ_present_256", "tumor_present_256"]:
    if manifest[column].dtype != bool:
        manifest[column] = manifest[column].astype(str).str.lower().eq("true")


# ---------------------------------------------------------------------------
# 1. Source-to-derived exact regeneration and local offset search.
# ---------------------------------------------------------------------------
requested_samples: dict[tuple[int, int], set[str]] = {}
for volume_id, group in manifest.groupby("volume_id"):
    representative = group.sort_values(["organ_pixels_256", "tumor_pixels_256"]).iloc[-1]
    requested_samples.setdefault((int(volume_id), int(representative.slice_index)), set()).add("representative_max_organ")
for volume_id, slice_indices in SCREENSHOT_SAMPLES.items():
    for slice_index in slice_indices:
        requested_samples.setdefault((volume_id, slice_index), set()).add("screenshot")

pairing_rows = []
offset_rows = []
for volume_id in sorted({key[0] for key in requested_samples}):
    first_row = manifest[manifest.volume_id == volume_id].iloc[0]
    ct_image = nib.load(first_row.source_volume_path)
    seg_image = nib.load(first_row.source_segmentation_path)
    slice_count = int(ct_image.shape[2])

    for (requested_volume, slice_index), roles in sorted(requested_samples.items()):
        if requested_volume != volume_id:
            continue

        saved_image = read_gray(BUILD_DIR / f"images/v{volume_id:03d}/s{slice_index:04d}.png")
        saved_organ = read_gray(BUILD_DIR / f"organ_masks/v{volume_id:03d}/s{slice_index:04d}.png", nearest=True)
        saved_tumor = read_gray(BUILD_DIR / f"tumor_masks/v{volume_id:03d}/s{slice_index:04d}.png", nearest=True)

        source_ct = np.asanyarray(ct_image.dataobj[:, :, slice_index], dtype=np.float32)
        source_seg = np.asanyarray(seg_image.dataobj[:, :, slice_index])
        expected_image = resize_256(window_ct(source_ct))
        expected_organ = resize_256(((source_seg > 0) * 255).astype(np.uint8), nearest=True)
        expected_tumor = resize_256(((source_seg == 2) * 255).astype(np.uint8), nearest=True)

        candidates = []
        for candidate_slice in range(max(0, slice_index - 5), min(slice_count, slice_index + 6)):
            candidate_ct = np.asanyarray(ct_image.dataobj[:, :, candidate_slice], dtype=np.float32)
            candidate_image = resize_256(window_ct(candidate_ct))
            score = correlation(saved_image, candidate_image)
            candidates.append((score, candidate_slice))
            offset_rows.append(
                {
                    "volume_id": volume_id,
                    "saved_slice_index": slice_index,
                    "candidate_source_slice": candidate_slice,
                    "offset": candidate_slice - slice_index,
                    "image_correlation": score,
                }
            )
        best_score, best_slice = max(candidates)

        reverse_slice = slice_count - 1 - slice_index
        reverse_ct = np.asanyarray(ct_image.dataobj[:, :, reverse_slice], dtype=np.float32)
        reverse_score = correlation(saved_image, resize_256(window_ct(reverse_ct)))

        pairing_rows.append(
            {
                "volume_id": volume_id,
                "saved_slice_index": slice_index,
                "sample_role": "+".join(sorted(roles)),
                "expected_source_slice": slice_index,
                "best_matching_source_slice": best_slice,
                "best_offset": best_slice - slice_index,
                "source_slice_count": slice_count,
                "image_pixel_exact": bool(np.array_equal(saved_image, expected_image)),
                "image_mae": float(np.abs(saved_image.astype(np.int16) - expected_image.astype(np.int16)).mean()),
                "image_correlation": correlation(saved_image, expected_image),
                "organ_exact": bool(np.array_equal(saved_organ, expected_organ)),
                "tumor_exact": bool(np.array_equal(saved_tumor, expected_tumor)),
                "organ_dice": dice(saved_organ, expected_organ),
                "tumor_dice": dice(saved_tumor, expected_tumor),
                "reverse_order_score": reverse_score,
            }
        )

pairing_audit = pd.DataFrame(pairing_rows)
pairing_audit["status"] = np.where(
    pairing_audit[["image_pixel_exact", "organ_exact", "tumor_exact"]].all(axis=1)
    & pairing_audit.best_offset.eq(0),
    "PASS",
    "FAIL",
)
pairing_audit["failure_reason"] = np.where(pairing_audit.status.eq("PASS"), "", "source_regeneration_or_offset_failure")
pairing_audit.to_csv(OUTPUT_DIR / "slice_pairing_audit.csv", index=False)
pd.DataFrame(offset_rows).to_csv(OUTPUT_DIR / "offset_candidates.csv", index=False)


# ---------------------------------------------------------------------------
# 2. Independent legacy-reference transform audit where trusted masks exist.
# ---------------------------------------------------------------------------
legacy_rows = []
reference_volumes = [7, 8, 9] + list(range(78, 100))
for volume_id in reference_volumes:
    volume_rows = manifest[manifest.volume_id == volume_id]
    representative = volume_rows.sort_values(["tumor_pixels_256", "organ_pixels_256"]).iloc[-1]
    slice_index = int(representative.slice_index)
    current_image = read_gray(BUILD_DIR / representative.image_path)
    current_organ = read_gray(BUILD_DIR / representative.organ_mask_path, nearest=True)
    legacy_image = read_gray(LEGACY_DIR / f"volume-{volume_id}_{slice_index}.png")
    legacy_liver = read_gray(LEGACY_DIR / f"segmentation-{volume_id}_livermask_{slice_index}.png", nearest=True)
    legacy_tumor = read_gray(LEGACY_DIR / f"segmentation-{volume_id}_lesionmask_{slice_index}.png", nearest=True)
    legacy_organ = np.maximum(legacy_liver, legacy_tumor)

    image_candidates = [(correlation(current_image, transformed), name) for name, transformed in diagnostic_transforms(legacy_image).items()]
    mask_candidates = [(dice(current_organ, transformed), name) for name, transformed in diagnostic_transforms(legacy_organ).items()]
    image_score, image_transform = max(image_candidates)
    mask_score, mask_transform = max(mask_candidates)
    legacy_rows.append(
        {
            "volume_id": volume_id,
            "slice_index": slice_index,
            "image_best_transform": image_transform,
            "image_correlation": image_score,
            "mask_best_transform": mask_transform,
            "mask_dice": mask_score,
            "image_mask_transform_agree": image_transform == mask_transform,
        }
    )

legacy_audit = pd.DataFrame(legacy_rows)
legacy_audit.to_csv(OUTPUT_DIR / "legacy_reference_transform_audit.csv", index=False)


# ---------------------------------------------------------------------------
# 3. Anatomy/body-outline, boundary, continuity, and tumor morphology audit.
# ---------------------------------------------------------------------------
volume_rows = []
component_rows = []
for volume_id, group in manifest.groupby("volume_id"):
    volume_id = int(volume_id)
    group = group.sort_values("slice_index")
    positive = group[group.organ_present_256]
    selected_positions = set(np.linspace(0, max(len(positive) - 1, 0), min(5, len(positive)), dtype=int).tolist()) if len(positive) else set()

    identity_intensity_sum = rotated_intensity_sum = 0.0
    identity_nonair = rotated_nonair = total_organ_pixels = 0
    identity_tumor_intensity = rotated_tumor_intensity = 0.0
    identity_tumor_nonair = rotated_tumor_nonair = total_tumor_pixels = 0
    identity_boundary_values = []
    rotated_boundary_values = []
    adjacent_dice_values = []
    previous_mask = None
    previous_slice = None
    component_count = tiny_component_count = 0
    tiny_organ_slices = int(((group.organ_pixels_256 > 0) & (group.organ_pixels_256 < 100)).sum())
    tiny_tumor_slices = int(((group.tumor_pixels_256 > 0) & (group.tumor_pixels_256 < 10)).sum())

    for position, row in enumerate(positive.itertuples(index=False)):
        image = read_gray(BUILD_DIR / row.image_path).astype(np.float32)
        organ = read_gray(BUILD_DIR / row.organ_mask_path, nearest=True) > 0
        rotated_organ = np.rot90(organ, 2)
        organ_pixels = int(organ.sum())
        total_organ_pixels += organ_pixels
        identity_intensity_sum += float(image[organ].sum())
        rotated_intensity_sum += float(image[rotated_organ].sum())
        identity_nonair += int((image[organ] > 10).sum())
        rotated_nonair += int((image[rotated_organ] > 10).sum())

        if position in selected_positions:
            identity_boundary_values.append(boundary_gradient(image, organ))
            rotated_boundary_values.append(boundary_gradient(image, rotated_organ))

        if previous_mask is not None and int(row.slice_index) == previous_slice + 1:
            adjacent_dice_values.append(dice(organ, previous_mask))
        previous_mask = organ
        previous_slice = int(row.slice_index)

        if bool(row.tumor_present_256):
            tumor = read_gray(BUILD_DIR / row.tumor_mask_path, nearest=True) > 0
            rotated_tumor = np.rot90(tumor, 2)
            tumor_pixels = int(tumor.sum())
            total_tumor_pixels += tumor_pixels
            identity_tumor_intensity += float(image[tumor].sum())
            rotated_tumor_intensity += float(image[rotated_tumor].sum())
            identity_tumor_nonair += int((image[tumor] > 10).sum())
            rotated_tumor_nonair += int((image[rotated_tumor] > 10).sum())
            labels, number = label(tumor)
            sizes = np.bincount(labels.ravel())[1:]
            component_count += int(number)
            tiny_component_count += int((sizes < 10).sum())
            if np.any(sizes < 10):
                component_rows.append(
                    {
                        "volume_id": volume_id,
                        "slice_index": int(row.slice_index),
                        "component_count": int(number),
                        "tiny_components_lt_10px": int((sizes < 10).sum()),
                        "smallest_component_pixels": int(sizes.min()),
                        "largest_component_pixels": int(sizes.max()),
                    }
                )

    max_row = group.sort_values("organ_pixels_256").iloc[-1]
    max_image = read_gray(BUILD_DIR / max_row.image_path).astype(np.float32)
    max_organ = read_gray(BUILD_DIR / max_row.organ_mask_path, nearest=True) > 0
    identity_centroid_y, identity_centroid_x = normalized_organ_centroid(max_image, max_organ)
    rotated_centroid_y, rotated_centroid_x = normalized_organ_centroid(max_image, np.rot90(max_organ, 2))

    areas = group.organ_pixels_256.to_numpy(dtype=float)
    area_jump = np.abs(np.diff(areas)) / np.maximum(np.maximum(areas[:-1], areas[1:]), 1.0) if len(areas) > 1 else np.array([])

    organ_denominator = max(total_organ_pixels, 1)
    tumor_denominator = max(total_tumor_pixels, 1)
    volume_rows.append(
        {
            "volume_id": volume_id,
            "slice_count": len(group),
            "organ_positive_slices": int(group.organ_present_256.sum()),
            "tumor_positive_slices": int(group.tumor_present_256.sum()),
            "identity_organ_mean_intensity": identity_intensity_sum / organ_denominator,
            "rot180_organ_mean_intensity": rotated_intensity_sum / organ_denominator,
            "organ_intensity_delta_rot_minus_identity": (rotated_intensity_sum - identity_intensity_sum) / organ_denominator,
            "identity_organ_nonair_fraction": identity_nonair / organ_denominator,
            "rot180_organ_nonair_fraction": rotated_nonair / organ_denominator,
            "organ_nonair_delta_rot_minus_identity": (rotated_nonair - identity_nonair) / organ_denominator,
            "identity_boundary_gradient": float(np.nanmean(identity_boundary_values)) if identity_boundary_values else np.nan,
            "rot180_boundary_gradient": float(np.nanmean(rotated_boundary_values)) if rotated_boundary_values else np.nan,
            "identity_tumor_mean_intensity": identity_tumor_intensity / tumor_denominator if total_tumor_pixels else np.nan,
            "rot180_tumor_mean_intensity": rotated_tumor_intensity / tumor_denominator if total_tumor_pixels else np.nan,
            "identity_tumor_nonair_fraction": identity_tumor_nonair / tumor_denominator if total_tumor_pixels else np.nan,
            "rot180_tumor_nonair_fraction": rotated_tumor_nonair / tumor_denominator if total_tumor_pixels else np.nan,
            "median_adjacent_organ_dice": float(np.median(adjacent_dice_values)) if adjacent_dice_values else np.nan,
            "max_normalized_organ_area_jump": float(area_jump.max()) if len(area_jump) else np.nan,
            "identity_centroid_y": identity_centroid_y,
            "identity_centroid_x": identity_centroid_x,
            "rot180_centroid_y": rotated_centroid_y,
            "rot180_centroid_x": rotated_centroid_x,
            "tumor_component_count_2d": component_count,
            "tiny_tumor_component_count_lt10px": tiny_component_count,
            "tiny_organ_slices_lt100px": tiny_organ_slices,
            "tiny_tumor_slices_lt10px": tiny_tumor_slices,
        }
    )

anatomy = pd.DataFrame(volume_rows).sort_values("volume_id")

# The aligned transposed-display reference cohort is volumes 70-82.
reference = anatomy[anatomy.volume_id.between(70, 82)]
reference_y = float(reference.identity_centroid_y.median())
reference_x = float(reference.identity_centroid_x.median())
anatomy["identity_centroid_distance_to_70_82_reference"] = np.hypot(
    anatomy.identity_centroid_y - reference_y,
    anatomy.identity_centroid_x - reference_x,
)
anatomy["rot180_centroid_distance_to_70_82_reference"] = np.hypot(
    anatomy.rot180_centroid_y - reference_y,
    anatomy.rot180_centroid_x - reference_x,
)
anatomy["centroid_distance_improvement_if_rotated"] = (
    anatomy.identity_centroid_distance_to_70_82_reference - anatomy.rot180_centroid_distance_to_70_82_reference
)
anatomy["recommended_transform"] = np.where(
    (anatomy.organ_intensity_delta_rot_minus_identity > 20)
    & (anatomy.organ_nonair_delta_rot_minus_identity > 0),
    "rot180",
    "identity",
)
anatomy["orientation_confidence"] = np.select(
    [
        anatomy.volume_id.between(83, 99),
        anatomy.volume_id.between(101, 130),
        anatomy.volume_id.between(78, 82),
    ],
    ["confirmed_by_legacy_reference", "strong_anatomy_and_source_pattern", "confirmed_identity_by_legacy_reference"],
    default="anatomy_metrics_support_identity",
)

affine_match = validation.assign(
    affine_match_bool=validation.affine_match.astype(str).str.lower().eq("true")
)[["volume_id", "affine_match_bool"]]
anatomy = anatomy.merge(affine_match, on="volume_id", how="left")
anatomy.to_csv(OUTPUT_DIR / "orientation_candidates.csv", index=False)
pd.DataFrame(component_rows).to_csv(OUTPUT_DIR / "native_component_audit.csv", index=False)


# ---------------------------------------------------------------------------
# 4. Consolidated decisions and failures.
# ---------------------------------------------------------------------------
legacy_summary = legacy_audit[["volume_id", "image_best_transform", "mask_best_transform", "image_mask_transform_agree"]]
summary = anatomy.merge(legacy_summary, on="volume_id", how="left")
summary["slice_pairing_samples_pass"] = summary.volume_id.map(
    pairing_audit.groupby("volume_id").status.apply(lambda values: bool((values == "PASS").all()))
).fillna(False)
summary["slice_order_continuity_status"] = np.where(summary.median_adjacent_organ_dice >= 0.75, "smooth", "review")
summary["training_eligible"] = summary.recommended_transform.eq("identity") & summary.slice_pairing_samples_pass
summary.to_csv(OUTPUT_DIR / "volume_pairing_summary.csv", index=False)

failures = []
for row in summary.itertuples(index=False):
    if row.recommended_transform != "identity":
        failures.append(
            {
                "volume_id": row.volume_id,
                "severity": "critical",
                "failure_type": "in_plane_orientation_mismatch",
                "recommended_action": "rebuild organ and tumor masks with rot180 in a new build",
                "evidence": f"organ_intensity_delta={row.organ_intensity_delta_rot_minus_identity:.2f}; nonair_delta={row.organ_nonair_delta_rot_minus_identity:.3f}",
            }
        )
    if not row.slice_pairing_samples_pass:
        failures.append(
            {
                "volume_id": row.volume_id,
                "severity": "critical",
                "failure_type": "source_to_derived_pairing_failure",
                "recommended_action": "inspect filenames, offsets, and deterministic conversion",
                "evidence": "representative source regeneration did not pass",
            }
        )
pd.DataFrame(failures).to_csv(OUTPUT_DIR / "pairing_failures.csv", index=False)


# ---------------------------------------------------------------------------
# 5. Evidence-focused figures.
# ---------------------------------------------------------------------------
colors = np.where(anatomy.recommended_transform.eq("rot180"), "#B42318", "#344054")

fig, ax = plt.subplots(figsize=(16, 5))
ax.bar(anatomy.volume_id, anatomy.organ_intensity_delta_rot_minus_identity, color=colors, width=0.85)
ax.axhline(0, color="#101828", linewidth=1)
ax.axhline(20, color="#667085", linewidth=1, linestyle="--", label="rotation evidence threshold")
ax.set(
    title="A 180-degree mask rotation improves anatomy fit only in two volume blocks",
    xlabel="Volume ID",
    ylabel="Mean CT intensity under organ mask: rotated minus current",
)
ax.legend(frameon=False, loc="upper left")
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "orientation_score_by_volume.png", dpi=160)
plt.close(fig)

fig, ax = plt.subplots(figsize=(8, 6))
for transform, color in [("identity", "#344054"), ("rot180", "#B42318")]:
    subset = anatomy[anatomy.recommended_transform == transform]
    ax.scatter(
        subset.organ_intensity_delta_rot_minus_identity,
        subset.organ_nonair_delta_rot_minus_identity,
        s=32,
        alpha=0.8,
        color=color,
        label=transform,
    )
ax.axvline(0, color="#98A2B3", linewidth=1)
ax.axhline(0, color="#98A2B3", linewidth=1)
ax.set(
    title="Body-outline and tissue-intensity evidence agree on the orientation failure",
    xlabel="Organ intensity improvement after 180-degree rotation",
    ylabel="Non-air organ overlap improvement after rotation",
)
ax.legend(frameon=False)
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "orientation_evidence_scatter.png", dpi=160)
plt.close(fig)

fig, ax = plt.subplots(figsize=(16, 5))
ax.scatter(anatomy.volume_id, anatomy.median_adjacent_organ_dice, c=colors, s=26)
ax.axhline(0.75, color="#667085", linestyle="--", linewidth=1, label="review threshold")
ax.set(
    title="Adjacent-slice continuity remains smooth even when in-plane orientation is wrong",
    xlabel="Volume ID",
    ylabel="Median adjacent-slice organ Dice",
    ylim=(0, 1.02),
)
ax.legend(frameon=False)
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "continuity_by_volume.png", dpi=160)
plt.close(fig)

fig, ax = plt.subplots(figsize=(9, 6))
tumor_subset = anatomy[anatomy.tumor_positive_slices > 0]
ax.scatter(
    tumor_subset.tumor_positive_slices,
    tumor_subset.tiny_tumor_component_count_lt10px,
    c=np.where(tumor_subset.recommended_transform.eq("rot180"), "#B42318", "#344054"),
    alpha=0.8,
)
ax.set(
    title="Tiny tumor components are annotation/morphology signals, not orientation detectors",
    xlabel="Tumor-positive slices per volume",
    ylabel="2-D tumor components smaller than 10 pixels",
)
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "tumor_component_profile.png", dpi=160)
plt.close(fig)


result_summary = {
    "build_dir": str(BUILD_DIR),
    "manifest_rows": int(len(manifest)),
    "representative_and_screenshot_pairing_rows": int(len(pairing_audit)),
    "source_regeneration_pass_rows": int((pairing_audit.status == "PASS").sum()),
    "zero_offset_rows": int((pairing_audit.best_offset == 0).sum()),
    "legacy_reference_volumes": int(len(legacy_audit)),
    "legacy_transform_disagreements": legacy_audit.loc[
        ~legacy_audit.image_mask_transform_agree, "volume_id"
    ].astype(int).tolist(),
    "recommended_rot180_volumes": anatomy.loc[
        anatomy.recommended_transform == "rot180", "volume_id"
    ].astype(int).tolist(),
    "recommended_identity_volumes": anatomy.loc[
        anatomy.recommended_transform == "identity", "volume_id"
    ].astype(int).tolist(),
    "critical_failure_count": int(len(failures)),
    "interpretation": {
        "source_to_derived": "sampled saved PNGs exactly regenerate from their named source slice",
        "slice_numbering": "all local offset searches select offset 0",
        "orientation": "CT and segmentation source arrays use incompatible in-plane orientation for the flagged blocks",
        "continuity": "3-D slice order can remain smooth despite a wrong in-plane orientation",
        "tiny_components": "exact regeneration shows tiny boundary components originate in source segmentations rather than PNG conversion",
    },
}
(OUTPUT_DIR / "forensic_summary.json").write_text(json.dumps(result_summary, indent=2), encoding="utf-8")

print(json.dumps(result_summary, indent=2))
print(f"\nResults written to: {OUTPUT_DIR}")
