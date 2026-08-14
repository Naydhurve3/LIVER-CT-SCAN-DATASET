"""Generate the directly runnable Step 03 final inference-policy freeze notebook."""

from pathlib import Path
import nbformat as nbf


PROJECT_ROOT = Path(r"D:\DATA SCIENCE AND ANALYTICS\PROJECTS\Liver")
PHASE_DIR = PROJECT_ROOT / "mark 1 (part 2)" / "step_03_final_inference_policy_freeze"
NOTEBOOK_PATH = PHASE_DIR / "step_03_final_inference_policy_freeze.ipynb"


def md(text: str):
    return nbf.v4.new_markdown_cell(text.strip())


def code(text: str):
    return nbf.v4.new_code_cell(text.strip())


cells = [
    md(r"""
# Step 03 — Final inference-policy freeze

**Mode:** lightweight artifact verification and contract freeze only. This notebook performs no training, no model inference, and no test loading.

It verifies the completed Step 01 and Step 02 gates, recomputes the immutable artifact chain, freezes the exact loader/ROI/fusion/threshold/metric policy, declares the final acceptance table before test access, and emits an authorization-readiness gate. A pass means only that the project is ready to **request explicit one-time test authorization**; it does not itself authorize or perform test evaluation.
"""),
    code(r"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import hashlib, json, platform, sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import display

PROJECT_ROOT = Path(r"D:\DATA SCIENCE AND ANALYTICS\PROJECTS\Liver")
PART2 = PROJECT_ROOT / "mark 1 (part 2)"
PHASE_DIR = PART2 / "step_03_final_inference_policy_freeze"
OUTPUT_DIR = PHASE_DIR / "outputs"
STEP1 = PART2 / "step_01_pretraining_dataset_characterization"
STEP2 = PART2 / "step_02_fusion_freeze_confirmation"
STEP1_OUT = STEP1 / "outputs"
STEP2_OUT = STEP2 / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42
TEST_IMAGES_ACCESSED = False
PHASE = "step_03_final_inference_policy_freeze"
EXPECTED_OUTPUTS = [
    "configuration.json", "provenance.json", "source_gate_summary.csv",
    "artifact_checksum_inventory.csv", "artifact_inventory_summary.json",
    "probability_cache_inventory.csv", "freeze_diff_check.csv",
    "final_acceptance_table.csv", "final_acceptance_contract.json",
    "final_inference_policy.json", "freeze_signature.json",
    "test_access_declaration.json", "expected_vs_actual.csv",
    "freeze_readiness_dashboard.png", "FINAL_INFERENCE_POLICY_FREEZE.md",
    "authorization_readiness.json", "gate_result.json"
]

def sha256(path: Path, chunk: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()

def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

def save_json(obj, name: str):
    path = OUTPUT_DIR / name
    path.write_text(json.dumps(obj, indent=2, sort_keys=True), encoding="utf-8")
    return path

def save_csv(df: pd.DataFrame, name: str):
    path = OUTPUT_DIR / name
    df.to_csv(path, index=False)
    return path

print(f"Phase: {PHASE}")
print(f"Outputs: {OUTPUT_DIR}")
print("TEST LOCK: no test image, mask, probability, path, or statistic is read.")
"""),
    md("## 1. Verify prerequisite gates and immutable sources"),
    code(r"""
step1_gate_path = STEP1_OUT / "pretraining_dataset_gate.json"
step2_gate_path = STEP2_OUT / "gate_result.json"
step2_policy_path = STEP2_OUT / "immutable_inference_policy.json"
step2_config_path = STEP2_OUT / "configuration.json"
step2_prov_path = STEP2_OUT / "provenance.json"

for required in [step1_gate_path, step2_gate_path, step2_policy_path, step2_config_path, step2_prov_path]:
    assert required.is_file(), f"Missing prerequisite: {required}"

step1_gate = load_json(step1_gate_path)
step2_gate = load_json(step2_gate_path)
step2_policy = load_json(step2_policy_path)
step2_config = load_json(step2_config_path)
step2_prov = load_json(step2_prov_path)

manifest_path = Path(step2_config["manifest_path"])
control_checkpoint = Path(step2_policy["checkpoint_paths"]["control"])
recall_checkpoint = Path(step2_policy["checkpoint_paths"]["recall_loss"])
roi_manifest = PROJECT_ROOT / "mark 1" / "mark_4_outputs" / "validation_roi_manifest.csv"
roi_generator_checkpoint = PROJECT_ROOT / "Practice" / "multitask_liver_tumor_outputs" / "multitask_best.pth"
EXPECTED_ROI_GENERATOR_SHA256 = "9c4160bbd68891f9dc4e5f04ceca4391f38c5869b3f81c72b95d4639e0572223"

source_gate_rows = [
    {"gate": "step_01_dataset_audit", "result_level": step1_gate["result_level"], "passed": bool(step1_gate["all_mandatory_targets_passed"]), "test_images_accessed": bool(step1_gate["test_images_accessed"])},
    {"gate": "step_02_fusion_confirmation", "result_level": step2_gate["result_level"], "passed": bool(step2_gate["all_mandatory_targets_passed"]), "test_images_accessed": bool(step2_gate["test_images_accessed"])},
]
source_gate_summary = pd.DataFrame(source_gate_rows)
save_csv(source_gate_summary, "source_gate_summary.csv")

assert step1_gate["all_mandatory_targets_passed"] and not step1_gate["test_images_accessed"]
assert step2_gate["result_level"] == "VALIDATION_FREEZE_PASS"
assert step2_gate["all_mandatory_targets_passed"] and not step2_gate["test_images_accessed"]
assert step2_policy["status"] == "FROZEN" and not step2_policy["test_images_accessed"]
assert manifest_path.is_file() and control_checkpoint.is_file() and recall_checkpoint.is_file() and roi_manifest.is_file()

recomputed = {
    "manifest": sha256(manifest_path),
    "control_checkpoint": sha256(control_checkpoint),
    "recall_checkpoint": sha256(recall_checkpoint),
    "validation_roi_manifest": sha256(roi_manifest),
    "roi_generator_checkpoint": sha256(roi_generator_checkpoint),
}
assert recomputed["manifest"] == step2_policy["manifest_sha256"]
assert recomputed["control_checkpoint"] == step2_policy["checkpoint_sha256"]["control"]
assert recomputed["recall_checkpoint"] == step2_policy["checkpoint_sha256"]["recall_loss"]
assert recomputed["validation_roi_manifest"] == step2_gate["input_artifact_hashes"]["validation_roi_manifest"]
assert recomputed["roi_generator_checkpoint"] == EXPECTED_ROI_GENERATOR_SHA256
display(source_gate_summary)
print("PASS: prerequisite gates and four immutable hashes verified.")
"""),
    md("## 2. Freeze final inference policy and predeclare final acceptance"),
    code(r"""
final_policy = {
    "schema_version": "1.0",
    "status": "FROZEN_PENDING_EXPLICIT_TEST_AUTHORIZATION",
    "frozen_utc": datetime.now(timezone.utc).isoformat(),
    "dataset": {
        "build_id": step2_policy["dataset_build_id"],
        "manifest_path": str(manifest_path),
        "manifest_sha256": recomputed["manifest"],
    },
    "model": {
        "architecture": step2_config["model"]["architecture"],
        "input_channels": 1, "output_channels": 1,
        "control_checkpoint": str(control_checkpoint),
        "control_checkpoint_sha256": recomputed["control_checkpoint"],
        "recall_checkpoint": str(recall_checkpoint),
        "recall_checkpoint_sha256": recomputed["recall_checkpoint"],
        "strict_state_loading": True,
    },
    "input": {
        "source_type": "derived 256x256 PNG produced from source CT",
        "channels_in_order": ["broad_window"],
        "hu_window": [-160, 240],
        "clipping": "clip(HU,-160,240)",
        "stored_representation": "uint8",
        "normalization": "float32(uint8)/255",
    },
    "roi": {
        "source": "predicted_liver", "threshold": 0.5,
        "component_rule": "largest_3d", "padding_pixels": 16,
        "crop_resize": [256, 256], "image_interpolation": "bilinear",
        "inverse_mapping": "bilinear score resize into frozen full-image box",
        "validation_roi_manifest_sha256": recomputed["validation_roi_manifest"],
        "generator": {
            "architecture": "MobileNetV2UNet", "input_channels": 1, "output_channels": 2,
            "checkpoint": str(roi_generator_checkpoint),
            "checkpoint_sha256": recomputed["roi_generator_checkpoint"],
            "strict_state_loading": True, "score_definition": "sigmoid(logit)[:,0]",
            "input_source": "derived broad-window uint8 PNG divided by 255",
            "per_slice_normalization": "positive-pixel median and IQR/1.349 robust z-score, fallback whole slice; clip [-3,3], map to [0,1]",
            "robust_z_clip": 3.0, "connectivity": 26,
            "empty_roi_fallback": "full_image_box_[0,256,0,256]",
        },
    },
    "probability": {
        "score_definition": "sigmoid(logit)",
        "description": "model probability score; not claimed to be calibrated",
        "inference_dtype": "float32",
        "fusion_equation": "maximum(control_probability, recall_probability)",
        "global_threshold": 0.70,
        "hard_prediction_rule": "fused_probability >= 0.70",
        "post_processing": "none",
    },
    "metrics": {
        "dice_formula": "(2*intersection + epsilon)/(truth_pixels + predicted_pixels + epsilon)",
        "epsilon": 1e-6,
        "primary_population": "tumour-positive patients",
        "empty_truth_convention": "reported separately; not included in positive-patient mean Dice",
        "q1_positive_slice_definition": {"derivation_split": "train", "truth_pixels_min_exclusive": 0, "truth_pixels_max_inclusive": 51.0},
        "bootstrap": {"seed": 42, "iterations": 10000, "resampling_unit": "patient"},
    },
    "runtime_reference": step2_policy["software"],
    "random_seed": 42,
    "immutable_after_test": ["dataset", "model", "input", "roi", "probability", "metrics"],
    "test_results_may_not_trigger": ["threshold tuning", "checkpoint selection", "fusion changes", "post-processing changes", "retraining"],
    "test_images_accessed": False,
}
save_json(final_policy, "final_inference_policy.json")

acceptance_rows = [
    # Six retained validation guardrails. V104/V116 are validation sentinels and cannot be redefined on test.
    {"order": 1, "scope": "validation_prerequisite", "metric": "mean_positive_patient_dice", "direction": ">=", "target": 0.3329, "mandatory": True, "rationale": "retained temporary patient guardrail"},
    {"order": 2, "scope": "validation_prerequisite", "metric": "volume_104_dice", "direction": ">=", "target": 0.05, "mandatory": True, "rationale": "validation-only hard-case sentinel"},
    {"order": 3, "scope": "validation_prerequisite", "metric": "volume_116_dice", "direction": ">=", "target": 0.01, "mandatory": True, "rationale": "validation-only hard-case sentinel"},
    {"order": 4, "scope": "validation_prerequisite", "metric": "q1_positive_slice_detection_pct", "direction": ">=", "target": 35.0, "mandatory": True, "rationale": "retained small-lesion guardrail"},
    {"order": 5, "scope": "validation_prerequisite", "metric": "positive_predicted_empty_pct", "direction": "<=", "target": 35.0, "mandatory": True, "rationale": "retained miss-rate guardrail"},
    {"order": 6, "scope": "validation_prerequisite", "metric": "empty_slice_false_positive_pct", "direction": "<=", "target": 20.0, "mandatory": True, "rationale": "retained false-positive guardrail"},
    # Cohort-independent final-test metrics, frozen before test access.
    {"order": 7, "scope": "final_test", "metric": "mean_positive_patient_dice", "direction": ">=", "target": 0.3329, "mandatory": True, "rationale": "minimum final patient-level performance"},
    {"order": 8, "scope": "final_test", "metric": "q1_positive_slice_detection_pct_train_edges", "direction": ">=", "target": 35.0, "mandatory": True, "rationale": "small-lesion sensitivity with frozen train-derived edge"},
    {"order": 9, "scope": "final_test", "metric": "positive_predicted_empty_pct", "direction": "<=", "target": 35.0, "mandatory": True, "rationale": "positive-slice miss guardrail"},
    {"order": 10, "scope": "final_test", "metric": "empty_slice_false_positive_pct", "direction": "<=", "target": 20.0, "mandatory": True, "rationale": "empty-slice false-positive guardrail"},
    {"order": 11, "scope": "final_test", "metric": "minimum_positive_patient_dice", "direction": ">=", "target": 0.01, "mandatory": True, "rationale": "prohibit catastrophic positive-patient failure"},
    {"order": 12, "scope": "final_test_integrity", "metric": "sample_coverage_fraction", "direction": "==", "target": 1.0, "mandatory": True, "rationale": "complete one-time evaluation"},
    {"order": 13, "scope": "final_test_integrity", "metric": "finite_probability_fraction", "direction": "==", "target": 1.0, "mandatory": True, "rationale": "numerical integrity"},
    {"order": 14, "scope": "final_test_integrity", "metric": "unique_sample_id_fraction", "direction": "==", "target": 1.0, "mandatory": True, "rationale": "no duplicated evaluations"},
]
acceptance_table = pd.DataFrame(acceptance_rows)
save_csv(acceptance_table, "final_acceptance_table.csv")
acceptance_contract = {
    "schema_version": "1.0", "status": "FROZEN_PENDING_OWNER_AUTHORIZATION",
    "declared_before_test_access": True, "test_images_accessed": False,
    "table_file": "final_acceptance_table.csv",
    "metric_definitions": {"q1_positive_slice_detection_pct_train_edges": "percentage of positive test slices with any true-positive predicted pixel among slices having 1 to 51 tumour pixels; 51 is the frozen train-only 25th percentile"},
    "interpretation": "All mandatory final_test and final_test_integrity rows define formal minimum final acceptance. Validation prerequisite rows must remain passed.",
    "historical_aspirational_targets": {"mean_positive_patient_dice": 0.406915, "q1_detection_pct": 45.0, "positive_predicted_empty_pct": 20.0, "empty_slice_false_positive_pct": 15.0, "volume_104_dice_validation_only": 0.50, "volume_116_dice_validation_only": 0.05},
    "aspirational_targets_are_mandatory": False,
    "failure_action": "Report FINAL_TEST_COMPLETE with failed acceptance; do not tune, retrain, or rerun based on test results.",
    "authorization_semantics": "The project owner's explicit one-time test authorization also accepts this predeclared table unless they revise it before any test access.",
}
save_json(acceptance_contract, "final_acceptance_contract.json")
display(acceptance_table)
"""),
    md("## 3. Checksum the complete frozen artifact chain"),
    code(r"""
authoritative = [
    (manifest_path, "dataset_manifest"), (control_checkpoint, "control_checkpoint"),
    (recall_checkpoint, "recall_checkpoint"), (roi_manifest, "validation_roi_manifest"),
    (roi_generator_checkpoint, "roi_generator_checkpoint"),
    (STEP1_OUT / "pretraining_dataset_gate.json", "step01_gate"),
    (STEP1_OUT / "DATASET_DATA_CARD.md", "dataset_data_card"),
    (STEP1_OUT / "sampling_policy.json", "sampling_policy"),
    (STEP1_OUT / "train_derived_lesion_bins.json", "train_derived_bins"),
    (step2_gate_path, "step02_gate"), (step2_policy_path, "step02_immutable_policy"),
    (step2_config_path, "step02_configuration"), (step2_prov_path, "step02_provenance"),
    (STEP2_OUT / "selected_gate_table.csv", "step02_gate_table"),
    (STEP2_OUT / "bootstrap_uncertainty.csv", "step02_uncertainty"),
    (STEP2_OUT / "patient_metrics.csv", "step02_patient_metrics"),
    (STEP2_OUT / "slice_metrics.csv", "step02_slice_metrics"),
    (STEP2_OUT / "lesion_or_size_metrics.csv", "step02_lesion_metrics"),
    (STEP2_OUT / "fresh_vs_historical_cache_equivalence.csv", "step02_cache_equivalence"),
    (STEP2_OUT / "hard_prediction_equivalence.csv", "step02_hard_equivalence"),
    (STEP2_OUT / "determinism_check.csv", "step02_determinism"),
    (STEP2_OUT / "cache_integrity.csv", "step02_cache_integrity"),
    (STEP2 / "step_02_fusion_freeze_confirmation.ipynb", "step02_notebook"),
    (PROJECT_ROOT / "src/framework/data/manifest_dataset.py", "loader_source"),
    (PROJECT_ROOT / "src/framework/models/mobilenetv2_unet.py", "model_source"),
]
cache_paths = sorted((STEP2_OUT / "probability_cache").glob("volume_*.npz"))
assert len(cache_paths) == 13, f"Expected 13 validation caches, found {len(cache_paths)}"
authoritative.extend((p, "step02_validation_probability_cache") for p in cache_paths)

inventory_rows = []
for path, role in authoritative:
    inventory_rows.append({"role": role, "path": str(path), "exists": path.is_file(), "bytes": path.stat().st_size if path.is_file() else None, "sha256": sha256(path) if path.is_file() else None})
inventory = pd.DataFrame(inventory_rows)
assert inventory.exists.all(), inventory.loc[~inventory.exists]
inventory_path = save_csv(inventory, "artifact_checksum_inventory.csv")

cache_inventory = inventory[inventory.role.eq("step02_validation_probability_cache")].copy()
cache_inventory.insert(0, "volume_id", cache_inventory.path.str.extract(r"volume_(\d+)\.npz")[0].astype(int))
save_csv(cache_inventory, "probability_cache_inventory.csv")

inventory_summary = {"artifact_count": int(len(inventory)), "cache_count": int(len(cache_inventory)), "total_bytes": int(inventory.bytes.sum()), "inventory_csv_sha256": sha256(inventory_path), "all_exist": bool(inventory.exists.all()), "test_images_accessed": False}
save_json(inventory_summary, "artifact_inventory_summary.json")
display(inventory[["role", "bytes", "sha256"]])
print(f"PASS: {len(inventory)} artifacts checksummed, including {len(cache_inventory)} validation caches.")
"""),
    md("## 4. Verify freeze equality and create signed package"),
    code(r"""
checks = [
    ("manifest_sha256", step2_policy["manifest_sha256"], final_policy["dataset"]["manifest_sha256"]),
    ("control_checkpoint_sha256", step2_policy["checkpoint_sha256"]["control"], final_policy["model"]["control_checkpoint_sha256"]),
    ("recall_checkpoint_sha256", step2_policy["checkpoint_sha256"]["recall_loss"], final_policy["model"]["recall_checkpoint_sha256"]),
    ("hu_window", str(step2_policy["input"]["hu_window"]), str(final_policy["input"]["hu_window"])),
    ("normalization", step2_policy["input"]["derived_normalization"], "uint8/255"),
    ("roi_threshold", step2_policy["roi"]["threshold"], final_policy["roi"]["threshold"]),
    ("roi_component", step2_policy["roi"]["component"], final_policy["roi"]["component_rule"]),
    ("roi_padding", step2_policy["roi"]["padding"], final_policy["roi"]["padding_pixels"]),
    ("roi_resize", str(step2_policy["roi"]["resize"]), str(final_policy["roi"]["crop_resize"])),
    ("roi_generator_checkpoint_sha256", EXPECTED_ROI_GENERATOR_SHA256, final_policy["roi"]["generator"]["checkpoint_sha256"]),
    ("fusion_equation", step2_policy["fusion"]["equation"], final_policy["probability"]["fusion_equation"]),
    ("global_threshold", step2_policy["global_threshold"], final_policy["probability"]["global_threshold"]),
    ("post_processing", step2_policy["post_processing"], final_policy["probability"]["post_processing"]),
    ("metric_epsilon", step2_policy["metric_epsilon"], final_policy["metrics"]["epsilon"]),
    ("random_seed", step2_policy["random_seed"], final_policy["random_seed"]),
    ("test_images_accessed", step2_policy["test_images_accessed"], final_policy["test_images_accessed"]),
]
freeze_diff = pd.DataFrame(checks, columns=["field", "step02_value", "step03_value"])
freeze_diff["passed"] = freeze_diff.apply(lambda r: r.step02_value == r.step03_value, axis=1)
save_csv(freeze_diff, "freeze_diff_check.csv")
assert freeze_diff.passed.all(), freeze_diff.loc[~freeze_diff.passed]

policy_path = OUTPUT_DIR / "final_inference_policy.json"
acceptance_path = OUTPUT_DIR / "final_acceptance_contract.json"
signature_payload = {
    "algorithm": "SHA-256", "created_utc": datetime.now(timezone.utc).isoformat(),
    "final_inference_policy_sha256": sha256(policy_path),
    "final_acceptance_contract_sha256": sha256(acceptance_path),
    "final_acceptance_table_sha256": sha256(OUTPUT_DIR / "final_acceptance_table.csv"),
    "artifact_checksum_inventory_sha256": sha256(OUTPUT_DIR / "artifact_checksum_inventory.csv"),
    "source_step02_gate_sha256": sha256(step2_gate_path),
    "test_images_accessed": False,
}
signature_path = save_json(signature_payload, "freeze_signature.json")

test_declaration = {
    "test_images_accessed": False, "test_masks_accessed": False,
    "test_probabilities_accessed": False, "test_statistics_computed": False,
    "test_loader_instantiated": False, "authorization_recorded": False,
    "statement": "Step 03 never opens the test split. Explicit owner authorization is still required for exactly one final evaluation."
}
save_json(test_declaration, "test_access_declaration.json")
display(freeze_diff)
print(f"Freeze signature: {sha256(signature_path)}")
"""),
    md("## 5. Authorization-readiness gate and dashboard"),
    code(r"""
validation_targets = step2_gate["target_passes"]
readiness = {
    "step01_gate_passed": bool(step1_gate["all_mandatory_targets_passed"]),
    "step02_validation_freeze_passed": bool(step2_gate["all_mandatory_targets_passed"] and step2_gate["result_level"] == "VALIDATION_FREEZE_PASS"),
    "all_six_validation_targets_passed": bool(len(validation_targets) == 6 and all(validation_targets.values())),
    "immutable_hashes_verified": True,
    "roi_generator_fully_frozen": bool(final_policy["roi"]["generator"]["checkpoint_sha256"] == EXPECTED_ROI_GENERATOR_SHA256 and final_policy["roi"]["generator"]["empty_roi_fallback"] == "full_image_box_[0,256,0,256]"),
    "artifact_inventory_complete": bool(inventory.exists.all()),
    "probability_cache_inventory_complete": bool(len(cache_inventory) == 13),
    "freeze_fields_identical": bool(freeze_diff.passed.all()),
    "final_acceptance_declared_before_test": True,
    "test_lock_intact": True,
}
all_ready = all(readiness.values())

expected_vs_actual = pd.DataFrame([
    {"requirement": k, "expected": True, "actual": bool(v), "passed": bool(v)}
    for k, v in readiness.items()
])
save_csv(expected_vs_actual, "expected_vs_actual.csv")

authorization_readiness = {
    "status": "READY_TO_REQUEST_EXPLICIT_ONE_TIME_TEST_AUTHORIZATION" if all_ready else "NOT_READY",
    "ready": all_ready, "requirements": readiness,
    "authorization_granted": False,
    "required_authorization_text": "I explicitly authorize one-time locked test evaluation using the Step 03 frozen policy and acceptance contract.",
    "test_images_accessed": False,
}
save_json(authorization_readiness, "authorization_readiness.json")

gate_result = {
    "status": "final_inference_policy_freeze_complete" if all_ready else "final_inference_policy_freeze_failed",
    "result_level": "VALIDATION_FREEZE_PASS" if all_ready else "FAILED_GATE",
    "selected_configuration": {"fusion": "maximum", "threshold": 0.70, "post_processing": "none", "policy_sha256": sha256(policy_path), "acceptance_contract_sha256": sha256(acceptance_path)},
    "selected_metrics": {"prerequisite_gates_passed": 2, "validation_targets_passed": int(sum(validation_targets.values())), "frozen_artifacts": int(len(inventory)), "validation_probability_caches": int(len(cache_inventory))},
    "targets": {k: True for k in readiness}, "target_passes": readiness,
    "all_mandatory_targets_passed": all_ready,
    "decision": "REQUEST_EXPLICIT_ONE_TIME_TEST_AUTHORIZATION" if all_ready else "HOLD_REPAIR_FREEZE_PACKAGE",
    "next_step": "step_04_one_time_locked_test_evaluation_after_explicit_authorization" if all_ready else PHASE,
    "manifest_sha256": recomputed["manifest"],
    "input_artifact_hashes": {"step01_gate": sha256(step1_gate_path), "step02_gate": sha256(step2_gate_path), "step02_policy": sha256(step2_policy_path), "freeze_signature": sha256(signature_path)},
    "test_images_accessed": False,
}
save_json(gate_result, "gate_result.json")

configuration = {
    "phase": PHASE, "mode": "NO_INFERENCE_POLICY_FREEZE", "random_seed": 42,
    "project_root": str(PROJECT_ROOT), "output_directory": str(OUTPUT_DIR),
    "manifest_path": str(manifest_path), "manifest_sha256": recomputed["manifest"],
    "roi_generator_checkpoint": str(roi_generator_checkpoint), "roi_generator_checkpoint_sha256": recomputed["roi_generator_checkpoint"],
    "allowed_splits": [], "test_loader_allowed": False,
    "frozen_policy_file": "final_inference_policy.json",
    "acceptance_contract_file": "final_acceptance_contract.json",
    "expected_outputs": EXPECTED_OUTPUTS, "test_images_accessed": False,
}
save_json(configuration, "configuration.json")
provenance = {
    "created_utc": datetime.now(timezone.utc).isoformat(), "phase": PHASE,
    "python": sys.version, "platform": platform.platform(),
    "numpy": np.__version__, "pandas": pd.__version__,
    "source_gate_hashes": {"step01": sha256(step1_gate_path), "step02": sha256(step2_gate_path)},
    "manifest_sha256": recomputed["manifest"], "roi_generator_checkpoint_sha256": recomputed["roi_generator_checkpoint"], "freeze_signature_sha256": sha256(signature_path),
    "operations": ["read prerequisite JSON/CSV/PNG/notebook/source/checkpoint bytes for SHA-256", "write Step 03 contract artifacts"],
    "prohibited_operations_confirmed_absent": ["test loading", "training", "model inference", "threshold search", "checkpoint selection"],
    "test_images_accessed": False,
}
save_json(provenance, "provenance.json")

fig, ax = plt.subplots(figsize=(12, 6))
labels = list(readiness)
values = [int(readiness[k]) for k in labels]
colors = ["#2e7d32" if v else "#c62828" for v in values]
ax.barh(range(len(labels)), values, color=colors)
ax.set_yticks(range(len(labels)), [x.replace("_", " ") for x in labels])
ax.set_xlim(0, 1.08); ax.set_xticks([0, 1], ["FAIL", "PASS"]); ax.invert_yaxis()
ax.set_title("Step 03 final-policy freeze readiness — test remains locked")
for i, v in enumerate(values): ax.text(v + 0.02, i, "PASS" if v else "FAIL", va="center", fontsize=9)
fig.tight_layout(); fig.savefig(OUTPUT_DIR / "freeze_readiness_dashboard.png", dpi=160); plt.close(fig)

summary_md = f'''# Final Inference Policy Freeze

- Status: `{gate_result['status']}`
- Result level: `{gate_result['result_level']}`
- Frozen fusion: pixelwise maximum
- Frozen global threshold: `0.70`
- Post-processing: none
- Frozen artifacts checksummed: `{len(inventory)}`
- Validation probability caches inventoried: `{len(cache_inventory)}`
- Formal acceptance contract: `FROZEN_PENDING_OWNER_AUTHORIZATION`
- Historical stronger targets: research aspirations, not mandatory gates
- Test images accessed: `false`
- Authorization granted: `false`
- Decision: `{gate_result['decision']}`

This freeze prohibits test-driven threshold, checkpoint, fusion, post-processing, or training changes. A later test failure must be reported without tuning or rerunning on test.
'''
(OUTPUT_DIR / "FINAL_INFERENCE_POLICY_FREEZE.md").write_text(summary_md, encoding="utf-8")
display(expected_vs_actual)
print(json.dumps(gate_result, indent=2))
"""),
    md("## 6. Final completeness assertion"),
    code(r"""
missing = [name for name in EXPECTED_OUTPUTS if not (OUTPUT_DIR / name).is_file()]
assert not missing, f"Missing Step 03 outputs: {missing}"
gate = load_json(OUTPUT_DIR / "gate_result.json")
assert gate["all_mandatory_targets_passed"]
assert gate["decision"] == "REQUEST_EXPLICIT_ONE_TIME_TEST_AUTHORIZATION"
assert gate["test_images_accessed"] is False
print(f"PASS: {len(EXPECTED_OUTPUTS)}/{len(EXPECTED_OUTPUTS)} required outputs exist.")
print("Step 03 is complete. Test is still locked; explicit one-time authorization has not been granted.")
"""),
]

nb = nbf.v4.new_notebook(cells=cells)
nb["metadata"] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.11"},
    "project_contract": {"phase": "step_03_final_inference_policy_freeze", "test_images_accessed": False, "full_model_inference": False},
}
PHASE_DIR.mkdir(parents=True, exist_ok=True)
with NOTEBOOK_PATH.open("w", encoding="utf-8") as f:
    nbf.write(nb, f)
print(NOTEBOOK_PATH)
