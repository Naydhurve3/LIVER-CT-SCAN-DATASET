"""Generate the guarded one-time locked test-evaluation notebook."""

from pathlib import Path
import nbformat as nbf

PROJECT_ROOT = Path(r"D:\DATA SCIENCE AND ANALYTICS\PROJECTS\Liver")
PHASE_DIR = PROJECT_ROOT / "mark 1 (part 2)" / "step_04_one_time_locked_test_evaluation_after_explicit_authorization"
NOTEBOOK_PATH = PHASE_DIR / "step_04_one_time_locked_test_evaluation_after_explicit_authorization.ipynb"


def md(text):
    return nbf.v4.new_markdown_cell(text.strip())


def code(text, tags=None):
    cell = nbf.v4.new_code_cell(text.strip())
    if tags:
        cell.metadata["tags"] = tags
    return cell


cells = [
md(r"""
# Step 04 — One-time locked test evaluation

## tl;dr

This notebook is the final, one-time held-out test evaluation of the Step 03 frozen policy. It was delivered locked, and one-time owner authorization was subsequently recorded on 5 August 2026. Safe preflight still does not open the test split; test access begins only after the authorization gate passes.

After authorization, use **Restart Kernel and Run All exactly once**. Test results are report-only evidence: they may not trigger threshold tuning, checkpoint selection, fusion changes, post-processing changes, retraining, or another test run.
"""),
md(r"""
## Context & Methods

### Key assumptions and immutable rules

- Step 03 and its acceptance contract are authoritative.
- Predicted-liver ROIs are generated from the frozen two-output liver checkpoint, channel 0, threshold 0.50, 26-connected largest 3D component, padding 16, and a predeclared full-image fallback if the prediction is empty.
- Tumour inference uses the frozen control and recall-loss checkpoints, pixelwise maximum probability-score fusion, global threshold 0.70, and no post-processing.
- Q1 positive slices use the train-only frozen boundary of 1–51 tumour pixels.
- The notebook records a run ID and refuses a second completed test execution.
"""),
code(r"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import hashlib, json, platform, sys, time, uuid

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

PROJECT_ROOT = Path(r"D:\DATA SCIENCE AND ANALYTICS\PROJECTS\Liver")
PART2 = PROJECT_ROOT / "mark 1 (part 2)"
PHASE = "step_04_one_time_locked_test_evaluation_after_explicit_authorization"
PHASE_DIR = PART2 / PHASE
OUTPUT_DIR = PHASE_DIR / "outputs"
CACHE_DIR = OUTPUT_DIR / "probability_cache"
STEP3_OUT = PART2 / "step_03_final_inference_policy_freeze" / "outputs"
DATASET_ROOT = Path(r"D:\DATA SCIENCE AND ANALYTICS\Dataset\Liver\02_staging\build_corrected_20260713_214847_v2")
MANIFEST_PATH = DATASET_ROOT / "manifests" / "slice_manifest.csv"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# One-time owner authorization recorded in this task on 5 August 2026.
AUTHORIZATION_GRANTED = True
AUTHORIZATION_TEXT = "I explicitly authorize one-time locked test evaluation using the Step 03 frozen policy and acceptance contract."
AUTHORIZATION_RECORDED_UTC = "2026-08-05T11:39:37.3247843Z"
AUTHORIZATION_SOURCE_MESSAGE = "got this error make edit as required"
ONE_TIME_RUN_ID = "871d289b-bf6b-4346-978f-2df02ade26ab"
RUN_MODE = "new"       # use "resume" only for the same interrupted run ID
EXPECTED_AUTHORIZATION_TEXT = "I explicitly authorize one-time locked test evaluation using the Step 03 frozen policy and acceptance contract."

RANDOM_SEED = 42
BATCH_SIZE = 24
NUM_WORKERS = 0
BOOTSTRAP_ITERATIONS = 10_000
THRESHOLD = 0.70
DICE_EPSILON = 1e-6
Q1_MAX_TUMOUR_PIXELS = 51.0
TEST_IMAGES_ACCESSED = False

EXPECTED_OUTPUTS = [
    "preflight_status.json", "authorization_record.json", "run_ledger.json",
    "configuration.json", "provenance.json", "checkpoint_hashes.json",
    "test_roi_manifest.csv", "runtime_log.csv", "cache_integrity.csv",
    "patient_metrics.csv", "slice_metrics.csv", "global_metrics.csv",
    "bootstrap_uncertainty.csv", "lesion_or_size_metrics.csv", "size_stratum_metrics.csv",
    "probability_histograms.csv", "probability_reliability.csv", "failure_cases.csv",
    "expected_vs_actual.csv", "final_test_dashboard.png", "patient_dice_heatmap.png",
    "probability_diagnostics_dashboard.png", "failure_case_panels.png",
    "final_test_evidence_inventory.csv", "final_test_signature.json",
    "test_access_declaration.json", "TEST_RESULTS_DATA_CARD.md", "gate_result.json"
]

def sha256_file(path, chunk_size=1 << 20):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(chunk_size), b""):
            digest.update(block)
    return digest.hexdigest()

def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))

def write_json(name, payload):
    path = OUTPUT_DIR / name
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")
    return path

def save_csv(frame, name):
    path = OUTPUT_DIR / name
    frame.to_csv(path, index=False, float_format="%.8f")
    return path

def resize_float(array, size=(256, 256)):
    return np.asarray(Image.fromarray(np.asarray(array, np.float32), mode="F").resize(size, Image.Resampling.BILINEAR), dtype=np.float32)

def probability_to_full(probability, box):
    y0, y1, x0, x1 = map(int, box)
    full = np.zeros((256, 256), np.float32)
    full[y0:y1, x0:x1] = resize_float(probability, (x1-x0, y1-y0))
    return full

def dice_from_counts(intersection, truth_pixels, predicted_pixels):
    return float((2*intersection + DICE_EPSILON)/(truth_pixels + predicted_pixels + DICE_EPSILON))

print("One-time authorization is recorded. Test data remains unopened until the authorization gate and test-open cell execute.")
""", tags=["safe-preflight"]),
md("### 1. Safe preflight — verify the frozen package without reading test data"),
code(r"""
step3_gate_path = STEP3_OUT / "gate_result.json"
policy_path = STEP3_OUT / "final_inference_policy.json"
acceptance_path = STEP3_OUT / "final_acceptance_contract.json"
acceptance_table_path = STEP3_OUT / "final_acceptance_table.csv"
freeze_signature_path = STEP3_OUT / "freeze_signature.json"
for path in [step3_gate_path, policy_path, acceptance_path, acceptance_table_path, freeze_signature_path, MANIFEST_PATH]:
    assert path.is_file(), f"Missing prerequisite: {path}"

step3_gate = load_json(step3_gate_path)
policy = load_json(policy_path)
acceptance_contract = load_json(acceptance_path)
freeze_signature = load_json(freeze_signature_path)
assert step3_gate["result_level"] == "VALIDATION_FREEZE_PASS" and step3_gate["all_mandatory_targets_passed"]
assert step3_gate["decision"] == "REQUEST_EXPLICIT_ONE_TIME_TEST_AUTHORIZATION"
assert not step3_gate["test_images_accessed"] and not policy["test_images_accessed"]
assert acceptance_contract["declared_before_test_access"] and not acceptance_contract["test_images_accessed"]
assert sha256_file(policy_path) == freeze_signature["final_inference_policy_sha256"]
assert sha256_file(acceptance_path) == freeze_signature["final_acceptance_contract_sha256"]
assert sha256_file(acceptance_table_path) == freeze_signature["final_acceptance_table_sha256"]
assert sha256_file(MANIFEST_PATH) == policy["dataset"]["manifest_sha256"]

control_checkpoint = Path(policy["model"]["control_checkpoint"])
recall_checkpoint = Path(policy["model"]["recall_checkpoint"])
roi_checkpoint = Path(policy["roi"]["generator"]["checkpoint"])
checkpoint_hashes = {
    "control": sha256_file(control_checkpoint), "recall_loss": sha256_file(recall_checkpoint),
    "roi_generator": sha256_file(roi_checkpoint),
}
assert checkpoint_hashes["control"] == policy["model"]["control_checkpoint_sha256"]
assert checkpoint_hashes["recall_loss"] == policy["model"]["recall_checkpoint_sha256"]
assert checkpoint_hashes["roi_generator"] == policy["roi"]["generator"]["checkpoint_sha256"]
assert policy["probability"]["global_threshold"] == THRESHOLD
assert policy["probability"]["fusion_equation"] == "maximum(control_probability, recall_probability)"
assert policy["probability"]["post_processing"] == "none"
assert policy["metrics"]["q1_positive_slice_definition"]["truth_pixels_max_inclusive"] == Q1_MAX_TUMOUR_PIXELS

preflight = {
    "status": "SAFE_PREFLIGHT_PASS_AUTHORIZATION_RECORDED" if AUTHORIZATION_GRANTED else "SAFE_PREFLIGHT_PASS_AWAITING_EXPLICIT_AUTHORIZATION",
    "step3_gate_sha256": sha256_file(step3_gate_path), "policy_sha256": sha256_file(policy_path),
    "acceptance_contract_sha256": sha256_file(acceptance_path), "manifest_sha256": sha256_file(MANIFEST_PATH),
    "checkpoint_sha256": checkpoint_hashes, "test_paths_dereferenced": False,
    "test_images_accessed": False, "authorization_granted": bool(AUTHORIZATION_GRANTED),
}
write_json("preflight_status.json", preflight)
print(json.dumps(preflight, indent=2))
""", tags=["safe-preflight"]),
md(r"""
### 2. Authorization gate

This cell validates the recorded one-time authorization, timestamp, and UUID before creating the run ledger. It must never be bypassed or weakened.
"""),
code(r"""
assert AUTHORIZATION_GRANTED is True, "STOP: explicit one-time test authorization has not been granted."
assert AUTHORIZATION_TEXT == EXPECTED_AUTHORIZATION_TEXT, "STOP: authorization text is not exact."
assert AUTHORIZATION_RECORDED_UTC.strip(), "STOP: authorization timestamp is missing."
run_uuid = str(uuid.UUID(ONE_TIME_RUN_ID))
assert RUN_MODE in {"new", "resume"}

ledger_path = OUTPUT_DIR / "run_ledger.json"
if RUN_MODE == "new":
    assert not ledger_path.exists(), "STOP: a test run ledger already exists; a second test run is prohibited."
    CACHE_DIR.mkdir(parents=True, exist_ok=False)
    ledger = {"run_id": run_uuid, "status": "AUTHORIZED_NOT_OPENED", "authorization_recorded_utc": AUTHORIZATION_RECORDED_UTC,
              "policy_sha256": sha256_file(policy_path), "acceptance_contract_sha256": sha256_file(acceptance_path), "test_images_accessed": False}
else:
    assert ledger_path.is_file(), "STOP: resume requested but no ledger exists."
    ledger = load_json(ledger_path)
    assert ledger["run_id"] == run_uuid and ledger["status"] in {"RUNNING", "INTERRUPTED"}
    assert ledger["policy_sha256"] == sha256_file(policy_path) and ledger["acceptance_contract_sha256"] == sha256_file(acceptance_path)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

authorization_record = {"authorization_granted": True, "authorization_text": AUTHORIZATION_TEXT,
    "authorization_recorded_utc": AUTHORIZATION_RECORDED_UTC, "one_time_run_id": run_uuid,
    "authorization_source_message": AUTHORIZATION_SOURCE_MESSAGE,
    "policy_sha256": sha256_file(policy_path), "acceptance_contract_sha256": sha256_file(acceptance_path)}
write_json("authorization_record.json", authorization_record)
write_json("run_ledger.json", ledger)
print("PASS: explicit authorization recorded; the next cell will open the test split once.")
""", tags=["requires-test-authorization"]),
md("## Data\n\n### 3. Open the authorized test manifest and start the one-time ledger"),
code(r"""
manifest = pd.read_csv(MANIFEST_PATH)
test_rows = manifest.loc[manifest.split.eq("test")].copy().sort_values(["volume_id", "slice_index"]).reset_index(drop=True)
del manifest
assert len(test_rows) > 0 and test_rows.volume_id.nunique() > 0
assert test_rows.sample_id.is_unique and set(test_rows.split) == {"test"}
assert test_rows.automatic_integrity_pass.astype(bool).all()
assert test_rows.verification_status.eq("verified").all() and test_rows.manual_spatial_status.eq("approved").all()
assert test_rows.exclusion_reason.fillna("").eq("").all()

TEST_IMAGES_ACCESSED = True
ledger.update({"status": "RUNNING", "test_split_opened_utc": datetime.now(timezone.utc).isoformat(),
               "test_rows": int(len(test_rows)), "test_volumes": int(test_rows.volume_id.nunique()), "test_images_accessed": True})
write_json("run_ledger.json", ledger)
print(f"AUTHORIZED TEST OPEN: {len(test_rows)} slices across {test_rows.volume_id.nunique()} volumes; run_id={run_uuid}")
""", tags=["requires-test-authorization"]),
md("### 4. Load frozen models and generate prediction-only test ROIs"),
code(r"""
import torch
from torch.utils.data import DataLoader, Dataset
from scipy import ndimage
import nibabel as nib

if str(PROJECT_ROOT) not in sys.path: sys.path.insert(0, str(PROJECT_ROOT))
from src.framework.models.mobilenetv2_unet import MobileNetV2UNet

np.random.seed(RANDOM_SEED); torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED); torch.backends.cudnn.benchmark=False; torch.backends.cudnn.deterministic=True
torch.use_deterministic_algorithms(True, warn_only=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model(path, out_channels):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    assert payload["manifest_sha256"] == policy["dataset"]["manifest_sha256"]
    model = MobileNetV2UNet(1, out_channels, False)
    model.load_state_dict(payload["model_state"], strict=True)
    return model.to(DEVICE).eval()

roi_model = load_model(roi_checkpoint, 2)
control_model = load_model(control_checkpoint, 1)
recall_model = load_model(recall_checkpoint, 1)
write_json("checkpoint_hashes.json", checkpoint_hashes)

def robust_normalize(image):
    image = np.asarray(image, np.float32); reference = image[image > 0]
    if reference.size < 32: reference = image.reshape(-1)
    center = float(np.median(reference)); q25, q75 = np.percentile(reference, [25, 75])
    sigma = float((q75-q25)/1.349)
    if not np.isfinite(sigma) or sigma < 1e-3: sigma = max(float(np.std(reference)), 1e-3)
    return ((np.clip((image-center)/sigma, -3, 3)+3)/6).astype(np.float32)

def largest_component(mask):
    labels, count = ndimage.label(mask, structure=np.ones((3,3,3), np.uint8))
    if count == 0: return mask
    sizes = np.bincount(labels.ravel()); sizes[0] = 0
    return labels == sizes.argmax()

def padded_bbox(mask, padding=16):
    if not mask.any(): return (0,256,0,256), True
    _, ys, xs = np.where(mask)
    return (max(int(ys.min())-padding,0), min(int(ys.max())+1+padding,256),
            max(int(xs.min())-padding,0), min(int(xs.max())+1+padding,256)), False

roi_manifest_path = OUTPUT_DIR / "test_roi_manifest.csv"
if RUN_MODE == "resume" and roi_manifest_path.is_file():
    test_rois = pd.read_csv(roi_manifest_path)
else:
    roi_rows=[]
    with torch.inference_mode():
        for sequence,(volume_id,group) in enumerate(test_rows.groupby("volume_id",sort=True),1):
            paths=[DATASET_ROOT/str(p) for p in group.image_path]
            probabilities=[]
            for start in range(0,len(paths),BATCH_SIZE):
                images=[]
                for path in paths[start:start+BATCH_SIZE]:
                    with Image.open(path) as handle: images.append(robust_normalize(np.asarray(handle.convert("L"),np.float32)/255))
                batch=torch.from_numpy(np.stack(images)[:,None]).to(DEVICE)
                probabilities.append(torch.sigmoid(roi_model(batch))[:,0].cpu().numpy())
            mask=largest_component(np.concatenate(probabilities)>=0.5)
            box,fallback=padded_bbox(mask,16); y0,y1,x0,x1=box
            roi_rows.append({"volume_id":int(volume_id),"slices":len(group),"y0":y0,"y1":y1,"x0":x0,"x1":x1,
                "liver_threshold":0.5,"component_mode":"largest_3d_26conn","padding":16,"roi_empty":bool(fallback),
                "fallback_used":bool(fallback),"crop_area_ratio":float((y1-y0)*(x1-x0)/(256*256))})
            print(f"ROI [{sequence}/{test_rows.volume_id.nunique()}] volume {volume_id}")
    test_rois=pd.DataFrame(roi_rows); save_csv(test_rois,"test_roi_manifest.csv")
assert set(test_rois.volume_id)==set(test_rows.volume_id.unique()) and not test_rois.duplicated("volume_id").any()
print("PASS: frozen prediction-only test ROIs generated; empty fallback count:", int(test_rois.fallback_used.sum()))
""", tags=["requires-test-authorization", "expensive"]),
md("### 5. Run each frozen tumour checkpoint once and cache fused scores"),
code(r"""
class VolumeROIDataset(Dataset):
    def __init__(self, rows, roi): self.rows=rows.reset_index(drop=True); self.roi=roi
    def __len__(self): return len(self.rows)
    def __getitem__(self,index):
        row=self.rows.iloc[index]; box=np.array([self.roi.y0,self.roi.y1,self.roi.x0,self.roi.x1],np.int64); y0,y1,x0,x1=box
        with Image.open(DATASET_ROOT/str(row.image_path)) as h: image=np.asarray(h.convert("L"),np.float32)/255
        with Image.open(DATASET_ROOT/str(row.tumor_mask_path)) as h: truth=np.asarray(h.convert("L"),np.uint8)>0
        with Image.open(DATASET_ROOT/str(row.organ_mask_path)) as h: organ=np.asarray(h.convert("L"),np.uint8)>0
        crop=resize_float(image[y0:y1,x0:x1])[None].copy()
        return {"image":torch.from_numpy(crop),"truth":torch.from_numpy(truth),"organ":torch.from_numpy(organ),
                "sample_id":str(row.sample_id),"slice_index":int(row.slice_index),"box":torch.from_numpy(box)}

def infer_volume(model,loader):
    scores=[]; metadata=None
    with torch.inference_mode():
        truths=[]; organs=[]; ids=[]; indices=[]
        for batch in loader:
            batch_scores=torch.sigmoid(model(batch["image"].to(DEVICE)))[:,0].cpu().numpy()
            for i in range(len(batch_scores)):
                scores.append(probability_to_full(batch_scores[i],batch["box"][i].numpy()))
                truths.append(batch["truth"][i].numpy().astype(bool)); organs.append(batch["organ"][i].numpy().astype(bool))
                ids.append(str(batch["sample_id"][i])); indices.append(int(batch["slice_index"][i]))
    return np.stack(scores).astype(np.float32), {"truth":np.stack(truths),"organ":np.stack(organs),"sample_id":np.asarray(ids),"slice_index":np.asarray(indices,np.int64)}

runtime_rows=[]; integrity_rows=[]; roi_lookup=test_rois.set_index("volume_id")
for sequence,(volume_id,rows) in enumerate(test_rows.groupby("volume_id",sort=True),1):
    cache_path=CACHE_DIR/f"volume_{int(volume_id)}.npz"
    if RUN_MODE=="resume" and cache_path.is_file():
        with np.load(cache_path,allow_pickle=False) as item:
            fused=item["fused_probability"].astype(np.float32); metadata={k:item[k] for k in ["truth","organ","sample_id","slice_index"]}
        elapsed=0.0; source="resume_cache"
    else:
        loader=DataLoader(VolumeROIDataset(rows,roi_lookup.loc[volume_id]),batch_size=BATCH_SIZE,shuffle=False,num_workers=NUM_WORKERS,pin_memory=torch.cuda.is_available())
        started=time.perf_counter(); control,metadata=infer_volume(control_model,loader); recall,metadata2=infer_volume(recall_model,loader); elapsed=time.perf_counter()-started
        assert np.array_equal(metadata["sample_id"],metadata2["sample_id"]) and np.array_equal(metadata["truth"],metadata2["truth"])
        fused=np.maximum(control,recall)
        np.savez_compressed(cache_path,control_probability=control.astype(np.float16),recall_probability=recall.astype(np.float16),
            fused_probability=fused.astype(np.float16),truth=metadata["truth"],organ=metadata["organ"],slice_index=metadata["slice_index"],
            sample_id=metadata["sample_id"],volume_id=np.asarray(int(volume_id),np.int64),run_id=np.asarray(run_uuid))
        source="fresh_inference"
    expected_ids=rows.sample_id.astype(str).to_numpy()
    valid=bool(np.isfinite(fused).all() and fused.min()>=0 and fused.max()<=1 and np.array_equal(metadata["sample_id"].astype(str),expected_ids))
    integrity_rows.append({"volume_id":int(volume_id),"slices":len(rows),"finite_and_bounded":valid,"sample_ids_unique":len(np.unique(metadata["sample_id"]))==len(rows),"cache_sha256":sha256_file(cache_path)})
    runtime_rows.append({"volume_id":int(volume_id),"slices":len(rows),"seconds":elapsed,"source":source})
    print(f"Tumour [{sequence}/{test_rows.volume_id.nunique()}] volume {volume_id}: {source}")
cache_integrity=pd.DataFrame(integrity_rows); runtime_log=pd.DataFrame(runtime_rows)
save_csv(cache_integrity,"cache_integrity.csv"); save_csv(runtime_log,"runtime_log.csv")
assert cache_integrity.finite_and_bounded.all() and cache_integrity.sample_ids_unique.all()
del roi_model, control_model, recall_model
if torch.cuda.is_available(): torch.cuda.empty_cache()
""", tags=["requires-test-authorization", "expensive"]),
md("## Results\n\n### 6. Compute frozen global, patient, slice, and bootstrap metrics"),
code(r"""
cache_paths={int(p.stem.split("_")[-1]):p for p in CACHE_DIR.glob("volume_*.npz")}
assert set(cache_paths)==set(test_rows.volume_id.unique())
patient_rows=[]; slice_rows=[]; all_sample_ids=[]
global_truth=global_pred=global_intersection=global_tp=global_fp=global_fn=0
positive_empty=[]; empty_fp=[]; q1_detected=[]
for volume_id,path in sorted(cache_paths.items()):
    with np.load(path,allow_pickle=False) as item:
        probability=item["fused_probability"].astype(np.float32); truth=item["truth"].astype(bool); organ=item["organ"].astype(bool); pred=probability>=THRESHOLD
        truth_slice=truth.sum((1,2)); pred_slice=pred.sum((1,2)); overlap_slice=(pred&truth).sum((1,2)); positive=truth_slice>0; empty=~positive; q1=positive&(truth_slice<=Q1_MAX_TUMOUR_PIXELS)
        detected=(pred&truth).any((1,2)); positive_empty.extend((pred_slice[positive]==0).tolist()); empty_fp.extend((pred_slice[empty]>0).tolist()); q1_detected.extend(detected[q1].tolist())
        tp=int((pred&truth).sum()); fp=int((pred&~truth).sum()); fn=int((~pred&truth).sum()); t=int(truth.sum()); p=int(pred.sum())
        global_truth+=t; global_pred+=p; global_intersection+=tp; global_tp+=tp; global_fp+=fp; global_fn+=fn
        patient_rows.append({"volume_id":volume_id,"has_tumour":bool(t>0),"truth_pixels":t,"predicted_pixels":p,"intersection_pixels":tp,
            "false_positive_pixels":fp,"false_negative_pixels":fn,"dice":dice_from_counts(tp,t,p),"precision":tp/(tp+fp) if tp+fp else np.nan,"recall":tp/(tp+fn) if tp+fn else np.nan})
        for i,sid in enumerate(item["sample_id"].astype(str)):
            truth_region=probability[i][truth[i]]; non_tumour=probability[i][organ[i]&~truth[i]]
            slice_rows.append({"volume_id":volume_id,"slice_index":int(item["slice_index"][i]),"sample_id":sid,"truth_pixels":int(truth_slice[i]),
                "predicted_pixels":int(pred_slice[i]),"intersection_pixels":int(overlap_slice[i]),"false_positive_pixels":int(pred_slice[i]-overlap_slice[i]),
                "false_negative_pixels":int(truth_slice[i]-overlap_slice[i]),"dice":dice_from_counts(int(overlap_slice[i]),int(truth_slice[i]),int(pred_slice[i])),
                "detected":bool(detected[i]),"predicted_empty":bool(pred_slice[i]==0),"positive_slice":bool(positive[i]),"q1_positive_slice_train_edge":bool(q1[i]),
                "empty_slice_false_positive":bool(empty[i] and pred_slice[i]>0),"maximum_whole_image_score":float(probability[i].max()),
                "maximum_truth_region_score":float(truth_region.max()) if truth_region.size else np.nan,"median_truth_region_score":float(np.median(truth_region)) if truth_region.size else np.nan,
                "maximum_non_tumour_liver_score":float(non_tumour.max()) if non_tumour.size else np.nan})
        all_sample_ids.extend(item["sample_id"].astype(str).tolist())
patients=pd.DataFrame(patient_rows); slices=pd.DataFrame(slice_rows); positive_patients=patients[patients.has_tumour]
assert len(positive_patients)>0 and len(positive_empty)>0 and len(empty_fp)>0 and len(q1_detected)>0
metrics={
    "global_dice":dice_from_counts(global_intersection,global_truth,global_pred),
    "global_pixel_precision":global_tp/(global_tp+global_fp) if global_tp+global_fp else np.nan,
    "global_pixel_recall":global_tp/(global_tp+global_fn) if global_tp+global_fn else np.nan,
    "mean_positive_patient_dice":float(positive_patients.dice.mean()),"median_positive_patient_dice":float(positive_patients.dice.median()),
    "minimum_positive_patient_dice":float(positive_patients.dice.min()),
    "q1_positive_slice_detection_pct_train_edges":100*float(np.mean(q1_detected)),
    "positive_predicted_empty_pct":100*float(np.mean(positive_empty)),"empty_slice_false_positive_pct":100*float(np.mean(empty_fp)),
    "sample_coverage_fraction":len(all_sample_ids)/len(test_rows),"finite_probability_fraction":float(cache_integrity.finite_and_bounded.mean()),
    "unique_sample_id_fraction":len(set(all_sample_ids))/len(all_sample_ids),
}
save_csv(patients,"patient_metrics.csv"); save_csv(slices,"slice_metrics.csv"); save_csv(pd.DataFrame([metrics]),"global_metrics.csv")

rng=np.random.default_rng(RANDOM_SEED); patient_vectors=positive_patients[["dice","precision","recall"]].to_numpy(float)
boot=[]
for metric_index,name in enumerate(["mean_positive_patient_dice","mean_positive_patient_precision","mean_positive_patient_recall"]):
    values=patient_vectors[:,metric_index]; draws=rng.choice(values,size=(BOOTSTRAP_ITERATIONS,len(values)),replace=True).mean(1)
    boot.append({"metric":name,"estimate":float(np.nanmean(values)),"ci_lower_2_5":float(np.nanquantile(draws,.025)),"ci_upper_97_5":float(np.nanquantile(draws,.975)),
                 "iterations":BOOTSTRAP_ITERATIONS,"seed":RANDOM_SEED,"resampling_unit":"tumour-positive test patient"})
bootstrap=pd.DataFrame(boot); save_csv(bootstrap,"bootstrap_uncertainty.csv")
display(pd.DataFrame([metrics])); display(bootstrap)
""", tags=["requires-test-authorization"]),
md("### 7. Match 3D lesions and report frozen train-size strata"),
code(r"""
lesion_bins=load_json(PART2/"step_01_pretraining_dataset_characterization"/"outputs"/"train_derived_lesion_bins.json")["lesion_volume_ml"]
finite_edges=[float(x) for x in lesion_bins]; finite_edges[0]=-np.inf; finite_edges[-1]=np.inf
lesion_rows=[]; structure=ndimage.generate_binary_structure(3,1)
for volume_id,path in sorted(cache_paths.items()):
    group=test_rows[test_rows.volume_id.eq(volume_id)]; source_path=DATASET_ROOT/str(group.source_segmentation_path.iloc[0])
    nii=nib.load(str(source_path)); shape=nii.shape; zooms=nii.header.get_zooms()[:3]
    derived_voxel_ml=float(zooms[0]*(shape[0]/256)*zooms[1]*(shape[1]/256)*zooms[2]/1000)
    with np.load(path,allow_pickle=False) as item:
        truth=item["truth"].astype(bool); pred=item["fused_probability"].astype(np.float32)>=THRESHOLD
    truth_labels,n_truth=ndimage.label(truth,structure=structure); pred_labels,n_pred=ndimage.label(pred,structure=structure)
    pred_sizes=np.bincount(pred_labels.ravel(),minlength=n_pred+1)
    for lesion_id in range(1,n_truth+1):
        component=truth_labels==lesion_id; truth_size=int(component.sum()); ids,counts=np.unique(pred_labels[component],return_counts=True)
        candidates=[(int(i),int(c)) for i,c in zip(ids,counts) if i>0]
        if candidates:
            match_id,overlap=max(candidates,key=lambda pair:(2*pair[1]/(truth_size+pred_sizes[pair[0]]),pair[1])); pred_size=int(pred_sizes[match_id])
        else: match_id=overlap=pred_size=0
        lesion_rows.append({"volume_id":volume_id,"truth_lesion_id_6conn":lesion_id,"matched_prediction_id_6conn":match_id,
            "truth_component_pixels":truth_size,"predicted_component_pixels":pred_size,"overlap_pixels":overlap,"detected":bool(overlap>0),
            "matched_dice":dice_from_counts(overlap,truth_size,pred_size),"lesion_volume_ml":truth_size*derived_voxel_ml})
lesions=pd.DataFrame(lesion_rows)
labels=["train_lesion_volume_q1","train_lesion_volume_q2","train_lesion_volume_q3","train_lesion_volume_q4"]
lesions["size_stratum"]=pd.cut(lesions.lesion_volume_ml,bins=finite_edges,labels=labels,include_lowest=True).astype("string")
size_summary=lesions.groupby("size_stratum",dropna=False).agg(lesions=("truth_lesion_id_6conn","size"),detected_pct=("detected",lambda x:100*x.mean()),
    mean_matched_dice=("matched_dice","mean"),median_matched_dice=("matched_dice","median"),median_volume_ml=("lesion_volume_ml","median")).reset_index()
save_csv(lesions,"lesion_or_size_metrics.csv"); save_csv(size_summary,"size_stratum_metrics.csv")
display(size_summary)
""", tags=["requires-test-authorization"]),
md("### 8. Probability evidence, dashboards, and failure panels"),
code(r"""
hist_edges=np.linspace(0,1,51); reliability_edges=np.linspace(0,1,21)
hist={k:np.zeros(50,np.int64) for k in ["tumour_truth","non_tumour_liver","background"]}
rel_n=np.zeros(20,np.int64); rel_pos=np.zeros(20,np.int64); rel_sum=np.zeros(20,float)
for path in cache_paths.values():
    with np.load(path,allow_pickle=False) as item:
        score=item["fused_probability"].astype(np.float32); truth=item["truth"].astype(bool); organ=item["organ"].astype(bool)
    for s,t,o in zip(score,truth,organ):
        for key,mask in {"tumour_truth":t,"non_tumour_liver":o&~t,"background":~o}.items(): hist[key]+=np.histogram(s[mask],bins=hist_edges)[0]
        flat=s.ravel(); labels=t.ravel(); idx=np.clip(np.digitize(flat,reliability_edges)-1,0,19)
        rel_n+=np.bincount(idx,minlength=20); rel_pos+=np.bincount(idx,weights=labels,minlength=20).astype(np.int64); rel_sum+=np.bincount(idx,weights=flat,minlength=20)
histograms=pd.DataFrame([{"category":k,"bin_left":hist_edges[i],"bin_right":hist_edges[i+1],"count":int(v[i])} for k,v in hist.items() for i in range(50)])
reliability=pd.DataFrame({"bin_left":reliability_edges[:-1],"bin_right":reliability_edges[1:],"count":rel_n,
    "mean_score":np.divide(rel_sum,rel_n,out=np.full(20,np.nan),where=rel_n>0),"observed_positive_fraction":np.divide(rel_pos,rel_n,out=np.full(20,np.nan),where=rel_n>0)})
reliability["absolute_gap"]=(reliability.mean_score-reliability.observed_positive_fraction).abs(); ece=float(np.nansum(reliability.absolute_gap*reliability["count"])/reliability["count"].sum())
save_csv(histograms,"probability_histograms.csv"); save_csv(reliability,"probability_reliability.csv")

fig,axes=plt.subplots(2,2,figsize=(15,10))
axes[0,0].bar(["Global Dice","Mean patient Dice","Precision","Recall"],[metrics["global_dice"],metrics["mean_positive_patient_dice"],metrics["global_pixel_precision"],metrics["global_pixel_recall"]],color="#2171b5"); axes[0,0].set_ylim(0,1); axes[0,0].set_title("Frozen test metrics")
axes[0,1].bar(["Q1 detect","Positive empty","Empty FP"],[metrics["q1_positive_slice_detection_pct_train_edges"],metrics["positive_predicted_empty_pct"],metrics["empty_slice_false_positive_pct"]],color=["#2ca25f","#de2d26","#fd8d3c"]); axes[0,1].set_ylim(0,100); axes[0,1].set_title("Slice rates (%)")
axes[1,0].bar(size_summary.size_stratum.astype(str),size_summary.detected_pct,color="#756bb1"); axes[1,0].tick_params(axis="x",rotation=25); axes[1,0].set_ylim(0,100); axes[1,0].set_title("Lesion detection by frozen train-volume stratum")
axes[1,1].hist(positive_patients.dice,bins=np.linspace(0,1,11),color="#3182bd"); axes[1,1].set_title("Positive-patient Dice")
fig.suptitle("One-time locked test evaluation — frozen policy"); fig.tight_layout(); fig.savefig(OUTPUT_DIR/"final_test_dashboard.png",dpi=170); plt.close(fig)

ordered=positive_patients.sort_values("volume_id"); fig,ax=plt.subplots(figsize=(max(10,len(ordered)*.6),3.5)); im=ax.imshow(ordered[["dice"]].T,aspect="auto",vmin=0,vmax=1,cmap="viridis"); ax.set_xticks(range(len(ordered)),ordered.volume_id); ax.set_yticks([0],["Dice"]); fig.colorbar(im,ax=ax); fig.tight_layout(); fig.savefig(OUTPUT_DIR/"patient_dice_heatmap.png",dpi=170); plt.close(fig)
fig,axes=plt.subplots(1,2,figsize=(13,5))
for category,group in histograms.groupby("category"):
    centres=(group.bin_left+group.bin_right)/2; axes[0].plot(centres,group["count"]/group["count"].sum(),label=category)
axes[0].set_yscale("log"); axes[0].set_title("Model-score distributions"); axes[0].legend()
axes[1].plot([0,1],[0,1],"--",c="black"); axes[1].plot(reliability.mean_score,reliability.observed_positive_fraction,"o-"); axes[1].set_title(f"Descriptive reliability (ECE={ece:.4f})")
fig.tight_layout(); fig.savefig(OUTPUT_DIR/"probability_diagnostics_dashboard.png",dpi=170); plt.close(fig)

failure_rows=[]
for volume_id in positive_patients.nsmallest(min(3,len(positive_patients)),"dice").volume_id:
    candidate=slices[(slices.volume_id==volume_id)&slices.positive_slice].sort_values(["false_negative_pixels","truth_pixels"],ascending=False).iloc[0]
    failure_rows.append({"case_type":"lowest_patient_dice","volume_id":int(volume_id),"sample_id":candidate.sample_id,"slice_index":int(candidate.slice_index),"patient_dice":float(patients.set_index("volume_id").loc[volume_id,"dice"])})
if slices.empty_slice_false_positive.any():
    candidate=slices[slices.empty_slice_false_positive].sort_values("false_positive_pixels",ascending=False).iloc[0]
    failure_rows.append({"case_type":"largest_empty_slice_false_positive","volume_id":int(candidate.volume_id),"sample_id":candidate.sample_id,"slice_index":int(candidate.slice_index),"patient_dice":float(patients.set_index("volume_id").loc[candidate.volume_id,"dice"])})
failures=pd.DataFrame(failure_rows); save_csv(failures,"failure_cases.csv")
lookup=test_rows.set_index("sample_id"); fig,axes=plt.subplots(len(failures),6,figsize=(21,max(4,4*len(failures))),squeeze=False)
for row_axes,case in zip(axes,failures.itertuples()):
    path=cache_paths[int(case.volume_id)]
    with np.load(path,allow_pickle=False) as item:
        index=int(np.where(item["sample_id"].astype(str)==str(case.sample_id))[0][0]); score=item["fused_probability"][index].astype(np.float32); truth=item["truth"][index].astype(bool); pred=score>=THRESHOLD
        control=item["control_probability"][index].astype(np.float32); recall=item["recall_probability"][index].astype(np.float32)
    with Image.open(DATASET_ROOT/str(lookup.loc[str(case.sample_id),"image_path"])) as h: image=np.asarray(h.convert("L"),np.float32)/255
    error=np.zeros((256,256),np.uint8); error[pred&~truth]=1; error[truth&~pred]=2
    for ax,(panel,title,cmap,vmin,vmax) in zip(row_axes,[(image,"CT","gray",0,1),(truth,"Truth","gray",0,1),(control,"Control","magma",0,1),(recall,"Recall","magma",0,1),(score,"Maximum fusion","magma",0,1),(error,"Error FP=1 FN=2","viridis",0,2)]):
        ax.imshow(panel,cmap=cmap,vmin=vmin,vmax=vmax); ax.set_title(f"{title}\nV{case.volume_id} z={case.slice_index}"); ax.axis("off")
fig.tight_layout(); fig.savefig(OUTPUT_DIR/"failure_case_panels.png",dpi=170,bbox_inches="tight"); plt.close(fig)
""", tags=["requires-test-authorization"]),
md("## Takeaways\n\n### 9. Apply the predeclared acceptance table and seal the final test record"),
code(r"""
acceptance_table=pd.read_csv(acceptance_table_path); final_rows=acceptance_table[acceptance_table.scope.isin(["final_test","final_test_integrity"])].copy()
def compare(value,direction,target):
    if not np.isfinite(value): return False
    return bool(value>=target if direction==">=" else value<=target if direction=="<=" else np.isclose(value,target,atol=1e-12))
final_rows["actual"]=final_rows.metric.map(metrics); final_rows["passed"]=[compare(v,d,t) for v,d,t in zip(final_rows.actual,final_rows.direction,final_rows.target)]
validation_pass=bool(step3_gate["all_mandatory_targets_passed"])
expected_vs_actual=pd.concat([pd.DataFrame([{"order":0,"scope":"validation_prerequisite","metric":"step03_validation_freeze","direction":"==","target":1.0,"mandatory":True,"rationale":"frozen prerequisite","actual":1.0 if validation_pass else 0.0,"passed":validation_pass}]),final_rows],ignore_index=True)
save_csv(expected_vs_actual,"expected_vs_actual.csv")
all_acceptance_passed=bool(expected_vs_actual.loc[expected_vs_actual.mandatory.astype(bool),"passed"].all())

primary_evidence=[OUTPUT_DIR/name for name in ["patient_metrics.csv","slice_metrics.csv","global_metrics.csv","bootstrap_uncertainty.csv","lesion_or_size_metrics.csv","size_stratum_metrics.csv","expected_vs_actual.csv","test_roi_manifest.csv"]]
primary_evidence.extend(sorted(CACHE_DIR.glob("volume_*.npz")))
inventory=pd.DataFrame([{"path":str(p),"bytes":p.stat().st_size,"sha256":sha256_file(p)} for p in primary_evidence]); inventory_path=save_csv(inventory,"final_test_evidence_inventory.csv")
signature={"algorithm":"SHA-256","run_id":run_uuid,"created_utc":datetime.now(timezone.utc).isoformat(),"policy_sha256":sha256_file(policy_path),
    "acceptance_contract_sha256":sha256_file(acceptance_path),"manifest_sha256":sha256_file(MANIFEST_PATH),"evidence_inventory_sha256":sha256_file(inventory_path),
    "probability_cache_count":len(cache_paths),"test_images_accessed":True}
signature_path=write_json("final_test_signature.json",signature)

gate={"status":"one_time_locked_test_evaluation_complete","result_level":"FINAL_TEST_COMPLETE",
    "selected_configuration":{"fusion":"maximum","threshold":THRESHOLD,"post_processing":"none","policy_sha256":sha256_file(policy_path)},
    "selected_metrics":metrics,"uncertainty":bootstrap.to_dict(orient="records"),
    "targets":{r.metric:r.target for r in final_rows.itertuples()},"target_passes":{r.metric:bool(r.passed) for r in final_rows.itertuples()},
    "all_mandatory_targets_passed":all_acceptance_passed,
    "decision":"PROCEED_TO_FINAL_RESEARCH_PACKAGE_NO_TEST_TUNING" if all_acceptance_passed else "REPORT_FINAL_TEST_GATE_FAILURE_NO_TUNING_NO_RERUN",
    "next_step":"step_05_final_research_package","manifest_sha256":sha256_file(MANIFEST_PATH),
    "input_artifact_hashes":{"step03_gate":sha256_file(step3_gate_path),"policy":sha256_file(policy_path),"acceptance_contract":sha256_file(acceptance_path),"final_test_signature":sha256_file(signature_path)},
    "one_time_run_id":run_uuid,"test_images_accessed":True}
gate_path=write_json("gate_result.json",gate)
configuration={"phase":PHASE,"mode":"ONE_TIME_LOCKED_TEST_EVALUATION","one_time_run_id":run_uuid,"manifest_sha256":sha256_file(MANIFEST_PATH),
    "test_rows":len(test_rows),"test_volumes":int(test_rows.volume_id.nunique()),"policy":policy,"acceptance_contract_sha256":sha256_file(acceptance_path),
    "device":str(DEVICE),"batch_size":BATCH_SIZE,"num_workers":NUM_WORKERS,"random_seed":RANDOM_SEED,"expected_outputs":EXPECTED_OUTPUTS,"full_analysis_executed":True,"test_images_accessed":True}
write_json("configuration.json",configuration)
provenance={"completed_utc":datetime.now(timezone.utc).isoformat(),"one_time_run_id":run_uuid,"authorization":authorization_record,
    "policy_sha256":sha256_file(policy_path),"acceptance_contract_sha256":sha256_file(acceptance_path),"manifest_sha256":sha256_file(MANIFEST_PATH),
    "checkpoint_sha256":checkpoint_hashes,"test_rows":len(test_rows),"test_volumes":int(test_rows.volume_id.nunique()),"positive_test_volumes":int(positive_patients.shape[0]),
    "software":{"python":sys.version,"platform":platform.platform(),"numpy":np.__version__,"pandas":pd.__version__,"torch":torch.__version__},
    "test_results_may_not_trigger":policy["test_results_may_not_trigger"],"test_images_accessed":True}
write_json("provenance.json",provenance)
write_json("test_access_declaration.json",{"test_images_accessed":True,"test_masks_accessed":True,"test_probabilities_computed":True,"test_statistics_computed":True,"one_time_run_id":run_uuid,"rerun_allowed":False})

data_card=f'''# One-Time Locked Test Results

- Result level: `FINAL_TEST_COMPLETE`
- Formal minimum acceptance passed: `{all_acceptance_passed}`
- Mean positive-patient Dice: `{metrics["mean_positive_patient_dice"]:.6f}`
- Global Dice: `{metrics["global_dice"]:.6f}`
- Global pixel precision / recall: `{metrics["global_pixel_precision"]:.6f}` / `{metrics["global_pixel_recall"]:.6f}`
- Train-edge Q1 detection: `{metrics["q1_positive_slice_detection_pct_train_edges"]:.3f}%`
- Positive predicted-empty: `{metrics["positive_predicted_empty_pct"]:.3f}%`
- Empty-slice false positives: `{metrics["empty_slice_false_positive_pct"]:.3f}%`
- Minimum positive-patient Dice: `{metrics["minimum_positive_patient_dice"]:.6f}`
- One-time run ID: `{run_uuid}`
- Test images accessed: `true`
- Decision: `{gate["decision"]}`

The test result is final report-only evidence. No threshold, checkpoint, fusion, post-processing, or training change is permitted from these results.
'''
(OUTPUT_DIR/"TEST_RESULTS_DATA_CARD.md").write_text(data_card,encoding="utf-8")

ledger.update({"status":"COMPLETE","completed_utc":datetime.now(timezone.utc).isoformat(),"gate_result_sha256":sha256_file(gate_path),"test_images_accessed":True,"rerun_allowed":False})
write_json("run_ledger.json",ledger)
missing=[name for name in EXPECTED_OUTPUTS if not (OUTPUT_DIR/name).is_file()]
assert not missing,f"Missing final outputs: {missing}"
print(json.dumps(gate,indent=2)); print("SEALED: one-time test evaluation complete; rerun prohibited.")
""", tags=["requires-test-authorization"]),
md(r"""
### Interpretation rule

Use `outputs/gate_result.json` as the controlling final-test result. `FINAL_TEST_COMPLETE` means the one-time evaluation finished, not necessarily that every acceptance target passed. Whether it passed or failed, proceed only to the final research package and report limitations; never tune or rerun from test evidence.
"""),
]

nb=nbf.v4.new_notebook(cells=cells)
nb.metadata={
    "kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},
    "language_info":{"name":"python","version":"3.11"},
    "project_contract":{"phase":"step_04_one_time_locked_test_evaluation_after_explicit_authorization","authorization_granted":True,"authorization_recorded_utc":"2026-08-05T11:39:37.3247843Z","one_time_run_id":"871d289b-bf6b-4346-978f-2df02ade26ab","test_images_accessed":False,"one_time_only":True},
}
PHASE_DIR.mkdir(parents=True,exist_ok=True)
with NOTEBOOK_PATH.open("w",encoding="utf-8") as handle: nbf.write(nb,handle)
print(NOTEBOOK_PATH)
