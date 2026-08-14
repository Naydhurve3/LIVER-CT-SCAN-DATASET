"""Generate the no-inference Step 05 final research-package notebook."""

from pathlib import Path
import nbformat as nbf

PROJECT_ROOT = Path(r"D:\DATA SCIENCE AND ANALYTICS\PROJECTS\Liver")
PHASE_DIR = PROJECT_ROOT / "mark 1 (part 2)" / "step_05_final_research_package"
NOTEBOOK_PATH = PHASE_DIR / "step_05_final_research_package.ipynb"


def md(text):
    return nbf.v4.new_markdown_cell(text.strip())


def code(text):
    return nbf.v4.new_code_cell(text.strip())


cells = [
md(r"""
# Step 05 — Final research package

## Technical summary

This no-inference notebook packages the completed LiTS liver-tumour segmentation study. It verifies the sealed Step 01–04 evidence, independently reconciles the final test metrics, produces publication-ready figures and technical documentation, and closes the project without reopening source test data or changing the frozen model.

The final test execution was valid and complete, but formal model acceptance failed because one tumour-positive patient (V121) had effectively zero Dice. `FINAL_PROJECT_COMPLETE` in this phase means the research and reproducibility package is complete; it does **not** convert that failed model gate into a pass.
"""),
md(r"""
## Scope, assumptions, and evidence boundary

- Inputs are restricted to sealed Part 2 outputs and the signed Step 04 probability caches.
- No dataset manifest, source CT, source mask, checkpoint, loader, or model is opened.
- No inference, threshold sweep, checkpoint comparison, fusion change, post-processing experiment, training, or test rerun occurs.
- The Step 04 ledger remains immutable and `rerun_allowed: false`.
- Test-derived evidence is used only for reporting and failure characterization.
"""),
code(r"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import hashlib, json, platform, sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(r"D:\DATA SCIENCE AND ANALYTICS\PROJECTS\Liver")
PART2 = PROJECT_ROOT / "mark 1 (part 2)"
PHASE = "step_05_final_research_package"
PHASE_DIR = PART2 / PHASE
OUTPUT_DIR = PHASE_DIR / "outputs"
STEP1_OUT = PART2 / "step_01_pretraining_dataset_characterization" / "outputs"
STEP2_OUT = PART2 / "step_02_fusion_freeze_confirmation" / "outputs"
STEP3_OUT = PART2 / "step_03_final_inference_policy_freeze" / "outputs"
STEP4_OUT = PART2 / "step_04_one_time_locked_test_evaluation_after_explicit_authorization" / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TEST_SOURCE_FILES_REOPENED = False
TEST_IMAGES_ACCESSED_IN_THIS_PHASE = False
RANDOM_SEED = 42
EXPECTED_OUTPUTS = [
    "configuration.json", "provenance.json", "final_evidence_validation.csv",
    "final_metrics_summary.csv", "methods_and_parameters.csv", "experiment_timeline.csv",
    "decision_log.csv", "patient_results_table.csv", "lesion_stratum_results.csv",
    "v121_failure_evidence.csv", "chart_map.csv", "final_outcome_dashboard.png",
    "validation_to_test_comparison.png", "patient_failure_distribution.png",
    "FINAL_TECHNICAL_REPORT.md", "PAPER_DRAFT.md", "LIMITATIONS_AND_FAILURE_ANALYSIS.md",
    "REPRODUCIBILITY_INSTRUCTIONS.md", "FINAL_PROJECT_DATA_CARD.md",
    "expected_vs_actual.csv", "ARTIFACT_CHECKSUM_INVENTORY.csv",
    "final_package_signature.json", "gate_result.json"
]

def sha256_file(path, chunk_size=1 << 20):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda:f.read(chunk_size),b""): h.update(block)
    return h.hexdigest()

def load_json(path): return json.loads(Path(path).read_text(encoding="utf-8"))
def write_json(name,payload):
    path=OUTPUT_DIR/name; path.write_text(json.dumps(payload,indent=2,sort_keys=True,default=str),encoding="utf-8"); return path
def save_csv(frame,name):
    path=OUTPUT_DIR/name; frame.to_csv(path,index=False,float_format="%.8f"); return path

plt.rcParams.update({"figure.facecolor":"white","axes.facecolor":"#fbfcfe","axes.edgecolor":"#374151","axes.labelcolor":"#1f2937","text.color":"#111827","grid.color":"#d1d5db","grid.alpha":.45,"font.size":10})
print("Step 05 report-only package; no test source access or model execution.")
"""),
md("## 1. The sealed evidence chain is complete and internally consistent"),
code(r"""
step1_gate_path=STEP1_OUT/"pretraining_dataset_gate.json"
step2_gate_path=STEP2_OUT/"gate_result.json"
step3_gate_path=STEP3_OUT/"gate_result.json"
step4_gate_path=STEP4_OUT/"gate_result.json"
ledger_path=STEP4_OUT/"run_ledger.json"
signature_path=STEP4_OUT/"final_test_signature.json"
evidence_inventory_path=STEP4_OUT/"final_test_evidence_inventory.csv"
paths=[step1_gate_path,step2_gate_path,step3_gate_path,step4_gate_path,ledger_path,signature_path,evidence_inventory_path]
assert all(p.is_file() for p in paths)
step1=load_json(step1_gate_path); step2=load_json(step2_gate_path); step3=load_json(step3_gate_path); step4=load_json(step4_gate_path)
ledger=load_json(ledger_path); signature=load_json(signature_path); evidence_inventory=pd.read_csv(evidence_inventory_path)
assert step1["all_mandatory_targets_passed"] and step2["result_level"]=="VALIDATION_FREEZE_PASS" and step3["result_level"]=="VALIDATION_FREEZE_PASS"
assert step4["result_level"]=="FINAL_TEST_COMPLETE" and not step4["all_mandatory_targets_passed"]
assert ledger["status"]=="COMPLETE" and ledger["rerun_allowed"] is False and ledger["test_images_accessed"] is True
assert sha256_file(step4_gate_path)==ledger["gate_result_sha256"]
assert sha256_file(evidence_inventory_path)==signature["evidence_inventory_sha256"]
hash_failures=[]
for row in evidence_inventory.itertuples():
    path=Path(row.path)
    if not path.is_file() or sha256_file(path)!=row.sha256: hash_failures.append(str(path))
assert not hash_failures, hash_failures

evidence_checks=[
    ("step01_dataset_gate",step1["all_mandatory_targets_passed"]),
    ("step02_validation_freeze",step2["all_mandatory_targets_passed"]),
    ("step03_policy_freeze",step3["all_mandatory_targets_passed"]),
    ("step04_final_test_complete",step4["result_level"]=="FINAL_TEST_COMPLETE"),
    ("step04_ledger_complete",ledger["status"]=="COMPLETE"),
    ("step04_rerun_prohibited",ledger["rerun_allowed"] is False),
    ("gate_hash_matches_ledger",sha256_file(step4_gate_path)==ledger["gate_result_sha256"]),
    ("evidence_inventory_hash_matches_signature",sha256_file(evidence_inventory_path)==signature["evidence_inventory_sha256"]),
    ("all_inventory_files_match",not hash_failures),
    ("test_source_files_reopened_false",not TEST_SOURCE_FILES_REOPENED),
]
evidence_validation=pd.DataFrame(evidence_checks,columns=["check","passed"]); evidence_validation["expected"]=True
save_csv(evidence_validation,"final_evidence_validation.csv")
display(evidence_validation)
"""),
md("## 2. Aggregate performance was strong, but the catastrophic-patient guardrail failed"),
code(r"""
patients=pd.read_csv(STEP4_OUT/"patient_metrics.csv")
slices=pd.read_csv(STEP4_OUT/"slice_metrics.csv")
global_saved=pd.read_csv(STEP4_OUT/"global_metrics.csv").iloc[0]
bootstrap=pd.read_csv(STEP4_OUT/"bootstrap_uncertainty.csv")
lesions=pd.read_csv(STEP4_OUT/"lesion_or_size_metrics.csv")
strata=pd.read_csv(STEP4_OUT/"size_stratum_metrics.csv")
acceptance=pd.read_csv(STEP4_OUT/"expected_vs_actual.csv")
positive=patients[patients.has_tumour.astype(bool)].copy()

recomputed={
    "global_dice":float((2*patients.intersection_pixels.sum()+1e-6)/(patients.truth_pixels.sum()+patients.predicted_pixels.sum()+1e-6)),
    "global_pixel_precision":float(patients.intersection_pixels.sum()/patients.predicted_pixels.sum()),
    "global_pixel_recall":float(patients.intersection_pixels.sum()/patients.truth_pixels.sum()),
    "mean_positive_patient_dice":float(positive.dice.mean()),
    "median_positive_patient_dice":float(positive.dice.median()),
    "minimum_positive_patient_dice":float(positive.dice.min()),
    "q1_positive_slice_detection_pct_train_edges":100*float(slices.loc[slices.q1_positive_slice_train_edge.astype(bool),"detected"].mean()),
    "positive_predicted_empty_pct":100*float(slices.loc[slices.positive_slice.astype(bool),"predicted_empty"].mean()),
    "empty_slice_false_positive_pct":100*float(slices.loc[~slices.positive_slice.astype(bool),"empty_slice_false_positive"].mean()),
}
deltas={k:abs(v-float(global_saved[k])) for k,v in recomputed.items()}
assert max(deltas.values())<1e-7
metrics_summary=pd.DataFrame([{"metric":k,"value":v,"saved_value":float(global_saved[k]),"absolute_delta":deltas[k]} for k,v in recomputed.items()])
save_csv(metrics_summary,"final_metrics_summary.csv")
save_csv(patients,"patient_results_table.csv"); save_csv(strata,"lesion_stratum_results.csv")
formal_model_acceptance_passed=bool(step4["all_mandatory_targets_passed"])
failed_acceptance=acceptance[acceptance.mandatory.astype(bool)&~acceptance.passed.astype(bool)]
assert len(failed_acceptance)==1 and failed_acceptance.metric.iloc[0]=="minimum_positive_patient_dice"
assert int(positive.loc[positive.dice.idxmin(),"volume_id"])==121
display(metrics_summary); display(failed_acceptance)
"""),
md("## 3. V121 was a recognition failure, not an ROI-cropping failure"),
code(r"""
rois=pd.read_csv(STEP4_OUT/"test_roi_manifest.csv").set_index("volume_id")
cache_dir=STEP4_OUT/"probability_cache"
failure_rows=[]
for volume_id in [120,121,127]:
    cache_path=cache_dir/f"volume_{volume_id}.npz"
    assert str(cache_path) in set(evidence_inventory.path)
    with np.load(cache_path,allow_pickle=False) as item:
        truth=item["truth"].astype(bool); control=item["control_probability"].astype(np.float32)
        recall=item["recall_probability"].astype(np.float32); fused=item["fused_probability"].astype(np.float32)
    roi=rois.loc[volume_id]; support=np.zeros_like(truth,bool); support[:,int(roi.y0):int(roi.y1),int(roi.x0):int(roi.x1)]=True
    scores=fused[truth]
    failure_rows.append({"volume_id":volume_id,"patient_dice":float(patients.set_index("volume_id").loc[volume_id,"dice"]),
        "truth_pixels":int(truth.sum()),"roi_truth_containment":float((truth&support).sum()/truth.sum()),
        "maximum_control_truth_score":float(control[truth].max()),"maximum_recall_truth_score":float(recall[truth].max()),
        "maximum_fused_truth_score":float(scores.max()),"median_fused_truth_score":float(np.median(scores)),
        "truth_pixels_at_or_above_0_70":int(np.count_nonzero(scores>=.70))})
failure_evidence=pd.DataFrame(failure_rows); save_csv(failure_evidence,"v121_failure_evidence.csv")
v121=failure_evidence[failure_evidence.volume_id.eq(121)].iloc[0]
assert np.isclose(v121.roi_truth_containment,1.0) and v121.truth_pixels_at_or_above_0_70==0
display(failure_evidence)
"""),
md("## 4. Methods, parameters, and decisions remain traceable"),
code(r"""
policy=load_json(STEP3_OUT/"final_inference_policy.json")
methods_rows=[
    ("dataset","build_id",policy["dataset"]["build_id"]),("dataset","manifest_sha256",policy["dataset"]["manifest_sha256"]),
    ("input","channel","broad_window"),("input","hu_window","[-160,240]"),("input","normalization","uint8/255"),
    ("roi","generator_checkpoint_sha256",policy["roi"]["generator"]["checkpoint_sha256"]),("roi","threshold",0.5),
    ("roi","component","largest_3d_26conn"),("roi","padding_pixels",16),("roi","empty_fallback","full_image_box_[0,256,0,256]"),
    ("tumour_model","architecture","MobileNetV2UNet"),("tumour_model","control_checkpoint_sha256",policy["model"]["control_checkpoint_sha256"]),
    ("tumour_model","recall_checkpoint_sha256",policy["model"]["recall_checkpoint_sha256"]),
    ("inference","fusion","maximum(control_probability,recall_probability)"),("inference","threshold",0.70),("inference","post_processing","none"),
    ("metrics","epsilon",1e-6),("metrics","q1_positive_slice_max_pixels_train_only",51),("uncertainty","bootstrap_iterations",10000),
    ("uncertainty","bootstrap_seed",42),("uncertainty","resampling_unit","tumour-positive patient"),
]
methods=pd.DataFrame(methods_rows,columns=["section","parameter","value"]); save_csv(methods,"methods_and_parameters.csv")

timeline=pd.DataFrame([
    (1,"Practice correction and loader gates","PIPELINE_VERIFIED","Corrected manifest, strict loader and overfit evidence established"),
    (2,"Mark 1–4D diagnostics and bounded training","VALIDATION_ONLY","Diagnosed calibration/localization and produced control plus recall checkpoints"),
    (3,"Mark 4E fusion selection","TEMPORARY_CONTINUATION_PASS","Selected maximum fusion at threshold 0.70"),
    (4,"Part 2 Step 01 dataset characterization","DIAGNOSTIC_COMPLETE","12/12 audit requirements passed; data card and sampling policy frozen"),
    (5,"Part 2 Step 02 fusion confirmation","VALIDATION_FREEZE_PASS","Fresh deterministic inference reproduced all six validation targets"),
    (6,"Part 2 Step 03 final policy freeze","VALIDATION_FREEZE_PASS","Policy, acceptance table, ROI generator and artifact chain frozen"),
    (7,"Part 2 Step 04 one-time test evaluation","FINAL_TEST_COMPLETE","Seven of eight final targets passed; catastrophic-patient guardrail failed"),
    (8,"Part 2 Step 05 final research package","FINAL_PROJECT_COMPLETE","Report and reproducibility package completed without further test use"),
],columns=["sequence","phase","result_level","decision_or_evidence"])
save_csv(timeline,"experiment_timeline.csv")

decisions=pd.DataFrame([
    ("dataset","Retain corrected build","All integrity and leakage gates passed"),("sampling","Retain patient-aware existing sampler","Audit did not support a new sampling intervention"),
    ("fusion","Maximum control/recall scores","Only bounded policy satisfying six validation targets"),("threshold","Global 0.70","Frozen before test; no patient-specific thresholds"),
    ("training","Skip optional Part 2 refinement","Validation confirmation passed without evidence for another bounded intervention"),
    ("test","Run exactly once","Authorization recorded; ledger complete; rerun prohibited"),("final_outcome","Report failed formal acceptance","V121 Dice below catastrophic-patient floor"),
    ("post_test","No tuning or retraining","Test evidence is report-only under the frozen contract"),
],columns=["decision_area","decision","basis"]); save_csv(decisions,"decision_log.csv")
display(methods.head(10)); display(timeline)
"""),
md(r"""
## 5. Visual evidence

The figures below use neutral benchmark lines and one orange exception highlight. Bars start at zero; percentages and Dice are not mixed on one axis. Exact values remain available in the accompanying CSV tables.
"""),
code(r"""
BLUE="#2563eb"; BLUE_LIGHT="#93c5fd"; ORANGE="#f59e0b"; INK="#111827"; GREY="#6b7280"
fig,axes=plt.subplots(2,2,figsize=(16,11))

# Patient distribution with the catastrophic guardrail and mean target.
p=positive.sort_values("dice"); colors=[ORANGE if int(v)==121 else BLUE for v in p.volume_id]
axes[0,0].barh(p.volume_id.astype(str),p.dice,color=colors,edgecolor=INK,linewidth=.4)
axes[0,0].axvline(.01,color=INK,ls=":",label="catastrophic floor 0.01"); axes[0,0].axvline(.3329,color=GREY,ls="--",label="aggregate-mean target 0.3329 (context)")
axes[0,0].set_xlim(0,1); axes[0,0].set_xlabel("Patient Dice"); axes[0,0].set_title("Tumour-positive test-patient Dice (n=13)"); axes[0,0].legend(fontsize=8)

# Applicable final acceptance rows.
final_accept=acceptance[acceptance.scope.isin(["final_test","final_test_integrity"])].copy()
plot_accept=final_accept[final_accept.metric.isin(["mean_positive_patient_dice","q1_positive_slice_detection_pct_train_edges","positive_predicted_empty_pct","empty_slice_false_positive_pct","minimum_positive_patient_dice"])].copy()
plot_accept["target_normalized_margin"]=np.where(plot_accept.direction.eq(">="),plot_accept.actual/plot_accept.target,plot_accept.target/plot_accept.actual)
axes[0,1].barh(plot_accept.metric.str.replace("_"," "),plot_accept.target_normalized_margin,color=[BLUE if x else ORANGE for x in plot_accept.passed],edgecolor=INK,linewidth=.4)
axes[0,1].axvline(1,color=INK,ls="--",label="pass boundary"); axes[0,1].set_xlabel("Target-normalized performance (>1 passes)"); axes[0,1].set_title("Formal final acceptance metrics"); axes[0,1].legend(fontsize=8)
for index,row in enumerate(plot_accept.itertuples()):
    margin=float(row.target_normalized_margin)
    if not bool(row.passed):
        axes[0,1].scatter([.04],[index],marker="X",s=85,color=ORANGE,edgecolor=INK,zorder=4)
        axes[0,1].text(.12,index,"FAIL 0.00×",va="center",fontweight="bold",color=INK)
    else:
        axes[0,1].text(margin+.04,index,f"{margin:.2f}×",va="center",color=INK,fontsize=8)

# Lesion strata, same percent scale.
x=np.arange(len(strata)); width=.36
axes[1,0].bar(x-width/2,strata.detected_pct,width,label="Detected lesions (%)",color=BLUE,edgecolor=INK,linewidth=.4)
axes[1,0].bar(x+width/2,100*strata.mean_matched_dice,width,label="Mean matched Dice ×100",color=BLUE_LIGHT,edgecolor=INK,linewidth=.4)
axes[1,0].set_xticks(x,[f"Train Q{i+1}" for i in range(len(strata))]); axes[1,0].set_ylim(0,105); axes[1,0].set_ylabel("Percent / Dice ×100"); axes[1,0].set_title("Test lesions by frozen train-volume stratum (n=187)"); axes[1,0].legend(fontsize=8)

# Patient burden vs performance.
axes[1,1].scatter(positive.truth_pixels,positive.dice,c=[ORANGE if int(v)==121 else BLUE for v in positive.volume_id],s=55,edgecolor=INK,linewidth=.4)
for row in positive.itertuples(): axes[1,1].annotate(f"V{int(row.volume_id)}",(row.truth_pixels,row.dice),xytext=(4,3),textcoords="offset points",fontsize=8)
axes[1,1].set_xscale("log"); axes[1,1].set_ylim(-.03,1); axes[1,1].set_xlabel("Tumour pixels per patient (log scale)"); axes[1,1].set_ylabel("Patient Dice"); axes[1,1].set_title("Patient Dice versus tumour burden")
fig.suptitle("Final frozen-policy outcome: aggregate strength with one catastrophic patient miss",fontsize=15)
fig.tight_layout(); fig.savefig(OUTPUT_DIR/"final_outcome_dashboard.png",dpi=180,bbox_inches="tight"); plt.close(fig)

# Validation to test comparison; each panel retains its own correct unit.
comparison=[("Mean positive-patient Dice",step2["selected_metrics"]["mean_patient_dice"],recomputed["mean_positive_patient_dice"],.3329,"Dice"),
            ("Q1 positive-slice detection",step2["selected_metrics"]["q1_detected_pct"],recomputed["q1_positive_slice_detection_pct_train_edges"],35,"Percent"),
            ("Positive predicted-empty",step2["selected_metrics"]["positive_predicted_empty_pct"],recomputed["positive_predicted_empty_pct"],35,"Percent"),
            ("Empty-slice false positives",step2["selected_metrics"]["empty_slice_false_positive_pct"],recomputed["empty_slice_false_positive_pct"],20,"Percent")]
fig,axes=plt.subplots(2,2,figsize=(12,9))
for ax,(title,val,test,target,unit) in zip(axes.ravel(),comparison):
    ax.bar(["Validation","Test"],[val,test],color=[BLUE_LIGHT,BLUE],edgecolor=INK,linewidth=.4); ax.axhline(target,color=INK,ls="--",label=f"target {target:g}")
    ax.set_ylim(0,max(val,test,target)*1.3); ax.set_ylabel(unit); ax.set_title(title); ax.legend(fontsize=8)
fig.suptitle("Frozen validation and one-time test results (descriptive, not a tuning comparison)"); fig.tight_layout(); fig.savefig(OUTPUT_DIR/"validation_to_test_comparison.png",dpi=180,bbox_inches="tight"); plt.close(fig)

# Distribution figure with uncertainty.
mean_ci=bootstrap[bootstrap.metric.eq("mean_positive_patient_dice")].iloc[0]
fig,axes=plt.subplots(1,2,figsize=(13,5))
axes[0].hist(positive.dice,bins=np.linspace(0,1,11),color=BLUE_LIGHT,edgecolor=INK); axes[0].axvline(positive.dice.mean(),color=BLUE,lw=2,label=f"mean {positive.dice.mean():.3f}"); axes[0].axvline(positive.dice.median(),color=GREY,ls="--",label=f"median {positive.dice.median():.3f}"); axes[0].set_xlabel("Patient Dice"); axes[0].set_ylabel("Patients"); axes[0].set_title("Positive-patient Dice distribution"); axes[0].legend()
axes[1].errorbar([0],[mean_ci.estimate],yerr=[[mean_ci.estimate-mean_ci.ci_lower_2_5],[mean_ci.ci_upper_97_5-mean_ci.estimate]],fmt="o",color=BLUE,capsize=8); axes[1].axhline(.3329,color=INK,ls="--",label="formal mean target"); axes[1].set_xlim(-.5,.5); axes[1].set_xticks([0],["Test mean Dice"]); axes[1].set_ylim(0,1); axes[1].set_ylabel("Dice"); axes[1].set_title("10,000-resample patient bootstrap 95% interval"); axes[1].legend()
fig.tight_layout(); fig.savefig(OUTPUT_DIR/"patient_failure_distribution.png",dpi=180,bbox_inches="tight"); plt.close(fig)

chart_map=pd.DataFrame([
    ("final outcome","patient reliability and formal gate","comparison/distribution","horizontal bars, normalized bars, grouped bars, scatter","final_outcome_dashboard.png"),
    ("generalization","validation versus test descriptive comparison","comparison","four small-multiple bars","validation_to_test_comparison.png"),
    ("uncertainty","patient spread and bootstrap interval","distribution/uncertainty","histogram plus point interval","patient_failure_distribution.png"),
],columns=["report_segment","analytical_question","family","variant","output"]); save_csv(chart_map,"chart_map.csv")
print("Saved three publication-ready figures.")
"""),
md("## 6. Durable technical report, paper draft, and reproducibility package"),
code(r"""
mean_ci=bootstrap[bootstrap.metric.eq("mean_positive_patient_dice")].iloc[0]
q1=strata.iloc[0]
technical_report=f'''# Frozen LiTS Liver-Tumour Segmentation: Final Technical Report

## Technical summary

The project completed its one-time held-out test evaluation under a fully frozen two-checkpoint maximum-fusion policy. Aggregate performance was strong: global Dice was **{recomputed["global_dice"]:.4f}**, mean Dice across 13 tumour-positive patients was **{recomputed["mean_positive_patient_dice"]:.4f}** (patient-bootstrap 95% interval **{mean_ci.ci_lower_2_5:.4f}–{mean_ci.ci_upper_97_5:.4f}**), global precision was **{recomputed["global_pixel_precision"]:.4f}**, and global recall was **{recomputed["global_pixel_recall"]:.4f}**.

Formal model acceptance nevertheless failed. V121 had effectively zero Dice, below the predeclared minimum-patient floor of 0.01. The final result is therefore **FINAL_TEST_COMPLETE with failed formal acceptance**, not deployment-ready model success. No test-driven tuning or rerun is permitted.

## Aggregate generalization coexisted with severe patient heterogeneity

The median positive-patient Dice was **{recomputed["median_positive_patient_dice"]:.4f}**, but four of 13 positive patients scored below 0.3329. Large tumour-burden patients generally performed well, while very small or atypical cases produced unstable Dice. The one empty-tumour patient, V119, generated 968 false-positive pixels and is excluded from the positive-patient Dice mean but included in empty-slice false-positive reporting.

![Final outcome dashboard](final_outcome_dashboard.png)

The figure pairs exact patient outcomes with the formal pass boundary. Orange marks V121, the sole mandatory acceptance failure.

## Small lesions remain the principal segmentation weakness

Train-derived lesion-volume Q1 contained {int(q1.lesions)} held-out lesions. Detection was **{q1.detected_pct:.2f}%** and mean matched Dice was **{q1.mean_matched_dice:.3f}**. Detection rose to {strata.detected_pct.iloc[1]:.2f}%, {strata.detected_pct.iloc[2]:.2f}% and {strata.detected_pct.iloc[3]:.2f}% for Q2–Q4. Slice-level Q1 detection was **{recomputed["q1_positive_slice_detection_pct_train_edges"]:.2f}%** under the frozen 1–51-pixel definition.

## V121 was missed despite complete ROI containment

V121 contained 526 tumour pixels across three connected lesions. All tumour pixels were inside the frozen predicted-liver ROI, yet neither checkpoint produced a cached float16 tumour score at or above 0.70 in the truth region. The primary lesion was train-volume Q3, so the failure cannot be attributed only to a tiny-component definition. This is evidence of a recognition/localization failure, not predicted-ROI clipping.

## Scope, dataset, and metric definitions

The corrected dataset build was `{policy["dataset"]["build_id"]}` with manifest SHA-256 `{policy["dataset"]["manifest_sha256"]}`. The final test cohort contained 7,286 slices from 14 volumes, including 13 tumour-positive volumes. Positive-patient mean Dice excludes tumour-empty patients. Q1 positive slices contain 1–51 tumour pixels, with 51 derived exclusively from training data. Lesion strata use train-derived physical-volume quartile edges and 6-connected 3D truth components.

## Frozen model and evaluation methodology

Input was a single broad CT window `[-160,240]` HU stored as uint8 and divided by 255. Predicted-liver ROIs used the frozen two-output liver model, channel 0, robust per-slice normalization, threshold 0.50, largest 26-connected 3D component, padding 16, and full-image fallback for an empty ROI. Tumour scores were produced by control and recall-loss MobileNetV2UNet checkpoints and fused by pixelwise maximum. Hard predictions used one global threshold of 0.70 with no post-processing.

Step 02 independently reproduced the selected validation policy, including deterministic repeated inference and cache equivalence. Step 03 froze the complete policy and final acceptance table before test access. Step 04 opened the test split once under run UUID `{ledger["run_id"]}` and sealed the ledger as complete with rerun prohibited.

![Validation and test comparison](validation_to_test_comparison.png)

This comparison is descriptive evidence of generalization, not a basis for additional model selection. All policy choices were frozen before the test values existed.

## Uncertainty and robustness

The patient-bootstrap interval is wide because only 13 tumour-positive test patients are available and patient performance is heterogeneous. Integrity checks passed: all 14 cache files were finite and bounded, sample coverage and uniqueness were 100%, all signed evidence hashes matched, and independent metric reconciliation differed by less than 1e-7.

![Patient distribution and uncertainty](patient_failure_distribution.png)

The mean clears its formal aggregate target, but the distribution exposes a complete miss and several low-Dice patients that aggregate metrics obscure.

## Limitations

- Formal model acceptance failed because the catastrophic-patient floor was not met.
- Small-lesion matched Dice remains low even when detection occurs.
- The test cohort contains only 14 patients, limiting precision of patient-level uncertainty.
- The two-dimensional tumour model uses derived 256×256 broad-window PNG input and does not exploit full volumetric context.
- Pixelwise sigmoid scores are descriptive model scores, not clinically calibrated probabilities.
- This study does not establish clinical safety, external-domain generalization, or prospective performance.
- No external literature references are included in this internal draft; they must be added before submission.

## Recommended next steps

Archive the frozen test outcome and complete manuscript editing without reopening test data. Any future modelling work must be a separately versioned study with a new development protocol and a new untouched external evaluation cohort; it cannot reuse this test split for selection.

## Further questions

- Which imaging or lesion-appearance phenotype explains the V121 recognition failure?
- Would a future volumetric or multi-window design improve Q1 matched Dice under a newly locked external-evaluation protocol?
- How stable are these outcomes across institutions, scanners, and annotation conventions?
'''
(OUTPUT_DIR/"FINAL_TECHNICAL_REPORT.md").write_text(technical_report,encoding="utf-8")

paper=f'''# Maximum-Probability Fusion of Two MobileNetV2 U-Nets for Liver-Tumour Segmentation: A Reproducible Held-Out Evaluation

## Abstract

**Background:** Liver-tumour segmentation remains difficult for small and low-contrast lesions. **Methods:** We evaluated a frozen two-stage pipeline on the corrected LiTS build. A predicted-liver ROI preceded two MobileNetV2 U-Net tumour models. Their sigmoid scores were fused by pixelwise maximum and thresholded globally at 0.70. All preprocessing, checkpoints, acceptance targets and metrics were frozen before one-time test access. **Results:** Across 13 tumour-positive test patients, mean patient Dice was {recomputed["mean_positive_patient_dice"]:.4f} (95% patient-bootstrap interval {mean_ci.ci_lower_2_5:.4f}–{mean_ci.ci_upper_97_5:.4f}); global Dice was {recomputed["global_dice"]:.4f}, precision {recomputed["global_pixel_precision"]:.4f}, and recall {recomputed["global_pixel_recall"]:.4f}. Smallest-quartile lesion detection was {q1.detected_pct:.1f}%. One patient had effectively zero Dice despite complete ROI containment, causing failure of the predeclared minimum-patient acceptance gate. **Conclusion:** Maximum-probability fusion generalized well in aggregate but did not provide reliable patient-level performance. The outcome should be reported as a completed negative formal gate rather than deployment-ready success.

**Keywords:** liver tumour, CT, semantic segmentation, LiTS, MobileNetV2 U-Net, model fusion, reproducibility

## 1. Introduction

Automated liver-tumour delineation can support quantitative oncology workflows, but lesion size, contrast, multiplicity and acquisition variability create strong case-level heterogeneity. This study asked whether a validation-selected, maximum-probability fusion of complementary tumour models would reproduce under a strictly frozen one-time held-out evaluation.

## 2. Materials and methods

### 2.1 Dataset governance

The study used corrected build `{policy["dataset"]["build_id"]}`. Patient-disjoint training, validation and test splits were controlled by a SHA-256-locked manifest. Dataset integrity, geometry, label semantics, train-validation leakage, morphology, HU contrast, difficulty and focus-case phenotypes were audited before the final model freeze.

### 2.2 Preprocessing and ROI generation

Derived axial images represented a broad `[-160,240]` HU window at 256×256 resolution. A frozen two-output MobileNetV2 U-Net generated liver scores. The largest 26-connected 3D liver component at threshold 0.50 was projected to a padded bounding box; crops were resized to 256×256 and mapped back using bilinear score interpolation.

### 2.3 Tumour models and fusion

Control and recall-loss MobileNetV2 U-Net checkpoints produced sigmoid tumour scores. Fusion was `max(p_control,p_recall)`. A single global threshold of 0.70 and no post-processing were used for every patient.

### 2.4 Outcomes and uncertainty

The primary aggregate outcome was mean Dice over tumour-positive patients. Secondary outcomes included global Dice, pixel precision/recall, per-patient Dice, train-edge Q1 slice detection, positive predicted-empty rate, empty-slice false-positive rate, and matched 3D lesion performance by train-derived physical-volume quartiles. Uncertainty used 10,000 patient-bootstrap resamples with seed 42.

### 2.5 Governance and test lock

The inference policy and formal acceptance table were frozen before test authorization. Test data were evaluated once under run UUID `{ledger["run_id"]}`. The completed ledger prohibits rerun and test-driven tuning.

## 3. Results

The test cohort contained 7,286 slices from 14 volumes; 13 volumes were tumour-positive. Global Dice was {recomputed["global_dice"]:.4f}. Mean positive-patient Dice was {recomputed["mean_positive_patient_dice"]:.4f}, with median {recomputed["median_positive_patient_dice"]:.4f} and 95% bootstrap interval {mean_ci.ci_lower_2_5:.4f}–{mean_ci.ci_upper_97_5:.4f}. Global precision and recall were {recomputed["global_pixel_precision"]:.4f} and {recomputed["global_pixel_recall"]:.4f}. Q1 slice detection was {recomputed["q1_positive_slice_detection_pct_train_edges"]:.1f}%, positive predicted-empty {recomputed["positive_predicted_empty_pct"]:.1f}%, and empty-slice false positives {recomputed["empty_slice_false_positive_pct"]:.1f}%.

Seven of eight formal test/integrity targets passed. The minimum positive-patient Dice target failed: V121 had effectively zero Dice versus the 0.01 floor. V121 tumour pixels were fully contained by the ROI, but cached control and recall scores did not reach the global threshold in the truth region.

Smallest-quartile lesion detection was {q1.detected_pct:.1f}% with mean matched Dice {q1.mean_matched_dice:.3f}; larger quartiles achieved at least {strata.detected_pct.iloc[1:].min():.1f}% detection.

## 4. Discussion

The frozen fusion policy transferred from validation to strong aggregate test performance and exceeded the cohort-applicable historical aspirational values. However, the complete V121 miss and several other low-Dice patients show that aggregate overlap is insufficient for reliability claims. The smallest lesion stratum remains the clearest systematic weakness. Because the test was used once under a predeclared contract, these findings are descriptive failure evidence, not permission for retrospective selection.

## 5. Limitations

The held-out cohort is small; the pipeline is two-dimensional and based on derived broad-window images; probabilities are not clinically calibrated; and no external institutional cohort or prospective evaluation was available. Patient-level failure mechanisms require a future separately governed study.

## 6. Conclusion

Maximum-probability fusion provided strong aggregate LiTS performance but failed the predeclared catastrophic-patient guardrail. The technically correct conclusion is a completed held-out evaluation with failed formal model acceptance.

## Declarations and references to complete before submission

Add dataset licensing, ethics/applicability statement, author contributions, conflicts, funding, code availability, and literature references during manuscript preparation. No external citations were fabricated in this internal draft.
'''
(OUTPUT_DIR/"PAPER_DRAFT.md").write_text(paper,encoding="utf-8")

limitations=f'''# Limitations and Failure Analysis

## Formal outcome

Formal model acceptance failed solely because minimum positive-patient Dice was below 0.01. Aggregate and integrity targets passed.

## V121

- Patient Dice: {float(positive[positive.volume_id.eq(121)].dice.iloc[0]):.10f}
- Truth pixels: {int(v121.truth_pixels)}
- Frozen ROI containment: {v121.roi_truth_containment:.1%}
- Maximum cached fused score in truth: {v121.maximum_fused_truth_score:.8f}
- Truth pixels at or above 0.70: {int(v121.truth_pixels_at_or_above_0_70)}
- Interpretation: recognition failure, not ROI clipping.

## Other weak cases

V120 achieved high recall but excessive false-positive volume, producing Dice 0.1123. V127 contained only 73 tumour pixels and achieved partial overlap with Dice 0.0826. Four of 13 positive patients were below 0.3329.

## Lesion-size limitation

The smallest train-derived volume quartile had {q1.detected_pct:.1f}% detection and mean matched Dice {q1.mean_matched_dice:.3f}, markedly below larger strata.

## Prohibited response

Do not adjust the threshold, checkpoints, fusion, ROI, post-processing or training based on these test failures. Any follow-up model must start a new versioned study and use a new untouched evaluation cohort.
'''
(OUTPUT_DIR/"LIMITATIONS_AND_FAILURE_ANALYSIS.md").write_text(limitations,encoding="utf-8")

repro=f'''# Reproducibility Instructions

## Controlling artifacts

- Manifest SHA-256: `{policy["dataset"]["manifest_sha256"]}`
- Frozen policy SHA-256: `{sha256_file(STEP3_OUT/"final_inference_policy.json")}`
- Final acceptance contract SHA-256: `{signature["acceptance_contract_sha256"]}`
- One-time test run UUID: `{ledger["run_id"]}`
- Step 04 evidence inventory SHA-256: `{signature["evidence_inventory_sha256"]}`

## Reproduce the report package

Run `step_05_final_research_package.ipynb` from a fresh Python kernel. It reads only sealed Part 2 outputs, verifies every signed Step 04 evidence hash, recomputes summary metrics from saved tables, and regenerates reports and figures under this phase's `outputs/` folder.

## Do not reproduce the test inference

Step 04 is a completed one-time evaluation. Its ledger has `rerun_allowed: false`. Do not rerun Step 04, rebuild test ROIs, reload test source images/masks, change the frozen policy, or use the test cohort for further selection.
'''
(OUTPUT_DIR/"REPRODUCIBILITY_INSTRUCTIONS.md").write_text(repro,encoding="utf-8")

data_card=f'''# Final Project Data Card

- Project result level: `FINAL_PROJECT_COMPLETE`
- Final test result level: `FINAL_TEST_COMPLETE`
- Formal model acceptance passed: `false`
- Dataset build: `{policy["dataset"]["build_id"]}`
- Manifest SHA-256: `{policy["dataset"]["manifest_sha256"]}`
- Test cohort: 7,286 slices; 14 volumes; 13 tumour-positive volumes
- Frozen fusion: pixelwise maximum
- Frozen threshold: 0.70
- Global Dice: {recomputed["global_dice"]:.6f}
- Mean positive-patient Dice: {recomputed["mean_positive_patient_dice"]:.6f}
- Minimum positive-patient Dice: {recomputed["minimum_positive_patient_dice"]:.10f}
- Acceptance failure: V121 catastrophic-patient floor
- Test run UUID: `{ledger["run_id"]}`
- Test rerun allowed: `false`
- Test source files reopened in Step 05: `false`
- Final disposition: archive and report; no test-driven tuning
'''
(OUTPUT_DIR/"FINAL_PROJECT_DATA_CARD.md").write_text(data_card,encoding="utf-8")
print("Saved technical report, paper draft, limitations analysis, reproducibility instructions and data card.")
"""),
md("## 7. Final package gate and signed inventory"),
code(r"""
package_requirements={
    "sealed_evidence_verified":bool(evidence_validation.passed.all()),
    "metrics_independently_reconciled":bool(max(deltas.values())<1e-7),
    "formal_test_failure_preserved":bool(not formal_model_acceptance_passed),
    "v121_failure_characterized":bool(len(failure_evidence)==3 and np.isclose(v121.roi_truth_containment,1.0)),
    "methods_parameters_saved":bool((OUTPUT_DIR/"methods_and_parameters.csv").is_file()),
    "technical_report_saved":bool((OUTPUT_DIR/"FINAL_TECHNICAL_REPORT.md").is_file()),
    "paper_draft_saved":bool((OUTPUT_DIR/"PAPER_DRAFT.md").is_file()),
    "limitations_saved":bool((OUTPUT_DIR/"LIMITATIONS_AND_FAILURE_ANALYSIS.md").is_file()),
    "reproducibility_saved":bool((OUTPUT_DIR/"REPRODUCIBILITY_INSTRUCTIONS.md").is_file()),
    "figures_saved":all((OUTPUT_DIR/name).is_file() for name in ["final_outcome_dashboard.png","validation_to_test_comparison.png","patient_failure_distribution.png"]),
    "test_source_files_reopened_false":not TEST_SOURCE_FILES_REOPENED,
    "test_inference_rerun_false":True,
}
expected_vs_actual=pd.DataFrame([{"requirement":k,"expected":True,"actual":bool(v),"passed":bool(v)} for k,v in package_requirements.items()]); save_csv(expected_vs_actual,"expected_vs_actual.csv")
assert expected_vs_actual.passed.all()

source_artifacts=[step1_gate_path,step2_gate_path,step3_gate_path,STEP3_OUT/"final_inference_policy.json",STEP3_OUT/"final_acceptance_contract.json",step4_gate_path,ledger_path,signature_path,evidence_inventory_path,
                  STEP4_OUT/"patient_metrics.csv",STEP4_OUT/"slice_metrics.csv",STEP4_OUT/"global_metrics.csv",STEP4_OUT/"bootstrap_uncertainty.csv",STEP4_OUT/"lesion_or_size_metrics.csv",STEP4_OUT/"size_stratum_metrics.csv",STEP4_OUT/"test_roi_manifest.csv"]
package_outputs=[OUTPUT_DIR/name for name in EXPECTED_OUTPUTS if name not in {"configuration.json","provenance.json","ARTIFACT_CHECKSUM_INVENTORY.csv","final_package_signature.json","gate_result.json"}]
inventory_rows=[]
for role,paths in [("source",source_artifacts),("package_output",package_outputs)]:
    for path in paths:
        assert path.is_file(),path
        inventory_rows.append({"role":role,"path":str(path),"bytes":path.stat().st_size,"sha256":sha256_file(path)})
inventory=pd.DataFrame(inventory_rows); inventory_path=save_csv(inventory,"ARTIFACT_CHECKSUM_INVENTORY.csv")

signature_payload={"algorithm":"SHA-256","created_utc":datetime.now(timezone.utc).isoformat(),"inventory_sha256":sha256_file(inventory_path),
    "source_test_signature_sha256":sha256_file(signature_path),"source_test_run_id":ledger["run_id"],"formal_model_acceptance_passed":False,
    "test_source_files_reopened":False,"test_inference_rerun":False}
package_signature_path=write_json("final_package_signature.json",signature_payload)

configuration={"phase":PHASE,"mode":"NO_INFERENCE_FINAL_RESEARCH_PACKAGE","audience":"technical","source_steps":[1,2,3,4],
    "expected_outputs":EXPECTED_OUTPUTS,"random_seed":RANDOM_SEED,"formal_model_acceptance_passed":False,
    "test_source_files_reopened":False,"test_images_accessed_in_this_phase":False,"source_test_run_id":ledger["run_id"]}
write_json("configuration.json",configuration)
provenance={"completed_utc":datetime.now(timezone.utc).isoformat(),"phase":PHASE,"python":sys.version,"platform":platform.platform(),
    "numpy":np.__version__,"pandas":pd.__version__,"input_gate_hashes":{"step01":sha256_file(step1_gate_path),"step02":sha256_file(step2_gate_path),"step03":sha256_file(step3_gate_path),"step04":sha256_file(step4_gate_path)},
    "step04_signature_sha256":sha256_file(signature_path),"package_signature_sha256":sha256_file(package_signature_path),
    "source_test_evaluation_test_images_accessed":True,"test_source_files_reopened_in_this_phase":False,"test_images_accessed_in_this_phase":False}
write_json("provenance.json",provenance)

gate={"status":"final_research_package_complete","result_level":"FINAL_PROJECT_COMPLETE",
    "selected_configuration":{"fusion":"maximum","threshold":0.70,"post_processing":"none","policy_sha256":sha256_file(STEP3_OUT/"final_inference_policy.json")},
    "selected_metrics":{**recomputed,"test_positive_patients":13,"test_volumes":14,"final_test_targets_passed":7,"final_test_targets_total":8},
    "targets":{k:True for k in package_requirements},"target_passes":package_requirements,"all_mandatory_targets_passed":True,
    "formal_model_acceptance_passed":False,"formal_model_acceptance_failure":"minimum_positive_patient_dice",
    "decision":"ARCHIVE_FINAL_RESULT_FORMAL_MODEL_ACCEPTANCE_FAILED_NO_FURTHER_TEST_USE",
    "next_step":"archive_and_manuscript_revision_without_test_reuse","manifest_sha256":policy["dataset"]["manifest_sha256"],
    "input_artifact_hashes":{"step04_gate":sha256_file(step4_gate_path),"step04_signature":sha256_file(signature_path),"final_package_signature":sha256_file(package_signature_path)},
    "test_images_accessed":False,"source_test_evaluation_test_images_accessed":True,"test_source_files_reopened":False}
write_json("gate_result.json",gate)
missing=[name for name in EXPECTED_OUTPUTS if not (OUTPUT_DIR/name).is_file()]
assert not missing,missing
print(json.dumps(gate,indent=2)); print(f"PASS: {len(EXPECTED_OUTPUTS)}/{len(EXPECTED_OUTPUTS)} package outputs exist.")
"""),
md(r"""
## Final takeaway

The research workflow is complete and reproducible. The frozen model showed credible aggregate generalization but failed its predeclared patient-safety guardrail. The scientifically defensible disposition is to archive and report the negative formal gate, preserve the one-time test lock, and treat any future modelling as a new study with a new untouched evaluation cohort.
"""),
]

nb=nbf.v4.new_notebook(cells=cells)
nb.metadata={"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},"language_info":{"name":"python","version":"3.11"},
             "project_contract":{"phase":"step_05_final_research_package","mode":"no_inference","test_source_files_reopened":False,"formal_model_acceptance_passed":False}}
PHASE_DIR.mkdir(parents=True,exist_ok=True)
with NOTEBOOK_PATH.open("w",encoding="utf-8") as f: nbf.write(nb,f)
print(NOTEBOOK_PATH)
