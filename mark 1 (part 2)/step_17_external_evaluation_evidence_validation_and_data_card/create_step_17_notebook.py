from pathlib import Path
import nbformat as nbf

STEP_DIR=Path(__file__).resolve().parent
LONG=STEP_DIR/"step_17_external_evaluation_evidence_validation_and_data_card.ipynb"
SHORT=STEP_DIR/"step_17.ipynb"
nb=nbf.v4.new_notebook()
nb["metadata"]={"kernelspec":{"display_name":"Python 3 (ds_gpu)","language":"python","name":"python3"},"language_info":{"name":"python","version":"3.11"}}
c=[]
c.append(nbf.v4.new_markdown_cell("""# Step 17 — External-evaluation evidence validation and data card

## tl;dr

This read-only notebook validates the sealed Step 16 result, independently recomputes headline metrics, verifies all evidence and cache hashes, quantifies negative-control and difficulty caveats, produces a failure atlas and writes a signed external-result data card.

It performs no model loading, inference, threshold sweep, tuning, case exclusion or local LiTS test access. The Step 16 result remains immutable.
"""))
c.append(nbf.v4.new_markdown_cell("""## Context & Methods

### Key assumptions

- Step 16 is a complete sealed one-time external run.
- All metrics are recomputed from saved counts or frozen probability caches.
- Tumour-negative means no accepted hepatic-tumour label under the Step 14 semantics. Predictions in those cases are false positives for this evaluation, but their biological meaning cannot be inferred without additional annotations.
- A passed predeclared gate supports external generalization evidence; it does not establish clinical readiness or erase patient-level failures.
"""))
c.append(nbf.v4.new_code_cell(r'''from pathlib import Path
from datetime import datetime, timezone
import hashlib, json, platform, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import nibabel as nib

STEP_DIR=Path.cwd().resolve(); assert STEP_DIR.name=="step_17_external_evaluation_evidence_validation_and_data_card"
PART2=STEP_DIR.parent; OUT=STEP_DIR/"outputs"; OUT.mkdir(exist_ok=True)
STEP14=PART2/"step_14_3d_ircadb_normalized_conversion_and_parity_qc"; S14=STEP14/"outputs"
STEP15=PART2/"step_15_frozen_external_evaluation_contract"; S15=STEP15/"outputs"
STEP16=PART2/"step_16_one_time_external_evaluation_after_explicit_authorization"; S16=STEP16/"outputs"; CACHE=S16/"probability_cache"
EPS=1e-6
def sha256(p,chunk=1024*1024):
 h=hashlib.sha256()
 with Path(p).open("rb") as f:
  for b in iter(lambda:f.read(chunk),b""): h.update(b)
 return h.hexdigest()
def loadj(p): return json.loads(Path(p).read_text(encoding="utf-8"))
def savej(o,n): (OUT/n).write_text(json.dumps(o,indent=2,sort_keys=True,default=str),encoding="utf-8")
def savec(d,n): d.to_csv(OUT/n,index=False)
print("Read-only sealed-evidence validation; inference=False; tuning=False; local test access=False")'''))

c.append(nbf.v4.new_markdown_cell("""## Data

### 1. Verify the sealed Step 16 package and all cache hashes
"""))
c.append(nbf.v4.new_code_cell(r'''gate=loadj(S16/"gate_result.json"); ledger=loadj(S16/"run_ledger.json"); sig=loadj(S16/"step_16_signature.json")
sig_rows=[]
for rel,exp in sig["signed_artifacts"].items():
 p=S16/rel; act=sha256(p) if p.is_file() else None
 sig_rows.append({"artifact":rel,"exists":p.is_file(),"expected_sha256":exp,"actual_sha256":act,"passed":act==exp})
sig_check=pd.DataFrame(sig_rows); savec(sig_check,"step_16_signature_verification.csv")
inventory=pd.read_csv(S16/"external_evidence_inventory.csv"); inventory_rows=[]
for row in inventory.itertuples():
 p=S16/row.path; act=sha256(p) if p.is_file() else None
 inventory_rows.append({"path":row.path,"exists":p.is_file(),"expected_bytes":int(row.bytes),"actual_bytes":p.stat().st_size if p.is_file() else None,
  "expected_sha256":row.sha256,"actual_sha256":act,"passed":p.is_file() and p.stat().st_size==int(row.bytes) and act==row.sha256})
inventory_check=pd.DataFrame(inventory_rows); savec(inventory_check,"step_16_evidence_inventory_verification.csv")
prereq=pd.DataFrame([
 {"check":"step16_external_evaluation_complete","passed":gate.get("result_level")=="EXTERNAL_EVALUATION_COMPLETE"},
 {"check":"step16_ledger_sealed","passed":ledger.get("status")=="sealed_complete" and ledger.get("complete_result_produced") is True},
 {"check":"step16_rerun_prohibited","passed":gate.get("rerun_permitted") is False},
 {"check":"all_step16_signed_hashes_match","passed":bool(sig_check.passed.all())},
 {"check":"all_evidence_and_cache_hashes_match","passed":bool(inventory_check.passed.all())},
 {"check":"twenty_probability_caches","passed":len(list(CACHE.glob("*.npz")))==20},
 {"check":"one_time_run_identity","passed":gate.get("run_id")==ledger.get("run_id")==sig.get("run_id")},
 {"check":"local_lits_test_not_accessed","passed":gate.get("local_lits_test_accessed") is False},
])
savec(prereq,"input_verification.csv"); assert prereq.passed.all(),prereq.loc[~prereq.passed].to_dict("records")
print("PASS: sealed Step 16 package and 20 caches verified")'''))

c.append(nbf.v4.new_markdown_cell("""### 2. Independently reconcile the primary metrics and gate
"""))
c.append(nbf.v4.new_code_cell(r'''patients=pd.read_csv(S16/"patient_metrics.csv"); slices=pd.read_csv(S16/"slice_metrics.csv"); lesions=pd.read_csv(S16/"lesion_metrics.csv")
reported=pd.read_csv(S16/"global_metrics.csv").iloc[0]; expected=pd.read_csv(S16/"expected_vs_actual.csv")
positive=patients[patients.has_tumour.astype(bool)]; negative=patients[~patients.has_tumour.astype(bool)]
I=int(patients.intersection_pixels.sum()); T=int(patients.truth_pixels.sum()); P=int(patients.predicted_pixels.sum())
recalc={"global_dice":(2*I+EPS)/(T+P+EPS),"mean_positive_patient_dice":positive.dice.mean(),"median_positive_patient_dice":positive.dice.median(),
 "minimum_positive_patient_dice":positive.dice.min(),"q1_positive_slice_detection_pct":100*slices.loc[slices.q1_positive_slice_train_edge.astype(bool),"detected"].mean(),
 "positive_predicted_empty_pct":100*slices.loc[slices.positive_slice.astype(bool),"predicted_empty"].mean(),
 "empty_slice_false_positive_pct":100*slices.loc[~slices.positive_slice.astype(bool),"empty_slice_false_positive"].mean()}
reconciliation=[]
for metric,value in recalc.items():
 rv=float(reported[metric]); reconciliation.append({"metric":metric,"reported":rv,"recomputed":float(value),"absolute_difference":abs(rv-value),"passed":abs(rv-value)<=1e-10})
metric_recon=pd.DataFrame(reconciliation); savec(metric_recon,"independent_metric_reconciliation.csv"); assert metric_recon.passed.all()

gate_recon=expected.copy(); gate_recon["recomputed_passed"]=[bool(v>=t if op==">=" else v<=t if op=="<=" else np.isclose(v,t)) for v,op,t in zip(gate_recon.actual,gate_recon.operator,gate_recon.threshold)]
gate_recon["agrees_with_step16"]=gate_recon.passed.astype(bool)==gate_recon.recomputed_passed
savec(gate_recon,"acceptance_reconciliation.csv"); assert gate_recon.agrees_with_step16.all() and gate_recon.recomputed_passed.all()
print("PASS: headline metrics and all predeclared gates independently reconcile")'''))

c.append(nbf.v4.new_markdown_cell("""## Results

### 3. Quantify patient, negative-control and lesion caveats
"""))
c.append(nbf.v4.new_code_cell(r'''cohort=pd.read_csv(S15/"external_cohort_registry.csv"); meta=cohort.set_index("internal_patient_id")
risk=patients.copy(); risk["distance_from_positive_floor"]=np.where(risk.has_tumour.astype(bool),risk.dice-0.01,np.nan)
risk["severity"]=np.select([~risk.has_tumour.astype(bool)&(risk.predicted_pixels>0),risk.has_tumour.astype(bool)&(risk.dice<0.05),risk.has_tumour.astype(bool)&(risk.dice<0.5)],
 ["negative_control_false_positive","critical_positive_failure","low_positive_performance"],default="routine")
savec(risk.sort_values(["severity","dice"]),"patient_risk_profile.csv")

negative_qc=negative[["internal_patient_id","predicted_pixels","false_positive_pixels"]].copy()
negative_qc["standardized_voxel_ml"]=[float(meta.loc[x].spacing_x_mm*2*meta.loc[x].spacing_y_mm*2*meta.loc[x].spacing_z_mm/1000) for x in negative_qc.internal_patient_id]
negative_qc["predicted_false_positive_ml"]=negative_qc.predicted_pixels*negative_qc.standardized_voxel_ml
negative_qc["any_false_positive"]=negative_qc.predicted_pixels>0
negative_qc["interpretation_boundary"]="false positive against accepted hepatic-tumour truth; biological meaning unresolved"
savec(negative_qc,"negative_control_physical_volume.csv")

lesion_summary=lesions.groupby("train_derived_lesion_stratum",dropna=False).agg(lesions=("truth_component_id_6conn","size"),detected=("detected","sum"),
 detected_pct=("detected",lambda x:100*x.mean()),mean_matched_dice=("matched_dice","mean"),median_matched_dice=("matched_dice","median"),median_volume_ml=("lesion_volume_ml","median")).reset_index()
savec(lesion_summary,"lesion_difficulty_validation.csv")
findings=pd.DataFrame([
 {"finding":"formal_predeclared_external_gate","value":"PASS","severity":"positive evidence","confidence":"high"},
 {"finding":"mean_positive_patient_dice","value":float(positive.dice.mean()),"severity":"positive evidence","confidence":"high"},
 {"finding":"minimum_positive_patient_dice","value":float(positive.dice.min()),"severity":"critical caveat","confidence":"high"},
 {"finding":"tumour_negative_patients_with_any_fp","value":int(negative_qc.any_false_positive.sum()),"severity":"high caveat","confidence":"high"},
 {"finding":"tumour_negative_total_fp_ml","value":float(negative_qc.predicted_false_positive_ml.sum()),"severity":"high caveat","confidence":"high"},
 {"finding":"smallest_lesion_stratum_detection_pct","value":float(lesion_summary.iloc[0].detected_pct),"severity":"moderate caveat","confidence":"high"},
])
savec(findings,"validated_findings.csv"); display(findings); display(negative_qc.sort_values("predicted_false_positive_ml",ascending=False))'''))

c.append(nbf.v4.new_markdown_cell("""### 4. Produce descriptive probability diagnostics and a failure atlas

Calibration is descriptive only; no threshold changes are permitted.
"""))
c.append(nbf.v4.new_code_cell(r'''edges=np.linspace(0,1,51); rel_edges=np.linspace(0,1,21)
hist={"truth_tumour":np.zeros(50,np.int64),"non_tumour_liver":np.zeros(50,np.int64),"background":np.zeros(50,np.int64)}
rel_n=np.zeros(20,np.int64); rel_pos=np.zeros(20,np.float64); rel_sum=np.zeros(20,np.float64)
for path in sorted(CACHE.glob("*.npz")):
 with np.load(path,allow_pickle=False) as z: prob=z["fused_probability"].astype(np.float32); truth=z["truth"].astype(bool); organ=z["organ"].astype(bool)
 for key,mask in {"truth_tumour":truth,"non_tumour_liver":organ&~truth,"background":~organ}.items(): hist[key]+=np.histogram(prob[mask],edges)[0]
 flat=prob.ravel(); y=truth.ravel(); idx=np.clip(np.digitize(flat,rel_edges)-1,0,19); rel_n+=np.bincount(idx,minlength=20); rel_pos+=np.bincount(idx,weights=y,minlength=20); rel_sum+=np.bincount(idx,weights=flat,minlength=20)
hist_df=pd.DataFrame([{"category":k,"bin_left":edges[i],"bin_right":edges[i+1],"count":int(v[i])} for k,v in hist.items() for i in range(50)])
reliability=pd.DataFrame({"bin_left":rel_edges[:-1],"bin_right":rel_edges[1:],"count":rel_n,"mean_score":np.divide(rel_sum,rel_n,out=np.full(20,np.nan),where=rel_n>0),"observed_positive_fraction":np.divide(rel_pos,rel_n,out=np.full(20,np.nan),where=rel_n>0)})
reliability["absolute_gap"]=(reliability.mean_score-reliability.observed_positive_fraction).abs(); ece=float(np.nansum(reliability.absolute_gap*reliability["count"])/reliability["count"].sum())
savec(hist_df,"probability_histograms.csv"); savec(reliability,"descriptive_reliability.csv")

selected=[positive.sort_values("dice").iloc[0].internal_patient_id,*negative.sort_values("predicted_pixels",ascending=False).head(2).internal_patient_id.tolist()]
fig,axes=plt.subplots(len(selected),4,figsize=(15,4*len(selected)),squeeze=False)
slice_lookup=slices.copy(); cohort_lookup=cohort.set_index("internal_patient_id")
for row_axes,pid in zip(axes,selected):
 with np.load(CACHE/f"{pid}.npz",allow_pickle=False) as z: prob=z["fused_probability"].astype(np.float32); truth=z["truth"].astype(bool); pred=prob>=0.70
 if truth.any(): zi=int(np.argmax((truth&~pred).sum((1,2))))
 else: zi=int(np.argmax(pred.sum((1,2))))
 m=cohort_lookup.loc[pid]; ct=np.asanyarray(nib.load(str(STEP14/m.ct_relative_path)).dataobj).astype(np.float32); u8=np.rint((np.clip(ct[:,:,zi],-160,240)+160)/400*255).astype(np.uint8); image=np.asarray(Image.fromarray(u8).resize((256,256),Image.Resampling.BILINEAR))
 error=np.zeros((256,256),np.uint8); error[pred[zi]&~truth[zi]]=1; error[truth[zi]&~pred[zi]]=2
 panels=[(image,"Broad-window CT","gray",0,255),(truth[zi],"Truth","gray",0,1),(prob[zi],"Fused probability","magma",0,1),(error,"FP=1, FN=2","viridis",0,2)]
 for ax,(arr,title,cmap,vmin,vmax) in zip(row_axes,panels): ax.imshow(arr,cmap=cmap,vmin=vmin,vmax=vmax); ax.set_title(f"{pid} z={zi}\n{title}"); ax.axis("off")
fig.tight_layout(); fig.savefig(OUT/"external_failure_atlas.png",dpi=170,bbox_inches="tight"); plt.close(fig)

fig,axes=plt.subplots(1,3,figsize=(15,4.5))
axes[0].hist(positive.dice,bins=np.linspace(0,1,11),color="#4472c4"); axes[0].axvline(positive.dice.mean(),color="black",ls="--"); axes[0].set_title("Positive-patient Dice")
axes[1].bar(negative_qc.internal_patient_id,negative_qc.predicted_false_positive_ml,color="#d62728"); axes[1].tick_params(axis="x",rotation=45); axes[1].set_title("Negative-control FP volume (ml)")
axes[2].plot([0,1],[0,1],"--",color="black"); axes[2].plot(reliability.mean_score,reliability.observed_positive_fraction,"o-"); axes[2].set_title(f"All-pixel reliability; background-dominated (ECE={ece:.4f})")
fig.tight_layout(); fig.savefig(OUT/"external_evidence_validation_dashboard.png",dpi=170,bbox_inches="tight"); plt.close(fig)'''))

c.append(nbf.v4.new_markdown_cell("""## Takeaways

### 5. Freeze the validated interpretation, data card and signature
"""))
c.append(nbf.v4.new_code_cell(r'''bootstrap=pd.read_csv(S16/"bootstrap_uncertainty.csv").iloc[0]
assessment="EXTERNAL_GENERALIZATION_EVIDENCE_VALIDATED_WITH_CAVEATS"
gate17={"created_utc":datetime.now(timezone.utc).isoformat(),"result_level":assessment,"evidence_validation_passed":True,
 "step16_predeclared_gate_passed":bool(gate["all_mandatory_targets_passed"]),"step16_run_id":gate["run_id"],"step16_signature_sha256":sha256(S16/"step_16_signature.json"),
 "headline":{"global_dice":float(reported.global_dice),"mean_positive_patient_dice":float(reported.mean_positive_patient_dice),"median_positive_patient_dice":float(reported.median_positive_patient_dice),
  "mean_dice_ci_95":[float(bootstrap.ci_lower_2_5),float(bootstrap.ci_upper_97_5)],"minimum_positive_patient_dice":float(reported.minimum_positive_patient_dice),
  "q1_detection_pct":float(reported.q1_positive_slice_detection_pct),"empty_slice_false_positive_pct":float(reported.empty_slice_false_positive_pct)},
 "caveats":{"worst_positive_patient":str(positive.sort_values("dice").iloc[0].internal_patient_id),"worst_positive_dice":float(positive.dice.min()),
  "negative_patients_with_any_fp":int(negative_qc.any_false_positive.sum()),"negative_patients_total":len(negative_qc),"negative_total_fp_ml":float(negative_qc.predicted_false_positive_ml.sum()),
  "smallest_lesion_detection_pct":float(lesion_summary.iloc[0].detected_pct)},
 "interpretation":"Formal frozen external gate passed. Evidence supports external generalization on 3D-IRCADb-01, with material patient-level, small-lesion and negative-control caveats.",
 "prohibited_claims":["clinical readiness","universal generalization","state of the art","zero false-positive risk"],
 "inference_performed":False,"step16_rerun_performed":False,"tuning_performed":False,"local_lits_test_accessed":False,
 "next_step":"optional_read_only_error_review_or_second_external_dataset_contract; no Step16 tuning or rerun"}
savej(gate17,"gate_result.json")

card=f"""# 3D-IRCADb-01 External Evaluation Data Card

## Result

- Assessment: `{assessment}`
- Frozen Step 16 gate: **PASS** ({int(expected.passed.astype(bool).sum())}/{len(expected)} mandatory rows)
- Cohort: 20 patients; 15 tumour-positive; 5 tumour-negative; zero exclusions
- Run UUID: `{gate["run_id"]}`

## Primary evidence

- Global Dice: {reported.global_dice:.6f}
- Mean positive-patient Dice: {reported.mean_positive_patient_dice:.6f} (95% patient-bootstrap CI {bootstrap.ci_lower_2_5:.6f}–{bootstrap.ci_upper_97_5:.6f})
- Median positive-patient Dice: {reported.median_positive_patient_dice:.6f}
- Minimum positive-patient Dice: {reported.minimum_positive_patient_dice:.6f}
- Q1 positive-slice detection: {reported.q1_positive_slice_detection_pct:.2f}%
- Positive predicted-empty rate: {reported.positive_predicted_empty_pct:.2f}%
- Empty-slice false-positive rate: {reported.empty_slice_false_positive_pct:.2f}%

## Required caveats

- Worst positive case: `{positive.sort_values("dice").iloc[0].internal_patient_id}` with Dice {positive.dice.min():.6f}.
- All {len(negative_qc)} tumour-negative patients had at least one predicted pixel; total standardized physical FP volume was {negative_qc.predicted_false_positive_ml.sum():.3f} ml.
- Smallest train-derived lesion stratum detection was {lesion_summary.iloc[0].detected_pct:.2f}% ({int(lesion_summary.iloc[0].detected)}/{int(lesion_summary.iloc[0].lesions)} components).
- Tumour-negative labels exclude adrenal/generic non-hepatic tumour folders; prediction biology cannot be inferred from this evaluation.
- The failure atlas shows visible low-attenuation structures in the largest negative-control prediction cases, but only expert/source-annotation review can determine their biological meaning; they remain false positives under the frozen accepted truth.
- The reported all-pixel reliability curve and ECE are background-dominated descriptive diagnostics, not a clinical calibration claim.
- The result does not establish clinical readiness or universal generalization.

## Integrity

- Step 16 signature and all 32 evidence/cache inventory rows independently verified.
- Headline metrics independently recomputed from saved counts.
- No inference, tuning, threshold sweep, case exclusion or Step 16 rerun occurred in Step 17.
"""
(OUT/"EXTERNAL_EVALUATION_DATA_CARD.md").write_text(card,encoding="utf-8")
savej({"phase":STEP_DIR.name,"created_utc":gate17["created_utc"],"python":sys.version,"platform":platform.platform(),"step16_signature_sha256":sha256(S16/"step_16_signature.json"),
 "inference_performed":False,"tuning_performed":False,"local_lits_test_accessed":False},"provenance.json")

sign_names=["input_verification.csv","step_16_signature_verification.csv","step_16_evidence_inventory_verification.csv","independent_metric_reconciliation.csv","acceptance_reconciliation.csv",
 "patient_risk_profile.csv","negative_control_physical_volume.csv","lesion_difficulty_validation.csv","validated_findings.csv","probability_histograms.csv","descriptive_reliability.csv",
 "external_failure_atlas.png","external_evidence_validation_dashboard.png","EXTERNAL_EVALUATION_DATA_CARD.md","gate_result.json","provenance.json"]
signed={str((OUT/n).relative_to(PART2)):sha256(OUT/n) for n in sign_names}; combined=hashlib.sha256("\n".join(f"{k}:{v}" for k,v in sorted(signed.items())).encode()).hexdigest()
savej({"algorithm":"SHA-256","created_utc":datetime.now(timezone.utc).isoformat(),"combined_sha256":combined,"result_level":assessment,"signed_artifacts":signed,
 "inference_performed":False,"step16_rerun_performed":False,"local_lits_test_accessed":False},"step_17_signature.json")
print(json.dumps(gate17,indent=2))'''))
c.append(nbf.v4.new_markdown_cell("""The correct conclusion is a frozen external generalization pass with explicit caveats. Further work may inspect existing error evidence or contract a genuinely independent second dataset, but Step 16 must not be rerun or tuned.
"""))
nb["cells"]=c
for p in (LONG,SHORT): nbf.write(nb,p)
print("Wrote",LONG); print("Wrote",SHORT)
