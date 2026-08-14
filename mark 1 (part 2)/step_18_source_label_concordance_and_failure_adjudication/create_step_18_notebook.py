from pathlib import Path
import nbformat as nbf

STEP_DIR = Path(__file__).resolve().parent
LONG = STEP_DIR / "step_18_source_label_concordance_and_failure_adjudication.ipynb"
SHORT = STEP_DIR / "step_18.ipynb"
nb = nbf.v4.new_notebook()
nb["metadata"] = {"kernelspec": {"display_name": "Python 3 (ds_gpu)", "language": "python", "name": "python3"}, "language_info": {"name": "python", "version": "3.11"}}
c = []
c.append(nbf.v4.new_markdown_cell("""# Step 18 — Source-label concordance and failure adjudication

## tl;dr

This read-only audit compares the sealed Step 16 predictions for the five tumour-negative controls and worst positive case against the original 3D-IRCADb-01 source mask folders. It identifies whether predictions overlap excluded annotations, but it does **not** reinterpret biology, change the frozen hepatic-tumour truth, alter metrics, tune the model, or rerun inference.
"""))
c.append(nbf.v4.new_markdown_cell("""## Context & Methods

### Key Assumptions

- Step 16 and Step 17 are sealed, signed inputs.
- The accepted evaluation truth remains the Step 14 union of folders matching `^livertumors?\\d*$`.
- Generic `tumor`, adrenal/surrenal tumour, cyst, metastasectomy and organ masks are source annotations for concordance only.
- Overlap is descriptive; source-mask names and pixels are not expert adjudication.
- Nearest-neighbour resizing reproduces the frozen 256 × 256 mask grid.
- No local LiTS test image, mask, statistic or loader is accessed.
"""))
c.append(nbf.v4.new_code_cell(r'''from pathlib import Path
from datetime import datetime, timezone
import hashlib, json, platform, re, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import nibabel as nib
import pydicom

STEP_DIR=Path.cwd().resolve(); assert STEP_DIR.name=="step_18_source_label_concordance_and_failure_adjudication"
PART2=STEP_DIR.parent; OUT=STEP_DIR/"outputs"; OUT.mkdir(exist_ok=True)
S13=PART2/"step_13_3d_ircadb_ingestion_and_qc"/"outputs"
S14=PART2/"step_14_3d_ircadb_normalized_conversion_and_parity_qc"/"outputs"
S16=PART2/"step_16_one_time_external_evaluation_after_explicit_authorization"/"outputs"
S17=PART2/"step_17_external_evaluation_evidence_validation_and_data_card"/"outputs"
CACHE=S16/"probability_cache"; EXTRACTED=S13/"extracted"; THRESHOLD=0.70
SELECTED=["ircadb_05","ircadb_07","ircadb_11","ircadb_14","ircadb_18","ircadb_20"]
def sha256(p,chunk=1024*1024):
 h=hashlib.sha256()
 with Path(p).open("rb") as f:
  for b in iter(lambda:f.read(chunk),b""): h.update(b)
 return h.hexdigest()
def loadj(p): return json.loads(Path(p).read_text(encoding="utf-8"))
def savej(o,n): (OUT/n).write_text(json.dumps(o,indent=2,sort_keys=True,default=str),encoding="utf-8")
def savec(d,n): d.to_csv(OUT/n,index=False)
print("Diagnostic read-only audit; inference=False; tuning=False; Step16 rerun=False; local LiTS test access=False")'''))
c.append(nbf.v4.new_markdown_cell("""## Data

### 1. Verify sealed inputs and enumerate source labels
"""))
c.append(nbf.v4.new_code_cell(r'''g16=loadj(S16/"gate_result.json"); l16=loadj(S16/"run_ledger.json"); g17=loadj(S17/"gate_result.json")
checks=pd.DataFrame([
 {"check":"step16_complete","passed":g16.get("result_level")=="EXTERNAL_EVALUATION_COMPLETE"},
 {"check":"step16_sealed_and_rerun_prohibited","passed":l16.get("status")=="sealed_complete" and g16.get("rerun_permitted") is False},
 {"check":"step17_validated","passed":g17.get("result_level")=="EXTERNAL_GENERALIZATION_EVIDENCE_VALIDATED_WITH_CAVEATS"},
 {"check":"step17_points_to_step16_signature","passed":g17.get("step16_signature_sha256")==sha256(S16/"step_16_signature.json")},
 {"check":"six_selected_caches_exist","passed":all((CACHE/f"{p}.npz").is_file() for p in SELECTED)},
 {"check":"source_label_inventory_exists","passed":(S13/"label_inventory.csv").is_file()},
 {"check":"extracted_source_exists","passed":EXTRACTED.is_dir()},
 {"check":"local_lits_test_not_accessed","passed":g16.get("local_lits_test_accessed") is False},
])
savec(checks,"input_verification.csv"); assert checks.passed.all(),checks.loc[~checks.passed].to_dict("records")
labels=pd.read_csv(S13/"label_inventory.csv")
labels["internal_patient_id"]=labels.patient_id.str.extract(r"(\d+)$")[0].astype(int).map(lambda x:f"ircadb_{x:02d}")
source_inventory=labels[labels.internal_patient_id.isin(SELECTED)].copy()
def category(name):
 n=name.lower()
 if re.fullmatch(r"livertumors?\d*",n): return "accepted_hepatic_tumour"
 if n=="tumor": return "generic_tumour_excluded"
 if "surre" in n and "tumor" in n or "adrenal" in n and "tumor" in n: return "adrenal_tumour_excluded"
 if "metastasect" in n: return "metastasectomy_excluded"
 if "kyst" in n or "cyst" in n: return "cyst_excluded"
 if n=="liver": return "liver"
 return "other_structure"
source_inventory["semantic_category"]=source_inventory.label_name.map(category)
source_inventory["accepted_evaluation_truth"]=source_inventory.semantic_category.eq("accepted_hepatic_tumour")
savec(source_inventory,"selected_source_label_inventory.csv")
print(f"PASS: sealed evidence verified; {len(source_inventory)} source label folders inventoried")'''))
c.append(nbf.v4.new_markdown_cell("""## Results

### 2. Reconstruct source masks on the frozen grid and measure concordance

Each DICOM mask is aligned to CT slices by `InstanceNumber` and checked by spatial position. Resizing is nearest-neighbour only. Because source labels may overlap, category fractions must not be summed as if mutually exclusive.
"""))
c.append(nbf.v4.new_code_cell(r'''def deepest_named(root,name):
 hits=[p for p in root.rglob(name) if p.is_dir() and not any(q.name==name for q in p.iterdir() if q.is_dir())]
 assert len(hits)==1,(root,name,hits); return hits[0]
def read_rows(folder,pixels=False):
 rows=[]
 for p in folder.iterdir():
  if not p.is_file(): continue
  ds=pydicom.dcmread(p,force=True)
  if not hasattr(ds,"Rows"): continue
  ipp=np.asarray(getattr(ds,"ImagePositionPatient",[0,0,float(getattr(ds,"InstanceNumber",0))]),float)
  iop=np.asarray(getattr(ds,"ImageOrientationPatient",[1,0,0,0,1,0]),float)
  pos=float(np.dot(ipp,np.cross(iop[:3],iop[3:])))
  rows.append((pos,int(getattr(ds,"InstanceNumber",0)),p,ds))
 rows.sort(key=lambda x:(x[0],x[1],x[2].name)); assert rows
 if not pixels:return rows
 return rows,np.stack([(x[3].pixel_array>0).astype(np.uint8) for x in rows])
def aligned_mask(label_dir,ct_rows):
 mrows,m=read_rows(label_dir,True); idx={inst:i for i,(_,inst,_,_) in enumerate(mrows)}
 out=np.zeros((len(ct_rows),int(ct_rows[0][3].Rows),int(ct_rows[0][3].Columns)),np.uint8); errors=[]; missing=[]
 for zi,(pos,inst,_,_) in enumerate(ct_rows):
  if inst not in idx: missing.append(inst); continue
  mi=idx[inst]; out[zi]=m[mi]; errors.append(abs(pos-mrows[mi][0]))
 return out,{"source_slices":len(mrows),"matched_slices":len(ct_rows)-len(missing),"missing_instances":len(missing),"max_position_error_mm":max(errors) if errors else np.nan}
def resize_zyx(mask):
 # Step 14 serializes DICOM z,y,x as NIfTI x,y,z; Step 16 resizes each x,y slice.
 return np.stack([np.asarray(Image.fromarray(x.T).resize((256,256),Image.Resampling.NEAREST))>0 for x in mask])

cohort=pd.read_csv(PART2/"step_15_frozen_external_evaluation_contract"/"outputs"/"external_cohort_registry.csv").set_index("internal_patient_id")
rows=[]; align=[]
for pid in SELECTED:
 source_id=cohort.loc[pid,"source_patient_id"]; root=EXTRACTED/source_id
 ct_rows=read_rows(deepest_named(root,"PATIENT_DICOM")); mask_root=deepest_named(root,"MASKS_DICOM")
 with np.load(CACHE/f"{pid}.npz",allow_pickle=False) as z:
  pred=z["fused_probability"]>=THRESHOLD; truth=z["truth"].astype(bool)
 assert pred.shape[0]==len(ct_rows)
 for label_dir in sorted([p for p in mask_root.iterdir() if p.is_dir()],key=lambda p:p.name.lower()):
  native,qc=aligned_mask(label_dir,ct_rows); mask=resize_zyx(native)
  assert mask.shape==pred.shape
  inter=int((pred&mask).sum()); pp=int(pred.sum()); lp=int(mask.sum()); union=int((pred|mask).sum())
  rows.append({"internal_patient_id":pid,"source_patient_id":source_id,"label_name":label_dir.name,"semantic_category":category(label_dir.name),
   "accepted_evaluation_truth":category(label_dir.name)=="accepted_hepatic_tumour","prediction_pixels":pp,"label_pixels":lp,"overlap_pixels":inter,
   "prediction_explained_fraction":inter/pp if pp else np.nan,"label_recall_by_prediction":inter/lp if lp else np.nan,
   "dice":2*inter/(pp+lp) if pp+lp else 1.0,"jaccard":inter/union if union else 1.0})
  align.append({"internal_patient_id":pid,"label_name":label_dir.name,**qc,"shape_matches_prediction":mask.shape==pred.shape})
concordance=pd.DataFrame(rows); alignment=pd.DataFrame(align)
savec(concordance,"source_label_prediction_concordance.csv"); savec(alignment,"source_mask_alignment_qc.csv")
assert alignment.missing_instances.eq(0).all() and alignment.shape_matches_prediction.all() and alignment.max_position_error_mm.fillna(0).le(1e-3).all()
specific=concordance[~concordance.label_name.str.lower().isin(["skin","liver"])]
best=specific.sort_values(["internal_patient_id","overlap_pixels"],ascending=[True,False]).groupby("internal_patient_id").head(1)
savec(best,"highest_overlap_source_label_by_case.csv")
display(best[["internal_patient_id","label_name","semantic_category","prediction_explained_fraction","dice"]])'''))
c.append(nbf.v4.new_markdown_cell("""### 3. Summarize excluded-label overlap and create adjudication panels
"""))
c.append(nbf.v4.new_code_cell(r'''excluded=concordance[concordance.semantic_category.isin(["generic_tumour_excluded","adrenal_tumour_excluded","metastasectomy_excluded","cyst_excluded"])]
summary=[]
for pid in SELECTED:
 d=concordance[concordance.internal_patient_id.eq(pid)]; ex=excluded[excluded.internal_patient_id.eq(pid)]
 top=d[~d.label_name.str.lower().isin(["skin","liver"])].sort_values("overlap_pixels",ascending=False).iloc[0]
 with np.load(CACHE/f"{pid}.npz",allow_pickle=False) as z: pred=z["fused_probability"]>=THRESHOLD; truth=z["truth"].astype(bool)
 summary.append({"internal_patient_id":pid,"frozen_truth_status":"positive" if truth.any() else "negative_control","prediction_pixels":int(pred.sum()),
  "top_source_label":top.label_name,"top_source_category":top.semantic_category,"top_prediction_explained_fraction":float(top.prediction_explained_fraction),
  "excluded_annotation_labels_present":";".join(ex.label_name.tolist()),"maximum_single_excluded_overlap_fraction":float(ex.prediction_explained_fraction.max()) if len(ex) else 0.0,
  "adjudication_status":"source_overlap_measured_expert_review_required","frozen_metrics_changed":False})
case_summary=pd.DataFrame(summary); savec(case_summary,"case_adjudication_summary.csv")

plot_cases=["ircadb_07","ircadb_14","ircadb_18"]
panel_labels={"ircadb_07":"tumor","ircadb_14":"metastasectomie","ircadb_18":"livertumor"}
fig,axes=plt.subplots(3,4,figsize=(15,12))
for rr,pid in enumerate(plot_cases):
 top=concordance[concordance.internal_patient_id.eq(pid)&concordance.label_name.eq(panel_labels[pid])].iloc[0]
 source_id=cohort.loc[pid,"source_patient_id"]; root=EXTRACTED/source_id; ct_rows=read_rows(deepest_named(root,"PATIENT_DICOM")); mask_root=deepest_named(root,"MASKS_DICOM")
 label_dir=next(p for p in mask_root.iterdir() if p.is_dir() and p.name==top.label_name); source_mask=resize_zyx(aligned_mask(label_dir,ct_rows)[0])
 with np.load(CACHE/f"{pid}.npz",allow_pickle=False) as z: prob=z["fused_probability"].astype(np.float32); pred=prob>=THRESHOLD; truth=z["truth"].astype(bool)
 score=(pred&source_mask).sum((1,2)) if (pred&source_mask).any() else (truth&~pred).sum((1,2)); zi=int(np.argmax(score))
 ct=np.asanyarray(nib.load(str(PART2/"step_14_3d_ircadb_normalized_conversion_and_parity_qc"/cohort.loc[pid,"ct_relative_path"])).dataobj)
 u8=np.rint((np.clip(ct[:,:,zi],-160,240)+160)/400*255).astype(np.uint8); image=np.asarray(Image.fromarray(u8).resize((256,256),Image.Resampling.BILINEAR))
 overlay=np.zeros((256,256,3),float); overlay[...,0]=pred[zi]; overlay[...,1]=source_mask[zi]; overlay[...,2]=truth[zi]
 panels=[(image,"Broad-window CT","gray",0,255),(prob[zi],"Sealed probability","magma",0,1),(source_mask[zi],f"Source: {top.label_name}","gray",0,1),(overlay,"R=prediction G=source B=truth",None,0,1)]
 for ax,(arr,title,cmap,vmin,vmax) in zip(axes[rr],panels): ax.imshow(arr,cmap=cmap,vmin=vmin,vmax=vmax); ax.set_title(f"{pid} z={zi}\n{title}"); ax.axis("off")
fig.tight_layout(); fig.savefig(OUT/"source_label_concordance_panels.png",dpi=170,bbox_inches="tight"); plt.close(fig)

fig,ax=plt.subplots(figsize=(10,5)); temp=concordance[concordance.overlap_pixels.gt(0)&~concordance.label_name.str.lower().isin(["skin","liver"])].copy(); temp["label_case"]=temp.internal_patient_id+":"+temp.label_name
temp=temp.nlargest(15,"prediction_explained_fraction").sort_values("prediction_explained_fraction")
ax.barh(temp.label_case,temp.prediction_explained_fraction,color=np.where(temp.accepted_evaluation_truth,"#2ca02c","#d62728")); ax.set_xlim(0,1); ax.set_xlabel("Fraction of prediction overlapping source label"); ax.set_title("Largest source-label concordances (green = accepted truth)")
fig.tight_layout(); fig.savefig(OUT/"source_label_overlap_dashboard.png",dpi=170,bbox_inches="tight"); plt.close(fig)
display(case_summary)'''))
c.append(nbf.v4.new_markdown_cell("""## Takeaways

### 4. Freeze the interpretation, provenance and machine-readable gate
"""))
c.append(nbf.v4.new_code_cell(r'''meaningful=case_summary.maximum_single_excluded_overlap_fraction.gt(0.05)
result="SOURCE_LABEL_CONCORDANCE_COMPLETE_EXPERT_REVIEW_REQUIRED"
expected=pd.DataFrame([
 {"requirement":"sealed_inputs_verified","expected":True,"actual":bool(checks.passed.all()),"passed":bool(checks.passed.all())},
 {"requirement":"all_source_masks_aligned","expected":True,"actual":bool((alignment.missing_instances.eq(0)&alignment.shape_matches_prediction).all()),"passed":bool((alignment.missing_instances.eq(0)&alignment.shape_matches_prediction).all())},
 {"requirement":"six_cases_profiled","expected":6,"actual":int(case_summary.internal_patient_id.nunique()),"passed":case_summary.internal_patient_id.nunique()==6},
 {"requirement":"frozen_metrics_unchanged","expected":True,"actual":bool((~case_summary.frozen_metrics_changed).all()),"passed":bool((~case_summary.frozen_metrics_changed).all())},
 {"requirement":"no_inference_or_tuning","expected":True,"actual":True,"passed":True},
 {"requirement":"local_lits_test_not_accessed","expected":True,"actual":True,"passed":True},
])
savec(expected,"expected_vs_actual.csv"); assert expected.passed.all()
gate={"created_utc":datetime.now(timezone.utc).isoformat(),"result_level":result,"gate_passed":True,"cases_profiled":6,"source_label_folders_profiled":len(concordance),
 "cases_with_excluded_annotation_overlap_gt_5pct":case_summary.loc[meaningful,"internal_patient_id"].tolist(),
 "interpretation":"Source-label overlap was measured. It can explain annotation-semantic discordance but cannot establish biological truth without expert review.",
 "step16_result_changed":False,"step16_metrics_changed":False,"inference_performed":False,"tuning_performed":False,"step16_rerun_performed":False,"local_lits_test_accessed":False,
 "next_step":"optional blinded expert review of preserved panels and source annotations; otherwise stop analysis or freeze a second independent external dataset contract"}
savej(gate,"gate_result.json")
report="# Step 18 source-label concordance report\n\n"+f"Result: `{result}`.\n\n```csv\n"+case_summary.to_csv(index=False)+"```\n\n## Interpretation boundary\n\nThe source masks document annotation overlap only. They do not authorize retrospective relabeling, metric changes, threshold tuning, or biological claims. Expert review remains required.\n"
(OUT/"SOURCE_LABEL_CONCORDANCE_REPORT.md").write_text(report,encoding="utf-8")
provenance={"phase":STEP_DIR.name,"created_utc":gate["created_utc"],"python":sys.version,"platform":platform.platform(),"threshold":THRESHOLD,"selected_cases":SELECTED,
 "step16_signature_sha256":sha256(S16/"step_16_signature.json"),"step17_signature_sha256":sha256(S17/"step_17_signature.json"),"mask_interpolation":"nearest",
 "inference_performed":False,"tuning_performed":False,"local_lits_test_accessed":False}
savej(provenance,"provenance.json")
names=["input_verification.csv","selected_source_label_inventory.csv","source_label_prediction_concordance.csv","source_mask_alignment_qc.csv","highest_overlap_source_label_by_case.csv","case_adjudication_summary.csv","expected_vs_actual.csv","source_label_concordance_panels.png","source_label_overlap_dashboard.png","SOURCE_LABEL_CONCORDANCE_REPORT.md","gate_result.json","provenance.json"]
signed={str((OUT/n).relative_to(PART2)):sha256(OUT/n) for n in names}; combined=hashlib.sha256("\n".join(f"{k}:{v}" for k,v in sorted(signed.items())).encode()).hexdigest()
savej({"algorithm":"SHA-256","created_utc":datetime.now(timezone.utc).isoformat(),"combined_sha256":combined,"result_level":result,"signed_artifacts":signed,"local_lits_test_accessed":False},"step_18_signature.json")
print(json.dumps(gate,indent=2))'''))
c.append(nbf.v4.new_markdown_cell("""The sealed external evaluation remains complete and unchanged. Step 18 provides a reproducible source-annotation concordance record. The only scientifically justified follow-up is optional blinded expert review; no result-driven rerun or tuning is permitted."""))
nb["cells"] = c
for path in (LONG, SHORT): nbf.write(nb, path)
print("Wrote", LONG)
print("Wrote", SHORT)
