from pathlib import Path
import nbformat as nbf

STEP_DIR=Path(__file__).resolve().parent
LONG=STEP_DIR/"step_16_one_time_external_evaluation_after_explicit_authorization.ipynb"
SHORT=STEP_DIR/"step_16.ipynb"
nb=nbf.v4.new_notebook()
nb["metadata"]={"kernelspec":{"display_name":"Python 3 (ds_gpu)","language":"python","name":"python3"},"language_info":{"name":"python","version":"3.11"}}
c=[]
c.append(nbf.v4.new_markdown_cell("""# Step 16 — One-time external evaluation after explicit authorization

## tl;dr

This notebook implements the signed Step 15 contract for the complete 20-case 3D-IRCADb-01 cohort. It is guarded by exact authorization text and a one-time UUID ledger.

**Default state:** safe preflight only. It verifies signed inputs, checkpoint hashes and normalized paths, writes `EXTERNAL_EVALUATION_AUTHORIZATION_REQUIRED`, and performs no model loading or inference. Do not change model, threshold, fusion, post-processing, cohort or metrics after seeing results.
"""))
c.append(nbf.v4.new_markdown_cell("""## Context & Methods

### Authorization boundary

Only the exact authorization sentence frozen in Step 15 may enable the expensive cells. Authorization permits one external run under the frozen policy; it does not permit threshold sweeps, tuning, case exclusion, retraining or result-driven reruns. The local LiTS test split remains outside this notebook.
"""))
c.append(nbf.v4.new_code_cell(r'''from pathlib import Path
from datetime import datetime, timezone
import hashlib, json, platform, sys, time, uuid
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

STEP_DIR=Path.cwd().resolve(); assert STEP_DIR.name=="step_16_one_time_external_evaluation_after_explicit_authorization"
PART2=STEP_DIR.parent; PROJECT_ROOT=PART2.parent
OUT=STEP_DIR/"outputs"; CACHE=OUT/"probability_cache"; OUT.mkdir(exist_ok=True); CACHE.mkdir(exist_ok=True)
STEP14=PART2/"step_14_3d_ircadb_normalized_conversion_and_parity_qc"; S14=STEP14/"outputs"
STEP15=PART2/"step_15_frozen_external_evaluation_contract"; S15=STEP15/"outputs"

# These four values are patched only after the user supplies the exact sentence.
AUTHORIZATION_GRANTED=True
AUTHORIZATION_TEXT="I authorize the one-time Step 16 3D-IRCADb-01 external evaluation under the frozen Step 15 contract."
AUTHORIZATION_RECORDED_UTC="2026-08-05T14:49:18.3333060Z"
ONE_TIME_RUN_ID="61d1f140-c8b4-436a-928a-1e4b6f7c0b56"

EXPECTED_AUTHORIZATION_TEXT="I authorize the one-time Step 16 3D-IRCADb-01 external evaluation under the frozen Step 15 contract."
EXPECTED_MANIFEST="575a6fc391d63dc9bbbbb3317efe4d0f65edd57457ae63c1bfc6c2c615861889"

def sha256(p,chunk=1024*1024):
 h=hashlib.sha256()
 with Path(p).open("rb") as f:
  for b in iter(lambda:f.read(chunk),b""): h.update(b)
 return h.hexdigest()
def loadj(p): return json.loads(Path(p).read_text(encoding="utf-8"))
def savej(obj,name):
 p=OUT/name; p.write_text(json.dumps(obj,indent=2,sort_keys=True,default=str),encoding="utf-8"); return p
def savec(df,name):
 p=OUT/name; df.to_csv(p,index=False); return p

policy=loadj(S15/"frozen_external_evaluation_policy.json"); gate15=loadj(S15/"gate_result.json"); sig15=loadj(S15/"step_15_signature.json")
cohort=pd.read_csv(S15/"external_cohort_registry.csv"); acceptance=pd.read_csv(S15/"external_acceptance_contract.csv")
RUN_EVALUATION=bool(AUTHORIZATION_GRANTED and AUTHORIZATION_TEXT==EXPECTED_AUTHORIZATION_TEXT and AUTHORIZATION_RECORDED_UTC.strip() and ONE_TIME_RUN_ID.strip())
print("Run mode:","AUTHORIZED ONE-TIME EVALUATION" if RUN_EVALUATION else "SAFE PREFLIGHT — NO INFERENCE")'''))

c.append(nbf.v4.new_markdown_cell("""## Data

### 1. Verify the signed contract, checkpoints and complete cohort
"""))
c.append(nbf.v4.new_code_cell(r'''sig_rows=[]
for rel,exp in sig15["signed_artifacts"].items():
 p=PART2/rel; act=sha256(p) if p.is_file() else None
 sig_rows.append({"artifact":rel,"exists":p.is_file(),"expected_sha256":exp,"actual_sha256":act,"passed":act==exp})
sig_check=pd.DataFrame(sig_rows); savec(sig_check,"step_15_signature_verification.csv")

checkpoint_rows=[]
for role in ["control","recall","roi"]:
 path=Path(policy["model"][f"{role}_checkpoint"]); expected=policy["model"][f"{role}_sha256"]
 actual=sha256(path) if path.is_file() else None
 checkpoint_rows.append({"role":role,"path":str(path),"exists":path.is_file(),"expected_sha256":expected,"actual_sha256":actual,"passed":actual==expected})
checkpoint_check=pd.DataFrame(checkpoint_rows); savec(checkpoint_check,"checkpoint_verification.csv")

path_cols=["ct_relative_path","liver_relative_path","tumour_relative_path"]
path_rows=[]
for row in cohort.itertuples():
 for kind,col in [("ct","ct_relative_path"),("liver","liver_relative_path"),("tumour","tumour_relative_path")]:
  p=STEP14/getattr(row,col); path_rows.append({"internal_patient_id":row.internal_patient_id,"kind":kind,"path":str(p),"exists":p.is_file()})
path_check=pd.DataFrame(path_rows); savec(path_check,"normalized_path_verification.csv")

preflight=pd.DataFrame([
 {"check":"step15_contract_frozen","passed":gate15.get("contract_frozen") is True},
 {"check":"step15_result_level","passed":gate15.get("result_level")=="EXTERNAL_EVALUATION_CONTRACT_FROZEN_AWAITING_AUTHORIZATION"},
 {"check":"step15_signed_artifacts_match","passed":bool(sig_check.passed.all())},
 {"check":"three_checkpoint_hashes_match","passed":bool(checkpoint_check.passed.all())},
 {"check":"complete_20_case_cohort","passed":len(cohort)==20 and cohort.internal_patient_id.is_unique and cohort.evaluation_included.all()},
 {"check":"15_positive_5_negative","passed":int(cohort.primary_positive_population.sum())==15 and int(cohort.negative_control_population.sum())==5},
 {"check":"all_60_normalized_paths_exist","passed":len(path_check)==60 and bool(path_check.exists.all())},
 {"check":"manifest_identity","passed":gate15.get("manifest_sha256")==EXPECTED_MANIFEST},
 {"check":"policy_threshold_0_70","passed":policy["prediction"]["global_threshold"]==0.70},
 {"check":"policy_fusion_maximum","passed":policy["prediction"]["fusion"]=="maximum(control_probability, recall_probability)"},
 {"check":"policy_post_processing_none","passed":policy["prediction"]["post_processing"]=="none"},
 {"check":"local_lits_test_access_disabled","passed":policy["safety"]["local_lits_test_access_allowed"] is False},
])
savec(preflight,"preflight_checks.csv"); assert preflight.passed.all(),preflight.loc[~preflight.passed].to_dict("records")
auth={"authorization_granted":bool(AUTHORIZATION_GRANTED),"authorization_text":AUTHORIZATION_TEXT,"expected_authorization_text":EXPECTED_AUTHORIZATION_TEXT,
 "authorization_text_exact":AUTHORIZATION_TEXT==EXPECTED_AUTHORIZATION_TEXT,"authorization_recorded_utc":AUTHORIZATION_RECORDED_UTC,
 "one_time_run_id":ONE_TIME_RUN_ID,"run_evaluation":RUN_EVALUATION,"model_loaded":False,"inference_performed":False}
savej(auth,"authorization_state.json")
print("PASS: signed contract, checkpoints, cohort and paths verified")'''))

c.append(nbf.v4.new_markdown_cell("""### 2. Apply the authorization and one-time ledger gate

The default preflight completes without error. When authorization is absent, all later expensive cells print `SKIPPED` and make no model calls.
"""))
c.append(nbf.v4.new_code_cell(r'''LEDGER=OUT/"run_ledger.json"
if RUN_EVALUATION:
 run_uuid=str(uuid.UUID(ONE_TIME_RUN_ID))
 if LEDGER.is_file():
  prior=loadj(LEDGER)
  assert prior.get("status") not in {"metrics_complete","sealed_complete"},"STOP: the one-time external evaluation is already complete; rerun prohibited."
  assert prior.get("run_id")==run_uuid,"STOP: existing ledger belongs to a different run UUID."
  run_mode="resume_after_documented_technical_failure"
 else:
  run_mode="fresh"; savej({"run_id":run_uuid,"status":"started","started_utc":datetime.now(timezone.utc).isoformat(),
   "authorization_text":AUTHORIZATION_TEXT,"authorization_recorded_utc":AUTHORIZATION_RECORDED_UTC,
   "step15_policy_sha256":sha256(S15/"frozen_external_evaluation_policy.json"),"complete_result_produced":False},"run_ledger.json")
 print("AUTHORIZED:",run_uuid,run_mode)
else:
 run_uuid=None; run_mode="safe_preflight"
 status={"created_utc":datetime.now(timezone.utc).isoformat(),"result_level":"EXTERNAL_EVALUATION_AUTHORIZATION_REQUIRED",
  "preflight_passed":bool(preflight.passed.all()),"authorization_granted":False,"expected_authorization_text":EXPECTED_AUTHORIZATION_TEXT,
  "inference_performed":False,"formal_external_evaluation_performed":False,"local_lits_test_accessed":False,
  "next_step":"provide_exact_step16_authorization"}
 savej(status,"gate_result.json")
 preflight_files=["authorization_state.json","preflight_checks.csv","step_15_signature_verification.csv","checkpoint_verification.csv","normalized_path_verification.csv","gate_result.json"]
 signed={n:sha256(OUT/n) for n in preflight_files}; savej({"algorithm":"SHA-256","created_utc":datetime.now(timezone.utc).isoformat(),
  "result_level":status["result_level"],"signed_artifacts":signed,"inference_performed":False,"local_lits_test_accessed":False},"preflight_signature.json")
 print("SAFE STOP: exact authorization has not been supplied; expensive cells will be skipped.")'''))

c.append(nbf.v4.new_markdown_cell("""## Results

### 3. Load frozen models and run one deterministic pass

This cell is expensive only after authorization. It standardizes each native external volume to the frozen 256×256 pipeline, predicts the liver ROI, runs both tumour checkpoints, applies maximum fusion and saves one probability cache per patient.
"""))
c.append(nbf.v4.new_code_cell(r'''if not RUN_EVALUATION:
 print("SKIPPED: model loading and external inference require exact authorization.")
else:
 import torch, nibabel as nib
 from scipy import ndimage
 if str(PROJECT_ROOT) not in sys.path: sys.path.insert(0,str(PROJECT_ROOT))
 from src.framework.models.mobilenetv2_unet import MobileNetV2UNet
 seed=int(policy["execution"]["random_seed"]); np.random.seed(seed); torch.manual_seed(seed)
 if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed); torch.backends.cudnn.benchmark=False; torch.backends.cudnn.deterministic=True
 torch.use_deterministic_algorithms(True,warn_only=True); DEVICE=torch.device("cuda" if torch.cuda.is_available() else "cpu")
 BATCH=int(policy["execution"]["batch_size"]); THRESH=float(policy["prediction"]["global_threshold"])

 def load_model(role,out_channels):
  path=Path(policy["model"][f"{role}_checkpoint"]); payload=torch.load(path,map_location="cpu",weights_only=False)
  state=payload["model_state"] if isinstance(payload,dict) and "model_state" in payload else payload
  model=MobileNetV2UNet(1,out_channels,False); model.load_state_dict(state,strict=True); return model.to(DEVICE).eval()
 def resize_float(a,size=(256,256)):
  return np.asarray(Image.fromarray(np.asarray(a,np.float32),mode="F").resize(size,Image.Resampling.BILINEAR),np.float32)
 def resize_mask(a,size=(256,256)):
  return np.asarray(Image.fromarray(np.asarray(a,np.uint8)*255).resize(size,Image.Resampling.NEAREST),np.uint8)>0
 def standardize(row):
  ct=np.asanyarray(nib.load(str(STEP14/row.ct_relative_path)).dataobj).astype(np.float32)
  liver=np.asanyarray(nib.load(str(STEP14/row.liver_relative_path)).dataobj)>0
  tumour=np.asanyarray(nib.load(str(STEP14/row.tumour_relative_path)).dataobj)>0
  assert ct.shape==liver.shape==tumour.shape and ct.ndim==3 and np.isfinite(ct).all()
  images=[]; organs=[]; truths=[]
  for z in range(ct.shape[2]):
   u8=np.rint((np.clip(ct[:,:,z],-160,240)+160)/400*255).astype(np.uint8)
   images.append(np.asarray(Image.fromarray(u8).resize((256,256),Image.Resampling.BILINEAR),np.uint8).astype(np.float32)/255)
   organs.append(resize_mask(liver[:,:,z])); truths.append(resize_mask(tumour[:,:,z]))
  return np.stack(images),np.stack(organs),np.stack(truths)
 def robust(a):
  ref=a[a>0]; ref=ref if ref.size>=32 else a.ravel(); center=float(np.median(ref)); q1,q3=np.percentile(ref,[25,75]); sigma=float((q3-q1)/1.349)
  if not np.isfinite(sigma) or sigma<1e-3: sigma=max(float(np.std(ref)),1e-3)
  return ((np.clip((a-center)/sigma,-3,3)+3)/6).astype(np.float32)
 def predict_batches(model,arrays):
  result=[]
  with torch.inference_mode():
   for start in range(0,len(arrays),BATCH):
    batch=torch.from_numpy(np.asarray(arrays[start:start+BATCH],np.float32)[:,None]).to(DEVICE)
    result.append(torch.sigmoid(model(batch))[:,0].cpu().numpy())
  return np.concatenate(result)
 def largest3d(mask):
  labels,n=ndimage.label(mask,structure=np.ones((3,3,3),np.uint8))
  if n==0:return mask
  sizes=np.bincount(labels.ravel()); sizes[0]=0; return labels==sizes.argmax()
 def bbox(mask,pad=16):
  if not mask.any(): return (0,256,0,256),True
  _,ys,xs=np.where(mask); return (max(int(ys.min())-pad,0),min(int(ys.max())+1+pad,256),max(int(xs.min())-pad,0),min(int(xs.max())+1+pad,256)),False
 def to_full(score,box):
  y0,y1,x0,x1=box; full=np.zeros((256,256),np.float32); full[y0:y1,x0:x1]=resize_float(score,(x1-x0,y1-y0)); return full

 roi_model=load_model("roi",2); control_model=load_model("control",1); recall_model=load_model("recall",1)
 runtime=[]; integrity=[]; roi_rows=[]
 for sequence,row in enumerate(cohort.sort_values("internal_patient_id").itertuples(),1):
  cache_path=CACHE/f"{row.internal_patient_id}.npz"
  if cache_path.is_file():
   with np.load(cache_path,allow_pickle=False) as z: same=str(z["run_id"])==run_uuid and str(z["policy_sha256"])==sha256(S15/"frozen_external_evaluation_policy.json")
   assert same,"Existing cache does not match the authorized run and policy"; source="resume_cache"; elapsed=0.0
  else:
   started=time.perf_counter(); images,organ,truth=standardize(row)
   roi_prob=predict_batches(roi_model,[robust(x) for x in images]); roi_mask=largest3d(roi_prob>=0.5); box,fallback=bbox(roi_mask,16); y0,y1,x0,x1=box
   crops=np.stack([resize_float(x[y0:y1,x0:x1]) for x in images]); control=predict_batches(control_model,crops); recall=predict_batches(recall_model,crops)
   control_full=np.stack([to_full(x,box) for x in control]); recall_full=np.stack([to_full(x,box) for x in recall]); fused=np.maximum(control_full,recall_full)
   ids=np.asarray([f"{row.internal_patient_id}_z{z:04d}" for z in range(len(images))])
   np.savez_compressed(cache_path,control_probability=control_full.astype(np.float16),recall_probability=recall_full.astype(np.float16),
    fused_probability=fused.astype(np.float16),truth=truth,organ=organ,sample_id=ids,slice_index=np.arange(len(images)),
    patient_id=np.asarray(row.internal_patient_id),run_id=np.asarray(run_uuid),policy_sha256=np.asarray(sha256(S15/"frozen_external_evaluation_policy.json")))
   elapsed=time.perf_counter()-started; source="fresh_inference"
   roi_rows.append({"internal_patient_id":row.internal_patient_id,"slices":len(images),"y0":y0,"y1":y1,"x0":x0,"x1":x1,"fallback_used":fallback,"crop_area_ratio":((y1-y0)*(x1-x0))/(256*256)})
  with np.load(cache_path,allow_pickle=False) as z:
   finite=bool(np.isfinite(z["fused_probability"]).all()); slices=int(len(z["slice_index"])); unique=len(np.unique(z["sample_id"]))==slices
  integrity.append({"internal_patient_id":row.internal_patient_id,"slices":slices,"finite_probability":finite,"sample_ids_unique":unique,"cache_sha256":sha256(cache_path)})
  runtime.append({"internal_patient_id":row.internal_patient_id,"seconds":elapsed,"source":source}); print(f"[{sequence:02d}/20] {row.internal_patient_id}: {source}")
 savec(pd.DataFrame(integrity),"cache_integrity.csv"); savec(pd.DataFrame(runtime),"runtime_log.csv")
 if roi_rows: savec(pd.DataFrame(roi_rows),"external_roi_manifest.csv")
 assert pd.DataFrame(integrity)[["finite_probability","sample_ids_unique"]].all().all()
 del roi_model,control_model,recall_model
 if torch.cuda.is_available(): torch.cuda.empty_cache()
 ledger=loadj(LEDGER); ledger.update({"status":"inference_complete","inference_completed_utc":datetime.now(timezone.utc).isoformat(),"cache_count":20}); savej(ledger,"run_ledger.json")'''))

c.append(nbf.v4.new_markdown_cell("""### 4. Compute all frozen metrics and subgroup evidence
"""))
c.append(nbf.v4.new_code_cell(r'''if not RUN_EVALUATION:
 print("SKIPPED: metrics require authorized probability caches.")
else:
 from scipy import ndimage
 EPS=float(policy["metrics"]["epsilon"]); THRESH=float(policy["prediction"]["global_threshold"]); Q1_MAX=51
 def dice(i,t,p): return (2*i+EPS)/(t+p+EPS)
 patient_rows=[]; slice_rows=[]; lesion_rows=[]; all_ids=[]; global_t=global_p=global_i=global_tp=global_fp=global_fn=0
 positive_empty=[]; empty_fp=[]; q1_detect=[]
 cohort_lookup=cohort.set_index("internal_patient_id")
 for patient_id in sorted(cohort.internal_patient_id):
  path=CACHE/f"{patient_id}.npz"; assert path.is_file()
  with np.load(path,allow_pickle=False) as z:
   prob=z["fused_probability"].astype(np.float32); truth=z["truth"].astype(bool); organ=z["organ"].astype(bool); pred=prob>=THRESH; ids=z["sample_id"].astype(str); indices=z["slice_index"].astype(int)
  ts=truth.sum((1,2)); ps=pred.sum((1,2)); os=(truth&pred).sum((1,2)); pos=ts>0; emp=~pos; detected=(truth&pred).any((1,2)); q1=pos&(ts<=Q1_MAX)
  positive_empty.extend((ps[pos]==0).tolist()); empty_fp.extend((ps[emp]>0).tolist()); q1_detect.extend(detected[q1].tolist())
  tp=int((truth&pred).sum()); fp=int((~truth&pred).sum()); fn=int((truth&~pred).sum()); t=int(truth.sum()); p=int(pred.sum())
  global_t+=t; global_p+=p; global_i+=tp; global_tp+=tp; global_fp+=fp; global_fn+=fn
  meta=cohort_lookup.loc[patient_id]
  patient_rows.append({"internal_patient_id":patient_id,"has_tumour":t>0,"truth_pixels":t,"predicted_pixels":p,"intersection_pixels":tp,
   "false_positive_pixels":fp,"false_negative_pixels":fn,"dice":dice(tp,t,p),"precision":tp/(tp+fp) if tp+fp else np.nan,"recall":tp/(tp+fn) if tp+fn else np.nan,
   "train_derived_burden_stratum":meta.train_derived_burden_stratum,"tumour_volume_ml_external":meta.tumour_volume_ml_external,"spacing_z_mm":meta.spacing_z_mm,"tumour_minus_liver_hu":meta.tumour_minus_liver_hu})
  for k,sid in enumerate(ids):
   slice_rows.append({"internal_patient_id":patient_id,"slice_index":int(indices[k]),"sample_id":sid,"truth_pixels":int(ts[k]),"predicted_pixels":int(ps[k]),
    "intersection_pixels":int(os[k]),"dice":dice(int(os[k]),int(ts[k]),int(ps[k])),"positive_slice":bool(pos[k]),"q1_positive_slice_train_edge":bool(q1[k]),
    "detected":bool(detected[k]),"predicted_empty":bool(ps[k]==0),"empty_slice_false_positive":bool(emp[k] and ps[k]>0),"maximum_score":float(prob[k].max())})
  labels,n=ndimage.label(truth,structure=ndimage.generate_binary_structure(3,1)); pred_labels,npred=ndimage.label(pred,structure=ndimage.generate_binary_structure(3,1)); pred_sizes=np.bincount(pred_labels.ravel(),minlength=npred+1)
  voxel_ml=float(meta.spacing_x_mm*2*meta.spacing_y_mm*2*meta.spacing_z_mm/1000)
  for lid in range(1,n+1):
   comp=labels==lid; truth_size=int(comp.sum()); matched=np.unique(pred_labels[comp]); matched=matched[matched>0]
   best=(0,0,0.0)
   for mid in matched:
    overlap=int((comp&(pred_labels==mid)).sum()); pdice=dice(overlap,truth_size,int(pred_sizes[mid]));
    if pdice>best[2]: best=(int(mid),overlap,pdice)
   volume=truth_size*voxel_ml; edges=policy["metrics"]["subgroups"]["lesion_proxy_edges_ml"]
   stratum=["lesion_q1","lesion_q2","lesion_q3","lesion_q4"][int(np.digitize([volume],edges[1:-1],right=True)[0])]
   lesion_rows.append({"internal_patient_id":patient_id,"truth_component_id_6conn":lid,"truth_pixels":truth_size,"lesion_volume_ml":volume,"train_derived_lesion_stratum":stratum,
    "matched_prediction_id_6conn":best[0],"overlap_pixels":best[1],"detected":best[1]>0,"matched_dice":best[2]})
  all_ids.extend(ids.tolist())
 patients=pd.DataFrame(patient_rows); slices=pd.DataFrame(slice_rows); lesions=pd.DataFrame(lesion_rows); pos_pat=patients[patients.has_tumour]
 metrics={"global_dice":dice(global_i,global_t,global_p),"global_pixel_precision":global_tp/(global_tp+global_fp),"global_pixel_recall":global_tp/(global_tp+global_fn),
  "mean_positive_patient_dice":float(pos_pat.dice.mean()),"median_positive_patient_dice":float(pos_pat.dice.median()),"minimum_positive_patient_dice":float(pos_pat.dice.min()),
  "q1_positive_slice_detection_pct":100*float(np.mean(q1_detect)),"positive_predicted_empty_pct":100*float(np.mean(positive_empty)),"empty_slice_false_positive_pct":100*float(np.mean(empty_fp)),
  "sample_coverage_fraction":len(all_ids)/int(cohort.slices_z.sum()),"unique_patient_id_fraction":patients.internal_patient_id.nunique()/20,
  "finite_probability_fraction":float(pd.read_csv(OUT/"cache_integrity.csv").finite_probability.mean()),"checkpoint_hashes_match":float(checkpoint_check.passed.mean())}
 savec(patients,"patient_metrics.csv"); savec(slices,"slice_metrics.csv"); savec(lesions,"lesion_metrics.csv"); savec(pd.DataFrame([metrics]),"global_metrics.csv")
 patient_subgroups=patients.groupby("train_derived_burden_stratum",dropna=False).agg(patients=("internal_patient_id","size"),mean_dice=("dice","mean"),median_dice=("dice","median"),minimum_dice=("dice","min")).reset_index()
 lesion_subgroups=lesions.groupby("train_derived_lesion_stratum",dropna=False).agg(lesions=("truth_component_id_6conn","size"),detected_pct=("detected",lambda x:100*x.mean()),mean_matched_dice=("matched_dice","mean"),median_volume_ml=("lesion_volume_ml","median")).reset_index()
 negative=patients[~patients.has_tumour][["internal_patient_id","predicted_pixels","false_positive_pixels","dice"]].copy(); negative["any_false_positive"]=negative.predicted_pixels>0
 savec(patient_subgroups,"patient_burden_metrics.csv"); savec(lesion_subgroups,"lesion_stratum_metrics.csv"); savec(negative,"negative_control_metrics.csv")
 rng=np.random.default_rng(int(policy["metrics"]["bootstrap"]["seed"])); values=pos_pat.dice.to_numpy(); draws=rng.choice(values,size=(10000,len(values)),replace=True).mean(1)
 bootstrap=pd.DataFrame([{"metric":"mean_positive_patient_dice","estimate":values.mean(),"ci_lower_2_5":np.quantile(draws,.025),"ci_upper_97_5":np.quantile(draws,.975),"iterations":10000,"seed":42,"unit":"tumour-positive patient"}]); savec(bootstrap,"bootstrap_uncertainty.csv")
 print(pd.DataFrame([metrics]).to_string(index=False))'''))

c.append(nbf.v4.new_markdown_cell("""### 5. Apply the predeclared gate, create figures and seal the one-time result
"""))
c.append(nbf.v4.new_code_cell(r'''if not RUN_EVALUATION:
 print("SKIPPED: final gate and result sealing require an authorized completed run.")
else:
 def compare(v,op,t): return bool(v>=t if op==">=" else v<=t if op=="<=" else np.isclose(v,t))
 actuals={**metrics,"all_20_patient_rows_saved":float(len(patients)==20),"burden_and_negative_subgroups_saved":float(len(patient_subgroups)>0 and len(negative)==5)}
 evaluation=acceptance.copy(); evaluation["actual"]=evaluation.metric.map(actuals); evaluation["passed"]=[compare(v,o,t) for v,o,t in zip(evaluation.actual,evaluation.operator,evaluation.threshold)]
 savec(evaluation,"expected_vs_actual.csv"); all_pass=bool(evaluation.loc[evaluation.mandatory.astype(bool),"passed"].all())
 fig,axes=plt.subplots(2,2,figsize=(14,10))
 axes[0,0].bar(["Global","Mean patient","Precision","Recall"],[metrics["global_dice"],metrics["mean_positive_patient_dice"],metrics["global_pixel_precision"],metrics["global_pixel_recall"]],color="#4472c4"); axes[0,0].set_ylim(0,1); axes[0,0].set_title("Frozen external metrics")
 axes[0,1].bar(["Q1 detect","Positive empty","Empty FP"],[metrics["q1_positive_slice_detection_pct"],metrics["positive_predicted_empty_pct"],metrics["empty_slice_false_positive_pct"]],color=["#2ca02c","#d62728","#ff7f0e"]); axes[0,1].set_ylim(0,100); axes[0,1].set_title("Slice rates (%)")
 axes[1,0].bar(patient_subgroups.train_derived_burden_stratum.astype(str),patient_subgroups.mean_dice,color="#756bb1"); axes[1,0].tick_params(axis="x",rotation=25); axes[1,0].set_ylim(0,1); axes[1,0].set_title("Patient Dice by train-derived burden")
 axes[1,1].barh(evaluation.metric,evaluation.passed.astype(int),color=np.where(evaluation.passed,"#2ca02c","#d62728")); axes[1,1].set_xlim(0,1.05); axes[1,1].set_title("Predeclared gate")
 fig.tight_layout(); fig.savefig(OUT/"external_evaluation_dashboard.png",dpi=170,bbox_inches="tight"); plt.close(fig)
 failures=patients.sort_values("dice").head(5); savec(failures,"failure_cases.csv")
 evidence_names=["patient_metrics.csv","slice_metrics.csv","lesion_metrics.csv","global_metrics.csv","patient_burden_metrics.csv","lesion_stratum_metrics.csv","negative_control_metrics.csv","bootstrap_uncertainty.csv","expected_vs_actual.csv","cache_integrity.csv","runtime_log.csv","external_evaluation_dashboard.png"]
 inventory=pd.DataFrame([{"path":n,"bytes":(OUT/n).stat().st_size,"sha256":sha256(OUT/n)} for n in evidence_names]+[{"path":str(p.relative_to(OUT)),"bytes":p.stat().st_size,"sha256":sha256(p)} for p in sorted(CACHE.glob("*.npz"))]); savec(inventory,"external_evidence_inventory.csv")
 gate={"created_utc":datetime.now(timezone.utc).isoformat(),"result_level":"EXTERNAL_EVALUATION_COMPLETE","all_mandatory_targets_passed":all_pass,
  "decision":"REPORT_EXTERNAL_GENERALIZATION_PASS_NO_TUNING" if all_pass else "REPORT_EXTERNAL_GENERALIZATION_FAILURE_NO_TUNING_NO_RERUN",
  "selected_metrics":metrics,"target_passes":dict(zip(evaluation.metric,evaluation.passed.astype(bool))),"cohort":{"all":20,"positive":15,"negative":5,"excluded":0},
  "run_id":run_uuid,"inference_performed":True,"formal_external_evaluation_performed":True,"local_lits_test_accessed":False,"rerun_permitted":False}
 savej(gate,"gate_result.json")
 auth=loadj(OUT/"authorization_state.json"); auth.update({"model_loaded":True,"inference_performed":True,"formal_external_evaluation_performed":True,"completed_utc":gate["created_utc"]}); savej(auth,"authorization_state.json")
 ledger=loadj(LEDGER); ledger.update({"status":"sealed_complete","metrics_completed_utc":gate["created_utc"],"sealed_utc":datetime.now(timezone.utc).isoformat(),"complete_result_produced":True,"all_mandatory_targets_passed":all_pass}); savej(ledger,"run_ledger.json")
 sig_names=evidence_names+["external_evidence_inventory.csv","gate_result.json","run_ledger.json","authorization_state.json","preflight_checks.csv","checkpoint_verification.csv"]
 signed={n:sha256(OUT/n) for n in sig_names}; combined=hashlib.sha256("\n".join(f"{k}:{v}" for k,v in sorted(signed.items())).encode()).hexdigest()
 savej({"algorithm":"SHA-256","created_utc":datetime.now(timezone.utc).isoformat(),"combined_sha256":combined,"run_id":run_uuid,"signed_artifacts":signed,
  "result_level":"EXTERNAL_EVALUATION_COMPLETE","inference_performed":True,"local_lits_test_accessed":False},"step_16_signature.json")
 print(json.dumps(gate,indent=2))'''))

c.append(nbf.v4.new_markdown_cell("""## Takeaways

- Before authorization, a successful Run All is a signed preflight only.
- After authorization, the first complete result is final for this external cohort and policy, whether it passes or fails.
- Step 16 never opens the local LiTS test split and external results cannot retroactively change the completed LiTS acceptance record.
"""))
nb["cells"]=c
for p in (LONG,SHORT): nbf.write(nb,p)
print("Wrote",LONG); print("Wrote",SHORT)
