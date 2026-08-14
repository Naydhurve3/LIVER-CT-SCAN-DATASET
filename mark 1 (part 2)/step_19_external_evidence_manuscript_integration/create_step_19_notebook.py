from pathlib import Path
import nbformat as nbf

STEP_DIR=Path(__file__).resolve().parent
LONG=STEP_DIR/"step_19_external_evidence_manuscript_integration.ipynb"
SHORT=STEP_DIR/"step_19.ipynb"
nb=nbf.v4.new_notebook()
nb["metadata"]={"kernelspec":{"display_name":"Python 3 (ds_gpu)","language":"python","name":"python3"},"language_info":{"name":"python","version":"3.11"}}
c=[]
c.append(nbf.v4.new_markdown_cell("""# Step 19 — External-evidence manuscript integration

## tl;dr

This read-only notebook integrates the sealed LiTS held-out result and the independently validated 3D-IRCADb-01 external result into a paper-ready evidence update. It creates comparative tables, patient-distribution figures, claim boundaries, a revised manuscript draft, and a signed readiness gate.

It does not load a model, perform inference, reopen test source files, tune thresholds, revise truth, or submit a manuscript.
"""))
c.append(nbf.v4.new_markdown_cell("""## Context & Methods

### Key Assumptions

- The LiTS held-out evaluation remains a completed negative formal gate because minimum positive-patient Dice was 0.
- The 3D-IRCADb-01 external evaluation passed its separately frozen contract, but has patient-level, small-lesion, negative-control, and annotation-semantic caveats.
- Cohorts differ in acquisition, labels, tumour prevalence and denominators; descriptive differences are not causal or a head-to-head statistical comparison.
- Existing sealed summary tables may be read, but no LiTS test image, mask, probability cache, statistic recomputation from source data, or loader is opened.
"""))
c.append(nbf.v4.new_code_cell(r'''from pathlib import Path
from datetime import datetime, timezone
import hashlib,json,platform,sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

STEP_DIR=Path.cwd().resolve(); assert STEP_DIR.name=="step_19_external_evidence_manuscript_integration"
PART2=STEP_DIR.parent; OUT=STEP_DIR/"outputs"; OUT.mkdir(exist_ok=True)
S5=PART2/"step_05_final_research_package"/"outputs"
S7=PART2/"step_07_manuscript_evidence_and_declaration_scaffold"/"outputs"
S11=PART2/"step_11_owner_submission_gate_and_finalization_handoff"/"outputs"
S16=PART2/"step_16_one_time_external_evaluation_after_explicit_authorization"/"outputs"
S17=PART2/"step_17_external_evaluation_evidence_validation_and_data_card"/"outputs"
S18=PART2/"step_18_source_label_concordance_and_failure_adjudication"/"outputs"
MANIFEST_SHA="575a6fc391d63dc9bbbbb3317efe4d0f65edd57457ae63c1bfc6c2c615861889"
def sha256(p,chunk=1024*1024):
 h=hashlib.sha256()
 with Path(p).open("rb") as f:
  for b in iter(lambda:f.read(chunk),b""): h.update(b)
 return h.hexdigest()
def loadj(p): return json.loads(Path(p).read_text(encoding="utf-8"))
def savej(o,n): (OUT/n).write_text(json.dumps(o,indent=2,sort_keys=True,default=str),encoding="utf-8")
def savec(d,n): d.to_csv(OUT/n,index=False)
print("Manuscript integration only; inference=False; test source reopened=False; submission=False")'''))
c.append(nbf.v4.new_markdown_cell("""## Data

### 1. Verify sealed evidence and submission state
"""))
c.append(nbf.v4.new_code_cell(r'''g5=loadj(S5/"gate_result.json"); g11=loadj(S11/"gate_result.json"); g16=loadj(S16/"gate_result.json"); g17=loadj(S17/"gate_result.json"); g18=loadj(S18/"gate_result.json")
def verify_signed(sig_path,base):
 sig=loadj(sig_path); rows=[]
 for rel,expected in sig.get("signed_artifacts",{}).items():
  p=base/rel; actual=sha256(p) if p.is_file() else None
  rows.append({"signature":sig_path.name,"artifact":rel,"exists":p.is_file(),"expected_sha256":expected,"actual_sha256":actual,"passed":actual==expected})
 return pd.DataFrame(rows)
signed=pd.concat([verify_signed(S7/"step_07_signature.json",PART2),verify_signed(S17/"step_17_signature.json",PART2),verify_signed(S18/"step_18_signature.json",PART2)],ignore_index=True)
savec(signed,"input_signature_verification.csv")
inventory=pd.read_csv(S5/"ARTIFACT_CHECKSUM_INVENTORY.csv")
wanted={"final_metrics_summary.csv","patient_results_table.csv","PAPER_DRAFT.md","gate_result.json"}; inv=inventory[inventory.path.map(lambda x:Path(x).name in wanted)].copy()
inv["exists"]=inv.path.map(lambda x:Path(x).is_file()); inv["actual_sha256"]=inv.path.map(lambda x:sha256(x) if Path(x).is_file() else None); inv["passed"]=inv.sha256.eq(inv.actual_sha256)
savec(inv,"step05_evidence_verification.csv")
checks=pd.DataFrame([
 {"check":"manifest_identity","passed":g5.get("manifest_sha256")==MANIFEST_SHA},
 {"check":"lits_final_project_complete","passed":g5.get("result_level")=="FINAL_PROJECT_COMPLETE"},
 {"check":"lits_formal_failure_preserved","passed":g5.get("formal_model_acceptance_passed") is False},
 {"check":"external_evaluation_complete","passed":g16.get("result_level")=="EXTERNAL_EVALUATION_COMPLETE"},
 {"check":"external_evidence_validated","passed":g17.get("result_level")=="EXTERNAL_GENERALIZATION_EVIDENCE_VALIDATED_WITH_CAVEATS"},
 {"check":"source_concordance_complete","passed":g18.get("gate_passed") is True},
 {"check":"all_signed_inputs_match","passed":bool(signed.passed.all())},
 {"check":"step05_key_evidence_matches_inventory","passed":len(inv)>=4 and bool(inv.passed.all())},
 {"check":"owner_submission_gate_remains_closed","passed":g11.get("submission_ready") is False and g11.get("owner_gate_passed") is False},
])
savec(checks,"input_verification.csv"); assert checks.passed.all(),checks.loc[~checks.passed].to_dict("records")
print("PASS: sealed result packages verified; owner/submission gate remains closed")'''))
c.append(nbf.v4.new_markdown_cell("""## Results

### 2. Reconcile comparable metrics and cohort definitions
"""))
c.append(nbf.v4.new_code_cell(r'''test_raw=pd.read_csv(S5/"final_metrics_summary.csv").set_index("metric")["value"]
external=pd.read_csv(S16/"global_metrics.csv").iloc[0]; boot=pd.read_csv(S16/"bootstrap_uncertainty.csv").iloc[0]
test_pat=pd.read_csv(S5/"patient_results_table.csv"); ext_pat=pd.read_csv(S16/"patient_metrics.csv")
test_pos=test_pat[test_pat.has_tumour.astype(bool)].copy(); ext_pos=ext_pat[ext_pat.has_tumour.astype(bool)].copy()
cohorts=pd.DataFrame([
 {"evaluation":"LiTS held-out","patients":14,"positive_patients":13,"negative_controls":1,"slices":7286,"truth_semantics":"LiTS hepatic tumour label","formal_gate":"FAIL: minimum positive-patient Dice","role":"one-time internal held-out evaluation"},
 {"evaluation":"3D-IRCADb-01 external","patients":20,"positive_patients":15,"negative_controls":5,"slices":2823,"truth_semantics":"folders matching ^livertumors?\\d*$","formal_gate":"PASS under separately frozen external contract","role":"one-time external evaluation"},
])
savec(cohorts,"evaluation_cohort_summary.csv")
metric_map=[("global_dice","Global Dice"),("global_pixel_precision","Pixel precision"),("global_pixel_recall","Pixel recall"),("mean_positive_patient_dice","Mean positive-patient Dice"),("median_positive_patient_dice","Median positive-patient Dice"),("minimum_positive_patient_dice","Minimum positive-patient Dice"),("q1_positive_slice_detection_pct_train_edges","Q1 positive-slice detection (%)"),("positive_predicted_empty_pct","Positive predicted-empty (%)"),("empty_slice_false_positive_pct","Empty-slice false-positive (%)")]
rows=[]
for key,label in metric_map:
 ext_key=key.replace("_train_edges","")
 rows.append({"metric_key":key,"metric":label,"lits_heldout":float(test_raw[key]),"ircadb_external":float(external[ext_key]),"external_minus_lits":float(external[ext_key])-float(test_raw[key]),"comparison_type":"descriptive_only_different_cohorts"})
perf=pd.DataFrame(rows); savec(perf,"evaluation_performance_comparison.csv")
patient_distribution=pd.concat([pd.DataFrame({"evaluation":"LiTS held-out","patient_id":"V"+test_pos.volume_id.astype(str),"dice":test_pos.dice.astype(float)}),pd.DataFrame({"evaluation":"3D-IRCADb-01 external","patient_id":ext_pos.internal_patient_id,"dice":ext_pos.dice.astype(float)})],ignore_index=True)
savec(patient_distribution,"patient_dice_distribution.csv")
assert np.isclose(test_pos.dice.mean(),test_raw["mean_positive_patient_dice"],atol=1e-8)
assert np.isclose(ext_pos.dice.mean(),external.mean_positive_patient_dice,atol=1e-12)
display(perf)'''))
c.append(nbf.v4.new_markdown_cell("""### 3. Preserve failure evidence and annotation caveats
"""))
c.append(nbf.v4.new_code_cell(r'''lesion=pd.read_csv(S16/"lesion_stratum_metrics.csv"); neg=pd.read_csv(S17/"negative_control_physical_volume.csv"); adjud=pd.read_csv(S18/"case_adjudication_summary.csv")
failures=pd.DataFrame([
 {"evaluation":"LiTS held-out","failure":"catastrophic positive-patient miss","evidence":"V121 Dice 0.000000","interpretation":"failed predeclared minimum-patient guardrail","claim_boundary":"formal model acceptance remains failed"},
 {"evaluation":"3D-IRCADb-01 external","failure":"worst positive-patient performance","evidence":f"ircadb_18 Dice {external.minimum_positive_patient_dice:.6f}","interpretation":"passes 0.01 floor narrowly","claim_boundary":"external pass does not establish universal reliability"},
 {"evaluation":"3D-IRCADb-01 external","failure":"smallest-lesion detection","evidence":f"{int(lesion.iloc[0].detected_pct*lesion.iloc[0].lesions/100)}/{int(lesion.iloc[0].lesions)} ({lesion.iloc[0].detected_pct:.2f}%)","interpretation":"small lesions remain the shared weakness","claim_boundary":"no clinical sensitivity claim"},
 {"evaluation":"3D-IRCADb-01 external","failure":"negative-control predictions","evidence":f"5/5 cases; {neg.predicted_false_positive_ml.sum():.3f} ml total","interpretation":"false positive under frozen hepatic truth","claim_boundary":"biological meaning unresolved"},
 {"evaluation":"3D-IRCADb-01 external","failure":"source-label semantic discordance","evidence":"case 7: 86.39% generic tumor overlap; case 14: 88.93% metastasectomie overlap","interpretation":"excluded annotations explain much of two large negative-control predictions","claim_boundary":"no retrospective relabeling without expert review"},
])
savec(failures,"failure_mode_summary.csv")
claims=pd.DataFrame([
 {"claim_id":"C1","proposed_claim":"The frozen pipeline achieved LiTS global Dice 0.7677.","status":"supported","evidence":"Step 5 sealed held-out result","required_caveat":"formal acceptance failed because minimum patient Dice was 0"},
 {"claim_id":"C2","proposed_claim":"The pipeline achieved external global Dice 0.8480 and mean positive-patient Dice 0.7112.","status":"supported","evidence":"Steps 16-17 sealed and independently reconciled","required_caveat":"single public external cohort; 15 positive patients"},
 {"claim_id":"C3","proposed_claim":"The model generalized externally under the frozen 3D-IRCADb-01 contract.","status":"supported_with_caveats","evidence":"11/11 external gate rows passed","required_caveat":"does not reverse LiTS formal failure or establish clinical readiness"},
 {"claim_id":"C4","proposed_claim":"Small lesions remain a reproducible weakness.","status":"supported","evidence":"LiTS Q1 lesion detection 60.3%; external Q1 63.64%","required_caveat":"strata use train-derived size edges but datasets differ"},
 {"claim_id":"C5","proposed_claim":"The model is clinically ready.","status":"prohibited","evidence":"patient failures, false positives, no prospective/expert review","required_caveat":"not supportable"},
 {"claim_id":"C6","proposed_claim":"Case 7 and 14 external predictions are true tumours.","status":"prohibited","evidence":"source-label concordance is not expert adjudication","required_caveat":"retain frozen false-positive classification"},
 {"claim_id":"C7","proposed_claim":"The external dataset outperformed LiTS.","status":"prohibited_as_inferential_claim","evidence":"descriptive metric differences only","required_caveat":"cohorts and label semantics are not directly exchangeable"},
])
savec(claims,"manuscript_claim_matrix.csv"); display(failures); display(claims)'''))
c.append(nbf.v4.new_markdown_cell("""### 4. Create paper-ready visual evidence

Chart contract: static notebook figures; horizontal metric comparison and patient-level box/strip distribution; zero-based scales; two-root palette with marker/shape distinctions; exact cohorts and denominators in captions; final QA in exported PNGs.
"""))
c.append(nbf.v4.new_code_cell(r'''blue="#3366A3"; orange="#D17A22"; ink="#222222"; grid="#D9D9D9"
plot_keys=["global_dice","mean_positive_patient_dice","median_positive_patient_dice","minimum_positive_patient_dice"]
p=perf[perf.metric_key.isin(plot_keys)].copy(); y=np.arange(len(p)); w=.36
fig,ax=plt.subplots(figsize=(10,5.6)); ax.barh(y+w/2,p.lits_heldout,w,label="LiTS held-out (13 positive patients)",color=blue); ax.barh(y-w/2,p.ircadb_external,w,label="3D-IRCADb-01 (15 positive patients)",color=orange)
ax.set_yticks(y,p.metric); ax.set_xlim(0,1); ax.set_xlabel("Dice"); ax.set_title("Segmentation overlap metrics by evaluation cohort"); ax.grid(axis="x",color=grid,alpha=.7); ax.legend(frameon=False,loc="upper center",bbox_to_anchor=(.5,.98),ncol=2)
for i,(a,b) in enumerate(zip(p.lits_heldout,p.ircadb_external)): ax.text(a+.01,i+w/2,f"{a:.3f}",va="center",fontsize=9); ax.text(b+.01,i-w/2,f"{b:.3f}",va="center",fontsize=9)
fig.text(.01,.01,"Descriptive comparison only: cohorts and label semantics differ. Source: sealed Steps 5 and 16-18.",fontsize=9,color=ink); fig.tight_layout(rect=(0,.05,1,1)); fig.savefig(OUT/"comparative_results_figure.png",dpi=180,bbox_inches="tight"); plt.close(fig)

groups=[patient_distribution.loc[patient_distribution.evaluation.eq(e),"dice"].values for e in ["LiTS held-out","3D-IRCADb-01 external"]]
fig,ax=plt.subplots(figsize=(8.5,5.6)); bp=ax.boxplot(groups,tick_labels=["LiTS held-out\n(n=13)","3D-IRCADb-01 external\n(n=15)"],patch_artist=True,showmeans=True,showfliers=False,meanprops={"marker":"D","markerfacecolor":"white","markeredgecolor":ink})
for patch,color in zip(bp["boxes"],[blue,orange]): patch.set_facecolor(color); patch.set_alpha(.55)
rng=np.random.default_rng(42)
for i,vals in enumerate(groups,1): ax.scatter(i+rng.uniform(-.08,.08,len(vals)),vals,s=30,facecolors="white",edgecolors=[blue,orange][i-1],linewidths=1.2,zorder=3)
ax.set_ylim(0,1); ax.set_ylabel("Positive-patient Dice"); ax.set_title("Patient-level performance remains heterogeneous in both cohorts"); ax.grid(axis="y",color=grid,alpha=.7)
fig.text(.01,.01,"Each point is one tumour-positive patient; diamonds mark means. Source: sealed summary tables only.",fontsize=9,color=ink); fig.tight_layout(rect=(0,.05,1,1)); fig.savefig(OUT/"patient_dice_distribution_figure.png",dpi=180,bbox_inches="tight"); plt.close(fig)
chart_map=pd.DataFrame([
 {"section":"Primary comparative results","question":"How do the two sealed evaluations compare descriptively?","family":"comparison","chart":"grouped horizontal bar","fields":"evaluation, metric, value","claim":"external evidence is strong but does not erase held-out failure","palette":"blue/orange plus labels","artifact":"comparative_results_figure.png"},
 {"section":"Patient reliability","question":"How heterogeneous is positive-patient Dice?","family":"distribution","chart":"box plus jittered points","fields":"evaluation, patient_id, dice","claim":"case-level variation persists in both cohorts","palette":"blue/orange plus shape","artifact":"patient_dice_distribution_figure.png"},
]); savec(chart_map,"chart_map.csv")'''))
c.append(nbf.v4.new_markdown_cell("""### 5. Generate the revised technical manuscript package
"""))
c.append(nbf.v4.new_code_cell(r'''ext_ci=f"{boot.ci_lower_2_5:.4f}–{boot.ci_upper_97_5:.4f}"
draft=f"""# Maximum-Probability Fusion of Two MobileNetV2 U-Nets for Liver-Tumour Segmentation: Frozen Held-Out and External Evaluation

## Abstract

**Background:** Liver-tumour segmentation is vulnerable to small-lesion and patient-level failure. **Methods:** A predicted-liver ROI preceded two frozen MobileNetV2 U-Net tumour models. Pixelwise maximum fusion, threshold 0.70, and no post-processing were fixed before one-time evaluations on a 14-volume LiTS held-out cohort and the complete 20-case 3D-IRCADb-01 cohort. **Results:** LiTS global Dice was {test_raw['global_dice']:.4f}; mean positive-patient Dice was {test_raw['mean_positive_patient_dice']:.4f}, but minimum Dice was 0, failing the predeclared catastrophic-patient guardrail. External global Dice was {external.global_dice:.4f}; mean positive-patient Dice was {external.mean_positive_patient_dice:.4f} (95% bootstrap CI {ext_ci}), median {external.median_positive_patient_dice:.4f}, and minimum {external.minimum_positive_patient_dice:.4f}. External smallest-lesion detection was {lesion.iloc[0].detected_pct:.2f}%. All five external negative controls had predictions under the frozen hepatic-tumour truth; source-label concordance explained much of the two largest cases but did not change truth or metrics. **Conclusion:** The frozen method showed external generalization evidence and strong aggregate overlap, while persistent case-level and small-lesion failures preclude clinical-readiness claims. The LiTS formal acceptance result remains failed.

## 1. Introduction

This study evaluates whether a validation-selected maximum-probability fusion policy transfers under predeclared one-time held-out and external evaluations, while explicitly preserving patient-level failure evidence.

## 2. Materials and methods

### 2.1 Governance and cohorts

The corrected LiTS build used manifest SHA-256 `{MANIFEST_SHA}` with patient-disjoint splits. The one-time LiTS held-out evaluation included 14 volumes (13 tumour-positive). A separately frozen external contract included all 20 3D-IRCADb-01 cases (15 accepted hepatic-tumour positive and five negative controls), with zero exclusions. Neither evaluation was rerun or tuned after results were observed.

### 2.2 Pipeline and outcomes

The frozen pipeline used broad-window `[-160,240]` inputs at 256×256, a predicted-liver ROI, control and recall-loss tumour checkpoints, `max(p_control,p_recall)` fusion, threshold 0.70 and no post-processing. Outcomes included global overlap, patient Dice, train-derived small-lesion strata, empty predictions, false-positive behavior and 10,000-resample patient bootstrap uncertainty.

### 2.3 External label semantics

External hepatic truth used source folders matching `^livertumors?\\d*$`. Generic `tumor`, adrenal/surrenal tumour and `metastasectomie` annotations were excluded by the frozen contract. A later read-only concordance audit measured overlap with those annotations without relabeling cases.

## 3. Results

### 3.1 LiTS held-out evaluation failed its patient guardrail

LiTS global Dice was {test_raw['global_dice']:.4f}, precision {test_raw['global_pixel_precision']:.4f}, recall {test_raw['global_pixel_recall']:.4f}, and mean positive-patient Dice {test_raw['mean_positive_patient_dice']:.4f}. Minimum positive-patient Dice was 0 (V121), so formal model acceptance failed despite strong aggregate performance.

### 3.2 External evaluation passed its separate frozen contract with caveats

External global Dice was {external.global_dice:.4f}, precision {external.global_pixel_precision:.4f}, recall {external.global_pixel_recall:.4f}, and mean positive-patient Dice {external.mean_positive_patient_dice:.4f} (95% CI {ext_ci}). Median Dice was {external.median_positive_patient_dice:.4f}; minimum was {external.minimum_positive_patient_dice:.4f} for `ircadb_18`. Smallest-lesion detection was {lesion.iloc[0].detected_pct:.2f}% ({int(round(lesion.iloc[0].detected_pct*lesion.iloc[0].lesions/100))}/{int(lesion.iloc[0].lesions)}).

### 3.3 Negative-control predictions expose label and specificity limits

All five external negative controls contained predictions totaling {neg.predicted_false_positive_ml.sum():.3f} ml on the standardized grid. In cases 7 and 14, 86.39% and 88.93% of predicted pixels overlapped excluded `tumor` and `metastasectomie` masks. These remain false positives under frozen truth; biological interpretation requires expert review.

## 4. Discussion

The external pass supplies evidence that aggregate performance transfers beyond the corrected LiTS build, but it does not reverse the failed LiTS patient guardrail. Across both cohorts, small lesions and individual cases remain the dominant limitations. Descriptive differences between cohorts must not be interpreted as external superiority because acquisition, prevalence and label semantics differ.

## 5. Limitations

Both cohorts are small public datasets. The method is two-dimensional, probabilities are not clinically calibrated, negative-control source labels are not expert adjudication, and no prospective or institutional external study was performed. Owner declarations, exact 3D-IRCADb-01 scholarly citation, venue formatting, and submission authorization remain unresolved.

## 6. Conclusion

The frozen pipeline achieved strong aggregate overlap and passed a separately frozen public external evaluation, but failed the LiTS catastrophic-patient acceptance guardrail. The defensible conclusion is external generalization evidence with material reliability caveats—not clinical readiness.

## Submission blockers

The Step 11 owner/declaration gate remains closed. No submission or payment action is authorized.
"""
(OUT/"PAPER_DRAFT_EXTERNAL_VALIDATION_UPDATE.md").write_text(draft,encoding="utf-8")
limitations="""# Updated limitations and claim boundaries

1. The LiTS formal gate failed because one positive patient had Dice 0; external success cannot override this predeclared result.
2. 3D-IRCADb-01 has only 20 cases and uses source-folder label semantics that differ from LiTS.
3. Five negative controls all had predictions; source-label overlap is descriptive, not expert biological adjudication.
4. Smallest-lesion detection remained near 60% in both evaluations.
5. No prospective, institutional, clinical-workflow, reader, or calibration study was performed.
6. Cross-cohort metric differences are descriptive and not inferential evidence of superiority.
7. Clinical readiness, universal generalization, state-of-the-art performance, and zero false-positive risk are prohibited claims.
"""
(OUT/"UPDATED_LIMITATIONS_AND_CLAIM_BOUNDARIES.md").write_text(limitations,encoding="utf-8")
report=f"""# External-evidence technical report

## Technical summary

The external evaluation strengthens the evidence base but does not change the project's failed formal LiTS acceptance result. External global Dice was {external.global_dice:.4f}; mean positive-patient Dice was {external.mean_positive_patient_dice:.4f} (95% CI {ext_ci}). Patient-level, smallest-lesion and negative-control failures remain material.

## Key findings

- LiTS: global Dice {test_raw['global_dice']:.4f}; minimum positive-patient Dice 0; formal gate failed.
- 3D-IRCADb-01: global Dice {external.global_dice:.4f}; minimum {external.minimum_positive_patient_dice:.4f}; frozen external gate passed.
- Smallest-lesion detection: 60.3% LiTS and {lesion.iloc[0].detected_pct:.2f}% external.
- External negative controls: 5/5 with predictions, totaling {neg.predicted_false_positive_ml.sum():.3f} ml.

## Scope and definitions

Results come only from sealed summary artifacts. Cohorts, labels and denominators differ. Dice comparisons are descriptive.

## Robustness and limitations

Input signatures, key evidence hashes and headline means were independently rechecked. The analysis performs no new inference or source-data access. Clinical and inferential claims remain unsupported.

## Recommended next step

Complete owner declarations and verify the exact external-dataset citation if manuscript submission remains desired. Expert review of source-label discordance is optional but required before biological reinterpretation.

## Further questions

- Can an independent institutional cohort reproduce the result?
- Can a separately predeclared method improve small-lesion sensitivity without increasing negative-control errors?
"""
(OUT/"EXTERNAL_EVIDENCE_TECHNICAL_REPORT.md").write_text(report,encoding="utf-8")'''))
c.append(nbf.v4.new_markdown_cell("""## Takeaways

### 6. Freeze manuscript readiness and sign the package
"""))
c.append(nbf.v4.new_code_cell(r'''expected=pd.DataFrame([
 {"requirement":"sealed_inputs_verified","expected":True,"actual":bool(checks.passed.all()),"passed":bool(checks.passed.all())},
 {"requirement":"headline_means_reconciled","expected":True,"actual":True,"passed":True},
 {"requirement":"formal_lits_failure_preserved","expected":False,"actual":g5["formal_model_acceptance_passed"],"passed":g5["formal_model_acceptance_passed"] is False},
 {"requirement":"external_caveats_preserved","expected":True,"actual":len(failures)>=5,"passed":len(failures)>=5},
 {"requirement":"owner_submission_gate_closed","expected":False,"actual":g11["submission_ready"],"passed":g11["submission_ready"] is False},
 {"requirement":"no_inference_test_reopen_or_submission","expected":True,"actual":True,"passed":True},
])
savec(expected,"expected_vs_actual.csv"); assert expected.passed.all()
result="MANUSCRIPT_EXTERNAL_EVIDENCE_UPDATED_OWNER_INPUT_REQUIRED"
gate={"created_utc":datetime.now(timezone.utc).isoformat(),"result_level":result,"gate_passed":True,"formal_model_acceptance_passed":False,"external_contract_passed":True,
 "submission_ready":False,"owner_gate_passed":False,"manuscript_update_ready":True,"exact_external_dataset_citation_verified":False,
 "inference_performed":False,"test_source_files_reopened":False,"test_inference_rerun":False,"external_inference_rerun":False,"tuning_performed":False,"submission_action_performed":False,
 "next_step":"owner completes declarations and verifies exact 3D-IRCADb-01 scholarly citation; optional expert annotation review"}
savej(gate,"gate_result.json")
savej({"phase":STEP_DIR.name,"created_utc":gate["created_utc"],"python":sys.version,"platform":platform.platform(),"manifest_sha256":MANIFEST_SHA,
 "step05_gate_sha256":sha256(S5/"gate_result.json"),"step07_signature_sha256":sha256(S7/"step_07_signature.json"),"step16_signature_sha256":sha256(S16/"step_16_signature.json"),"step17_signature_sha256":sha256(S17/"step_17_signature.json"),"step18_signature_sha256":sha256(S18/"step_18_signature.json"),
 "inference_performed":False,"test_source_files_reopened":False,"submission_action_performed":False},"provenance.json")
names=["input_verification.csv","input_signature_verification.csv","step05_evidence_verification.csv","evaluation_cohort_summary.csv","evaluation_performance_comparison.csv","patient_dice_distribution.csv","failure_mode_summary.csv","manuscript_claim_matrix.csv","chart_map.csv","comparative_results_figure.png","patient_dice_distribution_figure.png","PAPER_DRAFT_EXTERNAL_VALIDATION_UPDATE.md","EXTERNAL_EVIDENCE_TECHNICAL_REPORT.md","UPDATED_LIMITATIONS_AND_CLAIM_BOUNDARIES.md","expected_vs_actual.csv","gate_result.json","provenance.json"]
signed={str((OUT/n).relative_to(PART2)):sha256(OUT/n) for n in names}; combined=hashlib.sha256("\n".join(f"{k}:{v}" for k,v in sorted(signed.items())).encode()).hexdigest()
savej({"algorithm":"SHA-256","created_utc":datetime.now(timezone.utc).isoformat(),"combined_sha256":combined,"result_level":result,"signed_artifacts":signed,"test_source_files_reopened":False,"submission_action_performed":False},"step_19_signature.json")
print(json.dumps(gate,indent=2))'''))
c.append(nbf.v4.new_markdown_cell("""The research evidence is now manuscript-integrated. The formal LiTS failure and external caveats remain explicit. The next action is administrative and scholarly: complete owner declarations and verify the exact 3D-IRCADb-01 citation; do not perform further result-driven inference or tuning."""))
nb["cells"]=c
for p in (LONG,SHORT): nbf.write(nb,p)
print("Wrote",LONG); print("Wrote",SHORT)
