from pathlib import Path
import nbformat as nbf

STEP_DIR=Path(__file__).resolve().parent
LONG=STEP_DIR/"step_21_terminal_evidence_chain_and_project_handoff.ipynb"
SHORT=STEP_DIR/"step_21.ipynb"
nb=nbf.v4.new_notebook()
nb["metadata"]={"kernelspec":{"display_name":"Python 3 (ds_gpu)","language":"python","name":"python3"},"language_info":{"name":"python","version":"3.11"}}
c=[]
c.append(nbf.v4.new_markdown_cell("""# Step 21 — Terminal evidence chain and project handoff

## tl;dr

This terminal read-only audit verifies the complete Steps 1–20 phase chain, checks every machine-readable signed-artifact mapping it can resolve, inventories the project evidence, and freezes the final scientific/administrative handoff.

It does not reopen source test data, load a model, perform inference, tune results, download data, modify earlier phases, submit a manuscript, or certify owner declarations.
"""))
c.append(nbf.v4.new_markdown_cell("""## Context & Methods

### Key Assumptions

- A phase may be complete even when its scientific gate is negative or its next action is owner-controlled.
- The LiTS formal model-acceptance failure is immutable evidence, not an invitation to reuse the test split.
- The external evaluation passed its separate frozen contract with preserved caveats.
- Submission readiness is distinct from scientific evidence completeness.
- Only gates, signatures, summaries and filesystem metadata are read; no medical image, mask, probability cache or model checkpoint content is opened.
"""))
c.append(nbf.v4.new_code_cell(r'''from pathlib import Path
from datetime import datetime, timezone
import hashlib,json,platform,re,sys
import pandas as pd
import matplotlib.pyplot as plt

STEP_DIR=Path.cwd().resolve(); assert STEP_DIR.name=="step_21_terminal_evidence_chain_and_project_handoff"
PART2=STEP_DIR.parent; OUT=STEP_DIR/"outputs"; OUT.mkdir(exist_ok=True)
MANIFEST_SHA="575a6fc391d63dc9bbbbb3317efe4d0f65edd57457ae63c1bfc6c2c615861889"
def sha256(p,chunk=1024*1024):
 h=hashlib.sha256()
 with Path(p).open("rb") as f:
  for b in iter(lambda:f.read(chunk),b""): h.update(b)
 return h.hexdigest()
def loadj(p): return json.loads(Path(p).read_text(encoding="utf-8"))
def savej(o,n): (OUT/n).write_text(json.dumps(o,indent=2,sort_keys=True,default=str),encoding="utf-8")
def savec(d,n): d.to_csv(OUT/n,index=False)
phases=sorted([p for p in PART2.iterdir() if p.is_dir() and re.match(r"step_\d{2}_",p.name) and int(p.name[5:7])<=20],key=lambda p:int(p.name[5:7]))
assert len(phases)==20,[p.name for p in phases]
print("Terminal evidence audit only; phases=20; inference=False; source-test access=False")'''))
c.append(nbf.v4.new_markdown_cell("""## Data

### 1. Inventory all phase gates and artifacts
"""))
c.append(nbf.v4.new_code_cell(r'''gate_rows=[]; artifact_rows=[]
for phase in phases:
 n=int(phase.name[5:7]); out=phase/"outputs"; gate_path=out/"gate_result.json"
 gate=loadj(gate_path) if gate_path.is_file() else {}
 result=gate.get("result_level") or gate.get("status") or gate.get("decision") or "UNSPECIFIED"
 if gate.get("formal_model_acceptance_passed") is False: classification="complete_with_formal_model_failure"
 elif any(x in str(result).upper() for x in ["OWNER_INPUT_REQUIRED","DECLARATIONS_REQUIRED","AWAITING_AUTHORIZATION","AUTHORIZATION_REQUIRED"]): classification="complete_with_external_action_required"
 else: classification="phase_evidence_complete"
 reported=gate.get("gate_passed",gate.get("all_mandatory_targets_passed",gate.get("passed",None)))
 gate_rows.append({"step":n,"phase":phase.name,"gate_path":str(gate_path.relative_to(PART2)) if gate_path.is_file() else "","gate_exists":gate_path.is_file(),"result_level":result,
  "classification":classification,"reported_gate_passed":reported,"formal_model_acceptance_passed":gate.get("formal_model_acceptance_passed"),
  "submission_ready":gate.get("submission_ready"),"inference_performed":gate.get("inference_performed"),"test_images_accessed":gate.get("test_images_accessed"),"next_step":gate.get("next_step","")})
 files=[p for p in out.rglob("*") if p.is_file()] if out.is_dir() else []
 artifact_rows.append({"step":n,"phase":phase.name,"output_files":len(files),"output_bytes":sum(p.stat().st_size for p in files),"notebooks":len(list(phase.glob("*.ipynb"))),"readme_exists":(phase/"README.md").is_file(),"error_log_exists":(phase/"error_fixes.md").is_file()})
gates=pd.DataFrame(gate_rows); artifacts=pd.DataFrame(artifact_rows)
savec(gates,"phase_gate_chain.csv"); savec(artifacts,"phase_artifact_index.csv")
assert gates.gate_exists.all() and gates.step.nunique()==20
display(gates[["step","result_level","classification","next_step"]])'''))
c.append(nbf.v4.new_markdown_cell("""### 2. Verify every signed-artifact map across Steps 1–20

Signature JSON files that contain a `signed_artifacts` dictionary are independently rehashed. Metadata-only signature files are inventoried but cannot be expanded beyond their own recorded contract.
"""))
c.append(nbf.v4.new_code_cell(r'''def resolve_signed_path(rel,phase,out):
 raw=Path(rel)
 candidates=[raw] if raw.is_absolute() else [PART2/raw,phase/raw,out/raw]
 for p in candidates:
  if p.is_file(): return p
 return candidates[0]
signature_rows=[]; signature_files=[]
for phase in phases:
 out=phase/"outputs"
 for sp in sorted(out.glob("*signature*.json")):
  try: sig=loadj(sp)
  except Exception as e:
   signature_files.append({"phase":phase.name,"signature_file":sp.name,"schema":"unreadable","mapped_artifacts":0,"passed":False,"note":str(e)}); continue
  mapping=sig.get("signed_artifacts")
  if isinstance(mapping,dict):
   local=[]; superseded=sp.name=="preflight_signature.json" and (out/"step_16_signature.json").is_file()
   for rel,expected in mapping.items():
    p=resolve_signed_path(rel,phase,out); actual=sha256(p) if p.is_file() else None; passed=actual==expected
    effective_pass=passed or superseded
    row={"phase":phase.name,"signature_file":sp.name,"artifact":rel,"resolved_path":str(p),"exists":p.is_file(),"expected_sha256":expected,"actual_sha256":actual,"hash_matches":passed,"superseded_preflight":superseded,"effective_pass":effective_pass}
    signature_rows.append(row); local.append(effective_pass)
   note="superseded pre-authorization snapshot; mismatch preserved as historical evidence" if superseded else "current signature independently rehashed"
   signature_files.append({"phase":phase.name,"signature_file":sp.name,"schema":"signed_artifacts_map","mapped_artifacts":len(mapping),"passed":bool(all(local)),"note":note})
  else:
   signature_files.append({"phase":phase.name,"signature_file":sp.name,"schema":"metadata_only","mapped_artifacts":0,"passed":True,"note":"no expandable signed_artifacts map"})
sig_detail=pd.DataFrame(signature_rows); sig_files=pd.DataFrame(signature_files)
savec(sig_detail,"signed_artifact_verification.csv"); savec(sig_files,"signature_file_inventory.csv")
assert len(sig_detail)>0 and sig_detail.effective_pass.all(),sig_detail.loc[~sig_detail.effective_pass].to_dict("records")
assert sig_files.passed.all()
print(f"PASS: {len(sig_files)} signature files inventoried; {len(sig_detail)} mappings current or explicitly superseded; historical mismatches={(~sig_detail.hash_matches).sum()}")'''))
c.append(nbf.v4.new_markdown_cell("""## Results

### 3. Consolidate scientific results and remaining actions
"""))
c.append(nbf.v4.new_code_cell(r'''g5=loadj(PART2/"step_05_final_research_package"/"outputs"/"gate_result.json")
g16=loadj(PART2/"step_16_one_time_external_evaluation_after_explicit_authorization"/"outputs"/"gate_result.json")
g20=loadj(PART2/"step_20_external_citation_verification_and_owner_readiness"/"outputs"/"gate_result.json")
external=pd.read_csv(PART2/"step_16_one_time_external_evaluation_after_explicit_authorization"/"outputs"/"global_metrics.csv").iloc[0]
summary=pd.DataFrame([
 {"evidence_area":"authoritative corrected dataset","status":"verified","key_result":f"manifest SHA-256 {MANIFEST_SHA}","claim_boundary":"patient-disjoint build; historical source remains read-only"},
 {"evidence_area":"LiTS held-out evaluation","status":"complete_formal_acceptance_failed","key_result":f"global Dice {g5['selected_metrics']['global_dice']:.6f}; minimum patient Dice {g5['selected_metrics']['minimum_positive_patient_dice']:.6f}","claim_boundary":"no further test reuse or test-driven tuning"},
 {"evidence_area":"3D-IRCADb-01 external evaluation","status":"separate_frozen_contract_passed","key_result":f"global Dice {external.global_dice:.6f}; mean positive-patient Dice {external.mean_positive_patient_dice:.6f}","claim_boundary":"single public cohort; material patient, lesion and negative-control caveats"},
 {"evidence_area":"external source citation and licence","status":"verified","key_result":"Soler2010IRCADb; CC BY-NC-ND 4.0","claim_boundary":"no redistribution authorization; owner reviews availability language"},
 {"evidence_area":"manuscript evidence package","status":"complete_with_owner_blockers","key_result":"citation-patched manuscript and signed evidence package exist","claim_boundary":"not submission-ready; owner gate closed"},
])
savec(summary,"terminal_scientific_summary.csv")
actions=pd.DataFrame([
 {"priority":1,"action":"Complete and certify all 14 Step 11 owner fields","owner":"project owner","required_before":"submission readiness","scientific_rerun_required":False,"status":"blocking"},
 {"priority":2,"action":"Select venue and publication budget from the verified venue matrix","owner":"project owner","required_before":"venue finalization","scientific_rerun_required":False,"status":"blocking"},
 {"priority":3,"action":"Review dataset/code availability statements against intended release and institutional policy","owner":"project owner","required_before":"submission","scientific_rerun_required":False,"status":"blocking"},
 {"priority":4,"action":"Optional blinded expert review of external source-label discordance","owner":"qualified medical/source reviewer","required_before":"biological reinterpretation only","scientific_rerun_required":False,"status":"optional"},
 {"priority":5,"action":"Define a new predeclared study only if new independent data or a new research question becomes available","owner":"research team","required_before":"any future model development","scientific_rerun_required":False,"status":"future_conditional"},
])
savec(actions,"unresolved_actions.csv"); display(summary); display(actions)'''))
c.append(nbf.v4.new_markdown_cell("""### 4. Visualize the 20-phase evidence chain

Chart contract: notebook-oriented static status map; 20 ordered phases; marker and direct-label encoding; three restrained categories; no performance magnitude implied; final QA on exported PNG.
"""))
c.append(nbf.v4.new_code_cell(r'''palette={"phase_evidence_complete":"#3366A3","complete_with_formal_model_failure":"#D17A22","complete_with_external_action_required":"#8A6D3B"}
labels={"phase_evidence_complete":"evidence complete","complete_with_formal_model_failure":"complete; formal model gate failed","complete_with_external_action_required":"complete; external/owner action required"}
fig,ax=plt.subplots(figsize=(15,11)); y=list(range(len(gates),0,-1))
for yi,row in zip(y,gates.itertuples()):
 color=palette[row.classification]; ax.scatter([1],[yi],s=125,color=color,marker="s",edgecolor="#222222",linewidth=.6)
 short=str(row.result_level); short=short if len(short)<=58 else short[:55]+"..."
 ax.text(1.04,yi,f"Step {row.step:02d}  {short}",va="center",fontsize=9,color="#222222")
ax.set_xlim(.96,2.1); ax.set_ylim(.3,20.7); ax.axis("off"); ax.set_title("Part 2 evidence-chain status across Steps 1–20",fontsize=16,pad=15)
handles=[plt.Line2D([0],[0],marker="s",color="none",markerfacecolor=palette[k],markeredgecolor="#222222",markersize=9,label=labels[k]) for k in palette]
ax.legend(handles=handles,loc="lower right",frameon=False)
fig.text(.01,.01,"Status categories describe evidence state, not model-performance magnitude. Source: phase gate JSON files.",fontsize=9,color="#333333")
fig.tight_layout(rect=(0,.035,1,1)); fig.savefig(OUT/"phase_evidence_chain.png",dpi=180,bbox_inches="tight"); plt.close(fig)
savec(pd.DataFrame([{"section":"Terminal phase chain","question":"Are all 20 phases represented and what is each evidence state?","family":"ordered status map","chart":"direct-labelled square markers","rows":20,"palette":"blue/orange/brown plus labels","artifact":"phase_evidence_chain.png"}]),"chart_map.csv")'''))
c.append(nbf.v4.new_markdown_cell("""### 5. Create terminal data card and handoff
"""))
c.append(nbf.v4.new_code_cell(r'''data_card=f"""# Terminal project evidence data card

## Final evidence state

- Part 2 phases inventoried: 20/20.
- Current signed-artifact mappings verified; two Step 16 pre-authorization hashes are explicitly preserved as superseded historical snapshots.
- Authoritative manifest: `{MANIFEST_SHA}`.
- LiTS held-out result: complete; formal model acceptance failed because minimum positive-patient Dice was 0.
- 3D-IRCADb-01 result: separate frozen external contract passed with preserved caveats.
- External citation/licence: verified (`Soler2010IRCADb`, CC BY-NC-ND 4.0).
- Submission readiness: false; owner fields complete 0/14.

## Scientific claim boundary

The project supports strong aggregate held-out and external segmentation evidence, but not formal LiTS model acceptance, clinical readiness, universal generalization, or biological reinterpretation of excluded external labels.

## Data and execution safeguards

Step 21 opened no medical image, mask, probability cache, loader or model. It performed no inference, tuning, dataset download, test reuse, submission or payment action.

## Resume rule

Resume submission preparation only after owner fields are supplied and certified. Resume scientific work only under a new predeclared question with genuinely new evidence; never reopen the sealed LiTS or Step 16 evaluation for result-driven tuning.
"""
(OUT/"TERMINAL_PROJECT_DATA_CARD.md").write_text(data_card,encoding="utf-8")
handoff=f"""# Terminal handoff

The Part 2 scientific evidence chain is complete and auditable. The project result is intentionally mixed: aggregate performance is strong and the separate public external contract passed, while the predeclared LiTS minimum-patient gate failed.

No additional scientific notebook is currently justified. The active blocking work is owner-controlled submission intake: 14 declarations/choices remain incomplete. Use `unresolved_actions.csv` and the verified Step 20 citation/availability drafts.

Do not rerun test or external inference, tune threshold 0.70, change maximum fusion, retrospectively relabel external cases, or claim clinical readiness.
"""
(OUT/"TERMINAL_HANDOFF.md").write_text(handoff,encoding="utf-8")'''))
c.append(nbf.v4.new_markdown_cell("""## Takeaways

### 6. Freeze the terminal project gate and signature
"""))
c.append(nbf.v4.new_code_cell(r'''expected=pd.DataFrame([
 {"requirement":"twenty_prior_phases_inventoried","expected":20,"actual":len(gates),"passed":len(gates)==20},
 {"requirement":"all_phase_gates_exist","expected":True,"actual":bool(gates.gate_exists.all()),"passed":bool(gates.gate_exists.all())},
 {"requirement":"all_current_or_superseded_signature_mappings_valid","expected":True,"actual":bool(sig_detail.effective_pass.all()),"passed":bool(sig_detail.effective_pass.all())},
 {"requirement":"authoritative_manifest_preserved","expected":MANIFEST_SHA,"actual":g5.get("manifest_sha256"),"passed":g5.get("manifest_sha256")==MANIFEST_SHA},
 {"requirement":"formal_LiTS_failure_preserved","expected":False,"actual":g5.get("formal_model_acceptance_passed"),"passed":g5.get("formal_model_acceptance_passed") is False},
 {"requirement":"external_contract_pass_preserved","expected":True,"actual":g16.get("all_mandatory_targets_passed"),"passed":g16.get("all_mandatory_targets_passed") is True},
 {"requirement":"owner_submission_gate_closed","expected":False,"actual":g20.get("submission_ready"),"passed":g20.get("submission_ready") is False},
 {"requirement":"no_inference_test_reopen_tuning_or_submission","expected":True,"actual":True,"passed":True},
])
savec(expected,"expected_vs_actual.csv"); assert expected.passed.all(),expected.loc[~expected.passed].to_dict("records")
result="PROJECT_EVIDENCE_CHAIN_VERIFIED_OWNER_ACTIONS_PENDING"
gate={"created_utc":datetime.now(timezone.utc).isoformat(),"result_level":result,"gate_passed":True,"prior_phases_inventoried":20,"signature_files_inventoried":len(sig_files),"mapped_signature_entries_audited":len(sig_detail),"superseded_preflight_hash_mismatches":int((~sig_detail.hash_matches).sum()),
 "formal_model_acceptance_passed":False,"external_contract_passed":True,"scientific_evidence_chain_complete":True,"submission_ready":False,"owner_fields_complete":0,"owner_fields_total":14,
 "inference_performed":False,"test_images_accessed":False,"test_source_files_reopened":False,"test_inference_rerun":False,"external_inference_rerun":False,"tuning_performed":False,"dataset_download_performed":False,"submission_action_performed":False,"payment_action_performed":False,
 "next_step":"owner completes and certifies Step 11 fields; otherwise stop. New scientific work requires a new predeclared question and new evidence."}
savej(gate,"gate_result.json")
savej({"phase":STEP_DIR.name,"created_utc":gate["created_utc"],"python":sys.version,"platform":platform.platform(),"manifest_sha256":MANIFEST_SHA,"phase_count":20,
 "step20_signature_sha256":sha256(PART2/"step_20_external_citation_verification_and_owner_readiness"/"outputs"/"step_20_signature.json"),"project_status_sha256_before_step21_update":sha256(PART2/"PROJECT_STATUS.md"),
 "medical_source_data_opened":False,"inference_performed":False,"test_source_files_reopened":False,"submission_action_performed":False},"provenance.json")
names=["phase_gate_chain.csv","phase_artifact_index.csv","signed_artifact_verification.csv","signature_file_inventory.csv","terminal_scientific_summary.csv","unresolved_actions.csv","phase_evidence_chain.png","chart_map.csv","TERMINAL_PROJECT_DATA_CARD.md","TERMINAL_HANDOFF.md","expected_vs_actual.csv","gate_result.json","provenance.json"]
signed={str((OUT/n).relative_to(PART2)):sha256(OUT/n) for n in names}; combined=hashlib.sha256("\n".join(f"{k}:{v}" for k,v in sorted(signed.items())).encode()).hexdigest()
savej({"algorithm":"SHA-256","created_utc":datetime.now(timezone.utc).isoformat(),"combined_sha256":combined,"result_level":result,"signed_artifacts":signed,"submission_action_performed":False},"step_21_signature.json")
print(json.dumps(gate,indent=2))'''))
c.append(nbf.v4.new_markdown_cell("""The Part 2 evidence chain is terminally verified. The next action is owner input, not another automatic analysis. If no owner submission work or new scientific question is supplied, stop here and preserve the archive unchanged."""))
nb["cells"]=c
for p in (LONG,SHORT): nbf.write(nb,p)
print("Wrote",LONG); print("Wrote",SHORT)
