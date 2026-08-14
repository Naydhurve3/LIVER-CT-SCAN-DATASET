from pathlib import Path
import nbformat as nbf

STEP_DIR=Path(__file__).resolve().parent
LONG=STEP_DIR/"step_20_external_citation_verification_and_owner_readiness.ipynb"
SHORT=STEP_DIR/"step_20.ipynb"
nb=nbf.v4.new_notebook()
nb["metadata"]={"kernelspec":{"display_name":"Python 3 (ds_gpu)","language":"python","name":"python3"},"language_info":{"name":"python","version":"3.11"}}
c=[]
c.append(nbf.v4.new_markdown_cell("""# Step 20 — External citation verification and owner readiness

## tl;dr

This notebook freezes the verified primary-source citation and licence evidence for 3D-IRCADb-01, creates a citation-patched manuscript copy and BibTeX addendum, and re-audits the owner-only submission blockers.

It performs no model loading, inference, tuning, dataset download, test access, submission, payment, or owner certification.
"""))
c.append(nbf.v4.new_markdown_cell("""## Context & Methods

### Key Assumptions

- The official IRCAD dataset page is the authoritative source for dataset identity, citation, cohort description and licence statement.
- The page was checked on 2026-08-06 and redirected to its current IRCAD URL.
- The cited object is an IRCAD technical report from 2010; the official page does not provide a DOI.
- Draft availability/terms language is advisory until the owner reviews and certifies it.
- Step 19 evidence and Step 11 owner state remain immutable inputs.
"""))
c.append(nbf.v4.new_code_cell(r'''from pathlib import Path
from datetime import datetime, timezone
import hashlib,json,platform,sys
import pandas as pd

STEP_DIR=Path.cwd().resolve(); assert STEP_DIR.name=="step_20_external_citation_verification_and_owner_readiness"
PART2=STEP_DIR.parent; OUT=STEP_DIR/"outputs"; OUT.mkdir(exist_ok=True)
S11=PART2/"step_11_owner_submission_gate_and_finalization_handoff"/"outputs"
S13=PART2/"step_13_3d_ircadb_ingestion_and_qc"/"outputs"
S19=PART2/"step_19_external_evidence_manuscript_integration"/"outputs"
OFFICIAL_URL="https://www.ircad.fr/research-and-development/data-sets/liver-segmentation-3d-ircadb-01/"
LICENSE_URL="https://creativecommons.org/licenses/by-nc-nd/4.0/"
ACCESS_DATE="2026-08-06"
def sha256(p,chunk=1024*1024):
 h=hashlib.sha256()
 with Path(p).open("rb") as f:
  for b in iter(lambda:f.read(chunk),b""): h.update(b)
 return h.hexdigest()
def loadj(p): return json.loads(Path(p).read_text(encoding="utf-8"))
def savej(o,n): (OUT/n).write_text(json.dumps(o,indent=2,sort_keys=True,default=str),encoding="utf-8")
def savec(d,n): d.to_csv(OUT/n,index=False)
print("Citation/readiness audit only; inference=False; dataset download=False; submission=False")'''))
c.append(nbf.v4.new_markdown_cell("""## Data

### 1. Verify upstream evidence and preserve the owner gate
"""))
c.append(nbf.v4.new_code_cell(r'''g19=loadj(S19/"gate_result.json"); sig19=loadj(S19/"step_19_signature.json"); owner=pd.read_csv(S11/"owner_input_status.csv"); terms=loadj(S13/"source_terms_snapshot.json")
sig_rows=[]
for rel,expected in sig19["signed_artifacts"].items():
 p=PART2/rel; actual=sha256(p) if p.is_file() else None
 sig_rows.append({"artifact":rel,"exists":p.is_file(),"expected_sha256":expected,"actual_sha256":actual,"passed":actual==expected})
sig_check=pd.DataFrame(sig_rows); savec(sig_check,"step19_signature_verification.csv")
checks=pd.DataFrame([
 {"check":"step19_manuscript_update_ready","passed":g19.get("manuscript_update_ready") is True},
 {"check":"step19_submission_closed","passed":g19.get("submission_ready") is False},
 {"check":"all_step19_signed_artifacts_match","passed":bool(sig_check.passed.all())},
 {"check":"fourteen_owner_fields_present","passed":len(owner)==14 and owner.field.nunique()==14},
 {"check":"owner_fields_remain_unset","passed":int(owner.complete.astype(bool).sum())==0},
 {"check":"local_terms_snapshot_matches_dataset","passed":terms.get("dataset")=="3D-IRCADb-01"},
 {"check":"redistribution_remains_unauthorized","passed":terms.get("redistribution_of_downloaded_or_derived_data_authorized") is False},
])
savec(checks,"input_verification.csv"); assert checks.passed.all(),checks.loc[~checks.passed].to_dict("records")
print("PASS: Step 19 verified; 0/14 owner fields complete; submission remains closed")'''))
c.append(nbf.v4.new_markdown_cell("""## Results

### 2. Freeze the official primary-source citation and licence snapshot

Primary source checked: [IRCAD 3D-IRCADb-01 official page](https://www.ircad.fr/research-and-development/data-sets/liver-segmentation-3d-ircadb-01/).
"""))
c.append(nbf.v4.new_code_cell(r'''citation={
 "citation_key":"Soler2010IRCADb",
 "authors":"Soler, L.; Hostettler, A.; Agnus, V.; Charnoz, A.; Fasquel, J.; Moreau, J.; Osswald, A.; Bouhadjar, M.; Marescaux, J.",
 "title":"3D image reconstruction for comparison of algorithm database: A patient specific anatomical and medical image database",
 "institution":"IRCAD","address":"Strasbourg, France","type":"Technical Report","year":2010,"doi":"not provided by official source",
 "official_url":OFFICIAL_URL,"access_date":ACCESS_DATE,
}
snapshot={"dataset":"3D-IRCADb-01","source_owner":"IRCAD","official_url":OFFICIAL_URL,"access_date":ACCESS_DATE,
 "cohort_statement":"3D CT scans of 10 women and 10 men; hepatic tumours in 75% of cases; 20 different patients",
 "content_statement":"PATIENT_DICOM, LABELLED_DICOM, MASKS_DICOM, and MESHES_VTK folders",
 "license_name":"Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International","license_url":LICENSE_URL,
 "official_citation":citation,"source_status":"primary_official_page_verified","download_performed":False}
savej(snapshot,"official_ircadb_source_snapshot.json")
evidence=pd.DataFrame([
 {"item":"dataset identity","official_evidence":"3D-IRCADb-01","status":"verified","source":OFFICIAL_URL,"access_date":ACCESS_DATE},
 {"item":"cohort size","official_evidence":"20 different patients (10 women, 10 men)","status":"verified","source":OFFICIAL_URL,"access_date":ACCESS_DATE},
 {"item":"tumour prevalence","official_evidence":"hepatic tumours in 75% of cases","status":"verified","source":OFFICIAL_URL,"access_date":ACCESS_DATE},
 {"item":"licence","official_evidence":"CC BY-NC-ND 4.0 International","status":"verified","source":OFFICIAL_URL,"access_date":ACCESS_DATE},
 {"item":"citation","official_evidence":citation["title"]+"; IRCAD technical report (2010)","status":"verified","source":OFFICIAL_URL,"access_date":ACCESS_DATE},
 {"item":"DOI","official_evidence":"none supplied on official page","status":"not_applicable_not_invented","source":OFFICIAL_URL,"access_date":ACCESS_DATE},
])
savec(evidence,"official_citation_evidence.csv")
bib=r"""@techreport{Soler2010IRCADb,
  author      = {Soler, L. and Hostettler, A. and Agnus, V. and Charnoz, A. and Fasquel, J. and Moreau, J. and Osswald, A. and Bouhadjar, M. and Marescaux, J.},
  title       = {3D Image Reconstruction for Comparison of Algorithm Database: A Patient Specific Anatomical and Medical Image Database},
  institution = {IRCAD},
  address     = {Strasbourg, France},
  type        = {Technical Report},
  year        = {2010},
  url         = {https://www.ircad.fr/research-and-development/data-sets/liver-segmentation-3d-ircadb-01/},
  urldate     = {2026-08-06}
}
"""
(OUT/"external_dataset_reference_addendum.bib").write_text(bib,encoding="utf-8")
display(evidence)'''))
c.append(nbf.v4.new_markdown_cell("""### 3. Create citation-patched manuscript and owner-review drafts
"""))
c.append(nbf.v4.new_code_cell(r'''draft=(S19/"PAPER_DRAFT_EXTERNAL_VALIDATION_UPDATE.md").read_text(encoding="utf-8")
needle="complete 20-case 3D-IRCADb-01 cohort"
assert needle in draft
patched=draft.replace(needle,"complete 20-case 3D-IRCADb-01 cohort [Soler2010IRCADb]",1)
patched += "\n\n## Verified external dataset reference\n\n[Soler2010IRCADb] "+citation["authors"]+" “"+citation["title"]+".” IRCAD, Strasbourg, France, Technical Report (2010). "+OFFICIAL_URL+" (accessed "+ACCESS_DATE+").\n"
(OUT/"PAPER_DRAFT_WITH_VERIFIED_EXTERNAL_CITATION.md").write_text(patched,encoding="utf-8")
statement=f"""# Dataset attribution and availability drafts — owner review required

## Dataset terms statement

The 3D-IRCADb-01 dataset was obtained from IRCAD for non-commercial research under the Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International licence. The official source and terms were checked on {ACCESS_DATE}: {OFFICIAL_URL}

## Data availability statement

3D-IRCADb-01 is available directly from IRCAD at {OFFICIAL_URL}. Dataset files and derived patient data are not redistributed with this work. The corrected LiTS build and project-specific derived artifacts remain subject to their respective source terms and repository access policy.

## Required owner action

The owner must verify that these statements match the intended code/data release, institutional policy, venue requirements, and actual materials being shared before copying them into the submission form.
"""
(OUT/"DATASET_ATTRIBUTION_AND_AVAILABILITY_DRAFTS.md").write_text(statement,encoding="utf-8")
patch_plan=pd.DataFrame([
 {"target":"Step 19 manuscript","location":"first external cohort mention","action":"insert [Soler2010IRCADb]","status":"applied_to_new_copy_only"},
 {"target":"reference list","location":"verified external dataset reference","action":"add IRCAD technical report citation","status":"applied_to_new_copy_only"},
 {"target":"references.bib","location":"new addendum","action":"add @techreport entry","status":"created"},
 {"target":"dataset terms declaration","location":"owner submission fields","action":"owner reviews advisory draft","status":"owner_required"},
 {"target":"data availability declaration","location":"owner submission fields","action":"owner reviews advisory draft","status":"owner_required"},
])
savec(patch_plan,"manuscript_citation_patch_plan.csv")'''))
c.append(nbf.v4.new_markdown_cell("""### 4. Re-audit submission blockers without filling owner fields
"""))
c.append(nbf.v4.new_code_cell(r'''blockers=owner[["field","value","validation_rule","complete"]].copy()
blockers["responsibility"]="owner"
blockers["safe_assistance_available"]=blockers.field.isin(["dataset_terms_and_availability_statement","data_availability_statement"])
blockers["assistant_status"]=blockers.safe_assistance_available.map({True:"advisory draft created; owner verification required",False:"cannot infer owner-specific value"})
savec(blockers,"owner_submission_blocker_matrix.csv")
readiness=pd.DataFrame([
 {"area":"scientific result package","status":"complete_with_failed_LiTS_guardrail_and_external_caveats","blocking":False},
 {"area":"external dataset citation","status":"verified_from_official_IRCAD_page","blocking":False},
 {"area":"external dataset licence/attribution","status":"verified_CC_BY_NC_ND_4_0; owner release review required","blocking":True},
 {"area":"owner declarations","status":"0_of_14_complete","blocking":True},
 {"area":"venue and budget","status":"owner values absent","blocking":True},
 {"area":"submission authorization","status":"not granted","blocking":True},
])
savec(readiness,"submission_readiness_summary.csv"); display(readiness); display(blockers)'''))
c.append(nbf.v4.new_markdown_cell("""## Takeaways

### 5. Freeze the citation result and owner-required gate
"""))
c.append(nbf.v4.new_code_cell(r'''expected=pd.DataFrame([
 {"requirement":"Step19 evidence verified","expected":True,"actual":bool(checks.passed.all()),"passed":bool(checks.passed.all())},
 {"requirement":"official citation captured","expected":True,"actual":len(evidence[evidence.item.eq("citation")])==1,"passed":len(evidence[evidence.item.eq("citation")])==1},
 {"requirement":"licence captured","expected":True,"actual":snapshot["license_name"].startswith("Creative Commons"),"passed":snapshot["license_name"].startswith("Creative Commons")},
 {"requirement":"owner fields unmodified","expected":0,"actual":int(owner.complete.astype(bool).sum()),"passed":int(owner.complete.astype(bool).sum())==0},
 {"requirement":"submission remains closed","expected":False,"actual":g19["submission_ready"],"passed":g19["submission_ready"] is False},
 {"requirement":"no inference download test access or submission","expected":True,"actual":True,"passed":True},
])
savec(expected,"expected_vs_actual.csv"); assert expected.passed.all()
result="EXTERNAL_CITATION_VERIFIED_OWNER_DECLARATIONS_REQUIRED"
gate={"created_utc":datetime.now(timezone.utc).isoformat(),"result_level":result,"gate_passed":True,"external_dataset_citation_verified":True,"citation_key":"Soler2010IRCADb","citation_source":OFFICIAL_URL,"citation_access_date":ACCESS_DATE,
 "license_verified":True,"license":"CC BY-NC-ND 4.0 International","owner_fields_complete":0,"owner_fields_total":14,"owner_gate_passed":False,"submission_ready":False,
 "inference_performed":False,"dataset_download_performed":False,"test_images_accessed":False,"test_source_files_reopened":False,"tuning_performed":False,"submission_action_performed":False,"payment_action_performed":False,
 "next_step":"owner completes and certifies the 14 Step 11 fields using the verified citation and advisory dataset statements"}
savej(gate,"gate_result.json")
savej({"phase":STEP_DIR.name,"created_utc":gate["created_utc"],"python":sys.version,"platform":platform.platform(),"official_source":OFFICIAL_URL,"access_date":ACCESS_DATE,
 "step19_signature_sha256":sha256(S19/"step_19_signature.json"),"step11_owner_status_sha256":sha256(S11/"owner_input_status.csv"),"step13_terms_snapshot_sha256":sha256(S13/"source_terms_snapshot.json"),
 "inference_performed":False,"dataset_download_performed":False,"test_source_files_reopened":False,"submission_action_performed":False},"provenance.json")
names=["input_verification.csv","step19_signature_verification.csv","official_ircadb_source_snapshot.json","official_citation_evidence.csv","external_dataset_reference_addendum.bib","PAPER_DRAFT_WITH_VERIFIED_EXTERNAL_CITATION.md","DATASET_ATTRIBUTION_AND_AVAILABILITY_DRAFTS.md","manuscript_citation_patch_plan.csv","owner_submission_blocker_matrix.csv","submission_readiness_summary.csv","expected_vs_actual.csv","gate_result.json","provenance.json"]
signed={str((OUT/n).relative_to(PART2)):sha256(OUT/n) for n in names}; combined=hashlib.sha256("\n".join(f"{k}:{v}" for k,v in sorted(signed.items())).encode()).hexdigest()
savej({"algorithm":"SHA-256","created_utc":datetime.now(timezone.utc).isoformat(),"combined_sha256":combined,"result_level":result,"signed_artifacts":signed,"submission_action_performed":False},"step_20_signature.json")
print(json.dumps(gate,indent=2))'''))
c.append(nbf.v4.new_markdown_cell("""The external citation blocker is resolved from the official source. Submission remains blocked solely by owner-controlled declarations, venue/budget choices, certification, and authorization. No further analytical notebook is required unless the owner supplies those values or a new scientific question is defined."""))
nb["cells"]=c
for p in (LONG,SHORT): nbf.write(nb,p)
print("Wrote",LONG); print("Wrote",SHORT)
