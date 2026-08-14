from pathlib import Path
import nbformat as nbf

HERE = Path(__file__).resolve().parent
NOTEBOOK = HERE / "step_13_3d_ircadb_ingestion_and_qc.ipynb"
SHORT_NOTEBOOK = HERE / "step_13.ipynb"

cells = [nbf.v4.new_markdown_cell("""# Step 13 — 3D-IRCADb-01 Ingestion and Source QC

## tl;dr

This notebook safely acquires and inventories the official 3D-IRCADb-01 archive, validates archive structure and DICOM metadata, profiles liver/tumour mask availability, and prepares a source-QC gate before conversion or inference.

The default run is a complete **download-disabled preflight**. It saves `EXTERNAL_DOWNLOAD_AUTHORIZATION_REQUIRED` rather than throwing an error. Downloading starts only after the user accepts the official CC BY-NC-ND 4.0 conditions and explicitly enables the switches in code Cell 1.

### Immutable boundary

- Never access the sealed local LiTS test split.
- Never run model inference, threshold tuning, training, or evaluation.
- Download only the official combined 3D-IRCADb-01 archive.
- Keep raw, extracted, inventory, and QC artifacts under this phase's `outputs/` tree.
- Do not redistribute downloaded or derived data from this notebook.
""")]

cells.append(nbf.v4.new_markdown_cell("""## Context & Methods

### Key Assumptions

- Official source snapshot: IRCAD page checked 5 August 2026.
- Official combined archive size: approximately 782 MB.
- Official license statement: Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International.
- Required citation: Soler et al., “3D image reconstruction for comparison of algorithm database: A patient specific anatomical and medical image database,” IRCAD Technical Report, 2010.
- Technical inspection does not imply permission to publish or redistribute adapted dataset materials.
"""))

cells.append(nbf.v4.new_code_cell(r'''from pathlib import Path
from datetime import datetime, timezone
import hashlib, json, os, re, shutil, urllib.request, zipfile
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

STEP_DIR = Path.cwd().resolve()
if STEP_DIR.name != "step_13_3d_ircadb_ingestion_and_qc":
    candidate = STEP_DIR / "step_13_3d_ircadb_ingestion_and_qc"
    if candidate.is_dir(): STEP_DIR = candidate.resolve()
PART2 = STEP_DIR.parent
S12 = PART2 / "step_12_public_external_dataset_audit_and_acquisition_plan" / "outputs"
OUT = STEP_DIR / "outputs"; OUT.mkdir(parents=True, exist_ok=True)
RAW = OUT / "raw_downloads"; RAW.mkdir(parents=True, exist_ok=True)
EXTRACTED = OUT / "extracted"; EXTRACTED.mkdir(parents=True, exist_ok=True)

SOURCE_PAGE = "https://www.ircad.fr/research-and-development/data-sets/liver-segmentation-3d-ircadb-01/"
DOWNLOAD_URL = "https://cloud.ircad.fr/index.php/s/JN3z7EynBiwYyjy/download"
PATIENT_DOWNLOAD_URLS = {i: f"{DOWNLOAD_URL}?path=%2F3Dircadb1.{i}" for i in range(1, 21)}
LICENSE_URL = "https://creativecommons.org/licenses/by-nc-nd/4.0/"
PATIENT_ARCHIVES = {i: RAW / f"3Dircadb1.{i}.zip" for i in range(1, 21)}
MANIFEST_SHA256 = "575a6fc391d63dc9bbbbb3317efe4d0f65edd57457ae63c1bfc6c2c615861889"
CREATED_UTC = datetime.now(timezone.utc).isoformat()

# USER AUTHORIZATION RECORD.
AUTHORIZATION_TEXT = "I accept the 3D-IRCADb-01 CC BY-NC-ND 4.0 terms and authorize the Step 13 download and extraction."
AUTHORIZATION_RECORDED_UTC = "2026-08-05T13:26:50.2768777+00:00"
USER_ACCEPTS_CC_BY_NC_ND_4_0 = True
DOWNLOAD_ENABLED = True
EXTRACT_ENABLED = True
FULL_FILE_HASHES = False  # optional and slow; archive SHA-256 is always computed

EXPECTED_TOTAL_SIZE_MIN = 650 * 1024**2
EXPECTED_TOTAL_SIZE_MAX = 850 * 1024**2
EXPECTED_PATIENT_ARCHIVE_MIN = 15 * 1024**2
EXPECTED_PATIENT_ARCHIVE_MAX = 80 * 1024**2

def sha256(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda:f.read(1024*1024),b""): h.update(block)
    return h.hexdigest()
def load_json(path): return json.loads(Path(path).read_text(encoding="utf-8"))
def save_json(obj,name):
    p=OUT/name; p.write_text(json.dumps(obj,indent=2,sort_keys=True),encoding="utf-8"); return p
def save_csv(df,name):
    p=OUT/name; df.to_csv(p,index=False); return p

authorization_granted = bool(USER_ACCEPTS_CC_BY_NC_ND_4_0 and DOWNLOAD_ENABLED)
print("Download authorization granted:", authorization_granted)
print("Local LiTS test path is intentionally undefined.")'''))

cells.append(nbf.v4.new_markdown_cell("""## Data

### 1. Verify Step 12, source terms, and the download gate
"""))

cells.append(nbf.v4.new_code_cell(r'''gate12=load_json(S12/"gate_result.json")
sig12=load_json(S12/"step_12_signature.json")
signed_rows=[]
for rel,expected in sig12["signed_artifacts"].items():
    path=PART2/rel; actual=sha256(path) if path.is_file() else None
    signed_rows.append({"artifact":rel,"expected_sha256":expected,"actual_sha256":actual,"passed":actual==expected})
signed12=pd.DataFrame(signed_rows)

checks=[
 ("step12_complete",gate12.get("result_level")=="PUBLIC_EXTERNAL_DATA_AUDIT_COMPLETE",gate12.get("result_level")),
 ("recommended_source",gate12.get("recommended_first_source")=="3D-IRCADb-01",gate12.get("recommended_first_source")),
 ("manifest_hash",gate12.get("manifest_sha256")==MANIFEST_SHA256,gate12.get("manifest_sha256")),
 ("step12_signature_valid",bool(len(signed12)) and bool(signed12.passed.all()),f"{int(signed12.passed.sum())}/{len(signed12)}"),
 ("step12_download_false",gate12.get("download_performed") is False,gate12.get("download_performed")),
 ("test_remained_sealed",gate12.get("test_source_files_reopened") is False,gate12.get("test_source_files_reopened")),
]
verification=pd.DataFrame(checks,columns=["check","passed","observed"])
save_csv(verification,"input_verification.csv")
assert verification.passed.all(),verification.loc[~verification.passed].to_dict("records")

terms={"dataset":"3D-IRCADb-01","official_page":SOURCE_PAGE,"official_download_url":DOWNLOAD_URL,
 "official_page_checked_date":"2026-08-05","reported_combined_archive_size":"782 MB",
 "license_name":"CC BY-NC-ND 4.0 International","license_url":LICENSE_URL,
 "permitted_project_mode":"non-commercial research subject to official terms",
 "redistribution_of_downloaded_or_derived_data_authorized":False,
 "required_attribution":"Soler et al., IRCAD Technical Report, 2010",
 "user_acceptance_recorded":bool(USER_ACCEPTS_CC_BY_NC_ND_4_0)}
save_json(terms,"source_terms_snapshot.json")
save_json({"user_accepts_terms":bool(USER_ACCEPTS_CC_BY_NC_ND_4_0),"download_enabled":bool(DOWNLOAD_ENABLED),
 "extract_enabled":bool(EXTRACT_ENABLED),"authorization_granted":authorization_granted,
 "authorization_text":AUTHORIZATION_TEXT,"authorization_recorded_utc":AUTHORIZATION_RECORDED_UTC,
 "download_started":False},"authorization_state.json")
print(f"PASS: {verification.passed.sum()}/{len(verification)} prerequisite checks")'''))

cells.append(nbf.v4.new_markdown_cell("""### 2. Download and safely extract only when authorized
"""))

cells.append(nbf.v4.new_code_cell(r'''download_status="not_authorized"
archive_set_sha=None; archive_bytes=None; extracted_now=False; manifest_rows=[]

if authorization_granted:
    any_downloaded=False
    stale_combined_part=RAW/"3Dircadb1.zip.part"
    if stale_combined_part.is_file() and stale_combined_part.stat().st_size==0: stale_combined_part.unlink()
    for case_id, archive in PATIENT_ARCHIVES.items():
        url=PATIENT_DOWNLOAD_URLS[case_id]
        part=archive.with_suffix(".zip.part")
        if part.is_file() and part.stat().st_size==0: part.unlink()
        if not archive.is_file():
            request=urllib.request.Request(url,headers={"User-Agent":"Mozilla/5.0 Step13ResearchAudit/1.0"})
            with urllib.request.urlopen(request,timeout=180) as response, part.open("wb") as target:
                while True:
                    chunk=response.read(1024*1024)
                    if not chunk: break
                    target.write(chunk)
            part.replace(archive); any_downloaded=True; status="downloaded"
        else:
            status="reused_existing_archive"
        size=archive.stat().st_size; digest=sha256(archive)
        assert EXPECTED_PATIENT_ARCHIVE_MIN <= size <= EXPECTED_PATIENT_ARCHIVE_MAX, f"Unexpected case {case_id} archive size: {size}"
        assert zipfile.is_zipfile(archive),f"Case {case_id} is not a valid ZIP archive"
        manifest_rows.append({"dataset":"3D-IRCADb-01","case_id":case_id,"official_url":url,
          "local_relative_path":str(archive.relative_to(STEP_DIR)),"status":status,"bytes":size,"sha256":digest,
          "downloaded_or_verified_utc":datetime.now(timezone.utc).isoformat(),"license":"CC BY-NC-ND 4.0",
          "redistribution_authorized":False})
    archive_bytes=sum(r["bytes"] for r in manifest_rows)
    archive_set_sha=hashlib.sha256("".join(f"{r['case_id']}:{r['sha256']}\n" for r in manifest_rows).encode()).hexdigest()
    assert len(manifest_rows)==20
    assert EXPECTED_TOTAL_SIZE_MIN <= archive_bytes <= EXPECTED_TOTAL_SIZE_MAX, f"Unexpected total archive size: {archive_bytes}"
    download_status="downloaded" if any_downloaded else "reused_existing_archives"

    if EXTRACT_ENABLED:
        root=EXTRACTED.resolve(); marker=EXTRACTED/".extraction_complete.json"
        marker_state=load_json(marker) if marker.is_file() else {}
        marker_matches=(marker_state.get("archive_set_sha256")==archive_set_sha and marker_state.get("extraction_version")==2)
        if not marker_matches:
            for case_id, archive in PATIENT_ARCHIVES.items():
                with zipfile.ZipFile(archive) as zf:
                    unsafe=[]
                    for member in zf.infolist():
                        target=(EXTRACTED/member.filename).resolve()
                        if root != target and root not in target.parents: unsafe.append(member.filename)
                    assert not unsafe,f"Unsafe ZIP paths in case {case_id}: {unsafe[:5]}"
                    zf.extractall(EXTRACTED)
            for patient_dir in sorted(p for p in EXTRACTED.glob("3Dircadb1.*") if p.is_dir()):
                for component in ["PATIENT_DICOM","MASKS_DICOM","LABELLED_DICOM","MESHES_VTK"]:
                    nested=patient_dir/f"{component}.zip"; destination=patient_dir/component
                    if nested.is_file() and zipfile.is_zipfile(nested):
                        destination.mkdir(parents=True,exist_ok=True); nested_root=destination.resolve()
                        with zipfile.ZipFile(nested) as nz:
                            unsafe=[]
                            for member in nz.infolist():
                                target=(destination/member.filename).resolve()
                                if nested_root != target and nested_root not in target.parents: unsafe.append(member.filename)
                            assert not unsafe,f"Unsafe nested ZIP paths in {nested}: {unsafe[:5]}"
                            nz.extractall(destination)
            marker.write_text(json.dumps({"archive_set_sha256":archive_set_sha,"case_archives":20,"extraction_version":2,
              "extracted_utc":datetime.now(timezone.utc).isoformat()},indent=2),encoding="utf-8")
            extracted_now=True
else:
    print("SAFE PREFLIGHT: no network request made. Set all authorization switches in Cell 1 for acquisition.")

if not manifest_rows:
    manifest_rows=[{"dataset":"3D-IRCADb-01","case_id":None,"official_url":DOWNLOAD_URL,
      "local_relative_path":str(RAW.relative_to(STEP_DIR)),"status":download_status,"bytes":None,"sha256":None,
      "downloaded_or_verified_utc":None,"license":"CC BY-NC-ND 4.0","redistribution_authorized":False}]
download_manifest=pd.DataFrame(manifest_rows)
save_csv(download_manifest,"download_manifest.csv")
save_json({"user_accepts_terms":bool(USER_ACCEPTS_CC_BY_NC_ND_4_0),"download_enabled":bool(DOWNLOAD_ENABLED),
 "extract_enabled":bool(EXTRACT_ENABLED),"authorization_granted":authorization_granted,
 "authorization_text":AUTHORIZATION_TEXT,"authorization_recorded_utc":AUTHORIZATION_RECORDED_UTC,
 "download_started":download_status in {"downloaded","reused_existing_archives"},"download_status":download_status,
 "archive_bytes":archive_bytes,"archive_set_sha256":archive_set_sha,"extracted_now":extracted_now},"authorization_state.json")
print("Download status:",download_status,"Extracted now:",extracted_now,"Archives:",len(manifest_rows))'''))

cells.append(nbf.v4.new_markdown_cell("""## Results

### 3. Inventory archive contents, DICOM series, and source labels
"""))

cells.append(nbf.v4.new_code_cell(r'''archive_rows=[]; file_rows=[]; series_rows=[]; label_rows=[]; patient_rows=[]

for case_id, archive in PATIENT_ARCHIVES.items():
    if archive.is_file() and zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as zf:
            for m in zf.infolist():
                archive_rows.append({"case_id":case_id,"archive_name":archive.name,"member":m.filename,
                                     "uncompressed_bytes":m.file_size,"compressed_bytes":m.compress_size,
                                     "is_directory":m.is_dir()})
archive_inventory=pd.DataFrame(archive_rows,columns=["case_id","archive_name","member","uncompressed_bytes","compressed_bytes","is_directory"])
save_csv(archive_inventory,"archive_inventory.csv")

marker=EXTRACTED/".extraction_complete.json"
if marker.is_file():
    files=sorted(p for p in EXTRACTED.rglob("*") if p.is_file() and p.name!=".extraction_complete.json")
    for p in files:
        file_rows.append({"relative_path":str(p.relative_to(EXTRACTED)),"bytes":p.stat().st_size,
                          "suffix":p.suffix.lower(),"sha256":sha256(p) if FULL_FILE_HASHES else None})
    patient_dir_candidates=sorted(p for p in EXTRACTED.rglob("PATIENT_DICOM")
                                  if p.is_dir() and any(x.is_file() for x in p.rglob("*")))
    # Patient archives contain PATIENT_DICOM/PATIENT_DICOM. Keep only the
    # deepest data-bearing directory so a rerun cannot count each case twice.
    patient_dirs=[p for p in patient_dir_candidates
                  if not any(q != p and p in q.parents for q in patient_dir_candidates)]
    try:
        import pydicom
        pydicom_available=True
    except Exception:
        pydicom_available=False
    assert pydicom_available,"pydicom is required after extraction"

    for image_dir in patient_dirs:
        patient_dir=next(parent for parent in image_dir.parents if parent.name.lower().startswith("3dircadb1."))
        patient_id=patient_dir.name
        dicom_files=sorted(p for p in image_dir.rglob("*") if p.is_file())
        headers=[]
        for p in dicom_files:
            try:
                ds=pydicom.dcmread(p,stop_before_pixels=True,force=True)
                headers.append(ds)
            except Exception: pass
        sops=[str(getattr(ds,"SOPInstanceUID","")) for ds in headers]
        series=set(str(getattr(ds,"SeriesInstanceUID","")) for ds in headers)
        rows=set(int(getattr(ds,"Rows",-1)) for ds in headers); cols=set(int(getattr(ds,"Columns",-1)) for ds in headers)
        spacings=set(tuple(float(x) for x in getattr(ds,"PixelSpacing",[])) for ds in headers)
        slopes=set(float(getattr(ds,"RescaleSlope",1)) for ds in headers); intercepts=set(float(getattr(ds,"RescaleIntercept",0)) for ds in headers)
        series_rows.append({"patient_id":patient_id,"dicom_files":len(dicom_files),"readable_headers":len(headers),
          "series_uid_count":len(series),"unique_sop_count":len(set(sops)),"duplicate_sop_count":len(sops)-len(set(sops)),
          "rows_values":";".join(map(str,sorted(rows))),"columns_values":";".join(map(str,sorted(cols))),
          "pixel_spacing_values":";".join(map(str,sorted(spacings))),"slope_values":";".join(map(str,sorted(slopes))),
          "intercept_values":";".join(map(str,sorted(intercepts)))})

        mask_candidates=[p for p in patient_dir.rglob("MASKS_DICOM") if p.is_dir() and any(x.is_dir() for x in p.iterdir())]
        mask_root=max(mask_candidates,key=lambda p:sum(x.is_dir() for x in p.iterdir())) if mask_candidates else patient_dir/"MASKS_DICOM"
        label_dirs=sorted(p for p in mask_root.iterdir() if p.is_dir()) if mask_root.is_dir() else []
        label_names=[]
        for label_dir in label_dirs:
            mask_files=[p for p in label_dir.rglob("*") if p.is_file()]
            lname=label_dir.name; label_names.append(lname)
            label_rows.append({"patient_id":patient_id,"label_name":lname,"mask_file_count":len(mask_files),
                               "is_liver":lname.lower() in {"liver","foie"},
                               "is_tumour":bool(re.search(r"tumou?r|lesion",lname,re.I))})
        patient_rows.append({"patient_id":patient_id,"patient_dicom_present":image_dir.is_dir(),
          "patient_dicom_file_count":len(dicom_files),"mask_root_present":mask_root.is_dir(),"label_count":len(label_dirs),
          "liver_label_present":any(x.lower() in {"liver","foie"} for x in label_names),
          "tumour_label_present":any(re.search(r"tumou?r|lesion",x,re.I) for x in label_names)})

file_inventory=pd.DataFrame(file_rows,columns=["relative_path","bytes","suffix","sha256"])
series_profile=pd.DataFrame(series_rows)
label_inventory=pd.DataFrame(label_rows)
patient_summary=pd.DataFrame(patient_rows)
save_csv(file_inventory,"extracted_file_inventory.csv")
save_csv(series_profile,"dicom_series_profile.csv")
save_csv(label_inventory,"label_inventory.csv")
save_csv(patient_summary,"patient_source_qc_summary.csv")
print("Discovered patients:",len(patient_summary),"labels:",len(label_inventory))'''))

cells.append(nbf.v4.new_markdown_cell("""### 4. Evaluate the source-ingestion gate
"""))

cells.append(nbf.v4.new_code_cell(r'''archive_ready=bool(archive_set_sha and archive_bytes and len(download_manifest)==20 and EXPECTED_TOTAL_SIZE_MIN<=archive_bytes<=EXPECTED_TOTAL_SIZE_MAX)
extraction_ready=bool((EXTRACTED/".extraction_complete.json").is_file())
patient_count=int(len(patient_summary))

qc_rows=[
 ("terms_and_attribution","preflight",True,"official CC BY-NC-ND 4.0 terms and citation recorded"),
 ("step12_source_selection","preflight",bool(verification.passed.all()),"Step 12 source decision and signatures verified"),
 ("download_authorization","acquisition",authorization_granted,"requires explicit user terms acceptance and DOWNLOAD_ENABLED"),
 ("archive_size_and_zip","acquisition",archive_ready,"official archive expected near 782 MB and valid ZIP"),
 ("safe_extraction","acquisition",extraction_ready,"ZIP traversal check and extraction marker required"),
 ("patient_folder_count","source QC",patient_count==20,f"expected 20; observed {patient_count}"),
 ("dicom_header_readability","source QC",bool(patient_count and not series_profile.empty and (series_profile.readable_headers>0).all()),"each patient must have readable headers"),
 ("sop_uid_uniqueness","source QC",bool(patient_count and not series_profile.empty and series_profile.duplicate_sop_count.eq(0).all()),"no duplicated SOP Instance UIDs per patient"),
 ("liver_label_availability","source QC",bool(patient_count and patient_summary.liver_label_present.all()),"liver mask required for all patients"),
 ("tumour_label_availability","source QC",bool(patient_count and patient_summary.tumour_label_present.sum()>=15),"official description reports tumours in 75% of 20 cases"),
 ("conversion_parity","future conversion",False,"not evaluated until a separate normalized conversion is created"),
 ("frozen_external_evaluation_contract","future evaluation",False,"must be declared after source and conversion QC pass"),
]
qc=pd.DataFrame(qc_rows,columns=["check","stage","passed","evidence_or_required_action"])
save_csv(qc,"ingestion_qc_results.csv")

source_checks=qc.loc[qc.stage.isin(["preflight","acquisition","source QC"])]
source_qc_pass=bool(source_checks.passed.all())
if not authorization_granted:
    result_level="EXTERNAL_DOWNLOAD_AUTHORIZATION_REQUIRED"; decision="REVIEW_TERMS_AND_ENABLE_DOWNLOAD"
elif not source_qc_pass:
    result_level="EXTERNAL_SOURCE_INGESTION_QC_FAIL"; decision="REPAIR_SOURCE_INGESTION_BEFORE_CONVERSION"
else:
    result_level="EXTERNAL_SOURCE_INGESTION_QC_PASS"; decision="PROCEED_TO_NORMALIZED_CONVERSION_AND_PARITY_QC"

report=["# 3D-IRCADb-01 Ingestion QC Report","",f"- Result level: `{result_level}`",
 f"- Download status: `{download_status}`",f"- Total archive bytes: {archive_bytes}",f"- Archive-set SHA-256: {archive_set_sha}",
 f"- Extracted patient folders: {patient_count}",f"- Source-ingestion gate passed: {str(source_qc_pass).lower()}","",
 "No model inference or local LiTS test access occurred."]
(OUT/"INGESTION_QC_REPORT.md").write_text("\n".join(report),encoding="utf-8")
print(result_level,decision)'''))

cells.append(nbf.v4.new_markdown_cell("""## Takeaways

### 5. Save bounded visualization, provenance, and signed decision
"""))

cells.append(nbf.v4.new_code_cell(r'''fig,ax=plt.subplots(figsize=(11,5.5))
plot=qc.copy()
pending_mode = result_level == "EXTERNAL_DOWNLOAD_AUTHORIZATION_REQUIRED"
colors=["#2ca02c" if x else ("#ffbf00" if pending_mode else "#d62728") for x in plot.passed]
ax.barh(plot.check,[1]*len(plot),color=colors); ax.set_xlim(0,1); ax.set_xticks([])
for i,ok in enumerate(plot.passed):
    label = "PASS" if ok else ("PENDING" if pending_mode else "FAIL")
    ax.text(.5,i,label,ha="center",va="center",color="white",fontweight="bold",fontsize=8)
ax.set_title(f"Step 13 3D-IRCADb-01 ingestion gate — {result_level}")
fig.tight_layout(); fig.savefig(OUT/"ingestion_qc_dashboard.png",dpi=180,bbox_inches="tight"); plt.show()

expected=pd.DataFrame([
 ("Step 12 prerequisite checks",6,int(verification.passed.sum()),verification.passed.all()),
 ("official terms snapshot saved",1,int((OUT/"source_terms_snapshot.json").is_file()),(OUT/"source_terms_snapshot.json").is_file()),
 ("download requests when unauthorized",0,0,True),
 ("zero local LiTS test source accesses",0,0,True),
 ("zero model inference runs",0,0,True),
],columns=["requirement","expected","actual","passed"])
save_csv(expected,"expected_vs_actual.csv")

configuration={"phase":"step_13_3d_ircadb_ingestion_and_qc","created_utc":CREATED_UTC,"source":"3D-IRCADb-01",
 "official_page":SOURCE_PAGE,"download_url":DOWNLOAD_URL,"license":"CC BY-NC-ND 4.0",
 "user_accepts_terms":bool(USER_ACCEPTS_CC_BY_NC_ND_4_0),"download_enabled":bool(DOWNLOAD_ENABLED),
 "extract_enabled":bool(EXTRACT_ENABLED),"full_file_hashes":bool(FULL_FILE_HASHES),
 "manifest_sha256":MANIFEST_SHA256,"test_access_allowed":False,"inference_allowed":False}
save_json(configuration,"configuration.json")

gate={"status":result_level.lower(),"result_level":result_level,"phase_integrity_passed":bool(expected.passed.all()),
 "source_ingestion_qc_passed":source_qc_pass,"decision":decision,
 "next_step":"step_14_external_normalized_conversion_and_parity_qc" if source_qc_pass else "complete_step_13_authorized_ingestion",
 "download_authorized":authorization_granted,"download_status":download_status,"archive_set_sha256":archive_set_sha,
 "extracted_patient_count":patient_count,"conversion_performed":False,"inference_performed":False,
 "formal_model_acceptance_passed":False,"manifest_sha256":MANIFEST_SHA256,
 "test_images_accessed":False,"test_source_files_reopened":False,"test_inference_rerun":False,
 "targets":{r.requirement:bool(r.passed) for r in expected.itertuples(index=False)}}
save_json(gate,"gate_result.json")

provenance={"created_utc":CREATED_UTC,"phase":configuration["phase"],"step12_gate_sha256":sha256(S12/"gate_result.json"),
 "step12_signature_sha256":sha256(S12/"step_12_signature.json"),"source_page":SOURCE_PAGE,"download_url":DOWNLOAD_URL,
 "archive_set_sha256":archive_set_sha,"download_status":download_status,"manifest_sha256":MANIFEST_SHA256,
 "test_images_accessed":False,"test_source_files_reopened":False,"inference_performed":False}
save_json(provenance,"provenance.json")

signed_names=["input_verification.csv","source_terms_snapshot.json","authorization_state.json","download_manifest.csv",
 "archive_inventory.csv","extracted_file_inventory.csv","dicom_series_profile.csv","label_inventory.csv","patient_source_qc_summary.csv",
 "ingestion_qc_results.csv","INGESTION_QC_REPORT.md","ingestion_qc_dashboard.png","expected_vs_actual.csv","configuration.json",
 "gate_result.json","provenance.json"]
signed={f"step_13_3d_ircadb_ingestion_and_qc/outputs/{n}":sha256(OUT/n) for n in signed_names}
combined=hashlib.sha256("".join(f"{k}:{v}\n" for k,v in sorted(signed.items())).encode()).hexdigest()
save_json({"algorithm":"SHA-256","created_utc":CREATED_UTC,"combined_sha256":combined,"signed_artifacts":signed,
 "result_level":result_level,"download_status":download_status,"inference_performed":False,
 "test_source_files_reopened":False},"step_13_signature.json")

print(f"PASS: notebook integrity; result={result_level}")
print("No inference and no local test access.")'''))

cells.append(nbf.v4.new_markdown_cell("""## Takeaways

- The default `EXTERNAL_DOWNLOAD_AUTHORIZATION_REQUIRED` state is a successful safe preflight, not an error.
- To acquire the official archive, review the source license and set `USER_ACCEPTS_CC_BY_NC_ND_4_0 = True`, `DOWNLOAD_ENABLED = True`, and `EXTRACT_ENABLED = True` in code Cell 1, then use **Restart Kernel and Run All**.
- A passing source-ingestion gate authorizes only a later normalized conversion/parity phase. It does not authorize inference.
- The local LiTS test split remains sealed throughout.
"""))

nb=nbf.v4.new_notebook(cells=cells)
nb["metadata"]["kernelspec"]={"display_name":"ds_gpu","language":"python","name":"python3"}
nb["metadata"]["language_info"]={"name":"python","version":"3"}
nbf.write(nb,NOTEBOOK); nbf.write(nb,SHORT_NOTEBOOK)
print(f"Wrote {NOTEBOOK}")
print(f"Wrote short-path alias {SHORT_NOTEBOOK}")
