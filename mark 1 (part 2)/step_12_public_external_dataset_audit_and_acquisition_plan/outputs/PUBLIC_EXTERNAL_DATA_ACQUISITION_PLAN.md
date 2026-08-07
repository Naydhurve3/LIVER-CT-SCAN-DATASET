# Public External-Data Acquisition Plan

## Decision

Start with **3D-IRCADb-01** as a small, independent, tumour-labelled feasibility cohort. Do not download it until the official reuse terms and citation requirements are recorded. If its conversion and label-QC gates pass, freeze a zero-tuning external evaluation contract before inference.

Use **HCC-TACE-Seg** second for a larger and harder domain-shift study. Its 28.57 GB DICOM/DICOM-SEG package, longitudinal phase selection and the documented HCC_001 dimension mismatch require a dedicated ingestion phase.

Do **not** use **MSD Task03 Liver** as independent external validation because the MSD publication identifies it as a subset of LiTS patients. It may be used only for format interoperability checks with an explicit overlap label.

Use **CHAOS CT** only for liver-ROI/domain-shift analysis because its CT subjects are healthy donors without tumours and its challenge test ground truth is withheld.

## Next authorized phase

Create a separate Step 13 ingestion-and-QC notebook only after the user chooses a source and confirms that its official terms are acceptable. That phase may download or inspect the chosen external source, but it must not reopen the sealed local LiTS test split.
