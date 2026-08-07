# Maximum-Probability Fusion of Two MobileNetV2 U-Nets for Liver-Tumour Segmentation: Frozen Held-Out and External Evaluation

## Abstract

**Background:** Liver-tumour segmentation is vulnerable to small-lesion and patient-level failure. **Methods:** A predicted-liver ROI preceded two frozen MobileNetV2 U-Net tumour models. Pixelwise maximum fusion, threshold 0.70, and no post-processing were fixed before one-time evaluations on a 14-volume LiTS held-out cohort and the complete 20-case 3D-IRCADb-01 cohort. **Results:** LiTS global Dice was 0.7677; mean positive-patient Dice was 0.5073, but minimum Dice was 0, failing the predeclared catastrophic-patient guardrail. External global Dice was 0.8480; mean positive-patient Dice was 0.7112 (95% bootstrap CI 0.5678–0.8278), median 0.8355, and minimum 0.0130. External smallest-lesion detection was 63.64%. All five external negative controls had predictions under the frozen hepatic-tumour truth; source-label concordance explained much of the two largest cases but did not change truth or metrics. **Conclusion:** The frozen method showed external generalization evidence and strong aggregate overlap, while persistent case-level and small-lesion failures preclude clinical-readiness claims. The LiTS formal acceptance result remains failed.

## 1. Introduction

This study evaluates whether a validation-selected maximum-probability fusion policy transfers under predeclared one-time held-out and external evaluations, while explicitly preserving patient-level failure evidence.

## 2. Materials and methods

### 2.1 Governance and cohorts

The corrected LiTS build used manifest SHA-256 `575a6fc391d63dc9bbbbb3317efe4d0f65edd57457ae63c1bfc6c2c615861889` with patient-disjoint splits. The one-time LiTS held-out evaluation included 14 volumes (13 tumour-positive). A separately frozen external contract included all 20 3D-IRCADb-01 cases (15 accepted hepatic-tumour positive and five negative controls), with zero exclusions. Neither evaluation was rerun or tuned after results were observed.

### 2.2 Pipeline and outcomes

The frozen pipeline used broad-window `[-160,240]` inputs at 256×256, a predicted-liver ROI, control and recall-loss tumour checkpoints, `max(p_control,p_recall)` fusion, threshold 0.70 and no post-processing. Outcomes included global overlap, patient Dice, train-derived small-lesion strata, empty predictions, false-positive behavior and 10,000-resample patient bootstrap uncertainty.

### 2.3 External label semantics

External hepatic truth used source folders matching `^livertumors?\d*$`. Generic `tumor`, adrenal/surrenal tumour and `metastasectomie` annotations were excluded by the frozen contract. A later read-only concordance audit measured overlap with those annotations without relabeling cases.

## 3. Results

### 3.1 LiTS held-out evaluation failed its patient guardrail

LiTS global Dice was 0.7677, precision 0.8444, recall 0.7038, and mean positive-patient Dice 0.5073. Minimum positive-patient Dice was 0 (V121), so formal model acceptance failed despite strong aggregate performance.

### 3.2 External evaluation passed its separate frozen contract with caveats

External global Dice was 0.8480, precision 0.8237, recall 0.8737, and mean positive-patient Dice 0.7112 (95% CI 0.5678–0.8278). Median Dice was 0.8355; minimum was 0.0130 for `ircadb_18`. Smallest-lesion detection was 63.64% (14/22).

### 3.3 Negative-control predictions expose label and specificity limits

All five external negative controls contained predictions totaling 52.936 ml on the standardized grid. In cases 7 and 14, 86.39% and 88.93% of predicted pixels overlapped excluded `tumor` and `metastasectomie` masks. These remain false positives under frozen truth; biological interpretation requires expert review.

## 4. Discussion

The external pass supplies evidence that aggregate performance transfers beyond the corrected LiTS build, but it does not reverse the failed LiTS patient guardrail. Across both cohorts, small lesions and individual cases remain the dominant limitations. Descriptive differences between cohorts must not be interpreted as external superiority because acquisition, prevalence and label semantics differ.

## 5. Limitations

Both cohorts are small public datasets. The method is two-dimensional, probabilities are not clinically calibrated, negative-control source labels are not expert adjudication, and no prospective or institutional external study was performed. Owner declarations, exact 3D-IRCADb-01 scholarly citation, venue formatting, and submission authorization remain unresolved.

## 6. Conclusion

The frozen pipeline achieved strong aggregate overlap and passed a separately frozen public external evaluation, but failed the LiTS catastrophic-patient acceptance guardrail. The defensible conclusion is external generalization evidence with material reliability caveats—not clinical readiness.

## Submission blockers

The Step 11 owner/declaration gate remains closed. No submission or payment action is authorized.
