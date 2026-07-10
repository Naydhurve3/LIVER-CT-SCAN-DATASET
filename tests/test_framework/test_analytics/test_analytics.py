import numpy as np
import torch

from src.framework.analytics.tumor_burden import TumorBurdenAnalyzer
from src.framework.analytics.patient_profile import PatientProfileBuilder
from src.framework.analytics.clinical_insights import ClinicalInsightEngine
from src.framework.analytics.report_generator import ReportGenerator
from src.framework.analytics.quality_checks import DataQualityChecker


def test_tumor_volume():
    analyzer = TumorBurdenAnalyzer(voxel_spacing=(1.0, 1.0, 1.0))
    slices = [np.zeros((64, 64), dtype=np.int32) for _ in range(10)]
    slices[3][20:40, 20:40] = 2
    vol = analyzer.tumor_volume(slices)
    assert vol > 0


def test_tumor_surface_area():
    analyzer = TumorBurdenAnalyzer()
    mask_3d = np.zeros((10, 64, 64), dtype=np.int32)
    mask_3d[3:7, 20:40, 20:40] = 2
    sa = analyzer.tumor_surface_area(mask_3d)
    assert sa > 0


def test_tumor_location_found():
    analyzer = TumorBurdenAnalyzer()
    slices = [np.zeros((64, 64), dtype=np.int32) for _ in range(10)]
    slices[2][30:40, 30:40] = 2
    loc = analyzer.tumor_location(slices)
    assert isinstance(loc, str)
    assert loc != "none"


def test_tumor_location_none():
    analyzer = TumorBurdenAnalyzer()
    slices = [np.zeros((64, 64), dtype=np.int32) for _ in range(10)]
    loc = analyzer.tumor_location(slices)
    assert loc == "none"


def test_summary():
    analyzer = TumorBurdenAnalyzer()
    slices = [np.zeros((64, 64), dtype=np.int32) for _ in range(10)]
    slices[3][20:40, 20:40] = 2
    summary = analyzer.summary(slices)
    assert "tumor_volume_mm3" in summary
    assert "num_tumor_slices" in summary
    assert "location" in summary


def test_per_slice_tumor_burden():
    analyzer = TumorBurdenAnalyzer()
    slices = [np.zeros((64, 64), dtype=np.int32) for _ in range(5)]
    slices[2][10:20, 10:20] = 2
    burden = analyzer.per_slice_tumor_burden(slices)
    assert len(burden) == 5
    assert burden[2] > 0
    assert all(b == 0 for i, b in enumerate(burden) if i != 2)


def test_patient_profile_builder():
    builder = PatientProfileBuilder()
    preds = [torch.zeros(64, 64) for _ in range(5)]
    preds[2][10:20, 10:20] = 1.0
    masks = [torch.zeros(64, 64) for _ in range(5)]
    masks[2][10:20, 10:20] = 2
    profile = builder.build_from_slices(1, preds, masks)
    assert profile["volume_id"] == 1
    assert profile["tumor_slices"] >= 1


def test_cohort_summary():
    builder = PatientProfileBuilder()
    for vid in range(3):
        preds = [torch.zeros(64, 64) for _ in range(5)]
        preds[2][10:20, 10:20] = 1.0
        masks = [torch.zeros(64, 64) for _ in range(5)]
        masks[2][10:20, 10:20] = 2
        builder.build_from_slices(vid, preds, masks)
    cohort = builder.cohort_summary()
    assert cohort["num_patients"] == 3


def test_clinical_insights_analyze_burden():
    engine = ClinicalInsightEngine()
    result = engine.analyze_tumor_burden(5000)
    assert "burden_category" in result
    assert "risk_level" in result


def test_clinical_insights_morphology():
    engine = ClinicalInsightEngine()
    assert engine.characterize_tumor_morphology(0.9) == "round"
    assert engine.characterize_tumor_morphology(0.6) == "irregular"
    assert engine.characterize_tumor_morphology(0.3) == "complex"


def test_clinical_insights_uncertainty_risk():
    engine = ClinicalInsightEngine()
    uncertainty = torch.tensor([[0.05, 0.05], [0.05, 0.05]])
    risk = engine.assess_uncertainty_risk(uncertainty)
    assert "risk_level" in risk


def test_clinical_insights_generate_summary():
    engine = ClinicalInsightEngine()
    summary = engine.generate_summary(
        {"burden_category": "small", "risk_level": "low"},
        {"risk_level": "low"},
        "round",
    )
    assert "Tumor burden" in summary
    assert "Morphology" in summary


def test_report_dataset_report(tmp_path):
    rg = ReportGenerator(output_dir=str(tmp_path))
    rg.generate_dataset_report({
        "total_volumes": 131, "tumor_slices_pct": 12.14,
    })
    assert (tmp_path / "dataset_report.txt").exists()
    assert (tmp_path / "dataset_report.txt").exists()


def test_report_model_report(tmp_path):
    rg = ReportGenerator(output_dir=str(tmp_path))
    path = rg.generate_model_report({"dice": 0.85, "iou": 0.74})
    assert "model_report" in path


def test_report_html(tmp_path):
    rg = ReportGenerator(output_dir=str(tmp_path))
    path = rg.generate_html_report("Test", {"Section": "Content"})
    assert (tmp_path / "report.html").exists()


def test_report_json(tmp_path):
    rg = ReportGenerator(output_dir=str(tmp_path))
    path = rg.generate_json_report({"key": "value"})
    assert (tmp_path / "report.json").exists()


def test_quality_image_integrity():
    qc = DataQualityChecker()
    images = [np.zeros((256, 256), dtype=np.float32) for _ in range(3)]
    result = qc.check_image_integrity(images)
    assert result["pass"] is True


def test_quality_mask_integrity():
    qc = DataQualityChecker()
    masks = [np.zeros((256, 256), dtype=np.int32) for _ in range(3)]
    result = qc.check_mask_integrity(masks)
    assert result["pass"] is True


def test_quality_split_balance():
    qc = DataQualityChecker()
    result = qc.check_split_balance({"train": [1, 2, 3], "val": [4, 5]})
    assert "train" in result


def test_quality_intensity_drift():
    qc = DataQualityChecker()
    result = qc.check_intensity_drift([0.1, 0.12, 0.11])
    assert "pass" in result


def test_quality_summary_report():
    qc = DataQualityChecker()
    result = qc.summary_report({"integrity": {"pass": True}})
    assert "PASS" in result
