class ClinicalInsightEngine:
    def __init__(self):
        self.insights = []

    def analyze_tumor_burden(self, tumor_volume_mm3, body_surface_area=1.8):
        insights = {}
        if tumor_volume_mm3 < 1000:
            insights["burden_category"] = "small"
            insights["risk_level"] = "low"
        elif tumor_volume_mm3 < 10000:
            insights["burden_category"] = "moderate"
            insights["risk_level"] = "medium"
        else:
            insights["burden_category"] = "large"
            insights["risk_level"] = "high"
        insights["tumor_volume"] = tumor_volume_mm3
        return insights

    def assess_uncertainty_risk(self, uncertainty_map, thresholds=(0.1, 0.3)):
        high_uncertainty = (uncertainty_map > thresholds[1]).float().mean().item()
        medium_uncertainty = ((uncertainty_map > thresholds[0]) & (uncertainty_map <= thresholds[1])).float().mean().item()
        if high_uncertainty > 0.3:
            risk = "high"
        elif high_uncertainty > 0.1 or medium_uncertainty > 0.3:
            risk = "medium"
        else:
            risk = "low"
        return {
            "risk_level": risk,
            "high_uncertainty_fraction": high_uncertainty,
            "medium_uncertainty_fraction": medium_uncertainty,
        }

    def characterize_tumor_morphology(self, sphericity):
        if sphericity > 0.8:
            return "round"
        elif sphericity > 0.5:
            return "irregular"
        return "complex"

    def generate_summary(self, tumor_burden, uncertainty_assessment, morphology):
        self.insights.append({
            "tumor_burden": tumor_burden,
            "uncertainty": uncertainty_assessment,
            "morphology": morphology,
        })
        lines = []
        lines.append(f"Tumor burden: {tumor_burden.get('burden_category', 'unknown')} "
                      f"({tumor_burden.get('risk_level', 'unknown')} risk)")
        lines.append(f"Morphology: {morphology}")
        lines.append(f"Uncertainty risk: {uncertainty_assessment.get('risk_level', 'unknown')}")
        return "\n".join(lines)
