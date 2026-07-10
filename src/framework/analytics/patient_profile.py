from collections import defaultdict

import numpy as np


class PatientProfileBuilder:
    def __init__(self):
        self.profiles = {}

    def build_from_slices(self, volume_id, slice_predictions, slice_masks, slice_uncertainties=None):
        tumor_areas = []
        tumor_uncertainties = []
        for i, pred in enumerate(slice_predictions):
            tumor_pixels = pred.sum()
            tumor_areas.append(tumor_pixels)
            if slice_uncertainties is not None:
                tumor_uncertainties.append(
                    slice_uncertainties[i][pred > 0.5].mean()
                    if pred.sum() > 0 else 0.0
                )
        profile = {
            "volume_id": volume_id,
            "num_slices": len(slice_predictions),
            "total_tumor_pixels": sum(tumor_areas),
            "tumor_slices": sum(1 for a in tumor_areas if a > 0),
            "mean_tumor_per_slice": float(np.mean(tumor_areas)) if tumor_areas else 0.0,
        }
        if tumor_uncertainties:
            profile["mean_uncertainty"] = float(np.mean(tumor_uncertainties))
        self.profiles[volume_id] = profile
        return profile

    def cohort_summary(self, profile_ids=None):
        profiles = [self.profiles[k] for k in (profile_ids or self.profiles)]
        if not profiles:
            return {}
        return {
            "num_patients": len(profiles),
            "mean_tumor_pixels": float(np.mean([p["total_tumor_pixels"] for p in profiles])),
            "std_tumor_pixels": float(np.std([p["total_tumor_pixels"] for p in profiles])),
            "mean_tumor_slices": float(np.mean([p["tumor_slices"] for p in profiles])),
            "total_tumor_pixels": sum(p["total_tumor_pixels"] for p in profiles),
        }
