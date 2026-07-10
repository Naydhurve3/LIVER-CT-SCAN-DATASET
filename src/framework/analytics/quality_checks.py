import numpy as np


class DataQualityChecker:
    def check_image_integrity(self, images, expected_shape=(256, 256)):
        issues = []
        for i, img in enumerate(images):
            if img.shape != expected_shape:
                issues.append(f"Image {i}: shape {img.shape} != {expected_shape}")
            if img.dtype != np.float32:
                issues.append(f"Image {i}: dtype {img.dtype} != float32")
            if np.isnan(img).any():
                issues.append(f"Image {i}: contains NaN")
            if np.isinf(img).any():
                issues.append(f"Image {i}: contains Inf")
        return {
            "total": len(images),
            "issues": issues,
            "pass": len(issues) == 0,
        }

    def check_mask_integrity(self, masks, expected_values=(0, 1)):
        issues = []
        for i, mask in enumerate(masks):
            unique = np.unique(mask)
            if not all(v in expected_values for v in unique):
                issues.append(f"Mask {i}: unexpected values {unique}")
            if mask.shape != masks[0].shape:
                issues.append(f"Mask {i}: shape {mask.shape} != {masks[0].shape}")
        return {
            "total": len(masks),
            "issues": issues,
            "pass": len(issues) == 0,
        }

    def check_split_balance(self, splits):
        total = sum(len(v) for v in splits.values())
        return {
            k: {"count": len(v), "pct": f"{len(v)/total*100:.1f}%"}
            for k, v in splits.items()
        }

    def check_intensity_drift(self, batch_means, threshold=0.1):
        if len(batch_means) < 2:
            return {"pass": True, "message": "Insufficient data for drift detection"}
        trend = np.polyfit(range(len(batch_means)), batch_means, 1)[0]
        return {
            "pass": abs(trend) < threshold,
            "trend": float(trend),
            "mean": float(np.mean(batch_means)),
            "std": float(np.std(batch_means)),
        }

    def summary_report(self, checks):
        results = []
        for name, result in checks.items():
            status = "PASS" if result.get("pass", True) else "FAIL"
            results.append(f"[{status}] {name}")
        return "\n".join(results)
