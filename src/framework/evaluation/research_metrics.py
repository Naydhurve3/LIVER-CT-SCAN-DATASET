from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
from scipy.integrate import trapezoid

from src.framework.evaluation.surface_metrics import compute_all_surface_metrics


def confusion_counts(pred: np.ndarray, target: np.ndarray) -> Tuple[int, int, int, int]:
    pred = np.asarray(pred, dtype=bool)
    target = np.asarray(target, dtype=bool)
    tp = int(np.logical_and(pred, target).sum())
    fp = int(np.logical_and(pred, ~target).sum())
    fn = int(np.logical_and(~pred, target).sum())
    tn = int(np.logical_and(~pred, ~target).sum())
    return tp, fp, fn, tn


def dice_from_counts(tp: int, fp: int, fn: int) -> float:
    denominator = 2 * tp + fp + fn
    return 1.0 if denominator == 0 else (2.0 * tp) / denominator


def iou_from_counts(tp: int, fp: int, fn: int) -> float:
    denominator = tp + fp + fn
    return 1.0 if denominator == 0 else tp / denominator


def positive_slice_dice(pred_binary: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    positive = target.sum(dim=(1, 2, 3)) > 0
    if not positive.any():
        return torch.tensor(float("nan"), device=target.device)
    pred = pred_binary[positive]
    truth = target[positive]
    intersection = (pred * truth).sum(dim=(1, 2, 3))
    denominator = pred.sum(dim=(1, 2, 3)) + truth.sum(dim=(1, 2, 3))
    return ((2 * intersection) / denominator.clamp_min(1)).mean()


def tumor_only_dice(pred_binary: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    warnings.warn(
        "tumor_only_dice is deprecated; use positive_slice_dice",
        DeprecationWarning,
        stacklevel=2,
    )
    return positive_slice_dice(pred_binary, target)


@dataclass
class StreamingProbabilityMetrics:
    probability_bins: int = 4096
    calibration_bins: int = 20
    pos_hist: np.ndarray = field(init=False)
    neg_hist: np.ndarray = field(init=False)
    cal_count: np.ndarray = field(init=False)
    cal_prob: np.ndarray = field(init=False)
    cal_target: np.ndarray = field(init=False)
    brier_sum: float = 0.0
    pixel_count: int = 0

    def __post_init__(self) -> None:
        self.pos_hist = np.zeros(self.probability_bins, dtype=np.int64)
        self.neg_hist = np.zeros(self.probability_bins, dtype=np.int64)
        self.cal_count = np.zeros(self.calibration_bins, dtype=np.int64)
        self.cal_prob = np.zeros(self.calibration_bins, dtype=np.float64)
        self.cal_target = np.zeros(self.calibration_bins, dtype=np.float64)

    def update(self, probs: np.ndarray, targets: np.ndarray) -> None:
        p = np.asarray(probs, dtype=np.float32).reshape(-1)
        y = np.asarray(targets, dtype=np.uint8).reshape(-1)
        p = np.clip(p, 0.0, 1.0)
        probability_idx = np.minimum((p * self.probability_bins).astype(np.int64),
                                     self.probability_bins - 1)
        self.pos_hist += np.bincount(probability_idx[y == 1], minlength=self.probability_bins)
        self.neg_hist += np.bincount(probability_idx[y == 0], minlength=self.probability_bins)
        cal_idx = np.minimum((p * self.calibration_bins).astype(np.int64),
                             self.calibration_bins - 1)
        self.cal_count += np.bincount(cal_idx, minlength=self.calibration_bins)
        self.cal_prob += np.bincount(cal_idx, weights=p, minlength=self.calibration_bins)
        self.cal_target += np.bincount(cal_idx, weights=y, minlength=self.calibration_bins)
        self.brier_sum += float(np.square(p - y).sum())
        self.pixel_count += int(p.size)

    def compute(self) -> Dict[str, float]:
        pos_total = int(self.pos_hist.sum())
        neg_total = int(self.neg_hist.sum())
        tp = np.cumsum(self.pos_hist[::-1], dtype=np.float64)
        fp = np.cumsum(self.neg_hist[::-1], dtype=np.float64)
        recall = tp / max(pos_total, 1)
        precision = tp / np.maximum(tp + fp, 1)
        recall_prev = np.concatenate(([0.0], recall[:-1]))
        auprc = float(np.sum((recall - recall_prev) * precision)) if pos_total else float("nan")
        tpr = np.concatenate(([0.0], recall))
        fpr = np.concatenate(([0.0], fp / max(neg_total, 1)))
        auroc = float(trapezoid(tpr, fpr)) if pos_total and neg_total else float("nan")
        valid = self.cal_count > 0
        observed = np.zeros_like(self.cal_prob)
        confidence = np.zeros_like(self.cal_prob)
        observed[valid] = self.cal_target[valid] / self.cal_count[valid]
        confidence[valid] = self.cal_prob[valid] / self.cal_count[valid]
        ece = float(np.sum(self.cal_count[valid] * np.abs(observed[valid] - confidence[valid])) /
                    max(self.pixel_count, 1))
        return {
            "auprc": auprc,
            "auroc": auroc,
            "ece": ece,
            "brier": self.brier_sum / max(self.pixel_count, 1),
            "foreground_prevalence": pos_total / max(pos_total + neg_total, 1),
        }

    def plot_data(self, max_points: int = 128) -> Dict[str, List[float]]:
        tp = np.cumsum(self.pos_hist[::-1], dtype=np.float64)
        fp = np.cumsum(self.neg_hist[::-1], dtype=np.float64)
        recall = tp / max(int(self.pos_hist.sum()), 1)
        precision = tp / np.maximum(tp + fp, 1)
        indices = np.linspace(0, len(recall) - 1, min(max_points, len(recall))).astype(int)
        valid = self.cal_count > 0
        observed = np.zeros_like(self.cal_prob)
        confidence = np.zeros_like(self.cal_prob)
        observed[valid] = self.cal_target[valid] / self.cal_count[valid]
        confidence[valid] = self.cal_prob[valid] / self.cal_count[valid]
        return {
            "pr_recall": recall[indices].tolist(),
            "pr_precision": precision[indices].tolist(),
            "calibration_confidence": confidence[valid].tolist(),
            "calibration_observed": observed[valid].tolist(),
            "calibration_count": self.cal_count[valid].astype(float).tolist(),
        }


def model_probabilities(model: torch.nn.Module, images: torch.Tensor) -> torch.Tensor:
    output = model(images)
    if isinstance(output, dict):
        output = output.get("prediction", output.get("logits"))
    if not isinstance(output, torch.Tensor):
        raise TypeError("Model output must be a tensor or contain prediction/logits")
    if output.min().item() >= 0.0 and output.max().item() <= 1.0:
        return output
    return torch.sigmoid(output)


def select_threshold(model: torch.nn.Module, loader: Iterable, device: torch.device,
                     thresholds: Sequence[float]) -> Dict[str, object]:
    thresholds = [float(x) for x in thresholds]
    per_threshold: Dict[float, Dict[int, List[int]]] = {t: {} for t in thresholds}
    probability_metrics = StreamingProbabilityMetrics()
    model.eval()
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            targets = batch["mask"].cpu().numpy().astype(np.uint8)
            probs = model_probabilities(model, images).detach().cpu().numpy()
            probability_metrics.update(probs, targets)
            volume_ids = batch["volume_id"].cpu().tolist() if torch.is_tensor(batch["volume_id"]) else batch["volume_id"]
            for index, volume_id in enumerate(volume_ids):
                target = targets[index] > 0
                for threshold in thresholds:
                    counts = confusion_counts(probs[index] >= threshold, target)
                    totals = per_threshold[threshold].setdefault(int(volume_id), [0, 0, 0, 0])
                    for i, value in enumerate(counts):
                        totals[i] += value
    ranking = []
    probability_result = probability_metrics.compute()
    for threshold, volumes in per_threshold.items():
        positive_volume_dices = []
        total_tp = total_fp = total_fn = 0
        for tp, fp, fn, tn in volumes.values():
            total_tp += tp
            total_fp += fp
            total_fn += fn
            if tp + fn > 0:
                positive_volume_dices.append(dice_from_counts(tp, fp, fn))
        precision = total_tp / max(total_tp + total_fp, 1)
        ranking.append({
            "threshold": threshold,
            "positive_volume_dice": float(np.mean(positive_volume_dices)) if positive_volume_dices else float("nan"),
            "auprc": probability_result["auprc"],
            "precision": precision,
        })
    ranking.sort(key=lambda row: (
        -float(np.nan_to_num(row["positive_volume_dice"], nan=-1.0)),
        -float(np.nan_to_num(row["auprc"], nan=-1.0)),
        -row["precision"],
        row["threshold"],
    ))
    return {"selected_threshold": ranking[0]["threshold"], "ranking": ranking}


def evaluate_research_model(model: torch.nn.Module, loader: Iterable, device: torch.device,
                            threshold: float) -> Dict[str, object]:
    model.eval()
    global_prob = StreamingProbabilityMetrics()
    positive_prob = StreamingProbabilityMetrics()
    slice_rows: List[Dict[str, object]] = []
    volume_buffers: Dict[int, Dict[str, object]] = {}
    micro = [0, 0, 0, 0]
    positive_slice_dices: List[float] = []
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            targets = batch["mask"].cpu().numpy().astype(np.uint8)
            probs = model_probabilities(model, images).detach().cpu().numpy()
            preds = probs >= threshold
            global_prob.update(probs, targets)
            volume_ids = batch["volume_id"].cpu().tolist() if torch.is_tensor(batch["volume_id"]) else batch["volume_id"]
            slice_ids = batch["slice_id"].cpu().tolist() if torch.is_tensor(batch["slice_id"]) else batch["slice_id"]
            positive_batch = targets.reshape(targets.shape[0], -1).sum(axis=1) > 0
            if positive_batch.any():
                positive_prob.update(probs[positive_batch], targets[positive_batch])
            for index, volume_id in enumerate(volume_ids):
                target = targets[index] > 0
                pred = preds[index]
                counts = confusion_counts(pred, target)
                for i, value in enumerate(counts):
                    micro[i] += value
                tp, fp, fn, tn = counts
                dice = dice_from_counts(tp, fp, fn)
                iou = iou_from_counts(tp, fp, fn)
                has_tumor = bool(target.any())
                if has_tumor:
                    positive_slice_dices.append(dice)
                slice_rows.append({
                    "volume_id": int(volume_id), "slice_id": int(slice_ids[index]),
                    "has_tumor": has_tumor, "dice": dice, "iou": iou,
                    "tp": tp, "fp": fp, "fn": fn, "tn": tn,
                    "pred_fg_fraction": float(pred.mean()),
                    "true_fg_fraction": float(target.mean()),
                })
                buffer = volume_buffers.setdefault(int(volume_id), {
                    "counts": [0, 0, 0, 0], "pred": [], "target": []
                })
                for i, value in enumerate(counts):
                    buffer["counts"][i] += value
                buffer["pred"].append(pred[0].astype(np.uint8))
                buffer["target"].append(target[0].astype(np.uint8))

    volume_rows: List[Dict[str, object]] = []
    surface_omitted = 0
    for volume_id, buffer in sorted(volume_buffers.items()):
        tp, fp, fn, tn = buffer["counts"]
        pred_volume = np.stack(buffer["pred"])
        target_volume = np.stack(buffer["target"])
        positive = bool(target_volume.any())
        surfaces = {"hd95": float("nan"), "asd": float("nan"), "nsd": float("nan")}
        if positive and pred_volume.any():
            surfaces = compute_all_surface_metrics(pred_volume, target_volume)
        else:
            surface_omitted += 1
        volume_rows.append({
            "volume_id": volume_id, "has_tumor": positive,
            "dice": dice_from_counts(tp, fp, fn), "iou": iou_from_counts(tp, fp, fn),
            "tp": tp, "fp": fp, "fn": fn, "tn": tn, **surfaces,
        })

    tp, fp, fn, tn = micro
    positive_volumes = [row for row in volume_rows if row["has_tumor"]]
    surface_rows = [row for row in positive_volumes if not math.isnan(row["hd95"])]
    aggregate = {
        "threshold": threshold,
        "micro_dice": dice_from_counts(tp, fp, fn),
        "micro_iou": iou_from_counts(tp, fp, fn),
        "positive_slice_dice": float(np.mean(positive_slice_dices)) if positive_slice_dices else float("nan"),
        "macro_volume_dice": float(np.mean([row["dice"] for row in volume_rows])),
        "positive_volume_dice": float(np.mean([row["dice"] for row in positive_volumes])) if positive_volumes else float("nan"),
        "precision": tp / max(tp + fp, 1),
        "recall": tp / max(tp + fn, 1),
        "specificity": tn / max(tn + fp, 1),
        "pred_fg_fraction": (tp + fp) / max(tp + fp + fn + tn, 1),
        "true_fg_fraction": (tp + fn) / max(tp + fp + fn + tn, 1),
        "hd95": float(np.mean([row["hd95"] for row in surface_rows])) if surface_rows else float("nan"),
        "asd": float(np.mean([row["asd"] for row in surface_rows])) if surface_rows else float("nan"),
        "surface_metrics_omitted_volumes": surface_omitted,
        **global_prob.compute(),
    }
    aggregate.update({f"positive_slice_{key}": value for key, value in positive_prob.compute().items()})
    return {
        "aggregate": aggregate, "slices": slice_rows, "volumes": volume_rows,
        "plot_data": global_prob.plot_data(),
    }
