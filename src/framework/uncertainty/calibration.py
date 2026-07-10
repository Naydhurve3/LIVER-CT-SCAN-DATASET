import torch
import numpy as np


def expected_calibration_error(probs, targets, n_bins=10):
    confidences = probs.view(-1)
    accuracies = (probs > 0.5).float().view(-1).eq(targets.view(-1)).float()
    bin_boundaries = torch.linspace(0, 1, n_bins + 1, device=probs.device)
    ece = 0.0
    for i in range(n_bins):
        in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
        if in_bin.sum() > 0:
            bin_acc = accuracies[in_bin].mean()
            bin_conf = confidences[in_bin].mean()
            ece += (in_bin.sum() / confidences.numel()) * abs(bin_acc - bin_conf)
    return ece.item()


def maximum_calibration_error(probs, targets, n_bins=10):
    confidences = probs.view(-1)
    accuracies = (probs > 0.5).float().view(-1).eq(targets.view(-1)).float()
    bin_boundaries = torch.linspace(0, 1, n_bins + 1, device=probs.device)
    mce = 0.0
    for i in range(n_bins):
        in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
        if in_bin.sum() > 0:
            bin_acc = accuracies[in_bin].mean()
            bin_conf = confidences[in_bin].mean()
            mce = max(mce, abs(bin_acc - bin_conf))
    return mce


def brier_score(probs, targets):
    return ((probs - targets) ** 2).mean().item()


def reliability_diagram_data(probs, targets, n_bins=10):
    confidences = probs.view(-1).cpu().numpy()
    accuracies = (probs > 0.5).float().view(-1).eq(targets.view(-1)).float().cpu().numpy()
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_accs = []
    bin_confs = []
    bin_counts = []
    for i in range(n_bins):
        in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
        count = in_bin.sum()
        bin_counts.append(int(count))
        if count > 0:
            bin_accs.append(float(accuracies[in_bin].mean()))
            bin_confs.append(float(confidences[in_bin].mean()))
        else:
            bin_accs.append(0.0)
            bin_confs.append(0.0)
    return {
        'bin_confidences': bin_confs,
        'bin_accuracies': bin_accs,
        'bin_counts': bin_counts,
        'ece': expected_calibration_error(probs, targets),
        'mce': maximum_calibration_error(probs, targets),
        'brier': brier_score(probs, targets),
    }
