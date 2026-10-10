"""Image-level old-class filtering from paper Eqs. (11)--(14)."""
import torch


@torch.no_grad()
def filter_old_class_labels(predicted_old, old_cam_peaks):
    """Retain old candidates at or above their within-image mean CAM peak.

    An empty candidate set stays empty; a singleton is always retained. New
    tags are intentionally not inputs, so their ground-truth values cannot be
    altered by this filter. Return the retained labels and per-image thresholds.
    """
    if predicted_old.ndim != 2 or old_cam_peaks.shape != predicted_old.shape:
        raise ValueError("Old labels and CAM peaks must share [batch, old_classes].")
    present = predicted_old.bool()
    count = present.sum(dim=1, keepdim=True)
    peaks = old_cam_peaks.detach()
    threshold = torch.where(present, peaks, torch.zeros_like(peaks)).sum(dim=1, keepdim=True)
    threshold = threshold / count.clamp_min(1)
    retained = present & ((count <= 1) | (peaks >= threshold))
    return retained.to(predicted_old.dtype), threshold
