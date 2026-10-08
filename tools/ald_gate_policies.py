"""Optional, isolated ALD gate policies for future controlled experiments.

This module is not imported by the frozen primary training pipeline. Image-level
NEW labels may outlive the object after random cropping, so NEW rescue is an
experimental hypothesis rather than a required/default training rule.
"""

import math
from numbers import Real
from typing import NamedTuple

import torch


class ALDGateResult(NamedTuple):
    gate_labels: torch.Tensor
    legacy_gate_labels: torch.Tensor
    new_rescue_mask: torch.Tensor


_FLOAT_DTYPES = {torch.float16, torch.bfloat16, torch.float32, torch.float64}
_LABEL_DTYPES = _FLOAT_DTYPES | {torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64}


@torch.no_grad()
def gate_ald_classes(cam_peak, cls_labels, *, old_classes: int,
                     threshold: float = 12.5, policy="legacy") -> ALDGateResult:
    """Return independent final/legacy gates and a [B] actual NEW-rescue mask.

    Peaks must be finite, nonnegative floating tensors; labels must be binary
    integer or floating tensors. Both have shape [B,C] and the same device. The
    old/new boundary is ``old_classes`` (0 <= old_classes < C), with foreground
    classes only. Outputs preserve the label dtype/device and never alias inputs
    or each other. Threshold equality survives, as in the original ALD gate.

    Legacy retains positive classes meeting the threshold, then restores the
    strongest original positive class only if none survived. Ties select the
    lowest class index. NEW fallback adds one strongest original positive NEW
    only when no NEW survived legacy; all legacy selections remain unchanged.
    """
    if not isinstance(cam_peak, torch.Tensor) or not isinstance(cls_labels, torch.Tensor):
        raise TypeError("cam_peak and cls_labels must be tensors")
    if cam_peak.ndim != 2 or cls_labels.ndim != 2 or cam_peak.shape != cls_labels.shape:
        raise ValueError("cam_peak and cls_labels must share shape [B,C]")
    if cam_peak.device != cls_labels.device:
        raise ValueError("cam_peak and cls_labels must share a device")
    if cam_peak.device.type == "meta":
        raise ValueError("gate inputs must be on a materialized device")
    if cam_peak.dtype not in _FLOAT_DTYPES:
        raise TypeError("cam_peak must have float16, bfloat16, float32 or float64 dtype")
    if cls_labels.dtype not in _LABEL_DTYPES:
        raise TypeError("cls_labels must have a supported integer or floating dtype, excluding bool")
    if type(old_classes) is not int:
        raise TypeError("old_classes must be an integer, excluding bool")
    b, c = cam_peak.shape
    if not 0 <= old_classes < c:
        raise ValueError("old_classes must satisfy 0 <= old_classes < C")
    if not isinstance(threshold, Real) or isinstance(threshold, bool):
        raise TypeError("threshold must be a real scalar, excluding bool")
    if not math.isfinite(threshold) or threshold < 0:
        raise ValueError("threshold must be finite and nonnegative")
    if policy not in ("legacy", "new_fallback"):
        raise ValueError("policy must be 'legacy' or 'new_fallback'")
    if not torch.isfinite(cam_peak).all() or (cam_peak < 0).any():
        raise ValueError("cam_peak must contain finite nonnegative values")
    if not ((cls_labels == 0) | (cls_labels == 1)).all():
        raise ValueError("cls_labels must contain only binary 0/1 values")

    positive = cls_labels != 0
    legacy = cls_labels.clone()
    legacy[cam_peak < threshold] = 0
    fallback_rows = torch.nonzero(positive.any(dim=1) & ~legacy.bool().any(dim=1),
                                  as_tuple=True)[0]
    if fallback_rows.numel():
        scores = cam_peak[fallback_rows].masked_fill(~positive[fallback_rows], -torch.inf)
        selected = scores.argmax(dim=1)
        legacy[fallback_rows, selected] = 1

    gate = legacy.clone()
    rescue = torch.zeros(b, dtype=torch.bool, device=cam_peak.device)
    if policy == "new_fallback":
        new_positive = positive[:, old_classes:]
        rescue = new_positive.any(dim=1) & ~legacy[:, old_classes:].bool().any(dim=1)
        rescue_rows = torch.nonzero(rescue, as_tuple=True)[0]
        if rescue_rows.numel():
            scores = cam_peak[rescue_rows, old_classes:].masked_fill(
                ~new_positive[rescue_rows], -torch.inf)
            selected = scores.argmax(dim=1) + old_classes
            gate[rescue_rows, selected] = 1
    return ALDGateResult(gate, legacy, rescue)
