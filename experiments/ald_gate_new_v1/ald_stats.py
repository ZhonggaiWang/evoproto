"""Isolated ALD counters preserving original fallback before NEW rescue."""
import torch
from tools.ald_gate_policies import ALDGateResult


COUNT_KEYS = (
    "batch_images", "total_pixels", "valid_box_pixels",
    "before_new_cam_pixels", "after_new_cam_pixels", "rejected_new_cam_pixels",
    "rejected_teacher_old_pixels", "rejected_teacher_bg_pixels",
    "retained_ignore_pixels", "rejected_final_ignore_pixels",
    "rejected_final_old_pixels", "rejected_final_bg_pixels",
    "final_valid_pixels", "final_bg_pixels", "final_old_pixels", "final_new_pixels",
    "final_ignore_pixels", "padding_labeled_pixels", "positive_old_classes",
    "positive_new_classes", "gate_rejected_old_classes", "gate_rejected_new_classes",
    "fallback_images", "fallback_old_images", "fallback_new_images",
    "new_rescue_images", "new_rescued_class_occurrences", "direct_threshold_new_classes",
)


def valid_image_mask(labels, img_box):
    """Return [B,H,W] validity from integer [top,bottom,left,right] boxes."""
    if labels.ndim != 3:
        raise ValueError("labels must have shape [B,H,W]")
    b, h, w = labels.shape
    boxes = torch.as_tensor(img_box)
    if boxes.shape != (b, 4) or boxes.dtype.is_floating_point or boxes.dtype == torch.bool:
        raise ValueError("img_box must contain B integer [top,bottom,left,right] rows")
    boxes = boxes.detach().cpu().tolist()
    mask = torch.zeros_like(labels, dtype=torch.bool)
    for index, (top, bottom, left, right) in enumerate(boxes):
        if not (0 <= top <= bottom <= h and 0 <= left <= right <= w):
            raise ValueError("img_box must be within the final label resolution")
        mask[index, top:bottom, left:right] = True
    return mask


@torch.no_grad()
def supervision_counts(before, after, teacher, labels, rejected, retained, valid,
                       *, old_classes, total_classes, cls_labels,
                       gate_result=None, cam_peak=None, threshold=12.5, ignore_index=255):
    """Count on-device; the caller synchronizes only at a logging interval."""
    tensors = (before, after, teacher, rejected, retained, valid)
    if labels.ndim != 3 or any(x.shape != labels.shape or x.device != labels.device for x in tensors):
        raise ValueError("all pixel tensors must share [B,H,W] shape and device")
    if any(x.dtype != torch.bool for x in (rejected, retained, valid)):
        raise TypeError("rejected, retained and valid masks must be boolean")
    if cls_labels.shape != (labels.shape[0], total_classes) or cls_labels.device != labels.device:
        raise ValueError("image labels must have shape [B,total_foreground_classes]")
    if not 0 < old_classes < total_classes:
        raise ValueError("an incremental old/new foreground split is required")
    new_before = valid & (before > old_classes) & (before <= total_classes)
    new_after = valid & (after > old_classes) & (after <= total_classes)
    r = valid & rejected
    effective = valid & (labels != ignore_index)
    old = lambda x: (x > 0) & (x <= old_classes)
    positive = cls_labels > 0
    zero = torch.zeros((), dtype=torch.int64, device=labels.device)
    gate_old = gate_new = fallback = fallback_old = fallback_new = zero
    rescued_images = rescued_classes = direct_new = zero
    if gate_result is not None:
        if not isinstance(gate_result, ALDGateResult):
            raise TypeError("gate_result must be an ALDGateResult")
        gate_labels, legacy_gate_labels, rescue_mask = gate_result
        if (cam_peak is None or gate_labels.shape != cls_labels.shape
                or legacy_gate_labels.shape != cls_labels.shape or cam_peak.shape != cls_labels.shape):
            raise ValueError("gate and raw CAM peaks must match image labels")
        if (rescue_mask.shape != (labels.shape[0],) or rescue_mask.dtype != torch.bool):
            raise ValueError("NEW rescue mask must be a boolean [B] tensor")
        if gate_labels.dtype != cls_labels.dtype or legacy_gate_labels.dtype != cls_labels.dtype:
            raise TypeError("gate label dtypes must match image labels")
        if any(x.device != labels.device for x in (gate_labels, legacy_gate_labels, rescue_mask, cam_peak)):
            raise ValueError("gate diagnostics must share the label device")
        gate = gate_labels > 0
        legacy_gate = legacy_gate_labels > 0
        removed = positive & ~gate
        gate_old = removed[:, :old_classes].sum()
        gate_new = removed[:, old_classes:].sum()
        direct = positive & (cam_peak >= threshold)
        # Original fallback selects exactly one OLD or NEW. A later NEW rescue
        # must not turn an original OLD fallback into both OLD and NEW fallback.
        fallback_mask = positive.any(dim=1) & ~direct.any(dim=1)
        fallback = fallback_mask.sum()
        fallback_old = (fallback_mask & legacy_gate[:, :old_classes].any(dim=1)).sum()
        fallback_new = (fallback_mask & legacy_gate[:, old_classes:].any(dim=1)).sum()
        rescued_images = rescue_mask.sum()
        rescued_classes = (positive & gate & ~legacy_gate)[:, old_classes:].sum()
        direct_new = direct[:, old_classes:].sum()
    values = (
        zero.new_tensor(labels.shape[0]), zero.new_tensor(labels.numel()), valid.sum(),
        new_before.sum(), new_after.sum(), r.sum(),
        (r & old(teacher)).sum(), (r & (teacher == 0)).sum(),
        (valid & retained).sum(), (r & (labels == ignore_index)).sum(),
        (r & old(labels)).sum(), (r & (labels == 0)).sum(),
        effective.sum(), (effective & (labels == 0)).sum(),
        (effective & old(labels)).sum(),
        (effective & (labels > old_classes) & (labels <= total_classes)).sum(),
        (valid & (labels == ignore_index)).sum(), ((~valid) & (labels != ignore_index)).sum(),
        positive[:, :old_classes].sum(), positive[:, old_classes:].sum(),
        gate_old, gate_new, fallback, fallback_old, fallback_new,
        rescued_images, rescued_classes, direct_new,
    )
    return torch.stack(values).to(dtype=torch.int64)


def counts_record(counts):
    values = counts.detach().cpu().tolist()
    if len(values) != len(COUNT_KEYS):
        raise ValueError("incorrect counter vector length")
    record = dict(zip(COUNT_KEYS, (int(value) for value in values)))
    def ratio(numerator, denominator):
        return record[numerator] / record[denominator] if record[denominator] else None
    record.update(
        rejected_fraction_of_new_cam=ratio("rejected_new_cam_pixels", "before_new_cam_pixels"),
        rejected_fraction_of_valid_image=ratio("rejected_new_cam_pixels", "valid_box_pixels"),
        supervised_fraction_of_valid_image=ratio("final_valid_pixels", "valid_box_pixels"),
        retained_ignore_fraction_of_rejected=ratio("retained_ignore_pixels", "rejected_new_cam_pixels"),
        new_rescue_fraction_of_images=ratio("new_rescue_images", "batch_images"),
        new_rescued_fraction_of_positive_new_classes=ratio("new_rescued_class_occurrences", "positive_new_classes"),
        direct_threshold_fraction_of_positive_new_classes=ratio("direct_threshold_new_classes", "positive_new_classes"),
    )
    return record
