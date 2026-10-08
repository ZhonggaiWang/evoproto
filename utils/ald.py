"""Fuse filtered CAM labels with teacher labels without mutating either input.

The modes differ only in whether newly rejected current-class pixels retain
teacher supervision. Image-box validity and confidence estimation belong to
the caller, not this fusion module.
"""

from typing import NamedTuple

import torch


_INTEGER_DTYPES = frozenset(
    (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64)
)
_MODES = frozenset(("legacy", "preserve_rejected", "preserve_background"))


class ALDLabelFusionResult(NamedTuple):
    """Independent long labels and boolean masks on the input device.

    rejected_mask marks valid current-class CAM pixels changed to ignore by
    filtering. retained_ignore_mask is the subset actually left unsupervised
    by the selected mode. Existing CAM ignore regions are in neither mask.
    """

    labels: torch.Tensor
    rejected_mask: torch.Tensor
    retained_ignore_mask: torch.Tensor


def fuse_ald_labels(
    cam_before: torch.Tensor,
    cam_after: torch.Tensor,
    teacher_labels: torch.Tensor,
    *,
    total_classes: int,
    new_classes: int,
    mode: str = "legacy",
    ignore_index: int = 255,
) -> ALDLabelFusionResult:
    """Fuse [B, H, W] integer labels for an incremental foreground-class step.

    CAM IDs must be in 0..total_classes or equal ignore_index; teacher argmax
    IDs must be in 0..(total_classes - new_classes). Background is ID 0.
    Floating labels are rejected to prevent silent truncation: the caller
    should explicitly convert integral-valued CAM outputs to long beforehand.

    legacy reproduces get_mixed_label's values for valid inputs: teacher labels
    are overwritten only by valid current-class IDs in cam_after.
    preserve_rejected restores ignore on all newly rejected current-class
    pixels; preserve_background restores it only where teacher_labels == 0.
    To disable ALD, pass identical before/after CAMs with mode="legacy".

    Shape, dtype, device, scalar configuration and label ranges are checked.
    Inputs are never modified; outputs do not share storage with the inputs or
    with one another. Empty batches and empty rejection masks are supported.
    """
    for name, value in (
        ("total_classes", total_classes),
        ("new_classes", new_classes),
        ("ignore_index", ignore_index),
    ):
        if not isinstance(value, int) or isinstance(value, bool):
            raise TypeError(f"{name} must be an integer, not {type(value).__name__}.")
    if total_classes < 1 or not 1 <= new_classes <= total_classes:
        raise ValueError("Require total_classes >= 1 and 1 <= new_classes <= total_classes.")
    if ignore_index <= total_classes:
        raise ValueError("ignore_index must be outside the valid class IDs and above total_classes.")
    if ignore_index > torch.iinfo(torch.long).max:
        raise ValueError("Class IDs and ignore_index must fit the long output dtype.")
    if not isinstance(mode, str):
        raise TypeError("mode must be a string.")
    if mode not in _MODES:
        raise ValueError(f"Unknown ALD fusion mode {mode!r}; expected one of {sorted(_MODES)}.")

    inputs = (("cam_before", cam_before), ("cam_after", cam_after),
              ("teacher_labels", teacher_labels))
    for name, labels in inputs:
        if not isinstance(labels, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor.")
        if labels.ndim != 3:
            raise ValueError(f"{name} must have shape [B, H, W], got {tuple(labels.shape)}.")
        if labels.dtype not in _INTEGER_DTYPES:
            raise TypeError(f"{name} must have a supported integer dtype, got {labels.dtype}.")
    if any(labels.shape != cam_before.shape for _, labels in inputs[1:]):
        raise ValueError("All label tensors must have the same shape [B, H, W].")
    if any(labels.device != cam_before.device for _, labels in inputs[1:]):
        raise ValueError("All label tensors must be on the same device.")
    if cam_before.device.type == "meta":
        raise ValueError("Label tensors must be on a concrete device, not meta.")

    # Compare in long so uint8/int8 scalar promotion cannot wrap configuration
    # values (for example, uint8(255) == 511 would otherwise evaluate true).
    before = cam_before.to(dtype=torch.long)
    after = cam_after.to(dtype=torch.long)
    teacher = teacher_labels.to(dtype=torch.long)
    old_classes = total_classes - new_classes
    for name, labels in (("cam_before", before), ("cam_after", after)):
        valid = ((labels >= 0) & (labels <= total_classes)) | (labels == ignore_index)
        if not valid.all().item():
            raise ValueError(f"{name} contains IDs outside 0..{total_classes} and ignore_index.")
    if ((teacher < 0) | (teacher > old_classes)).any().item():
        raise ValueError(f"teacher_labels must contain argmax IDs in 0..{old_classes}.")

    rejected_mask = (
        (before > old_classes)
        & (before <= total_classes)
        & (after == ignore_index)
    )
    if mode == "preserve_rejected":
        retained_ignore_mask = rejected_mask.clone()
    elif mode == "preserve_background":
        retained_ignore_mask = rejected_mask & (teacher == 0)
    else:
        retained_ignore_mask = torch.zeros_like(rejected_mask)

    labels = teacher.clone()
    current_class = (after > old_classes) & (after <= total_classes)
    labels[current_class] = after[current_class]
    labels[retained_ignore_mask] = ignore_index
    return ALDLabelFusionResult(labels, rejected_mask, retained_ignore_mask)
