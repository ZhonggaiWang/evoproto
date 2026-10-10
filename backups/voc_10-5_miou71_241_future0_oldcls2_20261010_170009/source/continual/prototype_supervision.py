"""Shared fixed/paper ALD supervision and confusion estimation for VOC/COCO."""
import math
from typing import NamedTuple, Optional

import torch
import torch.distributed as dist
import torch.nn.functional as F
from tqdm import tqdm

from model.PAR import PAR
from utils import imutils
from utils.ald import fuse_ald_labels
from utils.ald_stats import valid_image_mask
from utils.camutils import (cam_to_label, multi_scale_cam2, multi_scale_cam2_filter,
                           cam_high_pass_filter, filter_cam_pesudo_label, refine_cams_with_bkg_v2)
from utils.evaluate import update_confusion_matrix
from utils.image_ald import filter_old_class_labels
from utils.prototype_confusion import symmetric_confusion


class IncrementalSupervision(NamedTuple):
    cls_labels: torch.Tensor
    cls_labels_before_ald: torch.Tensor
    cams: torch.Tensor
    cams_aux: torch.Tensor
    cam_peaks: torch.Tensor
    ald_thresholds: torch.Tensor
    refined_labels: torch.Tensor
    merged_labels: torch.Tensor
    old_logits: torch.Tensor
    old_labels: torch.Tensor
    old_prototypes: torch.Tensor
    valid_mask: torch.Tensor
    gate_labels: Optional[torch.Tensor]
    refined_labels_before_ald: torch.Tensor
    rejected_mask: torch.Tensor
    retained_ignore_mask: torch.Tensor
    ald_mode: str


def resolve_ald_mode(args):
    mode = getattr(args, 'ald_mode', None) or 'fixed'
    if mode == 'legacy':
        mode = 'fixed'
    if mode not in ('off', 'fixed', 'paper'):
        raise ValueError('ALD mode must be off, fixed or paper.')
    return mode if getattr(args, 'ald', True) else 'off'


def fixed_ald_threshold(args):
    threshold = getattr(args, 'ald_threshold', None)
    if threshold is None:
        threshold = 7.5 if getattr(args, 'dataset', 'voc') in ('coco', 'coco2voc') else 0.0
    if not math.isfinite(threshold) or threshold < 0:
        raise ValueError('Fixed ALD threshold must be finite and nonnegative.')
    return float(threshold)


@torch.no_grad()
def build_incremental_supervision(model, model_old, inputs, cls_labels, img_box, par,
                                  args, total_classes, new_classes):
    """Use identical filtered image tags and merged pixel targets in both paths."""
    old_cls, _, old_logits, _, old_prototypes = model_old(inputs, step0=True)
    # Threshold raw image-classification logits only; teacher pixel labels
    # below still use argmax over all old/background segmentation logits.
    old_cls_threshold = getattr(args, 'old_cls_threshold', 2.0)
    if not math.isfinite(old_cls_threshold):
        raise ValueError('Old-class image-logit threshold must be finite.')
    cls_label_old_pred = (old_cls > old_cls_threshold).long()
    new_tags = cls_labels[:, :total_classes][:, -new_classes:]
    before_ald = torch.cat((cls_label_old_pred, new_tags), dim=1)
    mode = resolve_ald_mode(args)
    if mode == 'fixed':
        # Original fixed ALD uses the aggregated CAM amplitude before its
        # per-class normalization, including all scales and flip fusion.
        cams, cams_aux, cam_peaks = multi_scale_cam2_filter(
            model, inputs=inputs, scales=args.cam_scales,
        )
        threshold = fixed_ald_threshold(args)
        thresholds = cam_peaks.new_full((inputs.shape[0], 1), threshold)
        gate_labels = cam_high_pass_filter(cam_peaks, before_ald, threshold)
        # Fixed ALD filters spatial labels; classification/CAM/PTC retain the
        # original teacher candidates and current-new image tags.
        train_tags = before_ald
    else:
        cams, cams_aux, cam_peaks = multi_scale_cam2(
            model, inputs=inputs, scales=args.cam_scales, return_peaks=True,
        )
        if mode == 'paper':
            old_count = total_classes - new_classes
            retained_old, thresholds = filter_old_class_labels(
                cls_label_old_pred, cam_peaks[:, :old_count],
            )
            train_tags = torch.cat((retained_old, new_tags), dim=1)
            gate_labels = train_tags
        else:
            train_tags = before_ald
            gate_labels = None
            thresholds = cam_peaks.new_zeros((inputs.shape[0], 1))

    valid_cam, _ = cam_to_label(
        cams, cls_label=train_tags, img_box=img_box, ignore_mid=True,
        bkg_thre=args.bkg_thre, high_thre=args.high_thre,
        low_thre=args.low_thre, ignore_index=args.ignore_index,
    )
    refined = refine_cams_with_bkg_v2(
        par, imutils.denormalize_img2(inputs), cams=valid_cam,
        cls_labels=train_tags, high_thre=args.high_thre,
        low_thre=args.low_thre, ignore_index=args.ignore_index, img_box=img_box,
    ).long()
    refined_before = refined
    if mode == 'fixed':
        refined = filter_cam_pesudo_label(refined, gate_labels, args.ignore_index)
    old_logits = F.interpolate(old_logits, size=refined.shape[-2:],
                               mode='bilinear', align_corners=False)
    old_labels = old_logits.argmax(dim=1)
    # Eq. (2): valid incoming-class CAM labels override the teacher; otherwise
    # retain its old/background prediction. In fixed mode rejected new CAM
    # pixels also inherit the teacher, matching the original legacy fusion.
    fusion = fuse_ald_labels(
        refined_before, refined, old_labels, total_classes=total_classes,
        new_classes=new_classes, mode='legacy', ignore_index=args.ignore_index,
    )
    valid = valid_image_mask(fusion.labels, img_box)
    merged = fusion.labels.masked_fill(~valid, args.ignore_index)
    return IncrementalSupervision(train_tags, before_ald, cams, cams_aux, cam_peaks,
                                  thresholds, refined, merged, old_logits, old_labels,
                                  old_prototypes, valid, gate_labels, refined_before,
                                  fusion.rejected_mask & valid, fusion.retained_ignore_mask & valid, mode)


@torch.no_grad()
def compute_confusion(trainer, model, data_loader, args):
    """Aggregate disagreement with the actual merged training targets."""
    was_training = model.training
    model.eval()
    try:
        par = PAR(num_iter=10, dilations=[1, 2, 4, 8, 12, 24]).to(trainer.device)
        classes = trainer.total_classes + 1
        counts = torch.zeros(classes, classes, dtype=torch.int64, device=trainer.device)
        for data in tqdm(data_loader, total=len(data_loader), ncols=100, ascii=" >="):
            _, inputs, cls_labels, img_box, _ = data
            inputs = inputs.to(trainer.device, non_blocking=True)
            cls_labels = cls_labels.to(trainer.device, non_blocking=True)
            supervision = build_incremental_supervision(
                model, trainer.model_old, inputs, cls_labels, img_box, par,
                args, trainer.total_classes, trainer.new_classes,
            )
            prediction_inputs = F.interpolate(inputs, size=[args.crop_size, args.crop_size],
                                              mode='bilinear', align_corners=False)
            segs = model(prediction_inputs, cal_sim=True)[1]
            segs = F.interpolate(segs, size=supervision.merged_labels.shape[-2:],
                                 mode='bilinear', align_corners=False)
            update_confusion_matrix(counts, segs.argmax(dim=1), supervision.merged_labels, classes)
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(counts, op=dist.ReduceOp.SUM)
        trainer.confusion_counts = counts
        return symmetric_confusion(counts)
    finally:
        model.train(was_training)
