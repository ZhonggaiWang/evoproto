import numpy as np
import sklearn.metrics as metrics
import torch


def multilabel_score(y_true, y_pred):
    return metrics.f1_score(y_true, y_pred)


def update_confusion_matrix(confusion_matrix, preds, gts, num_classes):
    """
    实时更新混淆矩阵
    preds: (N, H, W) LongTensor
    gts:   (N, H, W) LongTensor
    confusion_matrix: (num_classes, num_classes) LongTensor
    """
    preds = preds.view(-1)
    gts = gts.view(-1)
    mask = (gts >= 0) & (gts < num_classes)
    preds = preds[mask]
    gts = gts[mask]
    idx = gts * num_classes + preds
    cm = torch.bincount(idx, minlength=num_classes ** 2).reshape(num_classes, num_classes)
    confusion_matrix += cm

def compute_confusion(preds, gts, num_classes):
    """
    preds: (N, H, W) tensor，预测结果
    gts: (N, H, W) tensor，真实标签
    num_classes: 类别总数
    """
    confusion = torch.zeros((num_classes, num_classes), dtype=torch.int64)  # [gt_class, pred_class]

    for pred, gt in zip(preds, gts):
        pred = torch.tensor(pred)  # [H, W]
        gt = torch.tensor(gt)      # [H, W]
        mask = (gt >= 0) & (gt < num_classes)
        gt = gt[mask]
        pred = pred[mask]

        for i in range(num_classes):
            gt_mask = (gt == i)
            if gt_mask.sum() == 0:
                continue
            pred_of_i = pred[gt_mask]
            for j in range(num_classes):
                confusion[i, j] += (pred_of_i == j).sum()

    return confusion


def _fast_hist(label_true, label_pred, num_classes):
    mask = (label_true >= 0) & (label_true < num_classes)
    hist = np.bincount(
        num_classes * label_true[mask].astype(int) + label_pred[mask],
        minlength=num_classes ** 2,
    )
    return hist.reshape(num_classes, num_classes)

def scores(label_trues, label_preds, num_classes=21):
    hist = np.zeros((num_classes, num_classes))
    for lt, lp in zip(label_trues, label_preds):
        hist += _fast_hist(lt.flatten(), lp.flatten(), num_classes)
    acc = np.diag(hist).sum() / hist.sum()
    _acc_cls = np.diag(hist) / hist.sum(axis=1)
    acc_cls = np.nanmean(_acc_cls)
    iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
    valid = hist.sum(axis=1) > 0  # added
    mean_iu = np.nanmean(iu[valid])
    freq = hist.sum(axis=1) / hist.sum()
    cls_iu = dict(zip(range(num_classes), iu))

    return {
        "pAcc": acc,
        "mAcc": acc_cls,
        "miou": mean_iu,
        "iou": cls_iu,
    }

def pseudo_scores(label_trues, label_preds, num_classes=21):
    hist = np.zeros((num_classes, num_classes))
    for lt, lp in zip(label_trues, label_preds):
        lt = lt.flatten()
        lp = lp.flatten()
        lt[lp==255] = 255
        lp[lp==255] = 0
        hist += _fast_hist(lt, lp, num_classes)
    acc = np.diag(hist).sum() / hist.sum()
    acc_cls = np.diag(hist) / hist.sum(axis=1)
    acc_cls = np.nanmean(acc_cls)
    iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
    valid = hist.sum(axis=1) > 0  # added
    mean_iu = np.nanmean(iu[valid])
    freq = hist.sum(axis=1) / hist.sum()
    cls_iu = dict(zip(range(num_classes), iu))

    return {
        "pAcc": acc,
        "mAcc": acc_cls,
        "miou": mean_iu,
        "iou": cls_iu,
    }