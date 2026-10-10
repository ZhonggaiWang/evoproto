import pdb
import torch
import torch.nn.functional as F


def cam_to_label(cam, cls_label, img_box=None, bkg_thre=None, high_thre=None, low_thre=None, ignore_mid=False,
                 ignore_index=None):
    b, c, h, w = cam.shape
    # pseudo_label = torch.zeros((b,h,w))
    cls_label_rep = cls_label.unsqueeze(-1).unsqueeze(-1).repeat([1, 1, h, w])
    valid_cam = cls_label_rep * cam
    cam_value, _pseudo_label = valid_cam.max(dim=1, keepdim=False)
    _pseudo_label += 1
    _pseudo_label[cam_value <= bkg_thre] = 0

    if img_box is None:
        return _pseudo_label

    if ignore_mid:
        _pseudo_label[cam_value <= high_thre] = ignore_index
        _pseudo_label[cam_value <= low_thre] = 0
    pseudo_label = torch.ones_like(_pseudo_label) * ignore_index

    for idx, coord in enumerate(img_box):
        pseudo_label[idx, coord[0]:coord[1], coord[2]:coord[3]] = _pseudo_label[idx, coord[0]:coord[1],
                                                                  coord[2]:coord[3]]

    return valid_cam, pseudo_label


def cam_to_roi_mask2(cam, cls_label, hig_thre=None, low_thre=None):
    b, c, h, w = cam.shape
    # pseudo_label = torch.zeros((b,h,w))
    cls_label_rep = cls_label.unsqueeze(-1).unsqueeze(-1).repeat([1, 1, h, w])
    valid_cam = cls_label_rep * cam
    cam_value, _ = valid_cam.max(dim=1, keepdim=False)
    # _pseudo_label += 1
    roi_mask = torch.ones_like(cam_value, dtype=torch.int16)
    roi_mask[cam_value <= low_thre] = 0
    roi_mask[cam_value >= hig_thre] = 2

    return roi_mask


def get_valid_cam(cam, cls_label):
    b, c, h, w = cam.shape
    # pseudo_label = torch.zeros((b,h,w))
    cls_label_rep = cls_label.unsqueeze(-1).unsqueeze(-1).repeat([1, 1, h, w])
    valid_cam = cls_label_rep * cam

    return valid_cam


def ignore_img_box(label, img_box, ignore_index):
    pseudo_label = torch.ones_like(label) * ignore_index

    for idx, coord in enumerate(img_box):
        pseudo_label[idx, coord[0]:coord[1], coord[2]:coord[3]] = label[idx, coord[0]:coord[1], coord[2]:coord[3]]

    return pseudo_label


def crop_from_roi_neg(images, roi_mask=None, crop_num=8, crop_size=96):
    crops = []

    b, c, h, w = images.shape

    temp_crops = torch.zeros(size=(b, crop_num, c, crop_size, crop_size)).to(images.device)
    flags = torch.ones(size=(b, crop_num + 2)).to(images.device)
    margin = crop_size // 2

    for i1 in range(b):
        roi_index = (roi_mask[i1, margin:(h - margin), margin:(w - margin)] <= 1).nonzero()
        if roi_index.shape[0] < crop_num:
            roi_index = (roi_mask[i1, margin:(h - margin),
                         margin:(w - margin)] >= 0).nonzero()  ## if NULL then random crop
        rand_index = torch.randperm(roi_index.shape[0])
        crop_index = roi_index[rand_index[:crop_num], :]

        for i2 in range(crop_num):
            h0, w0 = crop_index[i2, 0], crop_index[i2, 1]  # centered at (h0, w0)
            temp_crops[i1, i2, ...] = images[i1, :, h0:(h0 + crop_size), w0:(w0 + crop_size)]
            temp_mask = roi_mask[i1, h0:(h0 + crop_size), w0:(w0 + crop_size)]
            if temp_mask.sum() / (crop_size * crop_size) <= 0.2:
                ## if ratio of uncertain regions < 0.2 then negative
                flags[i1, i2 + 2] = 0

    _crops = torch.chunk(temp_crops, chunks=crop_num, dim=1, )
    crops = [c[:, 0] for c in _crops]

    return crops, flags


@torch.no_grad()
def multi_scale_seg(model, inputs, scales):
    """Average signed segmentation logits over scales and aligned flip views."""
    b, _, h, w = inputs.shape
    fused = None
    view_scales = [1.0] + [s for s in scales if s != 1.0]
    for scale in view_scales:
        scaled = inputs if scale == 1.0 else F.interpolate(
            inputs, size=(int(scale * h), int(scale * w)),
            mode='bilinear', align_corners=False,
        )
        inputs_cat = torch.cat([scaled, scaled.flip(-1)], dim=0)
        logits = model(inputs_cat)[1]
        logits = F.interpolate(logits, size=(h, w), mode='bilinear', align_corners=False)
        logits = (logits[:b] + logits[b:].flip(-1)) / 2
        # Keep signed prototype logits and their shared class scale for argmax.
        fused = logits if fused is None else fused + logits
    return fused / len(view_scales)


def multi_scale_cam2(model, inputs, scales, return_peaks=False):
    '''process cam and aux-cam'''
    # cam_list, tscam_list = [], []
    b, c, h, w = inputs.shape
    with torch.no_grad():
        inputs_cat = torch.cat([inputs, inputs.flip(-1)], dim=0)

        _cam_aux, _cam = model(inputs_cat, cam_only=True)
        # Eq. (11) uses the original image's raw classifier CAM peaks, before
        # ReLU, interpolation, flip aggregation, or per-class normalization.
        raw_peaks = _cam[:b].flatten(2).amax(dim=2) if return_peaks else None

        _cam = F.interpolate(_cam, size=(h, w), mode='bilinear', align_corners=False)
        _cam = torch.max(_cam[:b, ...], _cam[b:, ...].flip(-1))
        _cam_aux = F.interpolate(_cam_aux, size=(h, w), mode='bilinear', align_corners=False)
        _cam_aux = torch.max(_cam_aux[:b, ...], _cam_aux[b:, ...].flip(-1))

        cam_list = [F.relu(_cam)]
        cam_aux_list = [F.relu(_cam_aux)]

        for s in scales:
            if s != 1.0:
                _inputs = F.interpolate(inputs, size=(int(s * h), int(s * w)), mode='bilinear', align_corners=False)
                inputs_cat = torch.cat([_inputs, _inputs.flip(-1)], dim=0)

                _cam_aux, _cam = model(inputs_cat, cam_only=True)

                _cam = F.interpolate(_cam, size=(h, w), mode='bilinear', align_corners=False)
                _cam = torch.max(_cam[:b, ...], _cam[b:, ...].flip(-1))
                _cam_aux = F.interpolate(_cam_aux, size=(h, w), mode='bilinear', align_corners=False)
                _cam_aux = torch.max(_cam_aux[:b, ...], _cam_aux[b:, ...].flip(-1))

                cam_list.append(F.relu(_cam))
                cam_aux_list.append(F.relu(_cam_aux))

        cam = torch.sum(torch.stack(cam_list, dim=0), dim=0)
        cam = cam + F.adaptive_max_pool2d(-cam, (1, 1))
        cam /= F.adaptive_max_pool2d(cam, (1, 1)) + 1e-5

        cam_aux = torch.sum(torch.stack(cam_aux_list, dim=0), dim=0)
        cam_aux = cam_aux + F.adaptive_max_pool2d(-cam_aux, (1, 1))
        cam_aux /= F.adaptive_max_pool2d(cam_aux, (1, 1)) + 1e-5

    if return_peaks:
        return cam, cam_aux, raw_peaks
    return cam, cam_aux

def multi_scale_cam2_filter(model, inputs, scales):
    # cam_list, tscam_list = [], []
    b, c, h, w = inputs.shape
    with torch.no_grad():
        inputs_cat = torch.cat([inputs, inputs.flip(-1)], dim=0)

        _cam_aux, _cam = model(inputs_cat, cam_only=True)

        _cam = F.interpolate(_cam, size=(h, w), mode='bilinear', align_corners=False)
        _cam = torch.max(_cam[:b, ...], _cam[b:, ...].flip(-1))
        _cam_aux = F.interpolate(_cam_aux, size=(h, w), mode='bilinear', align_corners=False)
        _cam_aux = torch.max(_cam_aux[:b, ...], _cam_aux[b:, ...].flip(-1))

        cam_list = [F.relu(_cam)]
        cam_aux_list = [F.relu(_cam_aux)]

        for s in scales:
            if s != 1.0:
                _inputs = F.interpolate(inputs, size=(int(s * h), int(s * w)), mode='bilinear', align_corners=False)
                inputs_cat = torch.cat([_inputs, _inputs.flip(-1)], dim=0)

                _cam_aux, _cam = model(inputs_cat, cam_only=True)

                _cam = F.interpolate(_cam, size=(h, w), mode='bilinear', align_corners=False)
                _cam = torch.max(_cam[:b, ...], _cam[b:, ...].flip(-1))
                _cam_aux = F.interpolate(_cam_aux, size=(h, w), mode='bilinear', align_corners=False)
                _cam_aux = torch.max(_cam_aux[:b, ...], _cam_aux[b:, ...].flip(-1))

                cam_list.append(F.relu(_cam))
                cam_aux_list.append(F.relu(_cam_aux))

        cam = torch.sum(torch.stack(cam_list, dim=0), dim=0)
        cam = cam + F.adaptive_max_pool2d(-cam, (1, 1))
        cam_max = F.adaptive_max_pool2d(cam, (1, 1)).squeeze(-1).squeeze(-1) #[b,c]
        cam /= F.adaptive_max_pool2d(cam, (1, 1)) + 1e-5


        cam_aux = torch.sum(torch.stack(cam_aux_list, dim=0), dim=0)
        cam_aux = cam_aux + F.adaptive_max_pool2d(-cam_aux, (1, 1))
        cam_aux /= F.adaptive_max_pool2d(cam_aux, (1, 1)) + 1e-5

    return cam, cam_aux, cam_max


def label_to_aff_mask(cam_label, ignore_index=255):
    b, h, w = cam_label.shape

    _cam_label = cam_label.reshape(b, 1, -1)
    _cam_label_rep = _cam_label.repeat([1, _cam_label.shape[-1], 1])
    _cam_label_rep_t = _cam_label_rep.permute(0, 2, 1)
    aff_label = (_cam_label_rep == _cam_label_rep_t).type(torch.long)

    for i in range(b):
        aff_label[i, :, _cam_label_rep[i, 0, :] == ignore_index] = ignore_index
        aff_label[i, _cam_label_rep[i, 0, :] == ignore_index, :] = ignore_index
    aff_label[:, range(h * w), range(h * w)] = ignore_index
    return aff_label


def refine_cams_with_bkg_v2(ref_mod=None, images=None, cams=None, cls_labels=None, high_thre=None, low_thre=None,
                            ignore_index=False, img_box=None, down_scale=2):
    b, _, h, w = images.shape
    _images = F.interpolate(images, size=[h // down_scale, w // down_scale], mode="bilinear", align_corners=False)

    bkg_h = torch.ones(size=(b, 1, h, w)) * high_thre
    bkg_h = bkg_h.to(cams.device)
    bkg_l = torch.ones(size=(b, 1, h, w)) * low_thre
    bkg_l = bkg_l.to(cams.device)

    bkg_cls = torch.ones(size=(b, 1))
    bkg_cls = bkg_cls.to(cams.device)
    cls_labels = torch.cat((bkg_cls, cls_labels), dim=1)

    refined_label = torch.ones(size=(b, h, w)) * ignore_index
    refined_label = refined_label.to(cams.device)
    refined_label_h = refined_label.clone()
    refined_label_l = refined_label.clone()

    cams_with_bkg_h = torch.cat((bkg_h, cams), dim=1)
    _cams_with_bkg_h = F.interpolate(cams_with_bkg_h, size=[h // down_scale, w // down_scale], mode="bilinear",
                                     align_corners=False)  # .softmax(dim=1)
    cams_with_bkg_l = torch.cat((bkg_l, cams), dim=1)
    _cams_with_bkg_l = F.interpolate(cams_with_bkg_l, size=[h // down_scale, w // down_scale], mode="bilinear",
                                     align_corners=False)  # .softmax(dim=1)

    for idx, coord in enumerate(img_box):
        valid_key = torch.nonzero(cls_labels[idx, ...])[:, 0]
        valid_cams_h = _cams_with_bkg_h[idx, valid_key, ...].unsqueeze(0).softmax(dim=1)
        valid_cams_l = _cams_with_bkg_l[idx, valid_key, ...].unsqueeze(0).softmax(dim=1)

        _refined_label_h = _refine_cams(ref_mod=ref_mod, images=_images[[idx], ...], cams=valid_cams_h,
                                        valid_key=valid_key, orig_size=(h, w))
        _refined_label_l = _refine_cams(ref_mod=ref_mod, images=_images[[idx], ...], cams=valid_cams_l,
                                        valid_key=valid_key, orig_size=(h, w))

        refined_label_h[idx, coord[0]:coord[1], coord[2]:coord[3]] = _refined_label_h[0, coord[0]:coord[1],
                                                                     coord[2]:coord[3]]
        refined_label_l[idx, coord[0]:coord[1], coord[2]:coord[3]] = _refined_label_l[0, coord[0]:coord[1],
                                                                     coord[2]:coord[3]]

    refined_label = refined_label_h.clone()
    refined_label[refined_label_h == 0] = ignore_index
    refined_label[(refined_label_h + refined_label_l) == 0] = 0

    return refined_label


def _refine_cams(ref_mod, images, cams, valid_key, orig_size):
    refined_cams = ref_mod(images, cams)
    refined_cams = F.interpolate(refined_cams, size=orig_size, mode="bilinear", align_corners=False)
    refined_label = refined_cams.argmax(dim=1)
    refined_label = valid_key[refined_label]

    return refined_label


def get_mixed_label(cam_label, old_segs_label, total_classes, new_classes):
    # b H W
    cam_label = cam_label.long()
    old_segs_label = old_segs_label.long()

    new_segs_label = torch.full_like(cam_label, 255).long()
    cam_old_classes = total_classes - new_classes
    cam_new_label = torch.full_like(cam_label, 255).long()
    cam_new_label[cam_label >= cam_old_classes + 1] = cam_label[cam_label >= cam_old_classes + 1]

    new_segs_label = old_segs_label
    new_segs_label[cam_new_label != 255] = cam_new_label[cam_new_label != 255]

    return new_segs_label



def cam_high_pass_filter(cam_channel_max, cls_label, cam_value_thre=20):
    low_activation_channel = cam_channel_max < cam_value_thre
    cls_label_filter = cls_label.clone()
    cls_label_filter[low_activation_channel] = 0

    b, c = cls_label.shape
    for i in range(b):
        if torch.sum(cls_label_filter[i]) == 0:
            valid_channels = cls_label[i].nonzero(as_tuple=False).squeeze(1)
            if valid_channels.numel() > 0:
                max_idx = torch.argmax(cam_channel_max[i, valid_channels])
                max_channel = valid_channels[max_idx]
                cls_label_filter[i, max_channel] = 1

    return cls_label_filter

def filter_cam_pesudo_label(cam_label, cam_filter_class, ignore_idx=255):

    B, H, W = cam_label.shape
    bg_class = torch.ones((B, 1), dtype=cam_filter_class.dtype, device=cam_filter_class.device)
    cam_filter_class = torch.cat((bg_class,cam_filter_class),dim=1)
    
    cam_label_filtered = cam_label.clone()

    for i in range(B):

        class_mask = cam_filter_class[i].bool()  # [C + 1]
        valid_classes = torch.arange(class_mask.size(0), device=cam_label.device)[class_mask]
        keep_mask = torch.isin(cam_label_filtered[i], valid_classes)  
        cam_label_filtered[i][~keep_mask] = ignore_idx

    return cam_label_filtered
