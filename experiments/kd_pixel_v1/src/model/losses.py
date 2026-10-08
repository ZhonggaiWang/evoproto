import pdb
import torch
import torch.nn as nn
from torch.autograd import Function
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import sys
import torch.distributed as dist
from model.pixel_kd import global_ratio
sys.path.append("./wrapper/bilateralfilter/build/lib.linux-x86_64-3.8")
# from bilateralfilter import bilateralfilter, bilateralfilter_batch

def get_masked_ptc_loss(inputs, mask):
    b, c, h, w = inputs.shape
    
    inputs = inputs.reshape(b, c, h*w)

    def cos_sim(x):
        x = F.normalize(x, p=2, dim=1, eps=1e-8)
        cos_sim = torch.matmul(x.transpose(1,2), x)
        return torch.abs(cos_sim)

    inputs_cos = cos_sim(inputs)

    pos_mask = mask == 1
    neg_mask = mask == 0
    loss = 0.5*(1 - global_ratio(torch.sum(pos_mask * inputs_cos), pos_mask.sum(), 1)) + 0.5 * global_ratio(torch.sum(neg_mask * inputs_cos), neg_mask.sum(), 1)
    return loss


def prototype_distillation_loss(student_prototypes, teacher_prototypes):
    """Align matching old foreground prototypes to a frozen teacher.

    Both tensors have shape [classes, feature_dim], with background at index 0.
    The student's first teacher_prototypes.shape[0] rows must use the same class
    order as the teacher. Newly added student classes are left unconstrained.
    """
    if student_prototypes.ndim != 2 or teacher_prototypes.ndim != 2:
        raise ValueError("Prototype tensors must have shape [classes, feature_dim].")
    if student_prototypes.shape[1] != teacher_prototypes.shape[1]:
        raise ValueError("Student and teacher prototype feature dimensions must match.")
    if student_prototypes.shape[0] < teacher_prototypes.shape[0]:
        raise ValueError("The student must contain all teacher classes in the same order.")

    num_old_classes = teacher_prototypes.shape[0]
    if num_old_classes <= 1:
        return student_prototypes.sum() * 0.0

    student_old = F.normalize(student_prototypes[1:num_old_classes], p=2, dim=1)
    teacher_old = F.normalize(teacher_prototypes.detach()[1:], p=2, dim=1)
    matching_similarity = (student_old * teacher_old).sum(dim=1).clamp(-1.0, 1.0)
    return (1.0 - matching_similarity).mean()


def prototype_separation_loss(prototypes, margin=0.0):
    """Penalize foreground prototype pairs whose cosine exceeds a fixed margin.

    Background (row 0) and self-pairs are excluded. Each unordered pair is
    counted once; pairs already separated by the margin have zero penalty.
    """
    if prototypes.ndim != 2:
        raise ValueError("Prototypes must have shape [classes, feature_dim].")
    if prototypes.shape[0] <= 2:
        return prototypes.sum() * 0.0

    foreground = F.normalize(prototypes[1:], p=2, dim=1)
    similarity = foreground @ foreground.t()
    pair_indices = torch.triu_indices(
        foreground.shape[0], foreground.shape[0], offset=1, device=prototypes.device
    )
    pair_similarity = similarity[pair_indices[0], pair_indices[1]].clamp(-1.0, 1.0)
    return F.relu(pair_similarity - margin).square().mean()


def get_seg_loss(pred, label, class_weight=None, ignore_index=255):
    ignore_mask = (label != ignore_index).float().unsqueeze(1)  # [B, 1, H, W]

    # one-hot label
    num_classes = pred.size(1)
    one_hot_label = F.one_hot(torch.clamp(label, max=num_classes - 1), num_classes=num_classes).permute(0, 3, 1,
                                                                                                         2).float()
    weight = class_weight.view(1, len(class_weight), 1, 1)  # reshape 成可广播形状 [1, 11, 1, 1]
    weight = weight.expand_as(pred)

    loss = F.binary_cross_entropy_with_logits(pred, one_hot_label, weight=weight, reduction='none')
    # Preserve the sum over classes while keeping all-ignore crops differentiable.
    loss = global_ratio((loss * ignore_mask).sum(), ignore_mask.sum())

    return loss


def get_type_seg_loss(pred, label, ignore_index=255):
    ignore_mask = ((label != ignore_index)&(label != 0)).float().unsqueeze(1)  # [B, 1, H, W]

    # one-hot label
    num_classes = pred.size(1)
    one_hot_label = F.one_hot(torch.clamp(label, max=num_classes - 1), num_classes=num_classes).permute(0, 3, 1,
                                                                                                        2).float()
    loss = F.binary_cross_entropy_with_logits(pred, one_hot_label, reduction='none')
    loss = (loss * ignore_mask).sum() / ignore_mask.sum()

    return loss

def get_energy_loss(img, logit, label, img_box, loss_layer, mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375]):

    pred_prob = F.softmax(logit, dim=1)
    crop_mask = torch.zeros_like(pred_prob[:,0,...])

    for idx, coord in enumerate(img_box):
        crop_mask[idx, coord[0]:coord[1], coord[2]:coord[3]] = 1

    _img = torch.zeros_like(img)
    _img[:,0,:,:] = img[:,0,:,:] * std[0] + mean[0]
    _img[:,1,:,:] = img[:,1,:,:] * std[1] + mean[1]
    _img[:,2,:,:] = img[:,2,:,:] * std[2] + mean[2]

    loss = loss_layer(_img, pred_prob, crop_mask, label.type(torch.uint8).unsqueeze(1), )

    return loss.cuda()
class CTCLoss_neg(nn.Module):
    def __init__(self, ncrops=10, temp=1.0,):
        super().__init__()
        self.temp = temp
        # self.center_momentum = center_momentum
        self.ncrops = ncrops
        # self.register_buffer("center", torch.zeros(1, out_dim))
        # we apply a warm up for the teacher temperature because
        # a too high temperature makes the training instable at the beginning
        # self.teacher_temp_schedule = np.concatenate((
        #     np.linspace(warmup_teacher_temp, teacher_temp, warmup_teacher_temp_epochs),
        #     np.ones(nepochs - warmup_teacher_temp_epochs) * teacher_temp
        # ))

    def forward(self, student_output, teacher_output, flags):
        """
        Cross-entropy between softmax outputs of the teacher and student networks.
        """

        b = flags.shape[0]

        student_out = student_output.reshape(self.ncrops, b, -1).permute(1,0,2)
        teacher_out = teacher_output.reshape(2, b, -1).permute(1,0,2)

        logits = torch.matmul(teacher_out, student_out.permute(0,2,1))
        logits = torch.exp(logits / self.temp)

        total_loss = 0
        for i in range(b):
            neg_logits = logits[i, :, flags[i]==0]
            pos_inds = torch.nonzero(flags[i])[:,0]
            loss = 0

            for j in pos_inds:
                pos_logit = logits[i, :, j]
                loss += -torch.log((pos_logit) / (pos_logit + neg_logits.sum(dim=1) + 1e-4))
            else:
                loss += -torch.log((1) / (1 + neg_logits.sum(dim=1) + 1e-4))
                
            total_loss += loss.sum() / 2 / (pos_inds.shape[0] + 1e-4)

        total_loss = total_loss / b

        return total_loss

class DenseEnergyLossFunction(Function):
    
    @staticmethod
    def forward(ctx, images, segmentations, sigma_rgb, sigma_xy, ROIs, unlabel_region):
        ctx.save_for_backward(segmentations)
        ctx.N, ctx.K, ctx.H, ctx.W = segmentations.shape
        Gate = ROIs.clone().to(ROIs.device)

        ROIs = ROIs.unsqueeze_(1).repeat(1,ctx.K,1,1)

        seg_max = torch.max(segmentations, dim=1)[0]
        Gate = Gate - seg_max
        Gate[unlabel_region] = 1
        Gate[Gate < 0] = 0
        Gate = Gate.unsqueeze_(1).repeat(1, ctx.K, 1, 1)

        segmentations = torch.mul(segmentations.cuda(), ROIs.cuda())
        ctx.ROIs = ROIs
        
        densecrf_loss = 0.0
        images = images.cpu().numpy().flatten()
        segmentations = segmentations.cpu().numpy().flatten()
        AS = np.zeros(segmentations.shape, dtype=np.float32)
        bilateralfilter_batch(images, segmentations, AS, ctx.N, ctx.K, ctx.H, ctx.W, sigma_rgb, sigma_xy)
        Gate = Gate.cpu().numpy().flatten()
        AS = np.multiply(AS, Gate)
        densecrf_loss -= np.dot(segmentations, AS)
    
        # averaged by the number of images
        densecrf_loss /= ctx.N
        
        ctx.AS = np.reshape(AS, (ctx.N, ctx.K, ctx.H, ctx.W))
        return Variable(torch.tensor([densecrf_loss]), requires_grad=True)
        
    @staticmethod
    def backward(ctx, grad_output):
        grad_segmentation = -2*grad_output*torch.from_numpy(ctx.AS)/ctx.N
        grad_segmentation = grad_segmentation.cuda()
        grad_segmentation = torch.mul(grad_segmentation, ctx.ROIs.cuda())
        return None, grad_segmentation, None, None, None, None
    

class DenseEnergyLoss(nn.Module):
    def __init__(self, weight, sigma_rgb, sigma_xy, scale_factor):
        super(DenseEnergyLoss, self).__init__()
        self.weight = weight
        self.sigma_rgb = sigma_rgb
        self.sigma_xy = sigma_xy
        self.scale_factor = scale_factor
    
    def forward(self, images, segmentations, ROIs, seg_label):
        """ scale imag by scale_factor """
        scaled_images = F.interpolate(images,scale_factor=self.scale_factor, recompute_scale_factor=True) 
        scaled_segs = F.interpolate(segmentations,scale_factor=self.scale_factor,mode='bilinear',align_corners=False, recompute_scale_factor=True)
        scaled_ROIs = F.interpolate(ROIs.unsqueeze(1),scale_factor=self.scale_factor, recompute_scale_factor=True).squeeze(1)
        scaled_seg_label = F.interpolate(seg_label,scale_factor=self.scale_factor,mode='nearest', recompute_scale_factor=True)
        unlabel_region = (scaled_seg_label.long() == 255).squeeze(1)

        return self.weight*DenseEnergyLossFunction.apply(
                scaled_images, scaled_segs, self.sigma_rgb, self.sigma_xy*self.scale_factor, scaled_ROIs, unlabel_region)
    
    def extra_repr(self):
        return 'sigma_rgb={}, sigma_xy={}, weight={}, scale_factor={}'.format(
            self.sigma_rgb, self.sigma_xy, self.weight, self.scale_factor
        )


def get_pixel_refine_cam_loss_v2(cam, gt_label, detach):
    b, h, w = gt_label.size()
    _, c, _, _ = cam.size()
    gt_label.cuda()
    cam.cuda()
    # cam=F.relu(cam)

    cambgmax = True
    # cambg=1-torch.mean(cam,dim=1,keepdim=True) if not cambgmax else 1-torch.max(cam,dim=1,keepdim=True)[0]
    cam_up = F.interpolate(cam, size=(h, w), mode='bilinear', align_corners=False)
    cam_up_flat = cam_up.view(b, c, -1).cuda()

    gt_label_flat = gt_label.view(b, -1)
    gt_valid_mask_flat = (gt_label_flat != 255).cuda()
    gt_label_fg = torch.zeros_like(cam_up_flat).cuda()
    for idx in range(b):
        for i in range(1, c + 1):
            gt_label_fg[idx, i - 1][gt_label_flat[idx] == i] = 1

    prc_loss = torch.tensor(0.).cuda()
    for i in range(b):
        prc_loss += F.multilabel_soft_margin_loss(cam_up_flat[i, :, gt_valid_mask_flat[i]],
                                                  gt_label_fg[i, :, gt_valid_mask_flat[i]])

    return prc_loss / b
