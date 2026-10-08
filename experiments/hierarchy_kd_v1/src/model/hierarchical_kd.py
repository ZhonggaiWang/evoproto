"""Directed old-mass preservation and evidence-grounded foreground rejection.

No pixel-GT interface. Routing uses the PREVIOUS observer snapshot; values only
rank directions, never scale a loss. Existing conditional old-class KD remains
in the trainer unchanged. Teacher background corresponds to student BG+new.
"""
import torch
import torch.nn.functional as F
from model.pixel_kd import distributed_sum, world_size


@torch.no_grad()
def select_mass_directions(observer, old_classes, min_updates=100,
                           min_row_images=8, min_pair_images=3, max_stale=200):
    c, device = observer.classes, observer.broad_ema_counts.device
    old = torch.zeros(c, dtype=torch.bool, device=device)
    bg = torch.zeros_like(old)
    if int(observer.updates) < min_updates:
        return old, bg
    age = int(observer.updates)-1-observer.broad_last_seen_update
    fresh = ((observer.broad_last_seen_update >= 0) & (age >= 0) & (age <= max_stale)
             & (observer.broad_image_observations >= min_row_images))
    counts = observer.broad_ema_counts
    rates = counts/counts.sum(1, keepdim=True).clamp_min(1e-12)
    eligible = (observer.broad_pair_image_observations >= min_pair_images) & (counts > 0)
    # Three directions per error family: bounded coverage, no rate-to-weight map.
    scores = rates[1:old_classes+1, 0].masked_fill(
        ~(eligible[1:old_classes+1, 0] & fresh[1:old_classes+1]), -1)
    ids = torch.argsort(scores, descending=True, stable=True)[:3]
    old[ids[scores[ids] >= 0]+1] = True
    scores = rates[0, 1:].masked_fill(~(eligible[0, 1:] & fresh[0]), -1)
    ids = torch.argsort(scores, descending=True, stable=True)[:3]
    bg[ids[scores[ids] >= 0]+1] = True
    return old, bg


def group_log_prob(logits, old_classes, temperature=2.):
    z = logits/temperature
    old = z[:, 1:old_classes+1].logsumexp(1)
    other = torch.cat((z[:, :1], z[:, old_classes+1:]), 1).logsumexp(1)
    return torch.stack((old, other), 1).log_softmax(1)


def balanced_mean(values, mask, groups, classes, weights=None):
    """Exact global sqrt-support normalization with DDP-averaged gradients."""
    if weights is None:
        weights = torch.ones_like(values)
    count = torch.zeros(classes, device=values.device).scatter_add_(0, groups.flatten(), mask.float().flatten())
    numerator = torch.zeros_like(count).scatter_add_(0, groups.flatten(), (values*mask*weights).flatten())
    counts = distributed_sum(count)
    scale = torch.where(counts > 0, counts.clamp_min(1).rsqrt(), 0)/counts.sqrt().sum().clamp_min(1)
    summed = distributed_sum(numerator)
    reported = (summed*scale).sum()
    local = world_size()*(numerator*scale).sum()
    return reported+(local-local.detach()), counts, reported


def hierarchical_kd_loss(student, teacher, par_labels, cams, img_box,
                         new_image_tags, old_directions, bg_directions,
                         temperature=2., high=.7):
    n,c,h,w = student.shape
    k = teacher.shape[1]
    old = k-1
    if not 1 < k < c or temperature <= 0 or new_image_tags.shape != (n,c-k):
        raise ValueError('Invalid incremental class boundary, temperature or new tags')
    with torch.no_grad():
        t = F.interpolate(teacher.detach(), (h,w), mode='bilinear', align_corners=False)
        p = t.argmax(1)
        pred = student.detach().argmax(1)
        y = F.interpolate(par_labels[:,None].float(), (h,w), mode='nearest')[:,0].long()
        cam = F.interpolate(cams.detach(), (h,w), mode='bilinear', align_corners=False).clamp(0,1)
        top = cam.topk(2, dim=1)
        winner, runner = top.values[:,0], top.values[:,1]
        valid = torch.zeros_like(par_labels, dtype=torch.float)
        for i, box in enumerate(img_box):
            a,b,l,r = [int(v) for v in box]; valid[i,a:b,l:r] = 1
        valid = F.interpolate(valid[:,None],(h,w),mode='nearest')[:,0] > 0
        newcam = cam[:,old:].amax(1)
        strict = (valid & (p>0) & (y==p) & (top.indices[:,0]+1==p)
                  & (winner>=high) & (winner>runner) & (newcam<high))
        confidence = t.sigmoid().gather(1,p[:,None])[:,0]
        reliability = (2*confidence-1).clamp_min(0).square()
        mass_mask = strict & old_directions[p] & (reliability>0)
        mass_weight = reliability*winner*(1-newcam).square()
        # A BG+new mixture, never teacher-BG -> student-BG on new objects.
        target_log = group_log_prob(t, old, temperature)
        target = target_log.exp()
        background = valid & (y==0) & (p==0) & (newcam<high)
        bg_mask = background & (pred>0) & bg_directions[pred]
        present = new_image_tags.detach().gather(1,(pred-k).clamp(0,c-k-1).flatten(1)).reshape_as(pred)
        absent_mask = valid & (p==0) & (pred>=k) & (present==0) & bg_directions[pred]
        # Give absent-class complementary supervision priority on overlap.
        bg_mask &= ~absent_mask
    mass_values = (target*(target_log-group_log_prob(student,old,temperature))).sum(1)*temperature**2
    mass_loss, mass_counts, mass_reported = balanced_mean(mass_values,mass_mask,p,c,mass_weight)
    winning_logit = student.gather(1,pred[:,None])[:,0]
    bg_values = F.softplus(winning_logit-student[:,0])
    # Exclude the absent class without assigning GT=BG to possible old objects.
    alternatives = student.masked_fill(F.one_hot(pred,c).permute(0,3,1,2).bool(), -torch.inf).logsumexp(1)
    absent_values = F.softplus(winning_logit-alternatives)
    rejection_values = torch.where(absent_mask, absent_values, bg_values)
    reject_loss, reject_counts, reject_reported = balanced_mean(rejection_values,bg_mask|absent_mask,pred,c)
    packed = distributed_sum(torch.stack((valid.sum(),strict.sum(),mass_mask.sum(),bg_mask.sum(),absent_mask.sum())).float())
    return mass_loss,reject_loss,{
        'mass_kd':float(mass_reported),'background_rejection':float(reject_reported),
        'valid_pixels':int(packed[0]),'strict_old_pixels':int(packed[1]),
        'mass_pixels':int(packed[2]),'background_pair_pixels':int(packed[3]),
        'absent_new_pixels':int(packed[4]),'mass_class_counts':mass_counts.tolist(),
        'rejection_class_counts':reject_counts.tolist(),
        'old_to_BG_sources':old_directions.nonzero().flatten().tolist(),
        'BG_to_foreground_targets':bg_directions.nonzero().flatten().tolist(),
        'teacher_BG_mapping':'student_BG_plus_new', 'rates_are_weights':False}
