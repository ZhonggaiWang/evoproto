"""Evidence-defined, probability-conserving correction targets. No pixel GT."""
import torch
import torch.nn.functional as F


@torch.no_grad()
def background_directions(observer):
    # Historical full-row probabilities rank only; support/freshness are gates.
    counts = observer.broad_ema_counts[0]
    rates = counts / counts.sum().clamp_min(1e-12)
    eligible = (observer.broad_pair_image_observations[0] >= 3) & (rates >= .01)
    eligible[0] = False
    if int(observer.updates) < 100 or int(observer.broad_image_observations[0]) < 8:
        eligible.zero_()
    age = int(observer.updates)-1-int(observer.broad_last_seen_update[0])
    if age < 0 or age > 200:
        eligible.zero_()
    chosen = torch.zeros_like(eligible)
    ids = rates.masked_fill(~eligible, -1).topk(min(3, len(rates))).indices
    chosen[ids] = eligible[ids]
    return chosen


@torch.no_grad()
def correction_targets(reference, old_teacher, refined, cams, boxes, new_tags,
                       pair_targets, bg_targets, temperature=2., high=.7):
    n, c, h, w = reference.shape
    k = old_teacher.shape[1]
    assert c == 21 and k == 16 and new_tags.shape == (n, c-k)
    assert pair_targets.shape == bg_targets.shape == (c,)
    t = F.interpolate(old_teacher.detach(), (h,w), mode='bilinear', align_corners=False)
    teacher_pred = t.argmax(1)
    reference_pred = reference.detach().argmax(1)
    y = F.interpolate(refined[:,None].float(), (h,w), mode='nearest')[:,0].long()
    valid = torch.zeros_like(refined, dtype=torch.float)
    for b, box in enumerate(boxes):
        a,z,l,r = [int(x) for x in box]; valid[b,a:z,l:r] = 1
    valid = F.interpolate(valid[:,None], (h,w), mode='nearest')[:,0] > 0
    cam = F.interpolate(cams.detach(), (h,w), mode='bilinear', align_corners=False).clamp(0,1)
    top = cam.topk(2, dim=1)
    strength = top.values[:,0]; gap = (strength-top.values[:,1]).clamp_min(0)
    competitor = pair_targets[y.clamp(0,c-1)]
    reliability = (2*t.sigmoid().gather(1, teacher_pred[:,None])[:,0]-1).clamp_min(0).square()
    evidence = strength*gap*reliability
    new_pair = (valid & (y>=k) & (y<c) & (top.indices[:,0]+1 == y) & (strength>=high)
                & (evidence>0) & (competitor>0) & (competitor<k)
                & (teacher_pred==competitor) & (reference_pred==competitor))
    present = new_tags.gather(1,(reference_pred-k).clamp(0,c-k-1).flatten(1)).reshape_as(y)
    absent = valid & (reference_pred>=k) & (present==0) & (teacher_pred==0)
    bg_pair = (valid & ~absent & (reference_pred>0) & bg_targets[reference_pred]
               & (y==0) & (teacher_pred==0) & (cam[:,k-1:].amax(1)<high))
    assert not (new_pair & (bg_pair | absent)).any()
    q = (reference.detach()/temperature).softmax(1)
    desired = q.clone()
    # Swap only the two probabilities: all other classes and pair sum unchanged.
    source = torch.where(new_pair, y, torch.zeros_like(y)).clamp(0,c-1)
    rival = reference_pred
    pair_mask = new_pair | bg_pair
    qs = q.gather(1,source[:,None]); qr = q.gather(1,rival[:,None])
    delta = (qr-qs)*pair_mask[:,None]
    desired.scatter_add_(1,source[:,None],delta)
    desired.scatter_add_(1,rival[:,None],-delta)
    # An absent class is impossible but the correct class may be foreground.
    # Condition on NOT that class, preserving all remaining relative probabilities.
    absent_onehot = F.one_hot(rival,c).permute(0,3,1,2).bool() & absent[:,None]
    restricted = q.masked_fill(absent_onehot,0)
    restricted = restricted/restricted.sum(1,keepdim=True).clamp_min(1e-12)
    desired = torch.where(absent[:,None],restricted,desired)
    weights = torch.where(new_pair,evidence,torch.ones_like(evidence))
    return dict(reference=q,target=desired,valid=valid,new_pair=new_pair,bg_pair=bg_pair,
                absent=absent,active=new_pair|bg_pair|absent,weights=weights,
                source=source,reference_prediction=reference_pred,pseudo=y)
