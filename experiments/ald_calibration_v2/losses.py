"""Partial-label soft calibration and uncertainty on classifier-only claims."""
import torch
import torch.nn.functional as F
from model.pixel_kd import global_ratio,pixel_kd_loss,distributed_sum
from targets import calibrated_targets

def segmentation_loss(student,reference,labels,old_logits,par,cams,trusted_old,state,new_tags,temperature=2.):
    z=calibrated_targets(reference,state==0,new_tags,temperature)
    logp=(student/temperature).log_softmax(1)
    def divergence(q):return (q*(q.clamp_min(1e-30).log()-logp)).sum(1)*temperature**2
    kl=divergence(z['reference']);projected=divergence(z['target'])
    retention=global_ratio(kl.sum(),kl.new_tensor(kl.numel()))
    correction=global_ratio(projected.sum(),projected.new_tensor(projected.numel()))
    gated=cams.detach().clone();gated[:,:15]*=trusted_old[:,None]
    h,w=labels.shape[-2:];boxes=torch.tensor([[0,h,0,w]]*len(labels))
    kd,stats=pixel_kd_loss(student,old_logits,par,gated,boxes,temperature)
    removed=global_ratio(z['removed_mass'].sum(),kl.new_tensor(kl.numel()))
    return retention+.1*correction+.1*kd,{'retention':float(retention.detach()),'projected_KL':float(correction.detach()),
        'conditional_KD':float(kd.detach()),'removed_mass_mean':float(removed),'KD_pixels':stats['kd_nonzero_pixels']}

def classification_loss(logits,reference,state,new_tags,spatial_absent):
    target=torch.cat([state.clamp_min(0),new_tags],1).to(logits.dtype)
    known=torch.cat([state>=0,torch.ones_like(new_tags,dtype=torch.bool)],1)
    target=target[:,None].expand(-1,2,-1).reshape_as(logits)
    known=known[:,None].expand(-1,2,-1).reshape_as(logits)
    loss=F.binary_cross_entropy_with_logits(logits,target,reduction='none')
    supervised=global_ratio((loss*known).sum(),known.sum().float())
    # Only ambiguous image classes with positive reference classification and
    # no spatial support in either model/view receive an uncertainty target.
    conflict=(state<0)&spatial_absent
    conflict=torch.cat([conflict,torch.zeros_like(new_tags,dtype=torch.bool)],1)[:,None].expand(-1,2,-1).reshape_as(logits)&(reference.detach()>0)
    neutral=F.binary_cross_entropy_with_logits(logits,torch.full_like(logits,.5),reduction='none')
    uncertainty=global_ratio((neutral*conflict).sum(),conflict.sum().float())
    q=reference.detach().sigmoid()
    kl=F.binary_cross_entropy_with_logits(logits,q,reduction='none')-F.binary_cross_entropy_with_logits(reference.detach(),q,reduction='none')
    retain=global_ratio(kl.sum(),kl.new_tensor(kl.numel()))
    return retain+.1*supervised+.1*uncertainty,{'classification_retention':float(retain.detach()),'classification_supervision':float(supervised.detach()),
        'uncertainty_supervision':float(uncertainty.detach()),'conflicts':int(distributed_sum(conflict.sum()))}
