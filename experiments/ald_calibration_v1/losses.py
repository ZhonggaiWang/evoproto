"""ALD trusts accepted labels, retains soft references on uncertain regions."""
import torch
import torch.nn.functional as F
from model.pixel_kd import global_ratio,distributed_sum,world_size,pixel_kd_loss

def balanced_mean(value,labels,mask,classes):
    ids=labels.clamp(0,classes-1).flatten()
    count=distributed_sum(value.new_zeros(classes).scatter_add_(0,ids,mask.float().flatten()))
    numerator=value.new_zeros(classes).scatter_add_(0,ids,(value*mask).flatten())
    inv=torch.where(count>0,count.clamp_min(1).rsqrt(),0.)
    den=count.sqrt().sum().clamp_min(1)
    reported=(distributed_sum(numerator)*inv).sum()/den
    local=world_size()*(numerator*inv).sum()/den
    return reported+local-local.detach(),count

def segmentation_loss(student,reference,labels,old_logits,par,cams,trusted_old,temperature=2.):
    q=(reference.detach()/temperature).softmax(1)
    kl=(q*(q.clamp_min(1e-30).log()-(student/temperature).log_softmax(1))).sum(1)*temperature**2
    retention=global_ratio(kl.sum(),kl.new_tensor(kl.numel()))
    mask=(labels>=0)&(labels<21)
    ce=F.cross_entropy(student,labels.clamp(0,20),reduction='none')
    supervised,counts=balanced_mean(ce,labels,mask,21)
    gated=cams.detach().clone();gated[:,:15]*=trusted_old[:,None]
    h,w=labels.shape[-2:];boxes=torch.tensor([[0,h,0,w]]*len(labels))
    kd,stats=pixel_kd_loss(student,old_logits,par,gated,boxes,temperature)
    # Weights fixed before evaluation; rates/CAM strengths are not treated as
    # calibrated probabilities or precision-derived learning weights.
    return retention+.1*supervised+.1*kd,{'retention':float(retention.detach()),'trusted_CE':float(supervised.detach()),
        'conditional_KD':float(kd.detach()),'trusted_counts':counts.tolist(),'KD_pixels':stats['kd_nonzero_pixels']}

def classification_loss(logits,reference,state,new_tags):
    # Two aligned image views, each with the same image-level target state.
    target=torch.cat([state.clamp_min(0),new_tags],1).to(logits.dtype)
    known=torch.cat([state>=0,torch.ones_like(new_tags,dtype=torch.bool)],1)
    target=target[:,None].expand(-1,2,-1).reshape_as(logits)
    known=known[:,None].expand(-1,2,-1).reshape_as(logits)
    loss=F.binary_cross_entropy_with_logits(logits,target,reduction='none')
    supervised=global_ratio((loss*known).sum(),known.sum().float())
    q=reference.detach().sigmoid()
    kl=F.binary_cross_entropy_with_logits(logits,q,reduction='none')-F.binary_cross_entropy_with_logits(reference.detach(),q,reduction='none')
    retain=global_ratio(kl.sum(),kl.new_tensor(kl.numel()))
    return retain+.1*supervised,{'classification_retention':float(retain.detach()),'classification_supervision':float(supervised.detach())}
