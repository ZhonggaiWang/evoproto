import torch
from model.pixel_kd import global_ratio,distributed_sum,world_size,pixel_kd_loss

@torch.no_grad()
def targets(reference,proposal,selected,temperature=2.):
    q=(reference.detach()/temperature).softmax(1);prediction=reference.argmax(1)
    active=selected&(proposal!=prediction)&(proposal>=0)&(proposal<21)
    target=q.clone();winner=q.gather(1,prediction[:,None])[:,0]
    candidate=proposal.clamp(0,20)
    previous=q.gather(1,candidate[:,None])[:,0]
    target.scatter_(1,candidate[:,None],torch.where(active,winner,previous)[:,None])
    target.scatter_(1,prediction[:,None],torch.where(active,previous,winner)[:,None])
    return {'reference':q,'target':target,'active':active,'prediction':prediction,'proposal':proposal}

def loss(student,reference,proposal,selected,old_logits,par,cams,trusted_old,temperature=2.):
    z=targets(reference,proposal,selected,temperature);q=z['target'];active=z['active']
    kl=(q*(q.clamp_min(1e-30).log()-(student/temperature).log_softmax(1))).sum(1)*temperature**2
    retained=~active;retention=global_ratio((kl*retained).sum(),retained.sum().float())
    group=torch.where(proposal>0,proposal,z['prediction']+21).clamp(0,41);ids=group.flatten()
    counts=distributed_sum(kl.new_zeros(42).scatter_add_(0,ids,active.float().flatten()))
    numerator=kl.new_zeros(42).scatter_add_(0,ids,(kl*active).flatten())
    inv=torch.where(counts>0,counts.clamp_min(1).rsqrt(),0.);den=counts.sqrt().sum().clamp_min(1)
    reported=(distributed_sum(numerator)*inv).sum()/den;local=world_size()*(numerator*inv).sum()/den
    correction=reported+local-local.detach()
    gated=cams.detach().clone();gated[:,:15]*=(trusted_old&~active)[:,None]
    h,w=par.shape[-2:];boxes=torch.tensor([[0,h,0,w]]*len(par))
    kd,stats=pixel_kd_loss(student,old_logits,par,gated,boxes,temperature)
    return retention+.1*correction+.1*kd,{'retention':float(retention.detach()),'correction':float(reported),
        'KD':float(kd.detach()),'KD_pixels':stats['kd_nonzero_pixels'],'correction_pixels':int(counts.sum()),'correction_counts':counts.tolist()}
