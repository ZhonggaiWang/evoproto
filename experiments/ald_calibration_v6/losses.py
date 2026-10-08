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

def loss(student,reference,proposal,selected,old_logits,par,cams,trusted_old,new_tags,temperature=2.):
    z=targets(reference,proposal,selected,temperature);q=z['target'];eligible=z['active']
    # A pseudo correction identifies a preferred class, not an exact margin.
    # Stop increasing that margin once it wins. Keep it excluded from old KD
    # even after it wins, so the teacher cannot immediately undo the correction.
    active=eligible&(student.detach().argmax(1)!=proposal)
    kl=(q*(q.clamp_min(1e-30).log()-(student/temperature).log_softmax(1))).sum(1)*temperature**2
    retained=~eligible;ref_ids=z['prediction'].flatten()
    retention_counts=distributed_sum(kl.new_zeros(21).scatter_add_(0,ref_ids,retained.float().flatten()))
    retention_numerator=kl.new_zeros(21).scatter_add_(0,ref_ids,(kl*retained).flatten())
    retention_inv=torch.where(retention_counts>0,retention_counts.clamp_min(1).rsqrt(),0.)
    retention_den=retention_counts.sqrt().sum().clamp_min(1)
    retention_reported=(distributed_sum(retention_numerator)*retention_inv).sum()/retention_den
    retention_local=world_size()*(retention_numerator*retention_inv).sum()/retention_den
    retention=retention_reported+retention_local-retention_local.detach()
    group=torch.where(proposal>0,proposal,z['prediction']+21).clamp(0,41);ids=group.flatten()
    counts=distributed_sum(kl.new_zeros(42).scatter_add_(0,ids,eligible.float().flatten()))
    active_counts=distributed_sum(kl.new_zeros(42).scatter_add_(0,ids,active.float().flatten()))
    numerator=kl.new_zeros(42).scatter_add_(0,ids,(kl*active).flatten())
    inv=torch.where(counts>0,counts.clamp_min(1).rsqrt(),0.);den=counts.sqrt().sum().clamp_min(1)
    reported=(distributed_sum(numerator)*inv).sum()/den;local=world_size()*(numerator*inv).sum()/den
    correction=reported+local-local.detach()
    assert torch.equal(student[:,:16],reference[:,:16]),'Old logits must remain exact in semantic residual training'
    negative,negative_counts=absence_loss(student,reference,new_tags)
    return retention+.1*correction+.1*negative,{'retention':float(retention.detach()),'correction':float(reported),
        'correction_pixels':int(active_counts.sum()),'correction_counts':active_counts.tolist(),
        'absence_loss':float(negative.detach()),'absence_reference_counts':negative_counts.tolist(),'eligible_counts':counts.tolist(),'eligible_pixels':int(counts.sum()),'retention_class_counts':retention_counts.tolist()}


def absence_loss(student,reference,new_tags):
    # Image GT constrains only the five current-new classes. Every old class
    # and BG remain possible; no hard alternative pseudo label is invented.
    allowed=torch.cat([torch.ones_like(new_tags[:,:1]).expand(-1,16),new_tags],1).bool()
    valid_logits=student.masked_fill(~allowed[:,:,None,None],-torch.inf)
    per=student.logsumexp(1)-valid_logits.logsumexp(1)
    ids=reference.detach().argmax(1).flatten()
    counts=distributed_sum(per.new_zeros(21).scatter_add_(0,ids,torch.ones_like(per).flatten()))
    num=per.new_zeros(21).scatter_add_(0,ids,per.flatten())
    inv=torch.where(counts>0,counts.clamp_min(1).rsqrt(),0.);den=counts.sqrt().sum().clamp_min(1)
    reported=(distributed_sum(num)*inv).sum()/den;local=world_size()*(num*inv).sum()/den
    return reported+local-local.detach(),counts
