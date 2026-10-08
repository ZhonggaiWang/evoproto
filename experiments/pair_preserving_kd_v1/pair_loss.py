"""Hierarchical distribution retention and evidence-targeted correction."""
import torch
from model.pixel_kd import global_ratio,distributed_sum,world_size


def hierarchical_kl(student,target,temperature=2.,old=15):
    logp=(student/temperature).log_softmax(1)
    q=target.detach();logq=q.clamp_min(1e-30).log()
    old_ids=list(range(1,old+1));other_ids=[0]+list(range(old+1,q.shape[1]))
    components=[];mass=student[:,0]*0
    for ids in [old_ids,other_ids]:
        qg=q[:,ids].sum(1);logqg=qg.clamp_min(1e-30).log()
        logpg=logp[:,ids].logsumexp(1)
        mass=mass+qg*(logqg-logpg)
        components.append((q[:,ids]*(logq[:,ids]-logqg[:,None]-logp[:,ids]+logpg[:,None])).sum(1))
    return (mass+components[0]+components[1])*temperature**2,tuple(x*temperature**2 for x in [mass,*components])


def pair_preserving_loss(student,z,temperature=2.):
    per_pixel,components=hierarchical_kl(student,z['target'],temperature)
    active=z['active'] & z['valid'];retained=z['valid'] & ~active
    retention=global_ratio((per_pixel*retained).sum(),retained.sum().float())
    # Binary trusted gates. Uncalibrated CAM/reliability scores do not dictate
    # exact gradient magnitudes; class sqrt-support balances the selected cases.
    group=torch.where(z['new_pair'],z['pseudo'],torch.where(z['bg_pair'],0,z['reference_prediction']+21)).clamp(0,41)
    counts=distributed_sum(torch.zeros(42,device=student.device).scatter_add_(0,group.flatten(),active.float().flatten()))
    numerator=torch.zeros(42,device=student.device).scatter_add_(0,group.flatten(),(per_pixel*active).flatten())
    inv=torch.where(counts>0,counts.clamp_min(1).rsqrt(),0.)
    den=counts.sqrt().sum().clamp_min(1)
    reported=(distributed_sum(numerator)*inv).sum()/den
    local=world_size()*(numerator*inv).sum()/den
    correction=reported+local-local.detach()
    loss=retention+.1*correction
    stats=distributed_sum(torch.stack([z['valid'].sum(),active.sum(),z['new_pair'].sum(),z['bg_pair'].sum(),z['absent'].sum()]).float())
    return loss,{'retention':float(retention.detach()),'correction':float(reported),'loss':float(loss.detach()),
        **dict(zip(['valid_pixels','active_pixels','new_pair_pixels','bg_pair_pixels','absent_pixels'],[int(v) for v in stats]))}
