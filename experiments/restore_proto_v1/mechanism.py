"""Confusion-guided prototype prediction distillation and local separation.
All graph evidence uses train image tags, CAMs and a previous-stage teacher.
No training segmentation masks are consumed at incremental stages.
"""
import torch
import torch.distributed as dist
import torch.nn.functional as F


def total(x):
    y=x.detach().clone()
    if dist.is_initialized(): dist.all_reduce(y)
    return y


def ratio(num, den):
    world=dist.get_world_size() if dist.is_initialized() else 1
    return (total(num)+world*(num-num.detach()))/total(den).clamp_min(1)


def resize(x,size):
    return F.interpolate(x,size=size,mode='bilinear',align_corners=False)


class ConfusionProto:
    def __init__(self, classes, old_count, device, mode='full'):
        self.classes,self.old_count,self.mode=classes,old_count,mode
        self.ema=torch.zeros(classes,classes,device=device)
        self.mass=torch.zeros(classes,device=device)
        self.seen=[[set() for _ in range(classes)] for _ in range(classes)]
        self.updates=0

    @torch.no_grad()
    def anchors(self, cams, aux, tags, teacher, par, valid, size):
        ca=resize(cams.detach(),size)*tags[:,:,None,None]
        cb=resize(aux.detach(),size)*tags[:,:,None,None]
        av,ai=ca.max(1);bv,bi=cb.max(1);a=ai+1
        p=F.interpolate(par[:,None].float(),size=size,mode='nearest')[:,0].long()
        v=F.interpolate(valid[:,None].float(),size=size,mode='nearest')[:,0].bool()
        tp=resize(teacher.detach(),size).argmax(1)
        trusted=v & (av>=.7) & (bv>=.7) & (ai==bi) & (p==a)
        trusted &= (a>=self.old_count) | (tp==a)
        return a.masked_fill(~trusted,255),ca,v,tp,p

    @torch.no_grad()
    def update(self, labels, logits, names):
        k=self.classes
        winner=logits.detach().argmax(1)
        valid=(labels>0)&(labels<k)&(winner>0)&(winner!=labels)
        ids=(labels.clamp_max(k-1)*k+winner)[valid]
        counts=torch.bincount(ids,minlength=k*k).reshape(k,k).float()
        anchors=torch.bincount(labels[labels<k],minlength=k).float()
        # Unique image support cannot be inflated by repeatedly sampling one crop.
        evidence=[]
        for i,name in enumerate(names):
            pairs=torch.unique((labels[i].clamp_max(k-1)*k+winner[i])[valid[i]]).tolist()
            evidence.extend((int(pair)//k,int(pair)%k,str(name)) for pair in pairs)
        if dist.is_initialized():
            gathered=[None]*dist.get_world_size();dist.all_gather_object(gathered,evidence)
            evidence=[x for group in gathered for x in group]
        for a,b,name in evidence: self.seen[a][b].add(name)
        counts=total(counts);anchors=total(anchors)
        self.ema.mul_(.99).add_(counts,alpha=.01)
        self.mass.mul_(.99).add_(anchors,alpha=.01)
        self.updates+=1

    def graph(self):
        k=self.classes
        support=torch.tensor([[len(x)>=3 for x in row] for row in self.seen],device=self.ema.device)
        c=self.ema/self.mass[:,None].clamp_min(1e-6)
        c=c*support
        c[0]=0;c[:,0]=0;c.fill_diagonal_(0)
        if self.mode=='without_confusion':
            c=torch.ones_like(c);c[0]=0;c[:,0]=0;c.fill_diagonal_(0)
            return c/c.sum(1,keepdim=True).clamp_min(1),torch.ones(k,device=c.device)
        # At most two empirically supported competitors per anchor class.
        vals,idx=c.topk(min(2,k),dim=1)
        sparse=torch.zeros_like(c).scatter(1,idx,vals)
        sparse=sparse/sparse.sum(1,keepdim=True).clamp_min(1e-6)
        degree=1+(c.sum(1)+c.sum(0)).clamp(0,2)
        return sparse,degree.detach()

    def loss(self, proto, teacher_proto, main, teacher_main, anchors, cams, valid, teacher_prediction, par, student_p, teacher_p):
        size=proto.shape[-2:];k=self.old_count
        teacher_proto=resize(teacher_proto.detach(),size)
        teacher_main=resize(teacher_main.detach(),size)
        main=resize(main,size)
        graph,degree=self.graph()
        # Old-foreground conditional KL leaves BG/new competition unconstrained.
        old_support=cams[:,:k-1].gather(1,(teacher_prediction-1).clamp(0,k-2)[:,None])[:,0]
        new_support=cams[:,k-1:].amax(1)
        trusted_old=valid & (teacher_prediction>0) & (old_support>=.25) & ~((par>=k)&(par<self.classes))
        confidence=teacher_main.sigmoid().gather(1,teacher_prediction[:,None])[:,0]
        reliability=(2*confidence-1).clamp_min(0).square()*old_support*(1-new_support).square()
        weights=trusted_old*reliability*degree[teacher_prediction]
        def kd(s,t):
            ts=t[:,1:k]/2.;ss=s[:,1:k]/2.
            per=(ts.softmax(1)*(ts.log_softmax(1)-ss.log_softmax(1))).sum(1)*4
            count=torch.zeros(k,device=s.device).scatter_add_(0,teacher_prediction.flatten(),trusted_old.float().flatten())
            counts=total(count)
            inv=torch.where(counts>0,counts.clamp_min(1).rsqrt(),0.)
            return ratio((per*weights*inv[teacher_prediction]).sum(),counts.sqrt().sum()/(dist.get_world_size() if dist.is_initialized() else 1))
        kd_main=kd(main,teacher_main)
        kd_proto=kd(proto,teacher_proto)
        anchor=(anchors>0)&(anchors<self.classes)
        safe=anchors.clamp_max(self.classes-1)
        edge=graph[safe].permute(0,3,1,2)
        def separation(logits):
            own=logits.gather(1,safe[:,None])
            # Stop pushing once a reliable anchor beats its confusing rival by 0.5.
            per=(F.relu(.5-own+logits)*edge).sum(1)
            count=torch.zeros(self.classes,device=logits.device).scatter_add_(0,safe.flatten(),(anchor*(edge.sum(1)>0)).float().flatten())
            counts=total(count)
            inv=torch.where(counts>0,counts.clamp_min(1).rsqrt(),0.)
            return ratio((per*anchor*inv[safe]).sum(),counts.sqrt().sum()/(dist.get_world_size() if dist.is_initialized() else 1))
        sep_proto=separation(proto);sep_main=separation(main)
        geometry=(1-(F.normalize(student_p[1:k],dim=1)*F.normalize(teacher_p.detach()[1:k],dim=1)).sum(1)).mean()
        active=self.mode!='without_proto'
        loss=.1*kd_main+.02*sep_main
        if active: loss=loss+.05*kd_proto+.05*sep_proto+.01*geometry
        else: loss=loss+0*(kd_proto+sep_proto+geometry)
        stats={'kd_main':float(kd_main.detach()),'kd_proto':float(kd_proto.detach()),'sep_main':float(sep_main.detach()),'sep_proto':float(sep_proto.detach()),'geometry':float(geometry.detach()),'edges':int((graph>0).sum()),'trusted_anchors':int(total(anchor.sum())),'kd_pixels':int(total(trusted_old.sum())),'updates':self.updates,'mode':self.mode}
        return loss,stats

    def state(self):
        return {'ema':self.ema.cpu().tolist(),'anchor_mass':self.mass.cpu().tolist(),'unique_image_support':[[len(x) for x in row] for row in self.seen],'updates':self.updates,'mode':self.mode}
