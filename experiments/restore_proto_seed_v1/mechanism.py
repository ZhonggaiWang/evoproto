"""Seed-supported correction of teacher old-class absorption of novel classes.
All inputs are model predictions, frozen teacher features and image-level tags.
No pixel annotations, image history features, or additional model are used.
"""
import torch
import torch.nn.functional as F
import torch.distributed as dist


def resize(x,size):return F.interpolate(x,size=size,mode='bilinear',align_corners=False)
def nearest(x,size):return F.interpolate(x[:,None].float(),size=size,mode='nearest')[:,0]

def flood(seed,allowed):
    region=seed.clone()
    for _ in range(seed.numel()):
        new=(F.max_pool2d(region[None,None].float(),3,stride=1,padding=1)[0,0]>0)&allowed
        new |= seed
        if torch.equal(new,region):break
        region=new
    return region


class SeedCorrection:
    def __init__(self,classes,old_count,device,mode='correct'):
        self.k,self.oc,self.mode=classes,old_count,mode
        self.count=torch.zeros(classes,old_count,device=device)
        self.mass=torch.zeros(classes,device=device)
        self.seen=[[set() for _ in range(old_count)] for _ in range(classes)]
        self.updates=0

    @torch.no_grad()
    def evidence(self,features,cams,aux,tags,teacher,par,valid):
        size=features.shape[-2:]
        ca=resize(cams.detach(),size)*tags[:,:,None,None]
        cb=resize(aux.detach(),size)*tags[:,:,None,None]
        av,ai=ca.max(1);bv,bi=cb.max(1);a=ai+1
        pp=nearest(par,size).long();vv=nearest(valid,size).bool()
        tp=resize(teacher.detach(),size).argmax(1)
        trusted=vv&(av>=.7)&(bv>=.7)&(ai==bi)&(pp==a)
        trusted &= (a>=self.oc)|(tp==a)
        anchor=a.masked_fill(~trusted,255)
        bg=vv&(tp==0)&(pp==0)&(av<.25)&(bv<.25)
        anchor[bg]=0
        return F.normalize(features.detach(),dim=1),ca,cb,tp,anchor,vv

    @torch.no_grad()
    def update(self,anchor,tp,names):
        valid=(anchor>=self.oc)&(anchor<self.k)
        codes=(anchor.clamp_max(self.k-1)*self.oc+tp)[valid]
        count=torch.bincount(codes,minlength=self.k*self.oc).reshape(self.k,self.oc).float()
        mass=torch.bincount(anchor[valid],minlength=self.k).float()
        evidence=[]
        for n,name in enumerate(names):
            for code in torch.unique((anchor[n].clamp_max(self.k-1)*self.oc+tp[n])[valid[n]]).tolist():
                evidence.append((code//self.oc,code%self.oc,str(name)))
        if dist.is_initialized():
            dist.all_reduce(count);dist.all_reduce(mass)
            gathered=[None]*dist.get_world_size();dist.all_gather_object(gathered,evidence)
            evidence=[x for group in gathered for x in group]
        for a,b,name in evidence:self.seen[a][b].add(name)
        self.count.mul_(.99).add_(count,alpha=.01);self.mass.mul_(.99).add_(mass,alpha=.01)
        self.updates+=1

    def graph(self):
        support=torch.tensor([[len(x)>=3 for x in row] for row in self.seen],device=self.count.device)
        rate=self.count/self.mass[:,None].clamp_min(1e-6)
        rate=rate*support;rate[:,0]=0;rate[:self.oc]=0
        values,indices=rate.topk(min(2,self.oc),dim=1)
        return torch.zeros_like(rate).scatter(1,indices,values)>0

    @torch.no_grad()
    def correct(self,evidence,labels,valid,active=True):
        f,ca,cb,tp,anchor,vv=evidence
        graph=self.graph();out=labels.clone();native_label=nearest(labels,f.shape[-2:]).long()
        proposals=torch.full_like(anchor,255);best=torch.full_like(ca[:,0],-2.)
        potential=0;seed_pixels=0;seed_classes=0
        for i in range(len(f)):
            refs={};radii={}
            for c in torch.unique(anchor[i]).tolist():
                if c==255:continue
                mask=anchor[i]==c
                if mask.sum()<2:continue
                feats=f[i,:,mask]
                q=F.normalize(feats.mean(1),dim=0)
                refs[c]=q
                radii[c]=torch.quantile(q@feats,.1)
            for c,q in refs.items():
                if c<self.oc:continue
                seed=anchor[i]==c;seed_pixels+=int(seed.sum());seed_classes+=1
                negative=[p for other,p in refs.items() if other!=c]
                if not negative:continue
                score=torch.einsum('c,chw->hw',q,f[i])
                negative_score=torch.einsum('kc,chw->khw',torch.stack(negative),f[i]).amax(0)
                allowed=vv[i]&(score>=radii[c])&(score>negative_score)
                allowed &= (ca[i,c-1]>=.25)&(cb[i,c-1]>=.25)
                region=flood(seed,allowed)
                target=region&(tp[i]>0)&(native_label[i]>0)&(native_label[i]<self.oc)
                # Never overwrite a corroborated old-class anchor.
                target &= ~((anchor[i]>0)&(anchor[i]<self.oc))
                potential+=int(target.sum())
                if self.mode!='no_graph':target &= graph[c,tp[i]]
                margin=score-negative_score
                take=target&(margin>best[i]);best[i][take]=margin[take];proposals[i][take]=c
        proposal=nearest(proposals,labels.shape[-2:]).long()
        changed=(proposal>=self.oc)&(proposal<self.k)&valid&(labels>0)&(labels<self.oc)
        if self.mode!='no_graph':
            changed &= graph[proposal.clamp_max(self.k-1),labels.clamp_max(self.oc-1)]
        if not active or self.mode=='off':changed=torch.zeros_like(changed)
        if self.mode=='ignore':out[changed]=255
        else:out[changed]=proposal[changed]
        stats=dict(seed_pixels=seed_pixels,seed_class_images=seed_classes,potential_native_pixels=potential,
                   corrected_pixels=int(changed.sum()),valid_pixels=int(valid.sum()),edges=int(graph.sum()))
        return out,changed,proposal,stats

    def state(self):
        return dict(count=self.count.cpu().tolist(),mass=self.mass.cpu().tolist(),
                    unique_image_support=[[len(x) for x in row] for row in self.seen],updates=self.updates,
                    interpretation='Row new class from reliable CAM/PAR seed; column frozen teacher prediction; background retained in denominator but never correction target')
