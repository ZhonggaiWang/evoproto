"""Read-only validation diagnosis at training checkpoints; never updates graph."""
import torch
import torch.distributed as dist
import torch.nn.functional as F
import numpy as np
from utils import imutils
from utils.camutils import multi_scale_cam2,cam_to_label,refine_cams_with_bkg_v2
from experiments.restore_proto_seed_v1.mechanism import nearest

@torch.no_grad()
def diagnose(net,teacher,module,par,dataset,device,limit=80):
    k,oc=module.k,module.oc
    keys=['valid','new_pixels','old_pixels','old_on_new_before','old_on_new_after','new_on_old_before','new_on_old_after','changed_valid','changed_to_true_new','changed_from_true_old','seed_valid','seed_correct']
    sums=torch.zeros(len(keys),device=device,dtype=torch.int64)
    indices=np.linspace(0,len(dataset)-1,min(limit,len(dataset)),dtype=int)
    rank=dist.get_rank();world=dist.get_world_size()
    for index in indices[rank::world]:
        name,image,gt,tags=dataset[int(index)]
        x=F.interpolate(torch.as_tensor(image,device=device)[None],size=(448,448),mode='bilinear',align_corners=False)
        cls,_,old,features,_=teacher(x,step0=True)
        old=F.interpolate(old,size=(448,448),mode='bilinear',align_corners=False)
        tags=torch.cat([(cls>0).float(),torch.as_tensor(tags,device=device)[None,oc-1:k-1]],1)
        cams,aux=multi_scale_cam2(net,x,scales=[1.,.5,1.5]);box=torch.tensor([[0,448,0,448]])
        ca,_=cam_to_label(cams,cls_label=tags,img_box=box,ignore_mid=True,bkg_thre=.5,high_thre=.7,low_thre=.25,ignore_index=255)
        labels=refine_cams_with_bkg_v2(par,imutils.denormalize_img2(x.clone()),cams=ca,cls_labels=tags,high_thre=.7,low_thre=.25,ignore_index=255,img_box=box).long()
        valid=torch.ones_like(labels,dtype=torch.bool);before=old.argmax(1);new=(labels>=oc)&(labels<k);before[new]=labels[new]
        e=module.evidence(features,cams,aux,tags,old,labels,valid)
        after,changed,proposal,_=module.correct(e,before,valid,active=True)
        # GT is read by metrics only after proposal and supervision are fixed.
        target=nearest(torch.as_tensor(gt,device=device)[None],(448,448)).long();known=(target>=0)&(target<k)
        tn=known&(target>=oc);to=known&(target>0)&(target<oc);m=changed&known
        seed=e[4];ngt=nearest(target,seed.shape[-2:]).long();sv=(seed>=oc)&(seed<k)&(ngt<k)
        counts=[known.sum(),tn.sum(),to.sum(),(tn&(before>0)&(before<oc)).sum(),(tn&(after>0)&(after<oc)).sum(),(to&(before>=oc)&(before<k)).sum(),(to&(after>=oc)&(after<k)).sum(),m.sum(),(m&(target==proposal)).sum(),(m&to).sum(),sv.sum(),(sv&(seed==ngt)).sum()]
        sums+=torch.stack(counts)
    dist.all_reduce(sums)
    result=dict(zip(keys,map(int,sums.tolist())))
    result.update(images=len(indices),pixel_gt_used_in_selection=False,graph_updated=False,prospective_correction=True)
    return result
