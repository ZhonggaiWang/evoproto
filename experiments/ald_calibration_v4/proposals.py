import torch
import torch.nn.functional as F

@torch.no_grad()
def proposals(presence,teacher_views,reference_views,par,cams,new_tags,high=.7,low=.25):
    n,_,_,h,w=reference_views.shape
    p=F.interpolate(par[:,None].float(),(h,w),mode='nearest')[:,0].long()
    cam=F.interpolate(cams,(h,w),mode='bilinear',align_corners=False).clamp(0,1)
    # Unknown old CAMs must remain visible for the background veto; exclusion
    # from hard foreground labels must not manufacture low-CAM background.
    safe=torch.cat([~presence['negative'],new_tags.bool()],1)
    background_cams=cam*safe[:,:,None,None]
    admitted=torch.cat([presence['positive'],new_tags.bool()],1)
    fore=cam*admitted[:,:,None,None];top=fore.topk(2,dim=1)
    strong=(top.indices[:,0]+1==p)&(top.values[:,0]>=high)&(top.values[:,0]>top.values[:,1])
    tp=teacher_views.argmax(2);rp=reference_views.argmax(2);changed=p!=rp[:,0]
    bg=(p==0)&(tp==0).all(1)&(background_cams.amax(1)<low)&changed
    old=(p>0)&(p<16)&strong&(tp[:,0]==p)&(tp[:,1]==p)&changed
    new=(p>=16)&(p<21)&strong&changed
    return {'labels':p,'background':bg,'old':old,'new_teacher_old':new&(tp[:,0]>0),
        'new_teacher_BG':new&(tp[:,0]==0),'reference_prediction':rp[:,0],'reference_stable':rp[:,0]==rp[:,1]}
