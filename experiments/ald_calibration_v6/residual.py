"""Learn corrections outside the existing classifier's semantic row space."""
import torch
import torch.nn.functional as F

class SemanticResidual(torch.nn.Module):
    def __init__(self,weights):
        super().__init__()
        w=weights.detach().flatten(1).float()
        # Compute the full nullspace in double precision; preserve all21
        # existing class directions, not just examples in current-step data.
        _,s,vh=torch.linalg.svd(w.double(),full_matrices=True)
        rank=int((s>s.max()*max(w.shape)*torch.finfo(torch.float64).eps).sum())
        basis=vh[rank:].T.to(w.dtype).contiguous()
        assert basis.shape==(512,512-rank) and rank<=21
        self.register_buffer('basis',basis)
        self.register_buffer('reference_weights',w[:,:,None,None].clone())
        self.delta=torch.nn.Parameter(torch.zeros(5,basis.shape[1],1,1,device=w.device))
        self.rank=rank

    def features(self,x):return F.conv2d(x,self.basis.T[:,:,None,None])

    def forward(self,reference,residual_features):
        new=reference[:,16:]+F.conv2d(residual_features,self.delta)
        return torch.cat([reference[:,:16],new],1)

    def folded_new_weight(self):
        change=self.delta.flatten(1)@self.basis.T
        return self.reference_weights[16:]+change[:,:,None,None]
