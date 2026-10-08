"""Evidence-supported class exclusion; no conversion of uncertainty to BG."""
import torch

@torch.no_grad()
def calibrated_targets(reference,spatial_absent,new_tags,temperature=2.):
    assert spatial_absent.shape==(reference.shape[0],15) and new_tags.shape==(reference.shape[0],5)
    forbidden=torch.cat([torch.zeros_like(spatial_absent[:,:1]),spatial_absent.bool(),~new_tags.bool()],1)
    q=(reference.detach()/temperature).softmax(1)
    allowed=~forbidden[:,:,None,None]
    projected=q*allowed
    target=projected/projected.sum(1,keepdim=True).clamp_min(1e-30)
    # Every remaining class keeps its relative probability. A veto does not
    # identify the correct alternative, so never substitute hard background.
    return {'reference':q,'target':target,'forbidden':forbidden,'removed_mass':(q*~allowed).sum(1)}
