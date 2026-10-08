import torch

@torch.no_grad()
def retain_unsupported_old(proposed,teacher_prediction,presence_state):
    """A credible old image class vetoes new-class takeover.

    Unknown is not an absence label: it merely cannot veto the independent
    new-image-tag + strong refreshed CAM/PAR correction proposal.
    """
    ids=(teacher_prediction-1).clamp(0,14)
    supported=(presence_state==1).gather(1,ids.flatten(1)).reshape_as(teacher_prediction)
    return proposed&~supported
