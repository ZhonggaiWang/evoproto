"""Continue an audited confusion graph without inventing historical image IDs."""
import json
import torch
from experiments.restore_proto_v1.mechanism import ConfusionProto


class ContinuedConfusionProto(ConfusionProto):
    def __init__(self, classes, old_count, device, mode, path):
        super().__init__(classes, old_count, device, mode)
        saved=json.loads(open(path).read())
        assert saved['mode']==mode
        self.ema.copy_(torch.tensor(saved['ema'],device=device))
        self.mass.copy_(torch.tensor(saved['anchor_mass'],device=device))
        self.historical_support=torch.tensor(saved['unique_image_support'],device=device)
        assert self.historical_support.shape==(classes,classes)
        self.updates=int(saved['updates'])

    def graph(self):
        if self.mode=='without_confusion':return super().graph()
        current=torch.tensor([[len(s)>=3 for s in row] for row in self.seen],device=self.ema.device)
        # A previous >=3-image certificate remains valid. Counts from old/new
        # periods are never added: they may involve the same images.
        supported=(self.historical_support>=3)|current
        c=self.ema/self.mass[:,None].clamp_min(1e-6)*supported
        c[0]=0;c[:,0]=0;c.fill_diagonal_(0)
        values,indices=c.topk(min(2,self.classes),dim=1)
        sparse=torch.zeros_like(c).scatter(1,indices,values)
        sparse=sparse/sparse.sum(1,keepdim=True).clamp_min(1e-6)
        return sparse,(1+(c.sum(1)+c.sum(0)).clamp(0,2)).detach()

    def state(self):
        state=super().state()
        state['historical_unique_image_support']=self.historical_support.cpu().tolist()
        state['support_semantics']='Historical support >=3 OR at least 3 distinct refinement images; counts not summed.'
        return state
