import unittest
import torch
import torch.nn.functional as F
from experiments.restore_proto_v1.mechanism import ConfusionProto

class Tests(unittest.TestCase):
    def setUp(self): torch.manual_seed(7)
    def fixture(self,mode='full',new_only=False):
        m=ConfusionProto(4,3,'cpu',mode)
        m.ema[1,2]=1;m.ema[3,2]=1;m.mass[:]=2
        m.seen[1][2]={'a','b','c'};m.seen[3][2]={'a','b','c'}
        f=torch.randn(2,5,2,2,requires_grad=True)
        p=torch.randn(4,5,requires_grad=True)
        tp=p[:3].detach().clone().requires_grad_(True)
        z=torch.einsum('bdhw,kd->bkhw',F.normalize(f,dim=1),F.normalize(p,dim=1))/.1
        teacher=torch.randn(2,3,2,2,requires_grad=True)
        main=torch.randn(2,4,2,2,requires_grad=True)
        anchor=torch.ones(2,2,2,dtype=torch.long);anchor[1]=3
        cams=torch.ones(2,3,2,2)*.7;cams[:,2]=.1
        valid=torch.ones(2,2,2,dtype=torch.bool)
        pred=torch.ones(2,2,2,dtype=torch.long)
        teacher.data[:,1]=3
        par=anchor.clone()
        if new_only:par[:]=3
        loss,stats=m.loss(z,teacher,main,teacher,anchor,cams,valid,pred,par,p,tp)
        return loss,stats,f,p,tp,teacher,main
    def test_actual_prototype_and_feature_gradients(self):
        l,s,f,p,tp,t,main=self.fixture();l.backward()
        self.assertGreater(float(p.grad.abs().sum()),0)
        self.assertGreater(float(f.grad.abs().sum()),0)
        self.assertGreater(float(main.grad.abs().sum()),0)
        self.assertIsNone(tp.grad);self.assertIsNone(t.grad)
    def test_no_proto_ablation_removes_direct_proto_gradient(self):
        l,s,f,p,*_=self.fixture('without_proto');l.backward()
        self.assertEqual(float(p.grad.abs().sum()),0)
        self.assertEqual(float(f.grad.abs().sum()),0)
    def test_new_regions_veto_both_kds(self):
        _,s,*_=self.fixture(new_only=True)
        self.assertEqual(s['kd_proto'],0);self.assertEqual(s['kd_main'],0)
    def test_unique_images_required(self):
        m=ConfusionProto(4,3,'cpu')
        labels=torch.full((1,2,2),3,dtype=torch.long)
        logits=torch.zeros(1,4,2,2);logits[:,2]=5
        for _ in range(4):m.update(labels,logits,['same'])
        self.assertEqual(float(m.graph()[0].sum()),0)
        m.update(labels,logits,['second']);m.update(labels,logits,['third'])
        self.assertEqual(float(m.graph()[0][3,2]),1)
        self.assertEqual(float(m.graph()[0][:,0].sum()),0)
    def test_dual_cam_and_teacher_gates(self):
        m=ConfusionProto(4,3,'cpu')
        ca=torch.zeros(1,3,2,2);ca[:,2]=.9
        cb=ca.clone();tags=torch.ones(1,3)
        teacher=torch.zeros(1,3,2,2)
        par=torch.full((1,2,2),3);valid=torch.ones_like(par,dtype=torch.bool)
        a,*_=m.anchors(ca,cb,tags,teacher,par,valid,(2,2));self.assertTrue((a==3).all())
        cb[:,2]=0;cb[:,1]=.9
        a,*_=m.anchors(ca,cb,tags,teacher,par,valid,(2,2));self.assertTrue((a==255).all())
    def test_confusion_changes_objective(self):
        torch.manual_seed(7);a=self.fixture('full')[0]
        torch.manual_seed(7);b=self.fixture('without_confusion')[0]
        self.assertGreater(abs(float(a-b)),1e-7)

if __name__=='__main__':unittest.main()
