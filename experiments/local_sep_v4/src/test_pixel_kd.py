import unittest
import torch
from model.pixel_kd import pixel_kd_loss

def data():
    s=torch.zeros(1,4,2,2,requires_grad=True)
    t=torch.tensor([-2.,4.,-3.]).reshape(1,3,1,1).expand(1,3,2,2).clone().requires_grad_()
    labels=torch.ones(1,2,2,dtype=torch.long)
    cams=torch.zeros(1,3,2,2);cams[:,0]=1
    box=torch.tensor([[0,2,0,2]])
    return s,t,labels,cams,box

class Tests(unittest.TestCase):
    def test_old_common_offset_invariance(self):
        a=list(data());v=pixel_kd_loss(*a)[0]
        a[0]=a[0].detach().clone();a[0][:,1:3]+=50
        torch.testing.assert_close(v,pixel_kd_loss(*a)[0])
    def test_old_only_teacher_detached(self):
        args=data();v,_=pixel_kd_loss(*args);v.backward();s,t,*_=args
        self.assertGreater(s.grad[:,1:3].abs().sum().item(),0)
        self.assertEqual(s.grad[:,[0,3]].abs().sum().item(),0)
        self.assertIsNone(t.grad)
    def test_matching_outputs_zero(self):
        s,t,y,c,b=data();s=torch.cat((t,torch.randn(1,1,2,2)),1)
        self.assertLess(abs(pixel_kd_loss(s,t,y,c,b)[0].item()),1e-6)
    def test_new_outputs_independent(self):
        a=list(data());v=pixel_kd_loss(*a)[0];a[0]=a[0].detach().clone();a[0][:,[0,3]]=100
        torch.testing.assert_close(v,pixel_kd_loss(*a)[0])
    def test_new_region_veto(self):
        a=list(data());a[2].fill_(3);v,_=pixel_kd_loss(*a);v.backward()
        self.assertEqual(v.item(),0);self.assertEqual(a[0].grad.abs().sum().item(),0)
    def test_missing_old_cam_veto(self):
        a=list(data());a[3].zero_();self.assertEqual(pixel_kd_loss(*a)[0].item(),0)
    def test_continuous_new_cam_discount(self):
        a=list(data());v=pixel_kd_loss(*a)[0];a[3][:,2]=.5
        torch.testing.assert_close(pixel_kd_loss(*a)[0],v*.25)
    def test_teacher_bg_not_distilled(self):
        a=list(data());a[1]=a[1].detach().clone();a[1][:,0]=10
        self.assertEqual(pixel_kd_loss(*a)[0].item(),0)
    def test_all_invalid_finite_zero(self):
        a=list(data());a[4].zero_();v,_=pixel_kd_loss(*a);v.backward()
        self.assertEqual(v.item(),0);self.assertEqual(a[0].grad.abs().sum().item(),0)
    def test_padding_grad_zero(self):
        a=list(data());a[4]=torch.tensor([[0,1,0,1]]);v,_=pixel_kd_loss(*a);v.backward()
        self.assertGreater(a[0].grad[:,:,0,0].abs().sum().item(),0)
        self.assertEqual(a[0].grad[:,:,1,:].abs().sum().item(),0)

if __name__=='__main__':unittest.main()
