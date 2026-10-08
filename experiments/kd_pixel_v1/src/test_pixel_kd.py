import unittest, math
import torch
from model.pixel_kd import student_to_teacher_logprobs, pixel_kd_loss

def loss(s,t,labels=None,cam=None,box=None):
    b,c,h,w=s.shape;k=t.shape[1]
    if labels is None:labels=torch.ones((b,h,w),dtype=torch.long)
    if cam is None:cam=torch.zeros((b,c-1,h,w))
    if box is None:box=torch.tensor([[0,h,0,w]]*b)
    return pixel_kd_loss(s,t,labels,cam,box,1.0)

class PixelKDTests(unittest.TestCase):
    def test_background_mass_is_aggregated(self):
        s=torch.tensor([.1,.7,.2]).log().reshape(1,3,1,1)
        q=student_to_teacher_logprobs(s,2,1).exp().flatten()
        torch.testing.assert_close(q,torch.tensor([.3,.7]))

    def test_background_new_mass_redistribution_is_invariant(self):
        t=torch.tensor([.3,.7]).log().reshape(1,2,1,1)
        a=torch.tensor([.1,.7,.2]).log().reshape(1,3,1,1)
        b=torch.tensor([.25,.7,.05]).log().reshape(1,3,1,1)
        torch.testing.assert_close(loss(a,t)[0],loss(b,t)[0],atol=1e-6,rtol=0)
        self.assertLess(abs(float(loss(a,t)[0])),1e-6)

    def test_teacher_is_detached(self):
        t=torch.tensor([1.,4.]).reshape(1,2,1,1).requires_grad_()
        s=torch.zeros(1,3,1,1,requires_grad=True)
        value,_=loss(s,t);value.backward()
        self.assertIsNone(t.grad);self.assertGreater(float(s.grad.abs().sum()),0)

    def test_new_cam_par_regions_are_vetoed(self):
        t=torch.tensor([0.,5.]).reshape(1,2,1,1)
        s=torch.zeros(1,3,1,1,requires_grad=True)
        value,stats=loss(s,t,torch.full((1,1,1),2,dtype=torch.long))
        value.backward();self.assertEqual(float(value),0);self.assertEqual(float(s.grad.abs().sum()),0)
        self.assertEqual(stats['veto_new_pixels'],1)

    def test_strong_new_cam_softly_attenuates_conflict(self):
        t=torch.tensor([0.,5.]).reshape(1,2,1,1)
        s=torch.zeros(1,3,1,1,requires_grad=True)
        cam=torch.zeros(1,2,1,1);cam[:,1]=.85
        a=loss(s,t)[0];b=loss(s,t,cam=cam)[0]
        torch.testing.assert_close(b,a*.5)

    def test_padding_has_no_kd_gradient(self):
        t=torch.tensor([0.,5.]).reshape(1,2,1,1).expand(1,2,2,2)
        s=torch.zeros(1,3,2,2,requires_grad=True)
        value,_=loss(s,t,box=torch.tensor([[0,1,0,1]]));value.backward()
        self.assertGreater(float(s.grad[:,:,0,0].abs().sum()),0)
        self.assertEqual(float(s.grad[:,:,1,:].abs().sum()),0)

    def test_all_invalid_is_differentiable_zero(self):
        s=torch.randn(2,4,2,2,requires_grad=True);t=torch.randn(2,3,2,2)
        value,_=loss(s,t,box=torch.zeros(2,4,dtype=torch.long));value.backward()
        self.assertEqual(float(value),0);self.assertEqual(float(s.grad.abs().sum()),0)

    def test_class_balance_ignores_background_multiplicity(self):
        # Duplicating identical background pixels does not overwhelm old foreground.
        t=torch.tensor([[5.,0.],[0.,5.]]).T.reshape(1,2,1,2)
        s=torch.tensor([[1.,1.,0.],[0.,1.,2.]]).T.reshape(1,3,1,2)
        ref=loss(s,t)[0]
        ii=torch.tensor([0]*20+[1])
        expanded=loss(s[:,:,:,ii],t[:,:,:,ii])[0]
        torch.testing.assert_close(ref,expanded)

    def test_uniform_teacher_has_zero_confidence(self):
        s=torch.randn(1,4,3,3,requires_grad=True);t=torch.zeros(1,3,3,3)
        value,_=loss(s,t)
        self.assertLess(abs(float(value)),1e-6)

if __name__=='__main__':unittest.main()
