from pathlib import Path
import sys,os,importlib.util,json
R=Path('/ML-vePFS/infra_rd/kun/others/wzg/workspace/evoproto');E=R/'experiments/kd_parallel_v1'
sys.path.insert(0,str(R/'experiments/kd_pixel_v2'));from run_kd import environment
os.environ.update(environment('8card'));os.environ['CUDA_VISIBLE_DEVICES']='7'
sys.path.insert(0,str(E/'a_sigmoid/src'))
import torch
from model.model_seg_neg import network
from kd_runtime import atomic_json,digest,now
spec=importlib.util.spec_from_file_location('model.original_network',R/'experiments/kd_pixel_v2/src/model/model_seg_neg.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
torch.manual_seed(0);torch.set_float32_matmul_precision('high')
kwargs=dict(backbone='vit_base_patch16_224',num_classes=16,classes_list=[11,5],pretrained=False,init_momentum=.9,aux_layer=-3)
a=m.network(**kwargs).cuda();b=network(**kwargs).cuda();b.load_state_dict(a.state_dict())
records=[]
for training in [False,True]:
 a.train(training);b.train(training)
 for size in [224,448,672]:
  x=torch.randn(2,3,size,size,device='cuda')
  with torch.no_grad():
   torch.manual_seed(17);before=a(x,cam_only=True)
   torch.manual_seed(17);after=b(x,cam_only=True)
  for first,last in zip(before,after):torch.testing.assert_close(first,last,rtol=0,atol=0)
  records.append({'training':training,'size':size,'max_abs_difference':0})
 x=torch.randn(1,3,448,448,device='cuda')
 torch.manual_seed(17);before=a(x,crops=[])
 torch.manual_seed(17);after=b(x,crops=[])
 for first,last in zip(before,after):torch.testing.assert_close(first,last,rtol=0,atol=0)
 before[1].square().mean().backward();after[1].square().mean().backward()
 for (n,p),(n2,q) in zip(a.named_parameters(),b.named_parameters()):
  assert n==n2
  if p.grad is not None:torch.testing.assert_close(p.grad,q.grad,rtol=0,atol=0)
 a.zero_grad();b.zero_grad()
atomic_json(E/'cam_equivalence.json',{'utc':now(),'passed':True,'cam_cases':records,'ordinary_forward_and_backward':'exact tensor equality in eval and train mode','original_sha256':digest(R/'experiments/kd_pixel_v2/src/model/model_seg_neg.py'),'optimized_sha256':digest(E/'a_sigmoid/src/model/model_seg_neg.py')})
print('PASS: CAM outputs and normal forward/backward exactly equal')
