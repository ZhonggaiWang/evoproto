"""Final single-view student inference. No teacher, tags, graph, or postfilter."""
from common import *
import argparse,json,math
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from network import UnifiedNet
from datasets.transforms import normalize_img

def configure(gpu):
 torch.set_num_threads(1);torch.cuda.set_device(gpu);torch.manual_seed(0)
 torch.set_float32_matmul_precision('high');torch.backends.cudnn.allow_tf32=True;torch.backends.cudnn.benchmark=False

def load_student():
 path=U/'formal/model_final.pth';deployment=json.loads((U/'deployment.json').read_text())
 assert digest(path)==deployment['checkpoint_sha256']
 model=UnifiedNet();model.load_state_dict(torch.load(path,map_location='cpu',weights_only=True,mmap=True)['model_state'],strict=True)
 return model.cuda().eval().requires_grad_(False)

@torch.inference_mode()
def predict_tensor(model,image,protocol='aspect672'):
 h,w=image.shape[-2:]
 if protocol=='square448':size=(448,448)
 else:
  assert protocol=='aspect672';scale=672/math.sqrt(h*w);size=tuple(max(16,round(v*scale/16)*16) for v in (h,w))
 logits=model(F.interpolate(image,size=size,mode='bilinear',align_corners=False))[0]
 return F.interpolate(logits,size=(h,w),mode='bilinear',align_corners=False)

def main():
 p=argparse.ArgumentParser();p.add_argument('--image',required=True);p.add_argument('--output',required=True);p.add_argument('--gpu',type=int,default=0);p.add_argument('--protocol',choices=['square448','aspect672'],default='aspect672');a=p.parse_args()
 configure(a.gpu);rgb=np.asarray(Image.open(a.image).convert('RGB'))
 image=torch.from_numpy(normalize_img(rgb).transpose(2,0,1).copy())[None].cuda()
 mask=predict_tensor(load_student(),image,a.protocol).argmax(1)[0].cpu().numpy().astype(np.uint8)
 path=safe_path(Path(a.output));path.parent.mkdir(parents=True,exist_ok=True)
 if path.exists():raise FileExistsError(path)
 Image.fromarray(mask).save(path);print(json.dumps(dict(output=str(path),shape=list(mask.shape),protocol=a.protocol)))

if __name__=='__main__':main()
