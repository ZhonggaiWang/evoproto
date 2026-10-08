"""Production inference: one aspect-preserving view + anchored BG margin.

Input is ImageNet-normalized RGB NCHW (batch one). No image tags or GT.
The architecture and unchanged checkpoint are pinned by deployment.json.
"""
from pathlib import Path
import sys,os,math,argparse,json
R=Path('/ML-vePFS/infra_rd/kun/others/wzg/workspace/evoproto')
sys.path.insert(0,str(R/'experiments/kd_pixel_v2'))
from run_kd import environment
os.environ.update(environment('8card'))
from kd_runtime import safe_path,digest
S=R/'experiments/local_sep_v5/src';sys.path.insert(0,str(S))
import torch
import torch.nn.functional as F
from model.model_seg_neg import network
from refine import boundary_margin

CHECKPOINT=R/'runs/local_sep_v5/formal/10-5/step2/checkpoints/model_final.pth'
SHA256='0770ac07bcb343d510bbb6287526231352891e16f0c3e11e4b0d36a14fca4177'

def configure(gpu=0):
 torch.set_num_threads(1);torch.cuda.set_device(gpu);torch.manual_seed(0)
 torch.set_float32_matmul_precision('high');torch.backends.cudnn.allow_tf32=True
 torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True

def load_model():
 assert digest(CHECKPOINT)==SHA256,'Checkpoint changed'
 m=network(backbone='vit_base_patch16_224',num_classes=21,classes_list=[11,5,5],pretrained=False,init_momentum=.9,aux_layer=-3)
 m.load_state_dict(torch.load(CHECKPOINT,map_location='cpu',weights_only=True,mmap=True)['model_state'],strict=True)
 return m.cuda().eval().requires_grad_(False)

@torch.inference_mode()
def predict_tensor(model,image,refine=True):
 assert image.ndim==4 and image.shape[:2]==(1,3)
 h,w=image.shape[-2:];scale=672/math.sqrt(h*w)
 shape=(max(16,round(h*scale/16)*16),max(16,round(w*scale/16)*16))
 resized=F.interpolate(image,size=shape,mode='bilinear',align_corners=False)
 logits=model(resized)[1]
 logits=F.interpolate(logits,size=(h,w),mode='bilinear',align_corners=False)
 return boundary_margin(image,logits) if refine else logits

def main():
 p=argparse.ArgumentParser();p.add_argument('--image',required=True);p.add_argument('--output',required=True);p.add_argument('--gpu',type=int,default=0);a=p.parse_args()
 configure(a.gpu)
 from PIL import Image
 import numpy as np
 from datasets.transforms import normalize_img
 # Input read-only. All output is constrained to the authorized workspace.
 rgb=np.asarray(Image.open(a.image).convert('RGB'))
 inp=torch.from_numpy(normalize_img(rgb).transpose(2,0,1).copy())[None].float().cuda()
 mask=predict_tensor(load_model(),inp).argmax(1)[0].cpu().numpy().astype(np.uint8)
 output=safe_path(Path(a.output));output.parent.mkdir(parents=True,exist_ok=True)
 if output.exists():raise FileExistsError(output)
 Image.fromarray(mask).save(output)
 print(json.dumps(dict(output=str(output),checkpoint_sha256=SHA256,shape=list(mask.shape),classes=np.unique(mask).tolist())))

if __name__=='__main__':main()
