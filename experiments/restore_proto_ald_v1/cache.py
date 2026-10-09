"""Freeze three-state image evidence from this arm's own models and new tags."""
import argparse,hashlib,json,os,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import torch
import torch.distributed as dist
import torch.nn.functional as F
from model.model_seg_neg import network
from datasets.voc import VOC12ClsDataset
from utils.camutils import multi_scale_cam2
from experiments.restore_proto_ald_v1.evidence import calibrate_presence


EXPECTED_GPUS='GPU-82e069d7-189e-06b0-faed-b08e1794a981,GPU-2d0a204c-101b-1149-06e1-e3528d9f38bf'


def require_devices():
    if os.environ.get('CUDA_VISIBLE_DEVICES') != EXPECTED_GPUS:
        raise RuntimeError('Set CUDA_VISIBLE_DEVICES to the verified physical GPU 5/6 UUIDs: '+EXPECTED_GPUS)


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for data in iter(lambda:stream.read(8*1024*1024),b''):h.update(data)
    return h.hexdigest()


def load(path,stage,device):
    path=Path(path).resolve(strict=True);assert path.is_relative_to(ROOT)
    classes=[11]+[5]*stage
    net=network('vit_base_patch16_224',num_classes=sum(classes),classes_list=classes,pretrained=False,init_momentum=.9,aux_layer=-3)
    raw=torch.load(path,map_location='cpu',weights_only=True)
    net.load_state_dict({k.removeprefix('module.'):v for k,v in raw['model_state'].items()},strict=True)
    return net.to(device).eval().requires_grad_(False)


def main():
    p=argparse.ArgumentParser();p.add_argument('--teacher',required=True);p.add_argument('--reference',required=True);p.add_argument('--output',required=True);p.add_argument('--limit',type=int,default=0);a=p.parse_args()
    output=Path(a.output).resolve();assert output.is_relative_to(ROOT) and not output.exists()
    require_devices()
    dist.init_process_group('nccl');rank=dist.get_rank();world=dist.get_world_size();torch.cuda.set_device(rank);torch.set_num_threads(4)
    torch.set_float32_matmul_precision('high');torch.backends.cudnn.allow_tf32=True
    device=torch.device('cuda',rank);teacher=load(a.teacher,1,device);reference=load(a.reference,2,device)
    ds=VOC12ClsDataset(root_dir='/data/zhonggai/coco/PascalVOC12',name_list_dir=str(ROOT/'datasets/voc'),split='train',stage='train',tasks='10-5',step=2,aug=False)
    if a.limit:ds.name_list=ds.name_list[:a.limit]
    rows=[]
    with torch.inference_mode():
        for index in range(rank,len(ds),world):
            name,image,tags=ds[index]
            x=F.interpolate(image[None].to(device),(448,448),mode='bilinear',align_corners=False)
            new_tags=torch.as_tensor(tags[15:],device=device)[None]
            views=torch.cat([x,x.flip(-1)])
            old_cls,old_aux,old_dense,_,_=teacher(views,step0=True)
            _,_,ref_dense,_,_,_,_=reference(views,cam_grad=True)
            tv=torch.stack([old_dense[0],old_dense[1].flip(-1)])[None]
            rv=torch.stack([ref_dense[0],ref_dense[1].flip(-1)])[None]
            cams,_=multi_scale_cam2(reference,x,scales=[1.,.5,1.5])
            states=calibrate_presence(old_cls[None],old_aux[None],tv,rv,cams,new_tags)
            rows.append((str(name),states['state'][0].cpu().tolist()))
            if len(rows)%50==0:print(json.dumps({'rank':rank,'images':len(rows)}),flush=True)
    gathered=[None]*world;dist.all_gather_object(gathered,rows)
    if rank==0:
        states=dict(item for part in gathered for item in part)
        assert len(states)==len(ds) and set(states)==set(map(str,ds.name_list))
        counts={str(s):sum(v.count(s) for v in states.values()) for s in (-1,0,1)}
        result={'states':states,'images':len(ds),'teacher':str(Path(a.teacher).resolve()),'teacher_sha256':digest(a.teacher),'reference':str(Path(a.reference).resolve()),'reference_sha256':digest(a.reference),'counts':counts,'uses_old_gt_tags':False,'uses_pixel_gt':False,'time':time.time()}
        output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(result,indent=2))
        print(json.dumps({'images':len(ds),'counts':counts}),flush=True)
    dist.destroy_process_group()

if __name__=='__main__':main()
