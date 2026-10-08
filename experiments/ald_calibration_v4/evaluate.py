"""Four independent, resumable shards of checkpoint inference, without training."""
from pathlib import Path
import os,sys,argparse,json,time,traceback,socket
SRC=Path('/ML-vePFS/infra_rd/kun/others/wzg/workspace/evoproto/experiments/prototype_sep_v1/c_new_anchor/src')
ROOT=Path('/ML-vePFS/infra_rd/kun/others/wzg/workspace/evoproto')
sys.path.insert(0,str(SRC))
os.environ['PYTHONDONTWRITEBYTECODE']='1'
sys.dont_write_bytecode=True
cache=ROOT/'.runtime/prototype_sep_v1/8card_eval'
for name,sub in {'TMPDIR':'tmp','XDG_CACHE_HOME':'cache','MPLCONFIGDIR':'mpl','CUDA_CACHE_PATH':'cuda'}.items():
    p=ROOT.parents[1]/'.kd8tmp' if name=='TMPDIR' else cache/sub
    p.mkdir(parents=True,exist_ok=True);os.environ[name]=str(p)
os.environ['MPLBACKEND']='Agg'
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader,Subset
import tasks
from datasets import voc
from model.model_seg_neg import network
from utils.evaluate import _fast_hist
from kd_runtime import atomic_json,digest,now,safe_path

def evaluate(job, rank, shards, out):
    cfg=job['config']; step=job['step']
    torch.cuda.set_device(rank)
    torch.set_float32_matmul_precision('high');torch.backends.cudnn.allow_tf32=True
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    checkpoint=Path(job['checkpoint'])
    if digest(checkpoint)!=job['checkpoint_sha256']: raise RuntimeError('Checkpoint hash changed')
    classes=tasks.get_per_task_classes('voc',cfg['task'],step)
    k=sum(classes);old=k-classes[-1]-1
    model=network(backbone=cfg['backbone'],num_classes=k,classes_list=classes,
                  pretrained=False,init_momentum=cfg['momentum'],aux_layer=cfg['aux_layer'])
    state=torch.load(checkpoint,map_location='cpu',weights_only=True)
    model.load_state_dict(state['model_state'],strict=True);del state
    model.cuda().eval()
    parent_path=ROOT/'runs/pair_preserving_kd_v1/formal/10-5/step2/checkpoints/model_final.pth'
    assert digest(parent_path)=='3f43207d665ce6001aa149867435364e13baca5d47a0e2274dfb9187975dd28d'
    parent=torch.load(parent_path,map_location='cpu',weights_only=True,mmap=True)
    import copy
    parent_classifier=copy.deepcopy(model.classifier).eval().requires_grad_(False)
    parent_classifier.load_state_dict({key.removeprefix('classifier.'):v for key,v in parent['model_state'].items() if key.startswith('classifier.')},strict=True)
    del parent
    classification={key:np.zeros((20,2,2),dtype=np.int64) for key in ['reference','candidate']}
    ds=voc.VOC12SegDataset(root_dir=cfg['data_folder'],name_list_dir=cfg['list_folder'],
         split=cfg['val_set'],stage='val',aug=False,ignore_index=cfg['ignore_index'],
         num_classes=cfg['num_classes'],tasks=cfg['task'],step=step)
    ds.label_dir=cfg['val_label_dir']
    if cfg.get('val_limit'): ds.name_list=ds.name_list[:cfg['val_limit']]
    indices=list(range(rank,len(ds),shards))
    loader=DataLoader(Subset(ds,indices),batch_size=1,shuffle=False,num_workers=2)
    hist=np.zeros((k,k),dtype=np.int64);names=[]
    with torch.inference_mode():
        for names_batch,images,labels,cls_label in loader:
            images=F.interpolate(images.cuda(),size=(cfg['crop_size'],cfg['crop_size']),mode='bilinear',align_corners=False)
            cls,segs,feature,_=model(images)
            old_cls=parent_classifier(model.pooling(feature,(1,1))).flatten(1)
            # Classification GT is used only after independent model outputs.
            truth=cls_label.numpy().astype(bool)[0]
            for key,logit in [('reference',old_cls),('candidate',cls)]:
                positive=(logit>0).cpu().numpy()[0]
                for c in range(20):classification[key][c,int(truth[c]),int(positive[c])]+=1
            pred=F.interpolate(segs,size=labels.shape[1:],mode='bilinear',align_corners=False).argmax(1).cpu().numpy()[0]
            hist+=_fast_hist(labels.numpy()[0].flatten(),pred.flatten(),k)
            names.append(names_batch[0])
    atomic_json(out,{'rank':rank,'shards':shards,'images':names,'histogram':hist.tolist(),
             'checkpoint_sha256':job['checkpoint_sha256'],'step':step,'iteration':job['iteration'],
             'classification':{key:v.tolist() for key,v in classification.items()},
             'host':socket.gethostname(),'finished_utc':now()})
    del model;torch.cuda.empty_cache()

def merge(job,part_dir,shards):
    files=[part_dir/f'rank{i}.json' for i in range(shards)]
    if not all(p.exists() for p in files): return
    output=part_dir/'result.json'
    if output.exists(): return
    records=[json.loads(p.read_text()) for p in files]
    assert all(x['checkpoint_sha256']==job['checkpoint_sha256'] for x in records)
    names=[name for x in records for name in x['images']]
    assert len(set(names))==len(names), 'Duplicate validation images'
    cfg=job['config'];step=job['step']
    split=Path(cfg['list_folder'])/'incremental_split'/f"val_{cfg['task']}_step_{step+1}.txt"
    expected=split.read_text().splitlines()
    if cfg.get('val_limit'):expected=expected[:cfg['val_limit']]
    assert set(names)==set(expected), 'Missing or extra validation images'
    hist=np.sum([np.array(x['histogram'],dtype=np.int64) for x in records],axis=0)
    denominator=hist.sum(1)+hist.sum(0)-hist.diagonal()
    iou=np.divide(hist.diagonal(),denominator,out=np.full(len(hist),np.nan),where=denominator>0)*100
    old=sum(tasks.get_per_task_classes('voc',cfg['task'],step-1))-1
    initial=tasks.get_per_task_classes('voc',cfg['task'],0)[0]-1
    def mean(v):return float(np.nanmean(v)) if np.isfinite(v).any() else None
    result={'step':step,'iteration':job['iteration'],'images':len(names),'checkpoint_sha256':job['checkpoint_sha256'],
        'all_miou':mean(iou),'foreground_miou':mean(iou[1:]),'previous_foreground_miou':mean(iou[1:old+1]),
        'current_foreground_miou':mean(iou[old+1:]),'old_initial10_miou':mean(iou[1:initial+1]),
        'new_since_initial_miou':mean(iou[initial+1:]),'class_iou':{voc.class_list[i]:float(v) if np.isfinite(v) else None for i,v in enumerate(iou)},
        'histogram':hist.tolist(),'old_gt_to_new_pixels':int(hist[1:old+1,old+1:].sum()),
        'old_gt_to_background_pixels':int(hist[1:old+1,0].sum()),'old_gt_pixels':int(hist[1:old+1].sum()),
        'new_gt_to_old_pixels':int(hist[old+1:,1:old+1].sum()),'new_gt_pixels':int(hist[old+1:].sum()),
        'finished_utc':now(),'evaluation':'historical baseline resize448, single-scale main head, original GT space'}
    reference=job.get('reference_metrics')
    if reference:
        target=json.loads(Path(reference).read_text().splitlines()[-1])
        errors={key:abs(result['class_iou'][key]-value) for key,value in target['class_iou'].items() if value is not None}
        result['reference_max_class_iou_error']=max(errors.values())
        assert max(errors.values())<1e-6, errors
    atomic_json(output,result)
    print('EVALUATED',part_dir.name,{k:result[k] for k in ('all_miou','previous_foreground_miou','current_foreground_miou')},flush=True)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',required=True);parser.add_argument('--rank',type=int,required=True)
    parser.add_argument('--shards',type=int,default=4);parser.add_argument('--once',action='store_true');args=parser.parse_args()
    run=safe_path(args.run);queue=run/'eval_queue';results=run/'evaluations'
    while True:
        for path in sorted(queue.glob('*.json')):
            job=json.loads(path.read_text());parts=results/path.stem;output=parts/f'rank{args.rank}.json'
            if not output.exists():
                try:evaluate(job,args.rank,args.shards,output)
                except Exception:
                    atomic_json(parts/f'error_rank{args.rank}.json',{'error':traceback.format_exc(),'time':now()});raise
            if args.rank==0:merge(job,parts,args.shards)
        if args.once:break
        if (run/'training_complete.json').exists():
            if all((results/p.stem/'result.json').exists() for p in queue.glob('*.json')):break
        time.sleep(15)

if __name__=='__main__':main()
