from common import *
import json,random,time,math,argparse
import numpy as np
import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader,DistributedSampler
from network import load_parent
from data import TrainImages,worker_seed
from mechanism import calibrate,DirectedEvidence,relation_loss,ald_loss

def main():
 p=argparse.ArgumentParser();p.add_argument('--mode',choices=['smoke','formal'],required=True);a=p.parse_args()
 rank=int(os.environ['LOCAL_RANK']);torch.cuda.set_device(rank);torch.set_num_threads(1)
 dist.init_process_group('nccl');torch.manual_seed(0);np.random.seed(rank);random.seed(rank)
 torch.set_float32_matmul_precision('high');torch.backends.cudnn.allow_tf32=True;torch.backends.cudnn.benchmark=False
 run=safe_path(U/a.mode);run.mkdir(parents=True,exist_ok=True)
 teacher=load_parent().cuda().eval().requires_grad_(False);student=load_parent().cuda().train()
 # This stage calibrates the segmentation decision, retaining learned features.
 for name,p in student.named_parameters():p.requires_grad_(name.startswith('decoder.'))
 student.encoder.eval();student.classifier.eval();student.aux_classifier.eval()
 groups=[dict(params=[p for p in student.parameters() if p.requires_grad],lr=2e-5)]
 optimizer=torch.optim.AdamW(groups,weight_decay=.01)
 student=DDP(student,device_ids=[rank]);dataset=TrainImages()
 sampler=DistributedSampler(dataset,shuffle=True,seed=0,drop_last=True)
 loader=DataLoader(dataset,batch_size=4,sampler=sampler,num_workers=2,drop_last=True,worker_init_fn=worker_seed,pin_memory=True)
 graph=DirectedEvidence(21,len(dataset),rank);steps=8 if a.mode=='smoke' else 300
 source={p.name:digest(p) for p in E.glob('*.py')}
 if rank==0:
  assert not (run/'study.json').exists()
  atomic_json(run/'study.json',dict(parent=str(PARENT),parent_sha256=PARENT_SHA,source_sha256=source,steps=steps,global_batch=32,seed=0,teacher_count=1,new_parameters=0,optimizer='Fresh AdamW; cosine300; existing segmentation head2e-5; encoder/classifiers exactly frozen',objective='ALD + collapsed-group KD + one-sided within-pair SEP; one online directed graph; no image-classification update',train_images=dataset.names,scope='Unified within-stage output refinement warm-started from ALDv9; not a clean all-stage rerun or isolated causal comparison.',utc=now()))
 epoch=0;sampler.set_epoch(epoch);it=iter(loader);start=time.time();totals=torch.zeros(6,device=rank)
 for step in range(steps):
  try:ids,x,tags,valid=next(it)
  except StopIteration:epoch+=1;sampler.set_epoch(epoch);it=iter(loader);ids,x,tags,valid=next(it)
  ids,x,tags,valid=[v.to(rank,non_blocking=True) for v in (ids,x,tags,valid)]
  native_valid=F.avg_pool2d(valid[:,None],16,16)[:,0]>.999
  with torch.no_grad():
   ref=teacher(x);e=calibrate(ref,tags,native_valid)
   # Use preceding observations for eligibility; this batch cannot self-authorize.
   supported=graph.supported().clone()
  out=student(x);ald=ald_loss(out[0],e);kd,sep,details=relation_loss(out[0],ref[0],e,supported)
  loss=ald+kd+sep
  assert torch.isfinite(loss)
  optimizer.zero_grad(set_to_none=True);loss.backward()
  assert all(p.grad is None or torch.isfinite(p.grad).all() for p in student.parameters())
  factor=.5+.5*math.cos(math.pi*step/300)
  for group in optimizer.param_groups:group['lr']=2e-5*factor
  optimizer.step();graph.update(e,ref[0],ids)
  batch=torch.stack([e['valid'].sum(),(e['labels']!=255).sum(),e['foreground'].sum(),details['selected'].sum(),details['active'].sum(),((e['labels']>=16)&(e['labels']<21)).sum()]).float()
  dist.all_reduce(batch);totals+=batch
  if rank==0 and (step%25==0 or step+1==steps):
   record=dict(step=step+1,loss=float(loss),ald=float(ald),kd=float(kd),sep=float(sep),supported_pairs=graph.supported().nonzero().tolist(),cumulative_counts=totals.tolist(),seconds=time.time()-start)
   with safe_path(run/'metrics.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
   atomic_json(run/'status.json',dict(status='training',**record,utc=now()));print(json.dumps(record),flush=True)
 state=torch.cat([p.detach().flatten() for p in student.module.parameters()]);check=state.clone();dist.broadcast(check,0);assert torch.equal(state,check)
 assert all(p.grad is None for p in teacher.parameters())
 frozen=teacher.state_dict();current=student.module.state_dict()
 assert all(torch.equal(v,frozen[k]) for k,v in current.items() if not k.startswith('decoder.'))
 atomic_json(run/f'audit_rank{rank}.json',dict(passed=True,ddp_weights_exact=True,teacher_no_grad=True,finite_gradients=True,encoder_and_classifiers_bitwise_unchanged=True,steps=steps,utc=now()))
 if rank==0:
  ck=safe_path(run/'model_final.pth');torch.save(dict(model_state=student.module.state_dict(),optimizer_state=optimizer.state_dict(),graph_seen=graph.seen.cpu(),graph_counts=graph.counts().cpu(),steps=steps,parent_sha256=PARENT_SHA,source_sha256=source),ck)
  atomic_json(run/'graph.json',dict(unique_image_counts=graph.counts().tolist(),supported_pairs=graph.supported().nonzero().tolist(),minimum_distinct_images=3,training_images=dataset.names,utc=now()))
  atomic_json(run/'status.json',dict(status='training_complete',steps=steps,checkpoint=str(ck),checkpoint_sha256=digest(ck),elapsed_seconds=time.time()-start,utc=now()))
 dist.barrier();dist.destroy_process_group()

if __name__=='__main__':main()
