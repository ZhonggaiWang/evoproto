"""Bind a frozen learned adapter to bidirectionally supported training pairs."""
from pathlib import Path
import sys,os,json,subprocess
R=Path('/ML-vePFS/infra_rd/kun/others/wzg/workspace/evoproto');E=R/'experiments/local_sep_v5';U=R/'runs/local_sep_v5';S=E/'src'
sys.path.insert(0,str(R/'experiments/kd_pixel_v2'))
from run_kd import environment
os.environ.update(environment('8card'))
from kd_runtime import safe_path,atomic_json,digest,now
import torch
parts=[json.loads(safe_path(U/f'support_rank{i}.json').read_text()) for i in range(8)]
names=sum([x['images'] for x in parts],[]);assert len(names)==len(set(names))==2145
assert all(x['source_sha256']==digest(E/'support.py') for x in parts)
counts=sum(torch.tensor(x['unique_pair_images']) for x in parts)
route=(counts>=3)&(counts.T>=3);route.fill_diagonal_(False);route[0]=False;route[:,0]=False
assert route.any() and torch.equal(route,route.T)
atomic_json(U/'routing.json',{'utc':now(),'unique_images':2145,'image_names':names,'unique_pair_images':counts.tolist(),
 'routing':route.tolist(),'pairs':route.triu(1).nonzero().tolist(),'threshold':3,
 'GT_role':'No pixel GT, no old image GT, no validation-derived per-class whitelist. Bidirectional support from distinct current-stage training images, weak positive anchors and image-absence exclusions.',
 'parent_support_source_sha256':digest(E/'support.py')})
# Preserve the positive-boundary v3 adapter. V4's unconditional exclusion
# training suppressed supported positives and did not supply missing old data.
parent=R/'runs/local_sep_v3/formal/10-5/step2/checkpoints/model_final.pth'
assert digest(parent)=='c4686142579257698e75e2e7ac15d5744a829826ac05ddf2a8032b18e86aab89'
ck=torch.load(safe_path(parent),map_location='cpu',weights_only=True,mmap=True)
assert ck['iteration']==8900
p=safe_path(S/'model/pair_adapter.py');t=p.read_text()
t=t.replace('def redistribute(logits,raw,limit=math.log(2)):','def redistribute(logits,raw,limit=math.log(2),routing=None):')
t=t.replace(' da=da*permitted;db=db*permitted',' if routing is not None:\n  permitted &= routing[a,b]\n da=da*permitted;db=db*permitted')
t=t.replace('  self.spatial=nn.Conv2d','  self.register_buffer("routing",torch.ones(21,21,dtype=torch.bool))\n  self.spatial=nn.Conv2d')
t=t.replace('return redistribute(logits,raw)','return redistribute(logits,raw,routing=self.routing)')
p.write_text(t)
sys.path.insert(0,str(S))
from model.pair_adapter import redistribute
torch.manual_seed(0)
z=torch.randn(32,21,8,8);raw=torch.randn_like(z)*5
assert torch.equal(redistribute(z,raw,routing=torch.zeros_like(route)),z)
out=redistribute(z,raw,routing=route)
ids=z[:,1:].topk(2,1).indices+1
allowed=route[ids[:,0],ids[:,1]]&(z.argmax(1)>0)
assert torch.equal(out.permute(0,2,3,1)[~allowed],z.permute(0,2,3,1)[~allowed])
assert torch.allclose(torch.logsumexp(out.gather(1,ids),1),torch.logsumexp(z.gather(1,ids),1),atol=2e-6)
assert torch.equal(out.argmax(1)==0,z.argmax(1)==0)
ck['model_state']['decoder.local_sep.routing']=route
ck['routing_provenance']={'parent_checkpoint':str(parent),'parent_sha256':digest(parent),'routing_sha256':digest(U/'routing.json'),'added_optimizer_updates':0}
stage=safe_path(U/'formal/10-5/step2');stage.mkdir(parents=True)
path=safe_path(stage/'checkpoints/model_final.pth');path.parent.mkdir();torch.save(ck,path)
parent_state=torch.load(parent,map_location='cpu',weights_only=True,mmap=True)
assert all(torch.equal(v,ck['model_state'][k]) for k,v in parent_state['model_state'].items())
from model.model_seg_neg import network
study=json.loads((R/'runs/local_sep_v3/formal/study.json').read_text());cfg=study['config'];cfg['work_dir']=str(U/'formal');cfg['val_limit']=0
m=network(backbone=cfg['backbone'],num_classes=21,classes_list=[11,5,5],pretrained=False,init_momentum=cfg['momentum'],aux_layer=cfg['aux_layer'])
m.load_state_dict(ck['model_state'],strict=True)
source={str(p.relative_to(S)):digest(p) for p in S.rglob('*.py')}
atomic_json(E/'preflight.json',{'passed':True,'tests':['all_unsupported_logits_exact_identity','route_symmetric_distinct_training_support','supported_pair_mass_conservation','native_BG_decision_preserved','every_parent_tensor_unchanged','strict_architecture_checkpoint_load'],'source_sha256':source})
atomic_json(U/'formal/study.json',{'config':cfg,'source_sha256':source,'parent_checkpoint':str(parent),'parent_sha256':digest(parent),'actual_updates':0,'total_adapter_updates':300,'routing_sha256':digest(U/'routing.json'),'utc':now()})
atomic_json(U/'formal/training_complete.json',{'status':'no_additional_training','checkpoint':str(path),'checkpoint_sha256':digest(path),'actual_updates':0,'utc':now()})
job=json.loads((R/'runs/local_sep_v3/formal/eval_queue/step2_iter8900.json').read_text());job.update(checkpoint=str(path),checkpoint_sha256=digest(path),config=cfg)
atomic_json(U/'formal/eval_queue/step2_iter8900.json',job)
atomic_json(U/'formal/status.json',{'status':'evaluating','utc':now()})
workers=[]
for rank in range(8):
 with safe_path(U/'formal'/f'evaluator{rank}.log').open('x') as log:
  p=subprocess.Popen([str(R/'.runtime/env/bin/python'),'-B',str(S/'evaluate_kd.py'),'--run',str(U/'formal'),'--rank',str(rank),'--shards','8','--once'],cwd=R,env=dict(os.environ),stdout=log,stderr=subprocess.STDOUT)
 workers.append(p)
assert [p.wait() for p in workers]==[0]*8
from evaluate_kd import merge
merge(job,U/'formal/evaluations/step2_iter8900',8)
result=json.loads((U/'formal/evaluations/step2_iter8900/result.json').read_text());assert result['images']==1449
atomic_json(U/'formal/status.json',{'status':'complete','all_miou':result['all_miou'],'utc':now()})
print(json.dumps({'pairs':route.triu(1).nonzero().tolist(),'mIoU':result['all_miou']}),flush=True)
