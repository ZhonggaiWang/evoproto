"""Load the actual full-model final on CPU and verify a frozen identical reference."""
import sys,json,math
from pathlib import Path
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import torch
from experiments.restore_proto_ald_v1.Trainer import Trainer
from experiments.restore_proto_ald_v1.cache import load
parent=ROOT/'runs/restore_proto_v1/formal/full/10-5'
args=SimpleNamespace(**json.loads((parent/'step2/config.json').read_text()))
args.loss_warmup_iters=0;args.max_iters=1200;args.lr=2e-6;args.refine_ald=False
args.start_checkpoint=str(parent/'step2/checkpoints/model_final.pth');args.prev_checkpoint=str(parent/'step1/checkpoints/model_final.pth')
torch.set_num_threads(4)
# Normal Trainer initialization needs CUDA. Exercise its real refinement loader on CPU.
trainer=Trainer.__new__(Trainer);trainer.step=2;trainer.total_classes=20
trainer.model=load(args.start_checkpoint,2,'cpu').requires_grad_(True)
trainer.optimizer=Trainer.get_optimizer(trainer,args,trainer.model.get_param_groups())
trainer.load_refinement(args)
assert all(torch.equal(v,trainer.ald_reference.state_dict()[k]) for k,v in trainer.model.state_dict().items())
assert all(not p.requires_grad for p in trainer.ald_reference.parameters()) and not trainer.ald_reference.training
assert all(math.isclose(g['lr'],expected,rel_tol=1e-12) for g,expected in zip(trainer.optimizer.param_groups,[2e-6,2e-6,2e-5,2e-5]))
assert len(trainer.optimizer.state)==0
print(json.dumps({'passed':True,'all_reference_tensors_match_start':True,'reference_frozen':True,'reference_eval':True,'prototype_blocks':[list(p.shape) for p in trainer.ald_reference.decoder.class_prototypes.parameters()],'optimizer_state_count':len(trainer.optimizer.state)}))
