from pathlib import Path
import sys,ast
R=Path('/ML-vePFS/infra_rd/kun/others/wzg/workspace/evoproto')
E=R/'experiments/prototype_sep_v1'; S=E/'a_geometry/src'
sys.path.insert(0,str(S))
from kd_runtime import safe_path,atomic_json,now,digest

def replace(text,old,new):
    assert text.count(old)==1, (old,text.count(old))
    return text.replace(old,new)

p=safe_path(S/'continual/Trainer.py');s=p.read_text()
s=replace(s,'import datetime','from pathlib import Path\nimport datetime')
s=replace(s,'from model.pixel_kd import pixel_kd_loss','''from model.pixel_kd import pixel_kd_loss
from model.online_directed_confusion import OnlineDirectedConfusion
from model.geometry_pair_selector import GeometryPairSelector
from model.confusion_prototype_sep import confusion_prototype_sep_loss''')
s=replace(s,'        self.class_weight[0] = 1.0\n','''        self.class_weight[0] = 1.0
        self.confusion = OnlineDirectedConfusion(self.total_classes+1,
            momentum=args.confusion_momentum, high_threshold=args.high_thre,
            low_threshold=args.low_thre, stage=self.step).to(self.device)
        self.pair_selector = GeometryPairSelector(self.total_classes+1,
            old_classes=self.old_classes, stage=self.step,
            refresh_interval=args.pair_refresh_interval, min_row_images=args.pair_min_row_images,
            min_pair_images=args.pair_min_pair_images, min_rate=args.pair_min_rate,
            min_updates=args.pair_min_updates, ramp_updates=args.pair_ramp_updates,
            max_stale_updates=args.pair_max_stale_updates).to(self.device)
''')
s=replace(s,'            optim.global_step = start_iteration\n','''            optim.global_step = start_iteration
            if saved.get('online_confusion_state') is not None:
                self.confusion.load_state_dict(saved['online_confusion_state'],strict=True)
            if saved.get('geometry_selector_state') is not None:
                self.pair_selector.load_state_dict(saved['geometry_selector_state'],strict=True)
''')
start=s.index('                mixed_pseudo_label = get_mixed_label(',s.index('            for n_iter in range(start_iteration,'))
s=s[:start]+'''                # Select from past globally synchronized state. Rates only rank
                # supported pairs; they are never interpreted as exact strengths.
                pair_targets = self.pair_selector.update(self.confusion,n_iter+1)
                pair_ramp = self.pair_selector.ramp(self.confusion)
                self.confusion.update(segs,refined_pseudo_label,valid_cam,img_box)
                geometry_sep, geometry_stats = confusion_prototype_sep_loss(
                    new_prototypes,pair_targets,self.old_classes,margin=args.proto_margin)
                if args.local_rank==0 and (n_iter+1)%args.log_iters==0:
                    from kd_runtime import atomic_json
                    atomic_json(Path(args.work_dir)/'online_confusion.json',
                        {'training_iteration':n_iter+1,**self.confusion.export()})
'''+s[start:]
s=replace(s,'prototype_sep + pixel_kd)','prototype_sep + pixel_kd + geometry_sep)')
s=replace(s,' + args.w_pixel_kd * pixel_kd\n',' + args.w_pixel_kd * pixel_kd + args.w_geometry_sep * pair_ramp * geometry_sep\n')
marker="                    with open(osp.join(args.work_dir, 'kd_metrics.jsonl'), 'a') as handle:\n                        handle.write(json.dumps(kd_stats, allow_nan=False)+'\\n')\n"
s=replace(s,marker,marker+'''                    geometry_stats.update(step=self.step,iteration=n_iter+1,
                        active=n_iter>=args.loss_warmup_iters,ramp=pair_ramp,
                        weight=args.w_geometry_sep,selector=self.pair_selector.export())
                    with open(osp.join(args.work_dir,'geometry_metrics.jsonl'),'a') as handle:
                        handle.write(json.dumps(geometry_stats,allow_nan=False)+'\\n')
''')
old="'optimizer_state': optim.state_dict()}"
assert s.count(old)==2
s=s.replace(old,"'optimizer_state': optim.state_dict(), 'online_confusion_state': self.confusion.state_dict(), 'geometry_selector_state': self.pair_selector.state_dict()}")
p.write_text(s)

p=safe_path(S/'scripts/dist_train_voc_seg_neg.py');s=p.read_text().replace('kd_parallel_v1','prototype_sep_v1')
marker="parser.add_argument('--async_eval', action='store_true')"
s=replace(s,marker,marker+'''
parser.add_argument('--w_geometry_sep',default=0.1,type=float)
parser.add_argument('--confusion_momentum',default=0.98,type=float)
parser.add_argument('--pair_refresh_interval',default=50,type=int)
parser.add_argument('--pair_min_row_images',default=8,type=int)
parser.add_argument('--pair_min_pair_images',default=3,type=int)
parser.add_argument('--pair_min_rate',default=0.01,type=float)
parser.add_argument('--pair_min_updates',default=100,type=int)
parser.add_argument('--pair_ramp_updates',default=200,type=int)
parser.add_argument('--pair_max_stale_updates',default=200,type=int)
''')
s=replace(s,'        "model_state": trainer.model.state_dict(),\n','''        "model_state": trainer.model.state_dict(),
        "iteration":trainer.current_iteration,
        "optimizer_state":trainer.optimizer.state_dict(),
        "online_confusion_state":trainer.confusion.state_dict(),
        "geometry_selector_state":trainer.pair_selector.state_dict(),
''')
s=replace(s,"        parser.error('Invalid KD settings')","        parser.error('Invalid KD settings')\n    if args.w_geometry_sep<0:\n        parser.error('Invalid geometry SEP weight')")
p.write_text(s)

p=safe_path(S/'evaluate_kd.py');s=p.read_text()
s=replace(s,'ROOT=SRC.parents[2]',"ROOT=Path('/ML-vePFS/infra_rd/kun/others/wzg/workspace/evoproto')")
s=s.replace("cache=ROOT/'.runtime/kd_pixel_v1/4card'","cache=ROOT/'.runtime/prototype_sep_v1/8card_eval'")
s=s.replace("ROOT.parents[1]/'.kd4tmp'","ROOT.parents[1]/'.kd8tmp'")
p.write_text(s)
for p in S.rglob('*.py'):ast.parse(p.read_text(),filename=str(p))
assert digest(S/'model/pixel_kd.py')==digest(R/'experiments/kd_parallel_v1/b_relational/src/model/pixel_kd.py')
atomic_json(E/'integration.json',{'utc':now(),'base':'optimized conditional old foreground KD byte-identical',
    'geometry':'student old prototypes detached only for SEP; new prototypes learn; foreground pairs deduplicated; cosine squared hinge margin0',
    'selection':'past broad EMA ranked with trusted pair exposures, foreground old-new/new-new only',
    'matrix_strength':'rates never loss weights; unchanged lambda0.1; bounded cosine hinge stops once separated',
    'resume':'restore model+optimizer; same-stage observer and selector restore exact; reset observer at common step1 warmup lacking state',
    'GT':'not read by any training selector/loss; diagnostic/evaluation only'})
print('Integrated prototype geometry SEP; optimized KD unchanged')
