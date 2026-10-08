from pathlib import Path
import sys
R=Path('/ML-vePFS/infra_rd/kun/others/wzg/workspace/evoproto')
E=R/'experiments/online_confusion_v1';sys.path.insert(0,str(E/'src'))
from kd_runtime import safe_path,atomic_json,now
p=safe_path(E/'src/continual/Trainer.py');s=p.read_text()
s=s.replace('from model.pixel_kd import pixel_kd_loss','from model.pixel_kd import pixel_kd_loss\nfrom model.online_directed_confusion import OnlineDirectedConfusion')
needle='        self.class_weight[0] = 1.0\n'
assert s.count(needle)==1
s=s.replace(needle,needle+'''        self.confusion = (OnlineDirectedConfusion(self.total_classes+1,
            momentum=args.confusion_momentum, high_threshold=args.high_thre,
            low_threshold=args.low_thre, stage=self.step).to(self.device)
            if args.online_confusion and self.step > 0 else None)
''')
needle='            optim.global_step = start_iteration\n'
assert s.count(needle)==1
s=s.replace(needle,needle+'''            if self.confusion is not None:
                if saved.get('online_confusion_state') is not None:
                    self.confusion.load_state_dict(saved['online_confusion_state'], strict=True)
                else:
                    logging.info('Checkpoint predates online confusion; start observational state at resume iteration')
''')
needle='                mixed_pseudo_label = get_mixed_label(refined_pseudo_label, old_pixel_label,\n'
assert s.count(needle)==1
s=s.replace(needle,'''                if self.confusion is not None:
                    # Observe independent CAM/PAR anchors before teacher mixing;
                    # no labels or weights from this observer enter any loss.
                    self.confusion.update(segs, refined_pseudo_label, valid_cam, img_box)
                    if args.local_rank == 0 and (n_iter + 1) % args.log_iters == 0:
                        from kd_runtime import atomic_json
                        atomic_json(Path(args.work_dir)/'online_confusion.json',
                            {'training_iteration': n_iter+1, **self.confusion.export()})
'''+needle)
if 'from pathlib import Path' not in s:s='from pathlib import Path\n'+s
needle="'optimizer_state': optim.state_dict()"
s=s.replace(needle,needle+", 'online_confusion_state': self.confusion.state_dict() if self.confusion is not None else None")
p.write_text(s)
p=safe_path(E/'src/scripts/dist_train_voc_seg_neg.py');s=p.read_text()
needle='parser.add_argument("--confusion_reweight", action="store_true", help="enable existing dynamic class weighting")'
assert s.count(needle)==1
s=s.replace(needle,needle+'''\nparser.add_argument("--online_confusion", action="store_true", help="observe directed confusion without changing losses")
parser.add_argument("--confusion_momentum", default=0.98, type=float)''')
needle='        "model_state": trainer.model.state_dict(),\n'
assert s.count(needle)==1
s=s.replace(needle,needle+'''        "online_confusion_state": trainer.confusion.state_dict() if trainer.confusion is not None else None,
''')
p.write_text(s)
atomic_json(E/'integration.json',{'utc':now(),'observer':'Before mixed teacher labels; segmentation prediction never gates anchors','loss_changed':False,'gt_read_in_observer':False,'resume':'Restores matrix from incremental checkpoint, starts fresh if older checkpoint lacks observer; mismatched class/stage rejected','save':'Iteration and final checkpoints plus latest JSON','update':'Every normal incremental training batch, no extra network forward'})
print('Integrated online observer')
