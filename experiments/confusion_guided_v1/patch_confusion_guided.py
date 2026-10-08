from pathlib import Path
import sys
R=Path('/ML-vePFS/infra_rd/kun/others/wzg/workspace/evoproto');E=R/'experiments/confusion_guided_v1'
sys.path.insert(0,str(E/'a_sep/src'))
from kd_runtime import safe_path,atomic_json,now
for arm in ['a_sep','b_pairkd']:
    S=E/arm/'src';p=safe_path(S/'continual/Trainer.py');s=p.read_text()
    needle='from model.online_directed_confusion import OnlineDirectedConfusion'
    assert s.count(needle)==1
    s=s.replace(needle,needle+'\nfrom model.directed_pair_selector import DirectedPairSelector\nfrom model.confusion_pair_losses import directed_pair_sep_loss')
    needle='            if args.online_confusion and self.step > 0 else None)\n'
    assert s.count(needle)==1
    s=s.replace(needle,needle+'''        if args.pair_mode != 'off' and self.confusion is None:
            raise ValueError('Confusion-guided losses require online confusion')
        self.pair_selector = (DirectedPairSelector(self.total_classes+1, stage=self.step,
            refresh_interval=args.pair_refresh_interval, min_row_images=args.pair_min_row_images,
            min_pair_images=args.pair_min_pair_images, min_rate=args.pair_min_rate,
            min_updates=args.pair_min_updates, ramp_updates=args.pair_ramp_updates,
            max_stale_updates=args.pair_max_stale_updates).to(self.device)
            if args.pair_mode != 'off' else None)
''')
    needle="                    logging.info('Checkpoint predates online confusion; start observational state at resume iteration')\n"
    assert s.count(needle)==1
    s=s.replace(needle,needle+'''            if self.pair_selector is not None and saved.get('pair_selector_state') is not None:
                self.pair_selector.load_state_dict(saved['pair_selector_state'], strict=True)
''')
    start=s.index('                if self.confusion is not None:\n',s.index('                refined_pseudo_label = refine_cams_with_bkg_v2'))
    end=s.index('                segs = F.interpolate(segs, size=refined_pseudo_label.shape[1:]',start)
    s=s[:start]+'''                # Preserve independent CAM/PAR anchors and consume only past
                # confusion; current errors cannot select their own hard pairs.
                pair_targets = None
                pair_ramp = 0.0
                pair_stats = {}
                pair_sep = 0.0 * (segs.sum() + type_seg.sum())
                pair_evidence = None
                if self.pair_selector is not None:
                    pair_targets = self.pair_selector.update(self.confusion, n_iter+1)
                    pair_ramp = self.pair_selector.ramp(self.confusion)
                if self.confusion is not None:
                    pair_evidence = self.confusion.update(segs, refined_pseudo_label, valid_cam, img_box)
                    if args.local_rank == 0 and (n_iter + 1) % args.log_iters == 0:
                        from kd_runtime import atomic_json
                        atomic_json(Path(args.work_dir)/'online_confusion.json',
                            {'training_iteration': n_iter+1, **self.confusion.export()})
                if self.pair_selector is not None:
                    pair_sep, pair_stats = directed_pair_sep_loss(segs, type_seg,
                        teacher_native_logits, pair_evidence, pair_targets)
                mixed_pseudo_label = get_mixed_label(refined_pseudo_label, old_pixel_label,
                                                     self.total_classes, self.new_classes)
                pixel_kd, kd_stats = pixel_kd_loss(segs, teacher_native_logits,
                    refined_pseudo_label, valid_cam, img_box, args.kd_temperature,
                    pair_targets=pair_targets if args.pair_mode == 'pairkd' else None,
                    pair_evidence=pair_evidence if args.pair_mode == 'pairkd' else None,
                    pair_blend=0.5*pair_ramp if args.pair_mode == 'pairkd' else 0.0)

'''+s[end:]
    s=s.replace('prototype_sep + pixel_kd)','prototype_sep + pixel_kd + pair_sep)')
    needle=' + args.w_pixel_kd * pixel_kd\n';assert s.count(needle)==1
    s=s.replace(needle,' + args.w_pixel_kd * pixel_kd + args.w_pair_sep * pair_ramp * pair_sep\n')
    needle="                        f.write" # Logging appends via separate small file.
    insert="                    with open(osp.join(args.work_dir, 'kd_metrics.jsonl'), 'a') as handle:\n                        handle.write(json.dumps(kd_stats, allow_nan=False)+'\\n')\n"
    assert s.count(insert)==1
    s=s.replace(insert,insert+'''                    if self.pair_selector is not None:
                        pair_stats.update(step=self.step, iteration=n_iter+1,
                            ramp=pair_ramp, weight=args.w_pair_sep, selector=self.pair_selector.export())
                        with open(osp.join(args.work_dir, 'pair_metrics.jsonl'), 'a') as handle:
                            handle.write(json.dumps(pair_stats, allow_nan=False)+'\\n')
''')
    needle="'online_confusion_state': self.confusion.state_dict() if self.confusion is not None else None"
    s=s.replace(needle,needle+", 'pair_selector_state': self.pair_selector.state_dict() if self.pair_selector is not None else None")
    p.write_text(s)
    p=safe_path(S/'scripts/dist_train_voc_seg_neg.py');s=p.read_text()
    needle='parser.add_argument("--confusion_momentum", default=0.98, type=float)';assert s.count(needle)==1
    s=s.replace(needle,needle+'''\nparser.add_argument("--pair_mode", choices=['off','sep','pairkd'], default='off')
parser.add_argument("--w_pair_sep", default=0.1, type=float)
parser.add_argument("--pair_refresh_interval", default=50, type=int)
parser.add_argument("--pair_min_row_images", default=8, type=int)
parser.add_argument("--pair_min_pair_images", default=3, type=int)
parser.add_argument("--pair_min_rate", default=0.01, type=float)
parser.add_argument("--pair_min_updates", default=100, type=int)
parser.add_argument("--pair_ramp_updates", default=200, type=int)
parser.add_argument("--pair_max_stale_updates", default=200, type=int)''')
    needle='        "online_confusion_state": trainer.confusion.state_dict() if trainer.confusion is not None else None,\n'
    assert s.count(needle)==1
    s=s.replace(needle,needle+'        "pair_selector_state": trainer.pair_selector.state_dict() if trainer.pair_selector is not None else None,\n')
    p.write_text(s)
atomic_json(E/'integration.json',{'utc':now(),'separation':'Main and prototype nativepixel logits, not prototypecosine alone','selection':'Consume past globally synchronized EMA; immutable pre-mixing PAR anchors; noGT','KDblend':'Perpixel max0.5 replacement with reliableoldoldbinaryKL, fullfallback elsewhere','state':'Estimator andselector exact same-stage checkpoint restore; resetperstage','source':'Isolatedcopies, frozenonlyafterverification'})
print('Integrated directed SEP and pairKD in isolated sources')
