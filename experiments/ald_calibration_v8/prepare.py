from pathlib import Path
import sys, os, json, ast
R=Path('/ML-vePFS/infra_rd/kun/others/wzg/workspace/evoproto')
E=R/'experiments/ald_calibration_v8'; U=R/'runs/ald_calibration_v8'; S=E/'src'
P=R/'experiments/prototype_sep_v1/c_new_anchor/src'
sys.path.insert(0,str(R/'experiments/kd_pixel_v2'))
from run_kd import environment
os.environ.update(environment('8card'))
from kd_runtime import safe_path, atomic_json, digest, now
import torch

assert not S.exists()
parent=json.loads((R/'runs/prototype_sep_v1/formal/c_new_anchor/study.json').read_text())
hashes={str(p.relative_to(P)):digest(p) for p in P.rglob('*.py') if '__pycache__' not in p.parts}
assert hashes==parent['source_sha256']
for p in P.rglob('*'):
    safe_path(p)
    if p.is_file() and '__pycache__' not in p.parts:
        q=safe_path(S/p.relative_to(P));q.parent.mkdir(parents=True,exist_ok=True)
        with q.open('xb') as out:out.write(p.read_bytes())
q=safe_path(S/'model/ald.py');q.write_bytes(safe_path(E/'ald.py').read_bytes())

def change(text, old, new):
    assert text.count(old)==1,(old[:100],text.count(old))
    return text.replace(old,new)

p=S/'continual/Trainer.py';text=p.read_text()
text=change(text,'from model.pixel_kd import pixel_kd_loss',
'''from model.pixel_kd import pixel_kd_loss
from model.ald import image_targets, fuse, gate_auxiliary, uncertain_loss, stats as ald_stats''')
text=change(text,'        model = DistributedDataParallel(model, device_ids=[args.local_rank], find_unused_parameters=True)',
'''        # The best prior student is a frozen self-training reference, not GT.
        from kd_runtime import safe_path, digest
        reference_path = safe_path(args.ald_reference)
        reference_saved = torch.load(reference_path, map_location='cpu', weights_only=True, mmap=True)
        self.ald_reference = network(backbone=args.backbone, num_classes=21,
            classes_list=[11,5,5], pretrained=False, init_momentum=args.momentum, aux_layer=args.aux_layer)
        self.ald_reference.load_state_dict(reference_saved['model_state'], strict=True)
        self.ald_reference = self.ald_reference.to(device).eval().requires_grad_(False)
        del reference_saved
        evidence = torch.load(safe_path(args.ald_evidence), map_location='cpu', weights_only=True)
        self.ald_states = {name: evidence['state'][i] for i,name in enumerate(evidence['names'])}
        assert len(self.ald_states) == 2145
        optim.max_iter = 8000  # Smoke uses the same LR trajectory as formal.
        model = DistributedDataParallel(model, device_ids=[args.local_rank], find_unused_parameters=True)''')
text=change(text,'            for n_iter in range(start_iteration, args.max_iters):\n                self.current_iteration = n_iter + 1',
'''            for n_iter in range(start_iteration, args.max_iters, args.ald_stride):
                # Iteration tracks sample exposure in legacy global-batch-8 units.
                # Adam advances once per actual global-batch-32 update.
                self.current_iteration = n_iter + args.ald_stride
                optim.global_step = n_iter''')
text=change(text,'                    teacher_native_logits = old_segs\n                cls_label_old_pred = (old_cls > 0).long()\n\n                cls_label_gt_new = cls_label[:, -self.new_classes:]\n                cls_label = torch.cat((cls_label_old_pred, cls_label_gt_new), dim=1)',
'''                    teacher_native_logits = old_segs
                cls_label_gt_new = cls_label[:, -self.new_classes:]
                with torch.no_grad():
                    _, reference_logits, _, _ = self.ald_reference(inputs)
                    ald_state = torch.stack([self.ald_states[str(name)] for name in img_name]).to(device)
                    ald_target, cls_label, ald_conflict = image_targets(ald_state, old_cls, old_cls_aux,
                        teacher_native_logits, reference_logits, cls_label_gt_new)''')
text=change(text,'''                mixed_pseudo_label = get_mixed_label(refined_pseudo_label, old_pixel_label,
                                                     self.total_classes, self.new_classes)''',
'''                ald_fused = fuse(refined_pseudo_label, teacher_native_logits, reference_logits, ald_state, img_box)
                mixed_pseudo_label = ald_fused['labels']''')
# Avoid modifying the unused step0 block, which has a different following comment.
text=change(text,'''                cls_loss = F.multilabel_soft_margin_loss(cls, cls_label)
                cls_loss_aux = F.multilabel_soft_margin_loss(cls_aux, cls_label)

                # # ctc_loss''',
'''                cls_loss = F.binary_cross_entropy_with_logits(cls, ald_target)
                cls_loss_aux = F.binary_cross_entropy_with_logits(cls_aux, ald_target)
                ald_soft = uncertain_loss(segs, reference_logits, ald_fused['unknown'],
                    ald_fused['valid'], ald_state, cls_label_gt_new, args.kd_temperature)
                ald_record = ald_stats(ald_fused, ald_state, ald_conflict, ald_soft)

                # # ctc_loss''')
text=change(text,'                self.confusion.update(segs,refined_pseudo_label,valid_cam,img_box)',
'''                # Observer starts fresh at warmup; source anchors remain PAR,
                # calibrated for old presence / reference BG, never current agreement.
                ald_anchors = gate_auxiliary(refined_pseudo_label, ald_state, reference_logits)
                self.confusion.update(segs,ald_anchors,valid_cam,img_box)''')
text=change(text,'                    refined_pseudo_label, valid_cam, img_box, args.kd_temperature)',
'''                    refined_pseudo_label, valid_cam, img_box, args.kd_temperature,
                    trusted_mask=ald_fused['trusted_old'])''')
text=change(text,'                aff_mask = label_to_aff_mask(pseudo_label_aux)',
'''                pseudo_label_aux = gate_auxiliary(pseudo_label_aux, ald_state, reference_logits)
                aff_mask = label_to_aff_mask(pseudo_label_aux)''')
text=change(text,' + args.w_pixel_kd * pixel_kd + protected_sep\n',
                   ' + args.w_pixel_kd * pixel_kd + protected_sep + args.w_seg * ald_soft\n')
text=change(text,'''                optim.zero_grad()
                loss.backward()
                if args.local_rank == 0 and (n_iter + 1) % args.log_iters == 0:''',
'''                optim.zero_grad()
                loss.backward()
                if n_iter == start_iteration or args.ald_smoke:
                    trainable = [p for p in self.model.parameters() if p.grad is not None]
                    assert trainable and all(torch.isfinite(p.grad).all() for p in trainable)
                    assert all(p.grad is None for p in self.ald_reference.parameters())
                    assert all(p.grad is None for p in model_old.parameters())
                if args.local_rank == 0 and (n_iter + 1) % args.log_iters == 0:
                    ald_record.update(iteration=self.current_iteration, global_batch=args.spg*dist.get_world_size())
                    with open(osp.join(args.work_dir,'ald_metrics.jsonl'),'a') as handle:
                        handle.write(json.dumps(ald_record,allow_nan=False)+'\\n')''')
# Only the incremental branch uses exposure-strided indices.
i=text.index('            for n_iter in range(start_iteration, args.max_iters, args.ald_stride):')
text=text[:i]+text[i:].replace('n_iter + 1','self.current_iteration').replace('n_iter+1','self.current_iteration')
text=change(text,'        return True\n',
'''        if args.ald_smoke:
            flat = torch.cat([p.detach().flatten() for p in self.model.parameters()])
            root = flat.clone(); dist.broadcast(root, 0)
            assert torch.equal(root, flat), 'DDP parameters differ across ranks'
            from kd_runtime import atomic_json
            atomic_json(Path(args.work_dir)/f'ald_smoke_rank{dist.get_rank()}.json',
                {'passed': True, 'finite_gradients': True, 'frozen_teachers_no_gradient': True,
                 'all_rank_parameters_exact': True, 'iteration': self.current_iteration})
        return True
''')
ast.parse(text);safe_path(p).write_text(text)

p=S/'model/pixel_kd.py';text=p.read_text()
text=change(text,'img_box, temperature=2.0):','img_box, temperature=2.0, trusted_mask=None):')
text=change(text,'        winner_confidence = teacher.sigmoid()',
'''        if trusted_mask is not None:
            trust = F.interpolate(trusted_mask[:,None].float(), size=size, mode='nearest')[:,0] > 0
            eligible &= trust
        winner_confidence = teacher.sigmoid()''')
ast.parse(text);safe_path(p).write_text(text)

p=S/'scripts/dist_train_voc_seg_neg.py';text=p.read_text()
text=change(text,'parser.add_argument("--backbone",',
'''parser.add_argument('--ald_reference', required=True)
parser.add_argument('--ald_evidence', required=True)
parser.add_argument('--ald_stride', type=int, default=4)
parser.add_argument('--ald_smoke', action='store_true')
parser.add_argument("--backbone",''')
text=change(text,"    args = parser.parse_args()",'''    args = parser.parse_args()
    if args.ald_stride != args.spg or (args.max_iters-2000) % args.ald_stride:
        parser.error('ALD stride must match batch per GPU and divide remaining exposure steps')''')
ast.parse(text);safe_path(p).write_text(text)

names=[];states=[];receipts=[]
for rank in range(8):
    path=R/'runs/ald_calibration_v1'/f'cache_receipt_rank{rank}.json'
    rec=json.loads(safe_path(path).read_text());assert rec['passed']
    assert digest(safe_path(rec['cache']))==rec['cache_sha256']
    cache=torch.load(rec['cache'],map_location='cpu',weights_only=True,mmap=True)
    names+=cache['names'];states.append(cache['state'].clone());receipts.append({'receipt':str(path),'sha256':digest(path),'cache_sha256':rec['cache_sha256']})
assert len(names)==len(set(names))==2145
target=safe_path(E/'image_evidence.pth');torch.save({'names':names,'state':torch.cat(states)},target)
atomic_json(E/'origin.json',{'utc':now(),'parent_source':str(P),'parent_source_sha256':hashes,
    'source_sha256':{str(p.relative_to(S)):digest(p) for p in S.rglob('*.py') if '__pycache__' not in p.parts},
    'image_evidence_sha256':digest(target),'evidence_receipts':receipts,
    'semantics':'Full weak training image states frozen from ALD-v1; augmented CAM/PAR and reference predictions computed online. No old image tags or pixel GT in optimizer.'})
print('Prepared early ALD source and 2145 weak image states')
