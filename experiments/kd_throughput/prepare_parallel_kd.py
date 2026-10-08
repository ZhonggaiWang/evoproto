from pathlib import Path
import sys,shutil,json,ast
R=Path('/ML-vePFS/infra_rd/kun/others/wzg/workspace/evoproto');P=R/'experiments/kd_pixel_v2'
sys.path.insert(0,str(P/'src'));from kd_runtime import safe_path,atomic_json,now,digest
E=safe_path(R/'experiments/kd_parallel_v1');assert not E.exists();E.mkdir()
for arm in ['a_sigmoid','b_relational']:
 S=E/arm/'src';shutil.copytree(P/'src',S,ignore=shutil.ignore_patterns('__pycache__'))
 p=S/'scripts/dist_train_voc_seg_neg.py';s=p.read_text().replace('kd_pixel_v2','kd_parallel_v1')
 s=s.replace('parser = argparse.ArgumentParser(', 'parser = argparse.ArgumentParser(')
 pos=s.index('parser.add_argument(')
 s=s[:pos]+"parser.add_argument('--resume_checkpoint', default='', type=str)\n"+s[pos:]
 # SOURCE_ROOT is based on file location; PROJECT_ROOT is explicitly absolute.
 p.write_text(s)
 p=S/'continual/Trainer.py';s=p.read_text()
 marker='        optim = self.optimizer\n'
 assert s.count(marker)==1
 s=s.replace(marker,marker+'''        start_iteration = 0
        if args.resume_checkpoint:
            from kd_runtime import safe_path
            resume_path = safe_path(args.resume_checkpoint)
            saved = torch.load(resume_path, map_location='cpu', weights_only=True)
            model.load_state_dict(saved['model_state'], strict=True)
            optim.load_state_dict(saved['optimizer_state'])
            start_iteration = int(saved['iteration'])
            optim.global_step = start_iteration
            if not 0 < start_iteration < args.max_iters:
                raise ValueError('Resume iteration outside active stage')
            logging.info('Resume student AND optimizer at iteration %d from %s; teacher remains previous stage', start_iteration, resume_path)
            del saved
''')
 marker='            for n_iter in range(args.max_iters):';pos=s.rindex(marker)
 s=s[:pos]+s[pos:].replace(marker,'            for n_iter in range(start_iteration, args.max_iters):',1)
 p.write_text(s)
 if arm=='b_relational':
  p=S/'model/pixel_kd.py';s=p.read_text()
  s=s.replace('Evidence-gated Bernoulli KD on old foreground outputs only.','Evidence-gated conditional old-class relational KD.')
  start=s.index('        scaled_teacher = teacher[:, 1:k] / temperature')
  end=s.index('    group = prediction.flatten()',start)
  s=s[:start]+'''        scaled_teacher = teacher[:, 1:k] / temperature
        target = scaled_teacher.softmax(1)
        log_target = scaled_teacher.log_softmax(1)
    log_student = (student[:, 1:k] / temperature).log_softmax(1)
    per_pixel = (target * (log_target-log_student)).sum(1) * temperature ** 2
'''+s[end:]
  s=s.replace('supported_old_foreground_bernoulli_KL','supported_old_foreground_conditional_KL')
  p.write_text(s)
  # Bernoulli matching and decoupling invariants also hold; relation KD additionally
  # must be invariant to a common additive offset of all old foreground logits.
  p=S/'test_pixel_kd.py';s=p.read_text();marker='    def test_old_only_teacher_detached(self):'
  s=s.replace(marker,'''    def test_old_common_offset_invariance(self):
        a=list(data());v=pixel_kd_loss(*a)[0]
        a[0]=a[0].detach().clone();a[0][:,1:3]+=50
        torch.testing.assert_close(v,pixel_kd_loss(*a)[0])
'''+marker);p.write_text(s)
 for p in S.rglob('*.py'):ast.parse(p.read_text(),filename=str(p))
 probe=(P/'probe_kd_gradients.py').read_text()
 probe=probe.replace("EXP=ROOT/'experiments/kd_pixel_v2';SRC=EXP/'src'",f"EXP=ROOT/'experiments/kd_parallel_v1/{arm}';SRC=EXP/'src'")
 probe=probe.replace("sys.path.insert(0,str(EXP));from run_kd import environment","sys.path.insert(0,str(ROOT/'experiments/kd_pixel_v2'));from run_kd import environment")
 probe=probe.replace("ROOT/'runs/kd_pixel_v1/10-5/step1/checkpoints/model_iter_2000.pth'","ROOT/'runs/kd_pixel_v2/10-5/step1/checkpoints/model_iter_2000.pth'")
 probe=probe.replace("ROOT/'runs/kd_pixel_v2/gradient_probe_warmup2000.json'",f"ROOT/'runs/kd_parallel_v1/{arm}/gradient_probe_warmup2000.json'")
 (E/arm/'probe.py').write_text(probe)
atomic_json(E/'protocol.json',{'utc':now(),'goal_amendment':str(R/'runs/kd_pixel_v2/goal_amendment.json'),
 'a_sigmoid':'Preserve absolute probabilities of supported old foreground classes using independent Bernoulli KL',
 'b_relational':'Preserve relative distinctions among supported old foreground classes using conditional categorical KL; invariant to common old-logit offsets',
 'shared':'same CAM evidence gates, square-root class support balance, lambda0.1, temperature2, globalbatch8, seed0; no background/new outputs in either KD',
 'resume':'same step1 warmup2000 student AND optimizer; teacher shared step0; subsequent dataloader stream restarts under chosen GPU grouping, not bitwise continuation',
 'comparison':'no baseline rerun; both full two-stage endpoints compared to archived BASE and prototype KD',
 'resource':'8card only; GPU partition chosen from measured full-training throughput'})
print(E)
