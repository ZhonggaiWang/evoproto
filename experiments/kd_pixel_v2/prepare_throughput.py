from pathlib import Path
import sys,shutil,json
R=Path('/ML-vePFS/infra_rd/kun/others/wzg/workspace/evoproto');E=R/'experiments/kd_pixel_v2'
sys.path.insert(0,str(E/'src'));from kd_runtime import safe_path,atomic_json,now
D=safe_path(R/'experiments/kd_throughput');assert not D.exists()
shutil.copytree(E/'src',D/'src',ignore=shutil.ignore_patterns('__pycache__'))
p=D/'src/continual/Trainer.py';s=p.read_text()
s=s.replace('import datetime','import datetime\nimport time')
s=s.replace('        if args.async_eval:\n',"        return 0.0, 'Throughput measurement: no validation'\n        if args.async_eval:\n",1)
marker='            for n_iter in range(args.max_iters):\n'
pos=s.rindex(marker);s=s[:pos]+s[pos:].replace(marker,marker+'''                if n_iter == 10:
                    torch.cuda.synchronize()
                    benchmark_start = time.perf_counter()
                    torch.cuda.reset_peak_memory_stats()
''',1)
marker='        if args.max_iters % args.eval_iters:\n'
s=s.replace(marker,'''        torch.cuda.synchronize()
        seconds = torch.tensor(time.perf_counter()-benchmark_start, device=device)
        peak = torch.tensor(torch.cuda.max_memory_allocated(), device=device)
        dist.all_reduce(seconds, op=dist.ReduceOp.MAX)
        dist.all_reduce(peak, op=dist.ReduceOp.MAX)
        if args.local_rank == 0:
            from kd_runtime import atomic_json, now
            atomic_json(osp.join(args.work_dir, 'throughput.json'), {
                'utc': now(), 'gpus': dist.get_world_size(), 'batch_per_gpu': args.spg,
                'global_batch': args.spg*dist.get_world_size(), 'measured_iterations': args.max_iters-10,
                'seconds': float(seconds), 'seconds_per_step': float(seconds)/(args.max_iters-10),
                'images_per_second': (args.max_iters-10)*args.spg*dist.get_world_size()/float(seconds),
                'images_per_gpu_second': (args.max_iters-10)*args.spg/float(seconds),
                'peak_allocated_gb': float(peak)/(1024**3),
                'scope': 'Disposable training trajectory; actual data, CAM/PAR, teacher, full losses, optimizer; no accuracy claim'})
''' + marker)
p.write_text(s)
p=D/'src/scripts/dist_train_voc_seg_neg.py';s=p.read_text().replace('kd_pixel_v2','kd_throughput')
s=s.replace('    torch.save(state, path)','    return  # no benchmark checkpoint needed')
p.write_text(s)
atomic_json(R/'runs/kd_pixel_v2/goal_amendment.json',{'utc':now(),'user_request':'Improve actual GPU efficiency; larger per-GPU batches and parallel meaningful KD strategies permitted; update goal',
 'priority':'images/sec and GPU-hours per useful candidate, not temperature',
 'constraints':['8card only','all writes within /ML-vePFS/infra_rd/kun/others/wzg','reuse step0 and available common warmup checkpoints','no seed search, hyperparameter grid, or baseline retraining'],
 'plan':['measure 8x1, 4x2, 2x4, 1x8 with global batch8','select GPU grouping by measured throughput','parallelize a small number of distinct KD mechanisms','evaluate each complete two-stage candidate against existing BASE/KD']})
print(D)
