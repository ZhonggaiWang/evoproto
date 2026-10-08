"""Record exact-source behavior/DDP receipts, then require actual network smoke."""
from pathlib import Path
import sys,os,json,subprocess,argparse
R=Path('/ML-vePFS/infra_rd/kun/others/wzg/workspace/evoproto');E=R/'experiments/prototype_sep_v1';S=E/'b_semantic/src'
sys.path.insert(0,str(R/'experiments/kd_pixel_v2'))
from run_kd import environment
sys.path.insert(0,str(S))
from kd_runtime import safe_path,atomic_json,now,digest
PY=R/'.runtime/env/bin/python'
p=argparse.ArgumentParser();p.add_argument('--finalize',action='store_true');args=p.parse_args()
source={str(f.relative_to(S)):digest(f) for f in S.rglob('*.py')}
origin=json.loads((E/'semantic_origin.json').read_text())
assert source==origin['conditional_source_sha256'],'Unrecorded source revision'
receipt=safe_path(E/'semantic_test_receipts.json')
if not args.finalize:
    assert json.loads((R/'runs/prototype_sep_v1/formal/a_geometry/status.json').read_text())['status']=='complete'
    assert not receipt.exists(),'Refuse duplicate preflight tests'
    env=environment('8card');env['PYTHONPATH']=str(S);env['CUDA_VISIBLE_DEVICES']='0,1,2,3,4,5,6,7'
    records={}
    cases=[('unit_tests',[str(PY),'-B',str(E/'test_semantic_protected_sep.py')],23),
        ('ddp_2rank',[str(PY),'-B','-m','torch.distributed.run','--standalone','--nproc_per_node=2',str(E/'test_semantic_protected_sep_ddp.py'),'--backend','gloo'],None),
        ('ddp_8rank',[str(PY),'-B','-m','torch.distributed.run','--standalone','--nproc_per_node=8',str(E/'test_semantic_protected_sep_ddp.py'),'--backend','nccl'],None)]
    for name,cmd,count in cases:
        log=safe_path(E/(name+'.log'))
        with log.open('x') as handle:
            child=subprocess.run(cmd,cwd=E,env=env,stdout=handle,stderr=subprocess.STDOUT,timeout=180)
        assert child.returncode==0,(name,child.returncode,str(log))
        record={'passed':True,'returncode':child.returncode,'log':str(log),'evidence_sha256':digest(log),
            'source_sha256':digest(E/('test_semantic_protected_sep.py' if count else 'test_semantic_protected_sep_ddp.py')),
            'module_sha256':digest(S/'model/semantic_protected_sep.py')}
        if count:
            assert 'Ran 23 tests' in log.read_text() and '\nOK\n' in log.read_text();record['count']=count
        else:
            result=[json.loads(line) for line in log.read_text().splitlines() if line.startswith('{')][-1]
            assert result['passed'];record['result']=result
        records[name]=record
    atomic_json(receipt,{'utc':now(),'passed':True,'source_sha256':source,**records})
    print(json.dumps({'passed':True,'receipt':str(receipt),'tests':['23 CPU','2rank Gloo','8rank NCCL']}))
else:
    tests=json.loads(receipt.read_text());assert tests['passed'] and tests['source_sha256']==source
    smoke=R/'runs/prototype_sep_v1/smoke/b_semantic'
    assert json.loads((smoke/'status.json').read_text())['status']=='complete'
    assert json.loads((smoke/'evaluation_workers_complete.json').read_text())['returncodes']==[0]*4
    guard=[json.loads(line) for line in (smoke/'10-5/step2/semantic_guard_metrics.jsonl').read_text().splitlines()]
    assert len(guard)==4 and all(row['semantic_gradient_used'] for row in guard)
    assert all(row['background_old_sep_gradient_zero'] for row in guard)
    assert any(row['conflicting_new_rows']>0 for row in guard),'Smoke must exercise actual projection'
    atomic_json(E/'semantic_preflight.json',{'utc':now(),'passed':True,'source_sha256':source,
        'runner_sha256':digest(E/'run_semantic_sep.py'),'smoke_complete':True,'smoke_run':str(smoke),
        'smoke_guard_rows':len(guard),'smoke_conflict_rows':[row['conflicting_new_rows'] for row in guard],
        **{key:tests[key] for key in ['unit_tests','ddp_2rank','ddp_8rank']}})
    print(json.dumps({'passed':True,'preflight':str(E/'semantic_preflight.json'),'actual_network_smoke_guard_rows':len(guard)}))
