"""Remove only audited, superseded experiment weights after the OLC comparison."""
import argparse,hashlib,json,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
CONTROL=ROOT/'runs/restore_proto_olc_v1/control'

def read(p):return json.loads(p.read_text())
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def write(p,v):
    temp=p.with_suffix('.tmp');temp.write_text(json.dumps(v,indent=2));temp.replace(p)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--execute',action='store_true');args=parser.parse_args()
    plan=read(CONTROL/'retention_plan.json');protected={Path(p).resolve() for p in plan['protected']}
    assert len(protected)==5 and all(p.is_relative_to(ROOT) for p in protected)
    dirs=[ROOT/f'runs/restore_proto_v1/formal/{arm}/10-5/step{s}' for arm in ['without_confusion','without_proto'] for s in [1,2]]
    dirs += [ROOT/f'runs/restore_proto_ald_v1/v1/formal/{arm}/10-5/step2' for arm in ['full_control','full_ald']]
    records=[]
    for d in dirs:
        assert d.resolve()==d and d.is_relative_to(ROOT)
        p=d/'checkpoints/model_final.pth';assert not p.is_symlink()
        p=p.resolve();assert p.is_relative_to(ROOT) and p not in protected
        old=read(d/'weight_retention.json') if (d/'weight_retention.json').exists() else None
        if not p.exists():
            assert old and old['status']=='deleted_after_verified_evaluation' and old['path']==str(p)
            records.append(old);continue
        receipt=read(d/('completion.json' if 'restore_proto_ald_v1' in str(d) else 'final_receipt.json'))
        expected=receipt.get('sha256',receipt.get('checkpoint_sha256'))
        digest=sha(p);assert digest==expected
        if (d/'evaluation.json').exists():
            evaluation=read(d/'evaluation.json');assert evaluation['checkpoint_sha256']==digest
            assert evaluation['histogram_label_dtype']=='int64' and not evaluation['prediction_uses_gt_tags']
        else:
            assert d.name=='step1' and 'restore_proto_v1' in str(d)
            next_stage=d.parent/'step2'
            assert read(next_stage/'predecessor.json')['sha256']==digest
            assert read(next_stage/'evaluation.json')['checkpoint_sha256']==read(next_stage/'final_receipt.json')['sha256']
        metrics=[json.loads(x) for x in (d/'metrics.jsonl').read_text().splitlines()]
        expected_updates=1200 if 'restore_proto_ald_v1' in str(d) else 8000
        assert metrics and max(m['iteration'] for m in metrics)==expected_updates
        assert all((d/n).exists() for n in ['config.json','train.log','relation_metrics.jsonl'])
        records.append(dict(path=str(p),sha256=digest,bytes=p.stat().st_size,status='verified_cleanup_candidate',reason='Superseded ablation or rejected extra-training ALD; logs, metrics and evaluation retained.'))
    if args.execute:
        assert read(ROOT/'runs/restore_proto_olc_v1/comparison.json')['status']=='complete'
        assert read(CONTROL/'postprocess_state.json')['status']=='completed'
        assert all(p.is_file() for p in protected)
        before={str(p):sha(p) for p in protected}
        for row in records:
            if row['status']!='verified_cleanup_candidate':continue
            p=Path(row['path']);assert sha(p)==row['sha256'] and p not in protected
            marker=p.parents[1]/'weight_retention.json'
            write(marker,dict(row,status='deletion_pending',time=time.time()))
            p.unlink()
            row.update(status='deleted_after_verified_evaluation',time=time.time());write(marker,row)
        assert before=={str(p):sha(p) for p in protected}
        write(CONTROL/'formal_weight_cleanup.json',dict(time=time.time(),records=records,protected_sha256=before,other_files_deleted=False,total_bytes=sum(r['bytes'] for r in records)))
    print(json.dumps(dict(executed=args.execute,records=records,total_bytes=sum(r['bytes'] for r in records))))
if __name__=='__main__':main()
