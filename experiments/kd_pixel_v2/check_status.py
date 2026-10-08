"""Persist verified process health, KD coverage, and archived-reference comparisons."""
from pathlib import Path
import os,sys,json,re,subprocess
R=Path('/ML-vePFS/infra_rd/kun/others/wzg/workspace/evoproto')
E=R/'experiments/kd_pixel_v2';U=R/'runs/kd_pixel_v2'
sys.path.insert(0,str(E/'src'))
from kd_runtime import atomic_json,now,digest

def main():
    os.chdir(R)
    processes={}
    for f in U.glob('coordinator_*.process.json'):
        d=json.loads(f.read_text());p=Path(f'/proc/{d["pid"]}/stat');c=Path(f'/proc/{d["pid"]}/cmdline')
        live=p.exists() and p.read_text().split()[21]==str(d['starttime']) and b'kd_pixel_v2/run_kd.py' in c.read_bytes()
        processes[f.stem]={'pid':d['pid'],'live':live,'host':d['host']}
    stages=[]
    for p in sorted(U.glob('10-5/step*/kd_metrics.jsonl')):
        rows=[json.loads(line) for line in p.read_text().splitlines()]
        active=[x for x in rows if x['active']]
        total=sum(x['valid_pixels'] for x in active)
        stages.append({'stage':p.parent.name,'latest_iteration':rows[-1]['iteration'] if rows else None,
            'active_logged_batches':len(active),'nonzero_kd_batches':sum(x['kd_nonzero_pixels']>0 for x in active),
            'active_logged_pixel_coverage':sum(x['kd_nonzero_pixels'] for x in active)/total if total else None,
            'active_logged_weight_per_valid_pixel':sum(x['weight_sum'] for x in active)/total if total else None,
            'latest_kd':rows[-1] if rows else None})
    # A manifest is evidence only after checking its files against actual content.
    study=json.loads((U/'study.json').read_text())
    mismatched=[]
    for name,expected in study['source_sha256'].items():
        p=E/'src'/name
        if digest(p)!=expected:mismatched.append(name)
    smoke=json.loads((R/'runs/kd_pixel_v2_smoke/training_complete.json').read_text())
    status={'utc':now(),'processes':processes,'stages':stages,'frozen_source_mismatches':mismatched,
      'shared_step0_unchanged':digest(Path(study['step0']['step0']))==study['step0']['step0_sha256'],
      'resource_policy':'8card only; no 4card task processes','smoke_stages_completed':smoke['stages'],
      'training_complete':(U/'training_complete.json').exists()}
    atomic_json(U/'verified_status.json',status)
    subprocess.run([sys.executable,'-B',str(E/'summarize_kd.py')],check=True)
    print(json.dumps({**status,'stages':[{k:v for k,v in x.items() if k!='latest_kd'} for x in stages]}))

if __name__=='__main__':main()
