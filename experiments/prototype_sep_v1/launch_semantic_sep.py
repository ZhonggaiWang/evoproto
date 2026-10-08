from pathlib import Path
import sys,json,subprocess,argparse
R=Path('/ML-vePFS/infra_rd/kun/others/wzg/workspace/evoproto');E=R/'experiments/prototype_sep_v1'
sys.path.insert(0,str(R/'experiments/kd_pixel_v2'))
from run_kd import environment
sys.path.insert(0,str(E/'b_semantic/src'))
from kd_runtime import safe_path,atomic_json,now
p=argparse.ArgumentParser();p.add_argument('--smoke',action='store_true');args=p.parse_args()
parent=R/'runs/prototype_sep_v1/formal/a_geometry'
assert json.loads((parent/'status.json').read_text())['status']=='complete'
assert not json.loads((R/'runs/prototype_sep_v1/result_analysis.json').read_text())['recommendation']['candidate_exceeds_required_endpoint']
tests=json.loads((E/'semantic_test_receipts.json').read_text());assert tests['passed']
if not args.smoke:assert json.loads((E/'semantic_preflight.json').read_text())['passed']
run=safe_path(R/'runs/prototype_sep_v1'/('smoke' if args.smoke else 'formal')/'b_semantic')
assert not (run/'coordinator.process.json').exists(),'Refuse duplicate launch'
for path in (E/'b_semantic/src').rglob('*'):safe_path(path)
run.mkdir(parents=True,exist_ok=True)
cmd=[str(R/'.runtime/env/bin/python'),'-B',str(E/'run_semantic_sep.py')]
if args.smoke:cmd.append('--smoke')
with open(safe_path(run/'coordinator.log'),'x') as log:
    child=subprocess.Popen(cmd,cwd=R,env=environment('8card'),stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
atomic_json(run/'coordinator.process.json',{'pid':child.pid,'starttime':Path(f'/proc/{child.pid}/stat').read_text().split()[21],
    'command':cmd,'utc':now()})
print(json.dumps({'pid':child.pid,'run':str(run),'smoke':args.smoke}))
