"""Read-only status summary backed by live process handles and experiment records."""
import json,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
CONTROL=ROOT/'runs/restore_proto_olc_v1/control'

def read(path):
    try:return json.loads(path.read_text())
    except (OSError,json.JSONDecodeError):return None

def live(pid,token):
    try:return token in Path(f'/proc/{int(pid)}/cmdline').read_bytes().decode().replace('\x00',' ')
    except (OSError,ValueError,TypeError):return False

def lines(path):
    result=[]
    try:
        for line in path.read_text().splitlines():
            try:result.append(json.loads(line))
            except json.JSONDecodeError:pass
    except OSError:pass
    return result

def main():
    state=read(CONTROL/'runner_state.json') or {}
    launch=read(CONTROL/'formal_launch.json') or {}
    guard=read(ROOT/'runs/restore_proto_v1/control/reservation_state.json') or {}
    result=dict(runner_state=state,runner_live=live(launch.get('pid'),'restore_proto_olc_v1/run.py'),guard_live=live(guard.get('pid'),'gpu_reservation_v2.py'),guard_mode=guard.get('mode'),stages={})
    for stage in [1,2]:
        d=ROOT/f'runs/restore_proto_olc_v1/formal/olc/10-5/step{stage}'
        r={};records=lines(d/'olc_metrics.jsonl')
        if (d/'train.log').exists():r['training_log_age_seconds']=time.time()-(d/'train.log').stat().st_mtime
        if records:r['last_logged_iteration']=records[-1]['iteration'];r['memory_images']=records[-1]['memory_images']
        metrics=lines(d/'metrics.jsonl')
        baseline={m['iteration']:m for m in lines(ROOT/f'runs/restore_proto_v1/formal/full/10-5/step{stage}/metrics.jsonl')}
        r['validations']=[dict(iteration=m['iteration'],main=m['all_miou'],prototype=m['prototype_miou'],cam=m['cam_miou'],auxiliary_cam=m['auxiliary_cam_miou'],baseline_main=baseline.get(m['iteration'],{}).get('all_miou'),main_delta=m['all_miou']-baseline[m['iteration']]['all_miou'] if m['iteration'] in baseline else None) for m in metrics]
        audits=[a for a in (read(p) for p in d.glob('olc_label_audit_*.json')) if a]
        if audits:
            a=max(audits,key=lambda a:a['iteration']);r['latest_label_audit']={k:a[k] for k in ['iteration','images','policies']}
        r['final_checkpoint_present']=(d/'checkpoints/model_final.pth').exists()
        result['stages'][str(stage)]=r
    result['postprocess']=read(CONTROL/'postprocess_state.json')
    post_launch=read(CONTROL/'postprocess_launch.json') or {}
    result['postprocess_live']=live(post_launch.get('pid'),'tools/monitor_olc_results.py')
    print(json.dumps(result,ensure_ascii=False))
if __name__=='__main__':main()
