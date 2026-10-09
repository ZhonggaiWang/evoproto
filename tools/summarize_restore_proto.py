"""Summarize actual completion and method diagnostics for the restoration study."""
from pathlib import Path
import json,time,os
ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'runs/restore_proto_v1'

def read(path):
    try:return json.loads(path.read_text())
    except (FileNotFoundError,json.JSONDecodeError):return None

def last(path):
    try:
        for line in reversed(path.read_text().splitlines()):
            try:return json.loads(line)
            except json.JSONDecodeError:pass
    except FileNotFoundError:pass
    return None

def process(pid_path,needle):
    try:
        pid=int(pid_path.read_text());raw=Path(f'/proc/{pid}/cmdline').read_bytes()
        return {'pid':pid,'live':bool(raw and needle.encode() in raw),'command':raw.decode().replace('\0',' ').strip()}
    except (FileNotFoundError,ValueError,PermissionError):return {'live':False}

def main():
    formal=RUN/'formal';manifest=read(formal/'manifest.json')
    result={'time':time.time(),'runner':process(RUN/'control/formal.pid','restore_proto_v1/run.py'),'reservation':process(RUN/'control/reservation.pid','gpu_reservation'),'reservation_state':read(RUN/'control/reservation_state.json'),'global_batch':manifest['batch_size'],'per_gpu_batch':manifest['batch_size']//len(manifest['gpus']),'stages':[],'complete':True}
    initial=read(formal/'shared_step0_evaluation.json')
    if initial:result['shared_step0']={k:v['miou'] for k,v in initial['results']['square448'].items()}
    for variant in manifest['variants']:
        for stage in [1,2]:
            d=formal/variant/'10-5'/f'step{stage}';metric=last(d/'metrics.jsonl');relation=last(d/'relation_metrics.jsonl');receipt=read(d/'final_receipt.json');evaluation=read(d/'evaluation.json')
            evaluation_valid=bool(evaluation and evaluation.get('histogram_label_dtype')=='int64' and metric and abs(evaluation['results']['square448']['main']['miou']-metric['all_miou'])<.2)
            final=d/'checkpoints/model_final.pth';done=bool(final.exists() and receipt and metric and metric['iteration']==manifest['iterations'] and (stage!=2 or evaluation_valid))
            row={'variant':variant,'stage':stage,'complete':done,'final_checkpoint_exists':final.exists(),'last_validation':metric,'latest_relations':relation,'prototype_relations_in_optimized_loss':bool(relation and relation['iteration']>2000 and variant!='without_proto')}
            row['effective_restoration']={'explicit_prototype_parameters':True,'incremental_prototype_losses':variant!='without_proto','confusion': 'uniform_foreground_control' if variant=='without_confusion' else 'observed_directed_confusion_top2','post_warmup_weights':{'main_BCE':.1,'prototype_BCE':0. if variant=='without_proto' else .1,'main_KD':.1,'prototype_KD':0. if variant=='without_proto' else .05,'main_SEP':.02,'prototype_SEP':0. if variant=='without_proto' else .05,'old_prototype_direction':0. if variant=='without_proto' else .01},'ALD':False}
            row['standalone_evaluation_verified_against_trainer']=evaluation_valid
            if evaluation_valid:row['final_protocol_miou']={m:{head:v['miou'] for head,v in branch.items()} for m,branch in evaluation['results'].items()}
            graph=read(d/'confusion.json')
            if graph:
                old=11 if stage==1 else 16;edges=[]
                for a,values in enumerate(graph['ema']):
                    for b,n in enumerate(values):
                        if a and b and a!=b and graph['unique_image_support'][a][b]>=3:
                            edges.append({'anchor':a,'rival':b,'rate':n/max(graph['anchor_mass'][a],1e-6),'unique_images':graph['unique_image_support'][a][b],'cross_old_new':(a<old)!=(b<old)})
                row['strongest_supported_confusions']=sorted(edges,key=lambda x:x['rate'],reverse=True)[:10]
            result['stages'].append(row);result['complete'] &= done
    result['completion_scope']='Formal restoration training/evaluation only; not the complete user goal.'
    result['note']='Legacy proto_kd/proto_sep in train.log are diagnostics with zero coefficients. The legacy confusion_reweight=false flag also does not disable ConfusionProto. Actual active restored terms are in relation_metrics.jsonl; before iteration2001 they are computed but multiplied by zero.'
    out=RUN/'progress.json';tmp=out.with_suffix('.tmp');tmp.write_text(json.dumps(result,indent=2));tmp.replace(out)
    print(json.dumps({'runner':result['runner']['live'],'reservation':result['reservation']['live'],'complete':result['complete'],'per_gpu_batch':result['per_gpu_batch'],'stages':[{'variant':s['variant'],'stage':s['stage'],'iteration':(s['latest_relations'] or {}).get('iteration'), 'complete':s['complete'],'miou':(s['last_validation'] or {}).get('all_miou')} for s in result['stages']]}))

if __name__=='__main__':main()
