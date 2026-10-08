"""Compare the single KD trajectory with archived, same-budget references."""
from pathlib import Path
import sys,json,math
ROOT=Path('/ML-vePFS/infra_rd/kun/others/wzg/workspace/evoproto')
EXP=ROOT/'experiments/kd_pixel_v2';SRC=EXP/'src';RUN=ROOT/'runs/kd_pixel_v2'
sys.path.insert(0,str(SRC))
from kd_runtime import atomic_json,now,digest

def historical(arm,step,iteration):
    p=ROOT/f'runs/fixed_baseline_v1/{arm}/10-5/step{step}/metrics.jsonl'
    rows=[json.loads(line) for line in p.read_text().splitlines()]
    row=next(x for x in rows if x['iteration']==iteration)
    values=list(row['class_iou'].values());old=10 if step==1 else 15
    assert len(values)==(16 if step==1 else 21) and all(isinstance(x,(int,float)) and math.isfinite(x) for x in values)
    return {'all_miou':sum(values)/len(values),'previous_foreground_miou':sum(values[1:old+1])/old,
            'current_foreground_miou':sum(values[old+1:])/5,'class_iou':row['class_iou'],'source':str(p),'source_sha256':digest(p)}

def main():
    rows=[]
    for p in sorted((RUN/'evaluations').glob('step*/result.json')):
        candidate=json.loads(p.read_text());step=candidate['step'];iteration=candidate['iteration']
        reference={arm:historical(arm,step,iteration) for arm in ['base','kd']}
        metrics=['all_miou','previous_foreground_miou','current_foreground_miou']
        row={'step':step,'iteration':iteration,'candidate':{key:candidate[key] for key in metrics},
             'reference':{arm:{key:values[key] for key in metrics} for arm,values in reference.items()},
             'delta_pp':{arm:{key:candidate[key]-reference[arm][key] for key in metrics} for arm in reference},
             'candidate_result_sha256':digest(p),'checkpoint_sha256':candidate['checkpoint_sha256'],
             'interpretation':'pre-KD warmup checkpoint; no gain attribution' if iteration<=2000 else 'intermediate' if iteration<8000 else 'stage endpoint',
             'per_class_delta_vs_base':{name:value-reference['base']['class_iou'][name] for name,value in candidate['class_iou'].items()},
             'pixel_error_rates_percent':{'old_to_background':100*candidate['old_gt_to_background_pixels']/candidate['old_gt_pixels'],
                'old_to_new':100*candidate['old_gt_to_new_pixels']/candidate['old_gt_pixels'],
                'new_to_old':100*candidate['new_gt_to_old_pixels']/candidate['new_gt_pixels']}}
        rows.append(row)
    result={'updated_utc':now(),'study':'kd_pixel_v2','training_complete':(RUN/'training_complete.json').exists(),
            'both_stage_endpoints_evaluated':sum(x['iteration']==8000 for x in rows)==2,
            'resource_policy':'8卡机 only; 4卡机 stopped per user request','comparisons':rows,
            'evidence_limits':['single seed and order','class grouping previous/current, not initial/cumulative','warmup is before active KD','do not claim final improvement from intermediate checkpoints']}
    atomic_json(RUN/'comparison.json',result)
    for row in rows:
        print(json.dumps({key:row[key] for key in ['step','iteration','candidate','delta_pp','interpretation']}))

if __name__=='__main__':main()
