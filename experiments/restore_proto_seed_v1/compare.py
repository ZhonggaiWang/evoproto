"""Verify completed arms and compare identical inference protocols."""
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import numpy as np
from tools.compare_olc_results import load_final
from tools.compare_restore_proto_results import paired_bootstrap
from experiments.restore_proto_v1.run import sha,write
STUDY=ROOT/'runs/restore_proto_seed_v1'
BASE=ROOT/'runs/restore_proto_v1/formal/full/10-5'
def read(p):return json.loads(p.read_text())
def main():
    manifest=read(STUDY/'formal/manifest.json')
    for name,h in manifest['source_sha256'].items():assert sha(ROOT/name)==h,name
    bn,bh,bo,bf=load_final(BASE/'step2',ROOT/'runs/restore_proto_olc_v1/reference_full_gpu')
    result=dict(status='complete',iterations_per_stage=8000,global_batch=8,seed=0,arms={},comparisons={},limitations='Single seed; adaptively inspected validation. Image bootstrap does not estimate seed variance. Alpha .5 fixed in advance. Seed correction is training-only; segmentation inference does not use image tags.')
    histograms={'baseline':bh}
    configs=['task','dataset','seed','spg','crop_size','max_iters','warmup_iters','loss_warmup_iters','lr','wt_decay','betas','power','w_ptc','w_seg','w_proto_seg','ald_mode','variant']
    for arm in manifest['arms']:
        armroot=STUDY/'formal'/arm/'10-5';records={}
        for stage in [1,2]:
            d=armroot/f'step{stage}';conf=read(d/'config.json');base=read(BASE/f'step{stage}/config.json')
            for key in configs:assert conf[key]==base[key],(arm,stage,key)
            assert conf['seed_mode']==arm and conf['max_iters']==8000 and conf['spg']==4
            assert '--nproc_per_node=2' in read(d/'command.json')
            receipt=read(d/'final_receipt.json');assert sha(Path(receipt['path']))==receipt['sha256']
            predecessor=read(d/'predecessor.json');assert sha(Path(predecessor['path']))==predecessor['sha256']
            expected=manifest['initial_sha256'] if stage==1 else read(armroot/'step1/final_receipt.json')['sha256']
            assert predecessor['sha256']==expected
            curve=list(map(json.loads,(d/'metrics.jsonl').read_text().splitlines()));assert curve[-1]['iteration']==8000
            stats=list(map(json.loads,(d/'seed_metrics.jsonl').read_text().splitlines()))
            active=[r for r in stats if r['active']];selected=sum(r['corrected_pixels'] for r in active)
            # Zero activation is reported explicitly, not hidden as an improvement.
            records[str(stage)]=dict(checkpoint_sha256=receipt['sha256'],learning_curve=curve,module_activated=selected>0,sampled_training_pixel_fraction=selected/max(sum(r['valid_pixels'] for r in active),1),offline_audit=read(d/'seed_audit.json'),prototype_audit=read(d/'prototype_direction_audit.json'))
        names,h,official,fusion=load_final(armroot/'step2');assert np.array_equal(names,bn);assert fusion['device']==bf['device']=='cuda'
        histograms[arm]=h;result['arms'][arm]=dict(stages=records,final=fusion['results'])
    pairs=[('baseline',a) for a in manifest['arms']]
    if 'ignore' in histograms:pairs.append(('ignore','correct'))
    if 'no_graph' in histograms:pairs.append(('no_graph','correct'))
    for reference,candidate in pairs:
        rows={}
        for mode in ['square448','aspect672']:
            rows[mode]={}
            for alpha,head in [('0.0','main'),('0.5','fixed_half_prototype_fusion')]:
                rows[mode][head]=paired_bootstrap(histograms[reference][mode][alpha],histograms[candidate][mode][alpha],1000)
        result['comparisons'][reference+'_to_'+candidate]=rows
    write(STUDY/'comparison.json',result)
    write(ROOT/'experiments/restore_proto_seed_v1/results.json',result)
    print(json.dumps(result['comparisons']),flush=True)
if __name__=='__main__':main()
