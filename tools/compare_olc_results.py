"""Verify the full same-budget OLC comparison, then summarize paired results."""
import argparse,hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from tools.compare_restore_proto_results import paired_bootstrap,miou
BASE=ROOT/'runs/restore_proto_v1/formal/full/10-5'
CANDIDATE=ROOT/'runs/restore_proto_olc_v1/formal/olc/10-5'
OUTPUT=ROOT/'runs/restore_proto_olc_v1/comparison.json'

def read(path):return json.loads(path.read_text())

def write(value):
    temp=OUTPUT.with_suffix('.tmp');temp.write_text(json.dumps(value,indent=2));temp.replace(OUTPUT)

def load_final(directory):
    official=read(directory/'evaluation.json');fusion=read(directory/'fusion_evaluation.json');receipt=read(directory/'final_receipt.json')
    assert official['checkpoint_sha256']==fusion['checkpoint_sha256']==receipt['sha256']
    assert official['images']==fusion['images']==1449 and official['stage']==fusion['stage']==2
    assert official['histogram_label_dtype']=='int64' and not official['prediction_uses_gt_tags'] and not fusion['image_tags_used']
    hist={};difference={}
    with np.load(directory/'fusion_evaluation.npz',allow_pickle=False) as arrays:
        names=arrays['names'].copy();alphas=list(arrays['alphas'])
        assert len(names)==len(set(names))==1449
        assert alphas==fusion['alphas'] and all(a in alphas for a in [0.,.5,1.])
        for mode in ['square448','aspect672']:
            assert arrays[mode].shape==(1449,len(alphas),21,21)
            assert np.issubdtype(arrays[mode].dtype,np.integer) and (arrays[mode]>=0).all()
            totals=arrays[mode].sum(0)
            for i,a in enumerate(alphas):
                row=fusion['results'][mode][str(a)]
                assert np.array_equal(totals[i],row['histogram'])
                assert abs(float(miou(totals[i]))-row['miou'])<1e-9
            difference[mode]={}
            for a,head in [(0.,'main'),(1.,'prototype')]:
                delta=fusion['results'][mode][str(a)]['miou']-official['results'][mode][head]['miou']
                assert abs(delta)<.01,(mode,head,delta)
                difference[mode][head]=delta
            hist[mode]={str(a):arrays[mode][:,alphas.index(a)].copy() for a in [0.,.5]}
    verification=dict(checkpoint_sha256=receipt['sha256'],images=1449,per_image_histograms_verified=True,fusion_minus_standalone_miou=difference,standalone_device=official.get('device','cuda'),endpoint_tolerance_pp=.01)
    if directory==CANDIDATE/'step2':
        (directory/'fusion_verification.json').write_text(json.dumps(verification,indent=2))
    return names,hist,official,fusion

def main():
    p=argparse.ArgumentParser();p.add_argument('--samples',type=int,default=1000);a=p.parse_args();assert a.samples>=100
    needed=[]
    for root in [BASE,CANDIDATE]:
        for stage in [1,2]:
            for name in ['config.json','final_receipt.json','predecessor.json','metrics.jsonl']:
                needed.append(root/f'step{stage}'/name)
        for name in ['evaluation.json','fusion_evaluation.json','fusion_evaluation.npz']:
            needed.append(root/'step2'/name)
    for stage in [1,2]:needed.append(CANDIDATE/f'step{stage}/olc_label_audit_8000.json')
    missing=[str(x.relative_to(ROOT)) for x in needed if not x.exists()]
    if missing:
        write(dict(status='waiting',missing=missing));print(json.dumps({'status':'waiting','missing':missing}));return
    manifest=read(ROOT/'runs/restore_proto_olc_v1/formal/manifest.json')
    assert manifest['iterations']==8000 and manifest['batch_size']==8 and manifest['gpus']==[5,6]
    for name,expected in manifest['source_sha256'].items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==expected, name
    compare_keys=['task','dataset','seed','spg','crop_size','max_iters','warmup_iters','loss_warmup_iters','lr','wt_decay','betas','power','w_ptc','w_seg','w_proto_seg','w_proto_kd','w_proto_sep','ald_mode','confusion_reweight','cam_scales','backbone','aux_layer','variant']
    learning={};lineage={}
    for stage in [1,2]:
        bd=BASE/f'step{stage}';cd=CANDIDATE/f'step{stage}';bc=read(bd/'config.json');cc=read(cd/'config.json')
        assert cc['olc'] and cc['max_iters']==8000 and cc['spg']==4 and cc['ald_mode']=='off'
        for key in compare_keys:assert bc[key]==cc[key],(stage,key,bc[key],cc[key])
        commands={name:read(path/'command.json') for name,path in [('baseline',bd),('olc',cd)]}
        assert all('--nproc_per_node=2' in cmd for cmd in commands.values())
        receipt=read(cd/'final_receipt.json')
        assert hashlib.sha256(Path(receipt['path']).read_bytes()).hexdigest()==receipt['sha256']
        previous=read(cd/'predecessor.json');lineage[str(stage)]=previous
        assert hashlib.sha256(Path(previous['path']).read_bytes()).hexdigest()==previous['sha256']
        if stage==1:assert previous['sha256']==read(bd/'predecessor.json')['sha256']
        else:assert previous['sha256']==read(CANDIDATE/'step1/final_receipt.json')['sha256']
        metrics={name:{row['iteration']:row for row in map(json.loads,(path/'metrics.jsonl').read_text().splitlines())} for name,path in [('baseline',bd),('olc',cd)]}
        assert max(metrics['baseline'])==max(metrics['olc'])==8000
        learning[str(stage)]={str(i):dict(baseline_main=metrics['baseline'][i]['all_miou'],olc_main=r['all_miou'],delta_pp=r['all_miou']-metrics['baseline'][i]['all_miou'],olc_prototype=r['prototype_miou'],olc_cam=r['cam_miou'],olc_auxiliary_cam=r['auxiliary_cam_miou']) for i,r in metrics['olc'].items() if i in metrics['baseline']}
    bn,bh,bo,bf=load_final(BASE/'step2');cn,ch,co,cf=load_final(CANDIDATE/'step2')
    assert np.array_equal(bn,cn)
    comparisons={}
    for mode in ['square448','aspect672']:
        comparisons[mode]={}
        for alpha,name in [('0.0','main'),('0.5','fixed_half_prototype_fusion')]:
            b=bf['results'][mode][alpha];c=cf['results'][mode][alpha]
            comparison=paired_bootstrap(bh[mode][alpha],ch[mode][alpha],a.samples)
            comparison.update(baseline_miou=b['miou'],olc_miou=c['miou'],old_delta_pp=c['previous_foreground']-b['previous_foreground'],new_delta_pp=c['current_foreground']-b['current_foreground'],class_delta_pp=[cv-bv for bv,cv in zip(b['class_iou'],c['class_iou'])])
            comparisons[mode][name]=comparison
    result=dict(status='complete',same_budget_verified=True,iterations_per_stage=8000,global_batch=8,seed=0,own_predecessor_lineage=lineage,learning_curves=learning,comparisons= comparisons,final_label_audits={str(s):read(CANDIDATE/f'step{s}/olc_label_audit_8000.json') for s in [1,2]},limitations='Single seed and adaptively inspected VOC validation set. Paired bootstrap measures image-sampling uncertainty, not training-seed variance. Fusion alpha 0.5 was fixed before OLC. CAM diagnostics use validation image tags, whereas segmentation does not. Offline label metrics do not supervise training.')
    write(result);print(json.dumps({'status':'complete','comparisons':comparisons}))
if __name__=='__main__':main()
