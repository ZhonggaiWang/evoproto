from explore import *

def metrics(h):
 h=np.asarray(h);den=h.sum(0)+h.sum(1)-h.diagonal();iou=np.divide(h.diagonal(),den,out=np.zeros(21,dtype=float),where=den>0)*100
 return dict(all_miou=float(iou.mean()),old15=float(iou[1:16].mean()),new5=float(iou[16:].mean()),class_iou=dict(zip(voc.class_list,iou.tolist())),pixels=int(h.sum()),errors=int(h.sum()-np.trace(h)),fg_to_bg=int(h[1:,0].sum()),bg_to_fg=int(h[0,1:].sum()),fg_to_fg=int(h[1:,1:].sum()-np.trace(h[1:,1:])),histogram=h.tolist())

if __name__=='__main__':
 records=[json.loads((U/f'diagnosis/rank{i}.json').read_text()) for i in range(8)]
 names=sum([r['images'] for r in records],[])
 expected=(R/'datasets/voc/incremental_split/val_10-5_step_3.txt').read_text().splitlines()
 assert len(names)==len(set(names))==1449 and set(names)==set(expected)
 assert len({r['source_sha256'] for r in records})==1
 results={}
 for mode in MODES:
  results[mode]={k:metrics(sum(np.array(r['histograms'][mode][k]) for r in records)) for k in records[0]['histograms'][mode]}
  results[mode]['forward_seconds']=sum(r['forward_seconds'][mode] for r in records)
 assert abs(results['square448']['all']['all_miou']-70.04202315102656)<1e-6
 atomic_json(U/'diagnosis/result.json',dict(results=results,images=1449,utc=now(),baseline_exactly_reproduced=True))
 print(json.dumps({m:{k:v for k,v in r['all'].items() if k not in ['histogram','class_iou']} for m,r in results.items()},indent=2))
 print('baseline_regions',json.dumps({k:{q:v for q,v in r.items() if q not in ['histogram','class_iou']} for k,r in results['square448'].items() if isinstance(r,dict)},indent=2))
