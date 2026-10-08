from merge import metrics
from explore import *
records=[json.loads((U/f'refinement/rank{i}.json').read_text()) for i in range(8)]
names=sum([r['images'] for r in records],[])
expected=(R/'datasets/voc/incremental_split/val_10-5_step_3.txt').read_text().splitlines()
assert len(names)==len(set(names))==1449 and set(names)==set(expected)
assert len({r['source_sha256'] for r in records})==1
results={m:{k:metrics(sum(np.array(r['histograms'][m][k]) for r in records)) for k in records[0]['histograms'][m]} for m in records[0]['histograms']}
assert abs(results['square448']['all']['all_miou']-70.04202315102656)<1e-6
atomic_json(U/'refinement/result.json',dict(results=results,images=1449,utc=now(),tests=records[0]['tests'],seconds={m:sum(r['seconds'][m] for r in records) for m in records[0]['seconds']}))
print(json.dumps({m:{k:v for k,v in r['all'].items() if k not in ['histogram','class_iou']} for m,r in results.items()},indent=2))
print('regions',json.dumps({m:{k:{q:v for q,v in r.items() if q in ['pixels','errors','fg_to_bg','bg_to_fg','fg_to_fg']} for k,r in h.items()} for m,h in results.items() if m in ['square448','square672','margin448','margin672']},indent=2))
