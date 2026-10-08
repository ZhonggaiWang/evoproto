from explore import *
from refine import boundary_margin,test

def true_regions(gt):
 # VOC void bands border most objects. Their valid neighbors are boundaries.
 edge=np.zeros_like(gt,dtype=bool)
 d=gt[1:]!=gt[:-1];edge[1:]|=d;edge[:-1]|=d
 d=gt[:,1:]!=gt[:,:-1];edge[:,1:]|=d;edge[:,:-1]|=d
 radius=max(2,round(.01*math.hypot(*gt.shape)))
 near=distance_transform_edt(~edge)<=radius if edge.any() else np.zeros_like(edge)
 masks=regions(gt);masks.update(boundary=near,interior=~near)
 return masks

def main():
 p=argparse.ArgumentParser();p.add_argument('--rank',type=int,required=True);a=p.parse_args()
 torch.set_num_threads(1);torch.cuda.set_device(a.rank)
 checks=test()
 # Trusted cache produced by explore.py here; VOC names are numpy.str_ keys.
 cache=torch.load(U/f'diagnosis/logits_rank{a.rank}.pth',map_location='cpu',weights_only=False)
 cfg=json.loads((R/'runs/local_sep_v5/formal/study.json').read_text())['config']
 ds=voc.VOC12SegDataset(root_dir=cfg['data_folder'],name_list_dir=cfg['list_folder'],split='val',stage='val',aug=False,ignore_index=255,num_classes=21,tasks='10-5',step=2);ds.label_dir=cfg['val_label_dir']
 modes=MODES+['mean448_672','margin448','margin672']
 hist={mode:{k:np.zeros((21,21),np.int64) for k in ['all','boundary','interior','small_foreground','medium_foreground','large_foreground']} for mode in modes}
 times={m:0. for m in modes};names=[];start=time.time()
 loader=DataLoader(Subset(ds,list(range(a.rank,len(ds),8))),batch_size=1,shuffle=False,num_workers=2)
 with torch.inference_mode():
  for nb,image,label,_ in loader:
   name=nb[0];image=image.cuda();hw=image.shape[-2:]
   z={m:F.interpolate(cache[name][m][None].cuda(),size=hw,mode='bilinear',align_corners=False) for m in MODES}
   z['mean448_672']=(z['square448']+z['square672'])/2
   for key,parent in [('margin448','square448'),('margin672','square672')]:
    torch.cuda.synchronize();t=time.time();z[key]=boundary_margin(image,z[parent]);torch.cuda.synchronize();times[key]+=time.time()-t
   preds={m:s.argmax(1)[0].cpu().numpy() for m,s in z.items()}
   gt=label[0].numpy();masks=true_regions(gt)
   for mode,pred in preds.items():
    hist[mode]['all']+=histogram(gt,pred)
    for key,mask in masks.items():hist[mode][key]+=histogram(gt,pred,mask)
   names.append(name)
   if len(names)%25==0:print(a.rank,len(names),round(time.time()-start,1),flush=True)
 atomic_json(U/f'refinement/rank{a.rank}.json',dict(images=names,histograms={m:{k:v.tolist() for k,v in hs.items()} for m,hs in hist.items()},seconds=times,tests=checks,source_sha256=digest(E/'refine.py'),utc=now()))
 print('DONE',a.rank,flush=True)

if __name__=='__main__':main()
