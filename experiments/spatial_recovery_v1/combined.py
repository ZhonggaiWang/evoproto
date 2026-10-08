from explore import *
from cached_eval import true_regions
from refine import boundary_margin

def main():
 p=argparse.ArgumentParser();p.add_argument('--rank',type=int,required=True);a=p.parse_args()
 torch.set_num_threads(1);torch.cuda.set_device(a.rank);torch.manual_seed(0)
 torch.set_float32_matmul_precision('high');torch.backends.cudnn.allow_tf32=True
 torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
 cfg=json.loads((R/'runs/local_sep_v5/formal/study.json').read_text())['config']
 ck=R/'runs/local_sep_v5/formal/10-5/step2/checkpoints/model_final.pth'
 assert digest(ck)=='0770ac07bcb343d510bbb6287526231352891e16f0c3e11e4b0d36a14fca4177'
 m=network(backbone=cfg['backbone'],num_classes=21,classes_list=[11,5,5],pretrained=False,init_momentum=cfg['momentum'],aux_layer=cfg['aux_layer'])
 m.load_state_dict(torch.load(ck,map_location='cpu',weights_only=True,mmap=True)['model_state'],strict=True);m.cuda().eval()
 ds=voc.VOC12SegDataset(root_dir=cfg['data_folder'],name_list_dir=cfg['list_folder'],split='val',stage='val',aug=False,ignore_index=255,num_classes=21,tasks='10-5',step=2);ds.label_dir=cfg['val_label_dir']
 loader=DataLoader(Subset(ds,list(range(a.rank,len(ds),8))),batch_size=1,shuffle=False,num_workers=2)
 cache=torch.load(U/f'diagnosis/logits_rank{a.rank}.pth',map_location='cpu',weights_only=False)
 modes=['aspect672','aspect672_margin','mean448_672_margin']
 hist={mode:{k:np.zeros((21,21),np.int64) for k in ['all','boundary','interior','small_foreground','medium_foreground','large_foreground']} for mode in modes}
 names=[];saved={};start=time.time();times={k:0. for k in ['forward','refine_aspect','refine_mean']}
 with torch.inference_mode():
  for nb,image,label,_ in loader:
   name=str(nb[0]);image=image.cuda();hw=image.shape[-2:];h,w=hw
   scale=672/math.sqrt(h*w);shape=(max(16,round(h*scale/16)*16),max(16,round(w*scale/16)*16))
   inp=F.interpolate(image,size=shape,mode='bilinear',align_corners=False)
   torch.cuda.synchronize();t=time.time();native=m(inp)[1];torch.cuda.synchronize();times['forward']+=time.time()-t
   saved[name]=native[0].cpu();z=F.interpolate(native,size=hw,mode='bilinear',align_corners=False)
   mean=sum(F.interpolate(cache[name][key][None].cuda(),size=hw,mode='bilinear',align_corners=False) for key in ['square448','square672'])/2
   torch.cuda.synchronize();t=time.time();aout=boundary_margin(image,z);torch.cuda.synchronize();times['refine_aspect']+=time.time()-t
   t=time.time();mout=boundary_margin(image,mean);torch.cuda.synchronize();times['refine_mean']+=time.time()-t
   preds={mode:out.argmax(1)[0].cpu().numpy() for mode,out in zip(modes,[z,aout,mout])}
   gt=label[0].numpy();masks=true_regions(gt)
   for mode,pred in preds.items():
    hist[mode]['all']+=histogram(gt,pred)
    for key,mask in masks.items():hist[mode][key]+=histogram(gt,pred,mask)
   names.append(name)
   if len(names)%25==0:print(a.rank,len(names),round(time.time()-start,1),flush=True)
 out=safe_path(U/'combined');out.mkdir(exist_ok=True)
 torch.save(saved,safe_path(out/f'logits_rank{a.rank}.pth'))
 atomic_json(out/f'rank{a.rank}.json',dict(images=names,histograms={m:{k:v.tolist() for k,v in h.items()} for m,h in hist.items()},seconds=times,utc=now(),source_sha256=digest(Path(__file__)),refiner_sha256=digest(E/'refine.py')))
 print('DONE',a.rank,flush=True)

if __name__=='__main__':main()
