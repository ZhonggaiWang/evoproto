from common import *
import json,random,numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from datasets.transforms import random_scaling,random_fliplr,random_crop,normalize_img

def config():return json.loads((R/'runs/ald_calibration_v9/formal/study.json').read_text())['config']

class TrainImages(Dataset):
 def __init__(self):
  self.cfg=config();folder=Path(self.cfg['list_folder'])
  self.names=(folder/'incremental_split/train_10-5_step_3.txt').read_text().splitlines()
  val=(folder/'incremental_split/val_10-5_step_3.txt').read_text().splitlines()
  assert len(self.names)==len(set(self.names))==2145 and not set(self.names)&set(val)
  raw=np.load(folder/'cls_labels_onehot.npy',allow_pickle=True).item()
  # Discard old-class ground-truth tags at dataset construction.
  self.tags={n:np.asarray(raw[n][15:],dtype=np.float32).copy() for n in self.names}
 def __len__(self):return len(self.names)
 def __getitem__(self,i):
  name=self.names[i];image=np.asarray(Image.open(Path(self.cfg['data_folder'])/'JPEGImages'/f'{name}.jpg').convert('RGB'))
  image=random_scaling(image,scale_range=[.5,2.]);image=random_fliplr(image)
  image,box=random_crop(image,crop_size=448,mean_rgb=[123,116,103])
  valid=np.zeros((448,448),dtype=np.float32);t,b,l,r=map(int,box);valid[t:b,l:r]=1
  x=normalize_img(image).transpose(2,0,1).copy()
  return i,torch.from_numpy(x),torch.from_numpy(self.tags[name]),torch.from_numpy(valid)

def worker_seed(worker):
 seed=torch.initial_seed()%2**32;random.seed(seed);np.random.seed(seed)
