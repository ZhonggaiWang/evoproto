"""Regression: class-index palette masks must not overflow histogram encoding."""
import importlib.util
from pathlib import Path
import torch
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('restore_eval',ROOT/'experiments/restore_proto_v1/evaluate.py')
evaluate=importlib.util.module_from_spec(spec);spec.loader.exec_module(evaluate)
for classes in (11,16,21):
    label=torch.tensor(list(range(classes))+[255],dtype=torch.uint8)
    pred=torch.tensor(list(range(classes))+[0],dtype=torch.int64)
    correct=evaluate.confusion_matrix(label,pred,classes)
    assert torch.equal(correct,torch.eye(classes,dtype=torch.int64))
    reference=np.zeros((classes,classes),dtype=np.int64)
    for truth,guess in zip(label.tolist(),pred.tolist()):
        if truth<classes:reference[truth,guess]+=1
    assert np.array_equal(correct.numpy(),reference)
    if classes==21:
        buggy=torch.bincount((label[:-1]*classes+pred[:-1]).flatten(),minlength=classes*classes).reshape(classes,classes)
        assert not torch.equal(buggy,correct)
        print('Original uint8 bug reproduced; all-class perfect predictions are now exactly diagonal.')
    print('PASS classes='+str(classes)+' including void label 255')
