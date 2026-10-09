"""CPU semantic checks before any optional refinement workload is admitted."""
import json,sys,tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import torch
from experiments.restore_proto_ald_v1.ald import image_targets,fuse,gate_auxiliary,uncertain_loss,partial_set_loss,foreground_relation_loss
from experiments.restore_proto_ald_v1.evidence import calibrate_presence
from experiments.restore_proto_ald_v1.continued_graph import ContinuedConfusionProto
from experiments.restore_proto_v1.mechanism import ConfusionProto

def main():
    torch.set_num_threads(2);torch.manual_seed(9)
    state=torch.full((1,15),-1,dtype=torch.long);state[0,0]=1;state[0,1]=0
    old=torch.full((1,15),-2.);old[:,2]=4
    teacher=torch.zeros(1,16,2,2);teacher[:,0]=2;teacher[0,1,0,0]=5;teacher[0,3,0,1]=5
    ref=torch.zeros(1,21,2,2);ref[:,0]=2;ref[0,16,1,0]=5;ref[0,1,0,0]=6
    tags=torch.ones(1,5)
    target,allowed,_=image_targets(state,old,old,teacher,ref,tags)
    assert target[0,0]==1 and target[0,1]==0 and 0<target[0,3]<.5
    assert allowed[0,2] and not allowed[0,1]
    old[0,4]=5;target,_,conflict=image_targets(state,old,old,teacher,ref,tags)
    assert target[0,4]==.5 and conflict[0,4]
    par=torch.zeros(1,2,2,dtype=torch.long);box=[[0,2,0,2]]
    fused=fuse(par,teacher,ref,state,box)
    assert fused['labels'].tolist()==[[[1,255],[255,0]]]
    par[0,0,0]=17;rejected=fuse(par,teacher,ref,state,[[0,1,0,2]])
    assert rejected['labels'][0,0,0]==255 and not rejected['trusted_old'].any()
    assert (rejected['labels'][:,1]==255).all() and not rejected['unknown'][:,1].any()
    ref_new=ref.clone();ref_new[0,17,0,0]=8
    assert fuse(par,teacher,ref_new,state,box)['labels'][0,0,0]==17
    assert gate_auxiliary(par,state,ref,calibrate_new=False)[0,0,0]==17
    assert gate_auxiliary(par,state,ref,calibrate_new=True)[0,0,0]==255
    student=torch.randn(1,21,2,2,requires_grad=True);reference=ref.clone().requires_grad_()
    loss=foreground_relation_loss(student,reference,fused['unknown'],fused['valid'],state,tags)
    loss.backward();assert reference.grad is None
    assert student.grad[:,0].abs().sum()==0 and student.grad[:,1:].sum(1).abs().max()<2e-7
    shifted=student.detach().clone();shifted[:,1:]+=3
    assert torch.allclose(loss,foreground_relation_loss(shifted,ref,fused['unknown'],fused['valid'],state,tags),atol=2e-6)
    assert partial_set_loss(student,fused['unknown'],fused['valid'],torch.ones_like(state),tags).abs()<1e-6
    empty=torch.zeros_like(par,dtype=torch.bool)
    assert uncertain_loss(student,ref,empty,empty,state,tags)==0
    assert torch.isfinite(uncertain_loss(student,ref,fused['unknown'],fused['valid'],torch.zeros_like(state),torch.zeros_like(tags)))
    print('PASS unknown/negative separation, conflict rejection, padding, independent new anchors, teacher stop-gradient and conditional-FG invariance')
    oldviews=torch.full((1,2,15),-1.);oldviews[:,:,0]=1.
    tv=torch.zeros(1,2,16,2,2);rv=torch.zeros(1,2,21,2,2);tv[:,:,0]=1;rv[:,:,1]=3
    cams=torch.zeros(1,20,2,2);cams[:,0]=1
    evidence=calibrate_presence(oldviews,oldviews,tv,rv,cams,torch.zeros(1,5))
    assert evidence['state'][0,0]==1 and (evidence['state'][0,1:]==0).all()
    rv[:,1,2]=4
    assert calibrate_presence(oldviews,oldviews,tv,rv,cams,torch.zeros(1,5))['state'][0,0]==-1
    print('PASS two-view weak presence requires stable dense support')
    base=ConfusionProto(21,16,'cpu');base.ema.uniform_(0,.2);base.mass.fill_(1)
    base.seen[1][2]={'a','b','c'};base.seen[2][3]={'a','b'}
    with tempfile.TemporaryDirectory(dir=ROOT/'.runtime/tmp') as folder:
        path=Path(folder)/'graph.json';path.write_text(json.dumps(base.state()))
        continued=ContinuedConfusionProto(21,16,'cpu','full',path)
        assert all(torch.equal(a,b) for a,b in zip(base.graph(),continued.graph()))
        assert all(not ids for row in continued.seen for ids in row)
        continued.seen[2][3]={'new_a'};assert continued.graph()[0][2,3]==0
        continued.seen[2][3]={'new_a','new_b','new_c'};assert continued.graph()[0][2,3]>0
    print('PASS historical graph preserved without adding overlapping image counts')

if __name__=='__main__':main()
