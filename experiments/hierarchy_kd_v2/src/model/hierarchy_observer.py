"""A separate confusion view aligned to the hierarchy loss evidence, without GT."""
import torch
from model.online_directed_confusion import OnlineDirectedConfusion
from model.hierarchical_kd import hierarchy_evidence


class HierarchyObserver(OnlineDirectedConfusion):
    SCHEMA=3
    def __init__(self,classes,old_classes,**kwargs):
        self.old_classes=int(old_classes)
        if not 1<=self.old_classes<classes-1:raise ValueError('Invalid class boundary')
        super().__init__(classes,**kwargs)
    def get_extra_state(self):
        return {**super().get_extra_state(),'old_classes':self.old_classes,
                'anchor_policy':'old_teacher_PAR_strong_CAM_agreement; teacher_BG_with_PAR_BG_or_absent_new_prediction'}
    @torch.no_grad()
    def update_evidence(self,student,teacher,par_labels,cams,img_box,new_image_tags):
        e=hierarchy_evidence(student,teacher,par_labels,cams,img_box,new_image_tags,high=self.high_threshold)
        anchors=torch.full_like(e['p'],255)
        anchors[e['mass_mask']]=e['p'][e['mass_mask']]
        anchors[e['background']|e['absent_mask']]=0
        h,w=student.shape[-2:]
        # Broad counts here mean ALL of these already evidence-gated anchors.
        # Additional CAM-accepted counts remain a separately labelled reference.
        return super().update(student,anchors,e['cams'],[[0,h,0,w]]*len(student))
    def export(self):
        result=super().export()
        result.update(definition='row=evidence-supported old class or teacher-background proxy; column=student argmax',
            evidence_policy=self.get_extra_state()['anchor_policy'],
            relation_views={'primary':'loss-evidence-aligned unweighted native-grid confusion',
                            'trusted_reference':'additional original CAM confidence filter, not used for hierarchy routing',
                            'usage':'ranking and support only; background proxy on absent-new predictions can include old objects, so loss rejects only absent class'})
        return result
