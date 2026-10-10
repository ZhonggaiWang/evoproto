import datetime
import logging
import os
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from datasets import coco as coco
from datasets import voc as voc
import os.path as osp
import tasks
from model.losses import get_masked_ptc_loss, get_seg_loss, CTCLoss_neg, DenseEnergyLoss, get_energy_loss, \
    rcpl_loss, cpa_loss
from model.model_seg_neg import network
from model.checkpoint import load_incremental_checkpoint
from continual.prototype_supervision import build_incremental_supervision, compute_confusion as compute_paper_confusion
from utils.prototype_confusion import confusion_counterparts, confusion_update_due
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from model.PAR import PAR
from utils import evaluate, imutils, optimizer
from utils.camutils import *
from utils.pyutils import AverageMeter, cal_eta, format_tabs, setup_logger




class Trainer:
    def __init__(self, args):
        self.args = args
        self.step = args.step
        self.task = args.task
        self.model = network(
            backbone=args.backbone,
            num_classes=sum(tasks.get_per_task_classes(args.dataset, args.task, args.step)),
            classes_list=tasks.get_per_task_classes(args.dataset, args.task, args.step),
            pretrained=args.pretrained,
            init_momentum=args.momentum,
            aux_layer=args.aux_layer,
            prototype_temperature=getattr(args, 'proto_temperature', 0.01),
        )
        self.device = torch.device(args.local_rank)
        self.total_classes = sum(tasks.get_per_task_classes(args.dataset, args.task, args.step)) - 1
        self.new_classes = tasks.get_per_task_classes(args.dataset, args.task, args.step)[-1]
        self.old_classes = self.total_classes - self.new_classes
        self.class_weight = torch.ones(self.total_classes + 1).to(self.device)
        self.class_weight[0] = 0.1
        self.confusion_matrix = torch.zeros(self.total_classes + 1, self.total_classes + 1, device=self.device)
        self.confusion_weights = torch.zeros_like(self.confusion_matrix)
        self.confusion_counterparts = None
        self.confusion_counts = None
        if args.step == 0:  # if step 0, we don't need to instance the model_old
            self.model_old = None
        else:  # instance model_old
            self.model_old = network(
                backbone=args.backbone,
                num_classes=sum(tasks.get_per_task_classes(args.dataset, args.task, args.step - 1)),
                classes_list=tasks.get_per_task_classes(args.dataset, args.task, args.step - 1),
                pretrained=args.pretrained,
                init_momentum=args.momentum,
                aux_layer=args.aux_layer,
                prototype_temperature=getattr(args, 'proto_temperature', 0.01),
            )
            # freeze old model and set eval mode
            for par in self.model_old.parameters():
                par.requires_grad = False
            self.model_old.eval()

        param_groups = self.model.get_param_groups()
        self.optimizer = self.get_optimizer(args, param_groups)
        
    def get_optimizer(self, args, param_groups):

        optim = getattr(optimizer, args.optimizer)(
            params=optimizer.prototype_parameter_groups(
                getattr(self, 'model', None), param_groups, args.lr, args.wt_decay,
                getattr(args, 'layer_decay', 0.9) if getattr(self, 'step', 0) > 0 else 1.0,
            ),
            lr=args.lr,
            weight_decay=args.wt_decay,
            betas=args.betas,
            warmup_iter=args.warmup_iters,
            max_iter=args.max_iters,
            warmup_ratio=optimizer.warmup_argument(args.optimizer, args.lr, args.warmup_lr),
            power=args.power)

        return optim
    
    def load_step_ckpt(self, path):
        if osp.exists(path):
            load_incremental_checkpoint(self.model, self.model_old, path, self.step)
            logging.info(f"[!] Previous model loaded from {path}")
        else:
            raise FileNotFoundError(f"Previous-step checkpoint is required: {path}")

    def validate(self, model=None, data_loader=None, args=None):
        preds, gts, cams, cams_aux = [], [], [], []
        model.eval()
        avg_meter = AverageMeter()
        with torch.no_grad():
            for _, data in tqdm(enumerate(data_loader), total=len(data_loader), ncols=100, ascii=" >="):
                name, inputs, labels, cls_label = data
                inputs = inputs.cuda()
                labels = labels.cuda()
                cls_label = cls_label.cuda()
                cls_label = cls_label[:, :self.total_classes]

                inputs = F.interpolate(inputs, size=[args.crop_size, args.crop_size], mode='bilinear',
                                       align_corners=False)

                cls, segs, _, _ = model(inputs, )

                if self.step > 0:
                    segs = multi_scale_seg(model, inputs, args.cam_scales)

                cls_pred = (cls > 0).type(torch.int16)
                _f1 = evaluate.multilabel_score(cls_label.cpu().numpy()[0], cls_pred.cpu().numpy()[0])
                avg_meter.add({"cls_score": _f1})

                _cams, _cams_aux = multi_scale_cam2(model, inputs, args.cam_scales)
                resized_cam = F.interpolate(_cams, size=labels.shape[1:], mode='bilinear', align_corners=False)
                cam_label = cam_to_label(resized_cam, cls_label, bkg_thre=args.bkg_thre, high_thre=args.high_thre,
                                         low_thre=args.low_thre, ignore_index=args.ignore_index)

                resized_cam_aux = F.interpolate(_cams_aux, size=labels.shape[1:], mode='bilinear', align_corners=False)
                cam_label_aux = cam_to_label(resized_cam_aux, cls_label, bkg_thre=args.bkg_thre,
                                             high_thre=args.high_thre, low_thre=args.low_thre,
                                             ignore_index=args.ignore_index)

                cls_pred = (cls > 0).type(torch.int16)
                _f1 = evaluate.multilabel_score(cls_label.cpu().numpy()[0], cls_pred.cpu().numpy()[0])
                avg_meter.add({"cls_score": _f1})

                resized_segs = F.interpolate(segs, size=labels.shape[1:], mode='bilinear', align_corners=False)
                preds += list(torch.argmax(resized_segs, dim=1).cpu().numpy().astype(np.int16))
                cams += list(cam_label.cpu().numpy().astype(np.int16))
                gts += list(labels.cpu().numpy().astype(np.int16))
                cams_aux += list(cam_label_aux.cpu().numpy().astype(np.int16))

                # valid_label = torch.nonzero(cls_label[0])[:, 0]
                # out_cam = torch.squeeze(resized_cam)[valid_label]
                # np.save(os.path.join(cfg.work_dir.pred_dir, name[0]+'.npy'), {"keys":valid_label.cpu().numpy(), "cam":out_cam.cpu().numpy()})

        cls_score = avg_meter.pop('cls_score')
        seg_score = evaluate.scores(gts, preds, self.total_classes + 1)
        cam_score = evaluate.scores(gts, cams, self.total_classes + 1)
        cam_aux_score = evaluate.scores(gts, cams_aux, self.total_classes + 1)
        model.train()

        tab_results = format_tabs([cam_score, cam_aux_score, seg_score], name_list=["CAM", "aux_CAM", "Seg_Pred"],
                                  cat_list=coco.class_list)

        return cls_score, tab_results


    def cal_sim(self, model=None, data_loader=None, args=None):
        return compute_paper_confusion(self, model, data_loader, args)

    def get_weight(self, con_matrix, total_classes, new_classes):
        return confusion_counterparts(
            con_matrix, old_count=total_classes - new_classes + 1,
            temperature=self.args.confusion_temperature,
        )
    
    def train(self ,args=None):
        # dist.init_process_group(backend=args.backend, )
        torch.cuda.set_device(args.local_rank)
        logging.info("Total gpus: %d, samples per gpu: %d..."%(dist.get_world_size(), args.spg))

        time0 = datetime.datetime.now()
        time0 = time0.replace(microsecond=0)

        train_step0_dataset = coco.CocoStep0Dataset(   #only 60 classes  x_filter.txt
            img_dir=args.coco_img_folder,
            label_dir=args.coco_label_folder,
            name_list_dir=args.coco_list_folder,
            split=args.train_set,
            stage='train',
            aug=True,
            # resize_range=cfg.dataset.resize_range,
            rescale_range=args.scales,
            crop_size=args.crop_size,
            img_fliplr=True,
            ignore_index=args.ignore_index,
            num_classes=args.num_classes,
        )
        
        train_step1_dataset = voc.Coco2VocClsDataset(
            root_dir=args.voc_data_folder,
            name_list_dir=args.voc_list_folder,
            split=args.train_set,
            stage='train',
            aug=True,
            # resize_range=cfg.dataset.resize_range,
            rescale_range=args.scales,
            crop_size=args.crop_size,
            img_fliplr=True,
            ignore_index=args.ignore_index,
            num_classes=args.num_classes,
            tasks=args.task,
            step=1,
        )

        val_step0_dataset = coco.CocoSegDataset(
            img_dir=args.coco_img_folder,
            label_dir=args.coco_label_folder,
            name_list_dir=args.coco_list_folder,
            split=args.coco_filter_val_set,
            stage='val',
            aug=False,
            ignore_index=args.ignore_index,
            num_classes=args.num_classes,
        )
        
        val_step1_coco_dataset = coco.CocoSegDataset(
            img_dir=args.coco_img_folder,
            label_dir=args.coco_label_folder,
            name_list_dir=args.coco_list_folder,
            split=args.coco_filter_val_set_step1,
            stage='val',
            aug=False,
            ignore_index=args.ignore_index,
            num_classes=args.num_classes,
        )

        cal_sim_loader = DataLoader(
            train_step1_dataset,
            batch_size=args.spg,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=False,
            drop_last=False)

        train_sampler_step0 = DistributedSampler(train_step0_dataset, shuffle=True)
        train_sampler_step1 = DistributedSampler(train_step1_dataset, shuffle=True)
        train_step0_loader = DataLoader(
            train_step0_dataset,
            batch_size=args.spg,
            #shuffle=True,
            num_workers=args.num_workers,
            pin_memory=False,
            drop_last=True,
            sampler=train_sampler_step0,
            prefetch_factor=4)
        
        train_step1_loader = DataLoader(
            train_step1_dataset,
            batch_size=args.spg,
            #shuffle=True,
            num_workers=args.num_workers,
            pin_memory=False,
            drop_last=True,
            sampler=train_sampler_step1,
            prefetch_factor=4)

        val_step0_loader = DataLoader(val_step0_dataset,
                                batch_size=1,
                                shuffle=False,
                                num_workers=args.num_workers,
                                pin_memory=False,
                                drop_last=False)
        
        val_step1_coco_loader = DataLoader(val_step1_coco_dataset,
                                batch_size=1,
                                shuffle=False,
                                num_workers=args.num_workers,
                                pin_memory=False,
                                drop_last=False)
        
        device = self.device
        model = self.model.to(device)
        # cfg.optimizer.learning_rate *= 2
        optim = self.optimizer
        logging.info('\nOptimizer: \n%s' % optim)
        model = DistributedDataParallel(model, device_ids=[args.local_rank], find_unused_parameters=True)
        
        train_sampler_step0.set_epoch(np.random.randint(args.max_iters))
        train_sampler_step1.set_epoch(np.random.randint(args.max_iters))
        
        
        train_loader_iter = iter(train_step1_loader)
        train_step0_loader_iter = iter(train_step0_loader)
        avg_meter = AverageMeter()
        
        if self.step == 0:
            for n_iter in range(args.max_iters):

                try:
                    name, inputs, labels, cls_label = next(train_step0_loader_iter)
                except:
                    train_sampler_step0.set_epoch(np.random.randint(args.max_iters))
                    train_step0_loader_iter = iter(train_step0_loader)
                    name, inputs, labels, cls_label = next(train_step0_loader_iter)

                inputs = inputs.to(device, non_blocking=True)
                inputs_denorm = imutils.denormalize_img2(inputs.clone())

                inputs = inputs.cuda()
                labels = labels.cuda()

                cls_label = cls_label.cuda()
                cls_label = cls_label[:, :self.total_classes]

                filter_idx = labels >= self.total_classes + 1
                labels[filter_idx] = 255

                _, cls, segs, fmap, cls_aux, _, new_prototypes = model(inputs, crops=None, cam_grad=True,
                                                                              n_iter=n_iter)

                contrastive_loss = rcpl_loss(
                    new_prototypes, self.confusion_weights,
                    mode=getattr(args, 'proto_sep_mode', 'paper'), margin=getattr(args, 'proto_margin', 0.0),
                )

                segs = F.interpolate(segs, size=labels.shape[-2:], mode='bilinear', align_corners=False)
                cls_loss = F.multilabel_soft_margin_loss(cls, cls_label)
                cls_loss_aux = F.multilabel_soft_margin_loss(cls_aux, cls_label)

                seg_loss = get_seg_loss(segs, labels.type(torch.long), class_weight=self.class_weight,
                                        ignore_index=args.ignore_index)
                loss = cls_loss + cls_loss_aux + args.w_seg * seg_loss + 0.1 * contrastive_loss

                optim.zero_grad()
                loss.backward()
                # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optim.step()
                avg_meter.add({
                    'cls_loss': cls_loss,
                    'seg_loss': seg_loss,
                    'contrastive_loss': contrastive_loss,
                })
                if (n_iter + 1) % args.log_iters == 0:

                    delta, eta = cal_eta(time0, n_iter + 1, args.max_iters)
                    cur_lr = optim.param_groups[0]['lr']

                    if args.local_rank == 0:
                        logging.info(
                            "Iter: %d; Elasped: %s; ETA: %s; LR: %.3e; cls_loss: %.4f, seg_loss: %.4f, contrastive_loss: %.4f" % (
                                n_iter + 1, delta, eta, cur_lr, avg_meter.pop('cls_loss'), avg_meter.pop('seg_loss'),
                                avg_meter.pop('contrastive_loss'),
                            ))

                if (n_iter + 1) % args.eval_iters == 0:
                    ckpt_name = os.path.join(args.ckpt_dir, "model_iter_%d.pth" % (n_iter + 1))
                    if args.local_rank == 0:
                        logging.info('Validating...')
                        if args.save_ckpt:
                            torch.save(model.state_dict(), ckpt_name)
                    val_cls_score, tab_results = self.validate(model=model, data_loader=val_step0_loader, args=args)
                    if args.local_rank == 0:
                        logging.info("val cls score: %.6f" % (val_cls_score))
                        logging.info("\n" + tab_results)

        else:
            model_old = self.model_old.to(device)
        # loss_layer = DenseEnergyLoss(weight=1e-7, sigma_rgb=15, sigma_xy=100, scale_factor=0.5)
            ncrops = 10

            par = PAR(num_iter=10, dilations=[1,2,4,8,12,24]).cuda(device)

            for n_iter in range(args.max_iters):
                if args.confusion_reweight and confusion_update_due(
                    n_iter, getattr(args, 'confusion_start_iter', 4000), args.confusion_interval,
                ):
                    self.confusion_matrix = self.cal_sim(model=model, data_loader=cal_sim_loader, args=args)
                    self.confusion_counterparts, self.confusion_weights = self.get_weight(
                        self.confusion_matrix, self.total_classes, self.new_classes,
                    )
                    logging.info('Prototype confusion refreshed after %d updates; counterparts=%s',
                                 n_iter, self.confusion_counterparts.tolist())

                try:
                    img_name, inputs, cls_label, img_box, crops = next(train_loader_iter)
                except:
                    train_sampler_step1.set_epoch(np.random.randint(args.max_iters))
                    train_loader_iter = iter(train_step1_loader)
                    img_name, inputs, cls_label, img_box, crops = next(train_loader_iter)

                inputs = inputs.to(device, non_blocking=True)
                inputs_denorm = imutils.denormalize_img2(inputs.clone())

                cls_label = cls_label.to(device, non_blocking=True)
                cls_label = cls_label[:, :self.total_classes]

                # cls_gt_label = torch.clone(cls_label)

                supervision = build_incremental_supervision(
                    model, model_old, inputs, cls_label, img_box, par,
                    args, self.total_classes, self.new_classes,
                )
                cls_label = supervision.cls_labels
                cams, cams_aux = supervision.cams, supervision.cams_aux
                old_prototypes = supervision.old_prototypes

                roi_mask = cam_to_roi_mask2(cams_aux.detach(), cls_label=cls_label, low_thre=args.low_thre,
                                            hig_thre=args.high_thre)

                local_crops, flags = crop_from_roi_neg(images=crops[2], roi_mask=roi_mask, crop_num=ncrops - 2,
                                                       crop_size=args.local_crop_size)
                roi_crops = crops[:2] + local_crops

                cls, segs, fmap, cls_aux, _, new_prototypes = model(inputs, crops=roi_crops, n_iter=n_iter)
                prototype_sep = rcpl_loss(
                    new_prototypes, self.confusion_weights,
                    mode=getattr(args, 'proto_sep_mode', 'paper'), margin=getattr(args, 'proto_margin', 0.0),
                )
                prototype_kd = cpa_loss(
                    new_prototypes, old_prototypes, self.confusion_weights,
                    mode=getattr(args, 'proto_kd_mode', 'direction'),
                    skip_background=getattr(args, 'cpa_skip_background', True),
                )
                contrastive_loss = prototype_sep + prototype_kd

                cls_loss = F.multilabel_soft_margin_loss(cls, cls_label)
                cls_loss_aux = F.multilabel_soft_margin_loss(cls_aux, cls_label)

                # # ctc_loss
                # ctc_loss = CTC_loss(out_s, out_t, flags)

                # seg_loss & reg_loss
                refined_pseudo_label = supervision.refined_labels
                mixed_pseudo_label = supervision.merged_labels

                segs = F.interpolate(segs, size=refined_pseudo_label.shape[1:], mode='bilinear', align_corners=False)
                seg_loss = get_seg_loss(segs, mixed_pseudo_label.type(torch.long), class_weight=self.class_weight,
                                        ignore_index=args.ignore_index)

                resized_cams_aux = F.interpolate(cams_aux, size=fmap.shape[2:], mode="bilinear", align_corners=False)
                _, pseudo_label_aux = cam_to_label(resized_cams_aux.detach(), cls_label=cls_label, img_box=img_box,
                                                   ignore_mid=True, bkg_thre=args.bkg_thre, high_thre=args.high_thre,
                                                   low_thre=args.low_thre, ignore_index=args.ignore_index)
                aff_mask = label_to_aff_mask(pseudo_label_aux)
                ptc_loss = get_masked_ptc_loss(fmap, aff_mask)

                # warmup
                if n_iter < args.loss_warmup_iters:
                    # Keep prototypes in the DDP graph during loss warmup.
                    loss = cls_loss + cls_loss_aux + args.w_ptc * ptc_loss + 0.0 * (seg_loss + prototype_kd + prototype_sep)
                else:
                    loss = cls_loss + cls_loss_aux + args.w_ptc * ptc_loss + args.w_seg * seg_loss + 0.1 * contrastive_loss

                if n_iter % 2000 == 0 and n_iter != 0:
                    if args.local_rank == 0:  # save model at the eval iteration
                        state = {
                            "model_state": self.model.state_dict(),
                        }
                        path = os.path.join(args.ckpt_dir, f"model_{n_iter}.pth")
                        torch.save(state, path)
                        logging.info("[!] Checkpoint saved.")

                cls_pred = (cls > 0).type(torch.int16)
                cls_score = evaluate.multilabel_score(cls_label.cpu().numpy()[0], cls_pred.cpu().numpy()[0])
                avg_meter.add({
                    'cls_loss': cls_loss,
                    'ptc_loss': ptc_loss,
                    'cls_loss_aux': cls_loss_aux,
                    'seg_loss': seg_loss,
                    'contrastive_loss': contrastive_loss,
                    'cls_score': cls_score,
                })
                optim.zero_grad()
                loss.backward()
                # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optim.step()
                if (n_iter + 1) % args.log_iters == 0:

                    delta, eta = cal_eta(time0, n_iter + 1, args.max_iters)
                    cur_lr = optim.param_groups[0]['lr']

                    if args.local_rank == 0:
                        logging.info(
                            "Iter: %d; Elasped: %s; ETA: %s; LR: %.3e; cls_loss: %.4f, cls_loss_aux: %.4f, ptc_loss: %.4f, seg_loss: %.4f, contrastive_loss: %.4f" % (
                                n_iter + 1, delta, eta, cur_lr, avg_meter.pop('cls_loss'),
                                avg_meter.pop('cls_loss_aux'),
                                avg_meter.pop('ptc_loss'), avg_meter.pop('seg_loss'),
                                avg_meter.pop('contrastive_loss'),))

                if (n_iter + 1) % args.eval_iters == 0:
                    ckpt_name = os.path.join(args.ckpt_dir, "model_iter_%d.pth" % (n_iter + 1))
                    if args.local_rank == 0:
                        logging.info('Validating...')
                        if args.save_ckpt:
                            torch.save(model.state_dict(), ckpt_name)
                    val_cls_score, tab_results = self.validate(model=model, data_loader=val_step1_coco_loader, args=args)
                    if args.local_rank == 0:
                        logging.info("val cls score: %.6f" % (val_cls_score))
                        logging.info("\n" + tab_results)
    
        return True
