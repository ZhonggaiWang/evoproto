import datetime
import json
import logging
import os
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
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
from utils.ald_stats import COUNT_KEYS, supervision_counts, counts_record
from utils.pyutils import AverageMeter, cal_eta, format_tabs, setup_logger
import os
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt
import torchvision.utils as vutils

import numpy as np

VOC_COLORMAP = np.array([
    [0, 0, 0],        # 0=background
    [128, 0, 0],      # 1=aeroplane
    [0, 128, 0],      # 2=bicycle
    [128, 128, 0],    # 3=bird
    [0, 0, 128],      # 4=boat
    [128, 0, 128],    # 5=bottle
    [0, 128, 128],    # 6=bus
    [128, 128, 128],  # 7=car
    [64, 0, 0],       # 8=cat
    [192, 0, 0],      # 9=chair
    [64, 128, 0],     # 10=cow
    [192, 128, 0],    # 11=diningtable
    [64, 0, 128],     # 12=dog
    [192, 0, 128],    # 13=horse
    [64, 128, 128],   # 14=motorbike
    [192, 128, 128],  # 15=person
    [0, 64, 0],       # 16=potted plant
    [128, 64, 0],     # 17=sheep
    [0, 192, 0],      # 18=sofa
    [128, 192, 0],    # 19=train
    [0, 64, 128]      # 20=tv/monitor
], dtype=np.uint8)

import os
import torchvision.transforms.functional as TF
from PIL import Image

def decode_segmap(label_mask, num_classes=21):
    VOC_COLORMAP = np.array([
        [0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0],
        [0, 0, 128], [128, 0, 128], [0, 128, 128], [128, 128, 128],
        [64, 0, 0], [192, 0, 0], [64, 128, 0], [192, 128, 0],
        [64, 0, 128], [192, 0, 128], [64, 128, 128], [192, 128, 128],
        [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0],
        [0, 64, 128]
    ], dtype=np.uint8)

    r = np.zeros_like(label_mask).astype(np.uint8)
    g = np.zeros_like(label_mask).astype(np.uint8)
    b = np.zeros_like(label_mask).astype(np.uint8)
    for label in range(0, num_classes):
        mask = label_mask == label
        r[mask] = VOC_COLORMAP[label, 0]
        g[mask] = VOC_COLORMAP[label, 1]
        b[mask] = VOC_COLORMAP[label, 2]
    return np.stack([r, g, b], axis=2)

def save_vis(image, gt, pred, cam, cam_aux, name, save_root):
    os.makedirs(os.path.join(save_root, "image"), exist_ok=True)
    os.makedirs(os.path.join(save_root, "gt"), exist_ok=True)
    os.makedirs(os.path.join(save_root, "pred"), exist_ok=True)
    os.makedirs(os.path.join(save_root, "cam"), exist_ok=True)
    os.makedirs(os.path.join(save_root, "cam_aux"), exist_ok=True)

    # 原图 image 是 Tensor
    img_pil = TF.to_pil_image(image.cpu())
    img_pil.save(os.path.join(save_root, "image", f"{name}.png"))

    # 如果是 Tensor，就先 .cpu().numpy()
    if hasattr(gt, 'cpu'):
        gt = gt.cpu().numpy()
    if hasattr(pred, 'cpu'):
        pred = pred.cpu().numpy()
    if hasattr(cam, 'cpu'):
        cam = cam.cpu().numpy()
    if hasattr(cam_aux, 'cpu'):
        cam_aux = cam_aux.cpu().numpy()

    gt_rgb = Image.fromarray(decode_segmap(gt))
    pred_rgb = Image.fromarray(decode_segmap(pred))
    cam_rgb = Image.fromarray(decode_segmap(cam))
    cam_aux_rgb = Image.fromarray(decode_segmap(cam_aux))

    gt_rgb.save(os.path.join(save_root, "gt", f"{name}.png"))
    pred_rgb.save(os.path.join(save_root, "pred", f"{name}.png"))
    cam_rgb.save(os.path.join(save_root, "cam", f"{name}.png"))
    cam_aux_rgb.save(os.path.join(save_root, "cam_aux", f"{name}.png"))

class Trainer:
    def __init__(self, args):
        self.args = args
        self.current_iteration = 0
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
        self.class_weight[0] = 1.0
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

                # save_vis(inputs[0], labels[0], preds[-1], cams[-1], cams_aux[-1], name[0],
                #          save_root=save_dir)

                # valid_label = torch.nonzero(cls_label[0])[:, 0]
                # out_cam = torch.squeeze(resized_cam)[valid_label]
                # np.save(os.path.join(cfg.work_dir.pred_dir, name[0]+'.npy'), {"keys":valid_label.cpu().numpy(), "cam":out_cam.cpu().numpy()})

        cls_score = avg_meter.pop('cls_score')
        seg_score = evaluate.scores(gts, preds, self.total_classes + 1)
        cam_score = evaluate.scores(gts, cams, self.total_classes + 1)
        cam_aux_score = evaluate.scores(gts, cams_aux, self.total_classes + 1)
        initial_foreground = tasks.get_per_task_classes(args.dataset, args.task, 0)[0] - 1
        ious = np.array([seg_score['iou'][index] for index in range(self.total_classes + 1)]) * 100
        def mean_iou(values):
            finite = values[np.isfinite(values)]
            return float(finite.mean()) if finite.size else None
        metrics = {
            'step': self.step, 'iteration': self.current_iteration,
            'old_miou': mean_iou(ious[1:initial_foreground + 1]),
            'new_miou': mean_iou(ious[initial_foreground + 1:]),
            'all_miou': mean_iou(ious), 'foreground_miou': mean_iou(ious[1:]),
            'class_iou': {voc.class_list[index]: float(value) if np.isfinite(value) else None
                          for index, value in enumerate(ious)},
            'classification_f1': float(cls_score),
            'cam_miou': float(cam_score['miou']) * 100,
            'w_proto_kd': args.w_proto_kd, 'w_proto_sep': args.w_proto_sep,
            'proto_margin': args.proto_margin, 'ald': args.ald,
            'proto_sep_mode': getattr(args, 'proto_sep_mode', 'paper'),
            'ald_mode': args.ald_mode, 'seed': args.seed,
            'ald_threshold': getattr(args, 'ald_threshold', 0.0) if args.ald_mode in ('fixed', 'legacy') else None,
            'future_class_label': getattr(args, 'future_class_label', 255),
            'confusion_reweight': args.confusion_reweight,
            'w_proto_seg': args.w_seg, 'w_seg': args.w_seg,
            'segmentation_head': 'patch_prototype',
            'seg_loss_type': 'cross_entropy',
            'prototype_dim': self.model.decoder.embed_dim,
            'proto_temperature': self.model.decoder.temperature,
            'proto_logit_scale': 1.0 / self.model.decoder.temperature,
            'prototype_temperature_trainable': True, 'prototype_init': 'unit_sphere',
            'proto_kd_mode': getattr(args, 'proto_kd_mode', 'direction'),
            'cpa_skip_background': getattr(args, 'cpa_skip_background', True),
            'layer_decay': getattr(args, 'layer_decay', 0.9) if self.step > 0 else 1.0,
            'confusion_temperature': args.confusion_temperature,
            'confusion_interval': args.confusion_interval,
            'confusion_start_iter': getattr(args, 'confusion_start_iter', 4000),
            'old_cls_threshold': getattr(args, 'old_cls_threshold', 2.0),
            'pixel_padding_ignored': True,
            'seg_tta': self.step > 0,
            'seg_tta_scales': [1.0] + [s for s in args.cam_scales if s != 1.0] if self.step > 0 else [1.0],
            'seg_tta_flip': self.step > 0,
            'seg_tta_fusion': 'flip_mean_scale_mean_logits' if self.step > 0 else 'single_view',
        }
        if args.local_rank == 0:
            with open(osp.join(args.work_dir, 'metrics.jsonl'), 'a', encoding='utf-8') as handle:
                handle.write(json.dumps(metrics, allow_nan=False) + '\n')
            logging.info('Baseline metrics: %s', json.dumps(metrics, allow_nan=False))
        model.train()

        tab_results = format_tabs([cam_score, cam_aux_score, seg_score], name_list=["CAM", "aux_CAM", "Seg_Pred"],
                                  cat_list=voc.class_list)

        return cls_score, tab_results

    def cal_sim(self, model=None, data_loader=None, args=None):
        return compute_paper_confusion(self, model, data_loader, args)

    def get_weight(self, con_matrix, total_classes, new_classes):
        return confusion_counterparts(
            con_matrix, old_count=total_classes - new_classes + 1,
            temperature=self.args.confusion_temperature,
        )

    def train(self, args):
        torch.cuda.set_device(args.local_rank)
        logging.info("Total gpus: %d, samples per gpu: %d..." % (dist.get_world_size(), args.spg))

        time0 = datetime.datetime.now()
        time0 = time0.replace(microsecond=0)
        train_dataset = voc.VOC12ClsDataset(
            root_dir=args.data_folder,
            name_list_dir=args.list_folder,
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
            step=args.step,
        )

        train_step0_dataset = voc.VOC12Step0Dataset(
            root_dir=args.data_folder,
            name_list_dir=args.list_folder,
            split=args.train_set,
            stage='train',
            aug=True,
            crop_size=args.crop_size,
            ignore_index=args.ignore_index,
            num_classes=args.num_classes,
            tasks=args.task,
            step=args.step,
        )

        val_dataset = voc.VOC12SegDataset(
            root_dir=args.data_folder,
            name_list_dir=args.list_folder,
            split=args.val_set,
            stage='val',
            aug=False,
            ignore_index=args.ignore_index,
            num_classes=args.num_classes,
            tasks=args.task,
            step=args.step,
        )

        train_dataset.label_dir = args.seg_label_dir
        train_step0_dataset.label_dir = args.seg_label_dir
        val_dataset.label_dir = args.val_label_dir
        if args.train_limit:
            train_dataset.name_list = train_dataset.name_list[:args.train_limit]
            train_step0_dataset.name_list = train_step0_dataset.name_list[:args.train_limit]
        if args.val_limit:
            val_dataset.name_list = val_dataset.name_list[:args.val_limit]

        train_sampler = DistributedSampler(train_dataset, shuffle=True)
        step0_sampler = DistributedSampler(train_step0_dataset, shuffle=True)
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.spg,
            # shuffle=True,
            num_workers=args.num_workers,
            pin_memory=False,
            drop_last=True,
            sampler=train_sampler,
            prefetch_factor=4)

        train_step0_loader = DataLoader(
            train_step0_dataset,
            batch_size=args.spg,
            # shuffle=True,
            num_workers=args.num_workers,
            pin_memory=False,
            drop_last=True,
            sampler=step0_sampler,
            prefetch_factor=4)

        val_loader = DataLoader(val_dataset,
                                batch_size=1,
                                shuffle=False,
                                num_workers=args.num_workers,
                                pin_memory=False,
                                drop_last=False)

        cal_sim_loader = DataLoader(
            train_dataset,
            batch_size=args.spg,
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
        active_sampler = step0_sampler if self.step == 0 else train_sampler
        active_sampler.set_epoch(np.random.randint(args.max_iters))
        train_loader_iter = iter(train_loader) if self.step > 0 else None
        avg_meter = AverageMeter()

        train_step0_loader_iter = iter(train_step0_loader) if self.step == 0 else None


        if self.step == 0:
            for n_iter in range(args.max_iters):
                self.current_iteration = n_iter + 1

                try:
                    name, inputs, labels, cls_label = next(train_step0_loader_iter)
                except StopIteration:
                    step0_sampler.set_epoch(np.random.randint(args.max_iters))
                    train_step0_loader_iter = iter(train_step0_loader)
                    name, inputs, labels, cls_label = next(train_step0_loader_iter)

                inputs = inputs.to(device, non_blocking=True)

                inputs = inputs.cuda()
                labels = labels.cuda()
                cls_label = cls_label.cuda()
                cls_label = cls_label[:, :self.total_classes]

                future_mask = (labels > self.total_classes) & (labels != args.ignore_index)
                labels[future_mask] = getattr(args, 'future_class_label', 255)
                if n_iter == 0 and args.local_rank == 0:
                    logging.info('Step0 future pixels: %d; target label: %d; ignore pixels after mapping: %d',
                                 future_mask.sum().item(), getattr(args, 'future_class_label', 255),
                                 (labels == args.ignore_index).sum().item())

                _, cls, segs, fmap, cls_aux, _, new_prototypes = model(inputs, crops=None, cam_grad=True, n_iter=n_iter)

                prototype_sep = rcpl_loss(
                    new_prototypes, self.confusion_weights,
                    mode=getattr(args, 'proto_sep_mode', 'paper'), margin=args.proto_margin,
                )

                segs = F.interpolate(segs, size=labels.shape[-2:], mode='bilinear', align_corners=False)
                cls_loss = F.multilabel_soft_margin_loss(cls, cls_label)
                cls_loss_aux = F.multilabel_soft_margin_loss(cls_aux, cls_label)

                seg_loss = get_seg_loss(segs, labels.type(torch.long), class_weight=self.class_weight, ignore_index=args.ignore_index)
                loss = cls_loss + cls_loss_aux + args.w_seg * seg_loss + args.w_proto_sep * prototype_sep
                if not torch.isfinite(loss):
                    raise FloatingPointError(f'Nonfinite loss at step 0 iteration {n_iter + 1}')

                optim.zero_grad()
                loss.backward()
                # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optim.step()
                avg_meter.add({
                    'cls_loss': cls_loss,
                    'seg_loss': seg_loss,
                    'proto_sep': prototype_sep,
                })
                if (n_iter + 1) % args.log_iters == 0:

                    delta, eta = cal_eta(time0, n_iter + 1, args.max_iters)
                    cur_lr = optim.param_groups[0]['lr']

                    if args.local_rank == 0:
                        logging.info(
                            "Iter: %d; Elasped: %s; ETA: %s; LR: %.3e; cls_loss: %.4f, seg_loss: %.4f, proto_sep: %.6f " % (
                            n_iter + 1, delta, eta, cur_lr, avg_meter.pop('cls_loss'), avg_meter.pop('seg_loss'),
                            avg_meter.pop('proto_sep')
                            ))

                if (n_iter + 1) % args.eval_iters == 0:
                    ckpt_name = os.path.join(args.ckpt_dir, "model_iter_%d.pth" % (n_iter + 1))
                    if args.local_rank == 0:
                        logging.info('Validating...')
                        if args.save_ckpt:
                            torch.save({'model_state': self.model.state_dict(), 'iteration': n_iter + 1}, ckpt_name)
                    val_cls_score, tab_results = self.validate(model=model, data_loader=val_loader, args=args)
                    if args.local_rank == 0:
                        logging.info("val cls score: %.6f" % (val_cls_score))
                        logging.info("\n" + tab_results)

        else:
            model_old = self.model_old.to(device)
            # loss_layer = DenseEnergyLoss(weight=1e-7, sigma_rgb=15, sigma_xy=100, scale_factor=0.5)
            ncrops = 10

            par = PAR(num_iter=10, dilations=[1, 2, 4, 8, 12, 24]).cuda()
            ald_counts = torch.zeros(len(COUNT_KEYS), dtype=torch.int64, device=device)
            ald_interval_batches = 0

            for n_iter in range(args.max_iters):
                self.current_iteration = n_iter + 1
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
                except StopIteration:
                    train_sampler.set_epoch(np.random.randint(args.max_iters))
                    train_loader_iter = iter(train_loader)
                    img_name, inputs, cls_label, img_box, crops = next(train_loader_iter)


                inputs = inputs.to(device, non_blocking=True)
                # import torchvision.transforms.functional as TF

                # input_image = TF.to_pil_image(inputs_denorm[0])
                # input_image.save('input_image denorm.png')

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

                prototype_kd = cpa_loss(
                    new_prototypes, old_prototypes, self.confusion_weights,
                    mode=getattr(args, 'proto_kd_mode', 'direction'),
                    skip_background=getattr(args, 'cpa_skip_background', True),
                )
                prototype_sep = rcpl_loss(
                    new_prototypes, self.confusion_weights,
                    mode=getattr(args, 'proto_sep_mode', 'paper'), margin=args.proto_margin,
                )

                old_pixel_label = supervision.old_labels

                cls_loss = F.multilabel_soft_margin_loss(cls, cls_label)
                cls_loss_aux = F.multilabel_soft_margin_loss(cls_aux, cls_label)

                # # ctc_loss
                # ctc_loss = CTC_loss(out_s, out_t, flags)

                # seg_loss & reg_loss
                refined_pseudo_label = supervision.refined_labels
                mixed_pseudo_label = supervision.merged_labels
                valid_pixels = supervision.valid_mask
                ald_counts += supervision_counts(
                    supervision.refined_labels_before_ald, refined_pseudo_label, old_pixel_label, mixed_pseudo_label,
                    supervision.rejected_mask, supervision.retained_ignore_mask, valid_pixels,
                    old_classes=self.old_classes, total_classes=self.total_classes,
                    cls_labels=supervision.cls_labels_before_ald,
                    gate_labels=supervision.gate_labels, cam_peak=supervision.cam_peaks,
                    threshold=supervision.ald_thresholds, ignore_index=args.ignore_index,
                    image_adaptive=supervision.ald_mode == 'paper',
                )
                ald_interval_batches += 1

                segs = F.interpolate(segs, size=refined_pseudo_label.shape[1:], mode='bilinear', align_corners=False)
                seg_loss = get_seg_loss(segs, mixed_pseudo_label.type(torch.long), class_weight=self.class_weight, ignore_index=args.ignore_index)

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
                    loss = cls_loss + cls_loss_aux + args.w_ptc * ptc_loss + args.w_seg * seg_loss + args.w_proto_kd * prototype_kd + args.w_proto_sep * prototype_sep
                if not torch.isfinite(loss):
                    raise FloatingPointError(f'Nonfinite loss at step {self.step} iteration {n_iter + 1}')

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
                    'cls_score': cls_score,
                    'proto_kd': prototype_kd,
                    'proto_sep': prototype_sep,
                })
                optim.zero_grad()
                loss.backward()
                # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optim.step()
                if (n_iter + 1) % args.log_iters == 0 or n_iter + 1 == args.max_iters:
                    # New study arms use one GPU. Reduce counts if this entry is used with DDP.
                    if dist.get_world_size() > 1:
                        dist.all_reduce(ald_counts, op=dist.ReduceOp.SUM)
                    if args.local_rank == 0:
                        diagnostic = counts_record(ald_counts)
                        diagnostic.update(step=self.step, iteration=n_iter + 1,
                                          interval_batches=ald_interval_batches,
                                          ald_mode=supervision.ald_mode,
                                          ald_scope=('image_labels' if supervision.ald_mode == 'paper' else
                                                     'pixel_labels' if supervision.ald_mode == 'fixed' else 'off'),
                                          segmentation_loss_active=n_iter >= args.loss_warmup_iters)
                        with open(osp.join(args.work_dir, 'ald_metrics.jsonl'), 'a', encoding='utf-8') as handle:
                            handle.write(json.dumps(diagnostic, allow_nan=False) + '\n')
                    ald_counts.zero_()
                    ald_interval_batches = 0
                if (n_iter + 1) % args.log_iters == 0:

                    delta, eta = cal_eta(time0, n_iter + 1, args.max_iters)
                    cur_lr = optim.param_groups[0]['lr']

                    if args.local_rank == 0:
                        logging.info(
                            "Iter: %d; Elasped: %s; ETA: %s; LR: %.3e; cls_loss: %.4f, cls_loss_aux: %.4f, ptc_loss: %.4f, seg_loss: %.4f, proto_kd: %.6f, proto_sep: %.6f" % (
                            n_iter + 1, delta, eta, cur_lr, avg_meter.pop('cls_loss'), avg_meter.pop('cls_loss_aux'),
                            avg_meter.pop('ptc_loss'), avg_meter.pop('seg_loss'),
                            avg_meter.pop('proto_kd'), avg_meter.pop('proto_sep')))

                if (n_iter + 1) % args.eval_iters == 0:
                    ckpt_name = os.path.join(args.ckpt_dir, "model_iter_%d.pth" % (n_iter + 1))
                    if args.local_rank == 0:
                        logging.info('Validating...')
                        if args.save_ckpt:
                            torch.save({'model_state': self.model.state_dict(), 'iteration': n_iter + 1}, ckpt_name)
                    val_cls_score, tab_results = self.validate(model=model, data_loader=val_loader, args=args)
                    if args.local_rank == 0:
                        logging.info("val cls score: %.6f" % (val_cls_score))
                        logging.info("\n" + tab_results)

        if args.max_iters % args.eval_iters:
            val_cls_score, tab_results = self.validate(model=model, data_loader=val_loader, args=args)
            if args.local_rank == 0:
                logging.info('Final validation classification F1: %.6f\n%s', val_cls_score, tab_results)
        return True
