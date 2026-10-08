from pathlib import Path
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
    get_type_seg_loss, prototype_distillation_loss, prototype_separation_loss
from model.model_seg_neg import network
from model.pixel_kd import pixel_kd_loss
from model.ald import image_targets, fuse, gate_auxiliary, uncertain_loss, stats as ald_stats
from model.online_directed_confusion import OnlineDirectedConfusion
from model.geometry_pair_selector import GeometryPairSelector
from model.confusion_prototype_sep import confusion_prototype_sep_loss
from model.semantic_protected_sep import semantic_protected_sep
from kd_runtime import publish_evaluation
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from model.PAR import PAR
from utils import evaluate, imutils, optimizer
from utils.camutils import *
from utils.evaluate import compute_confusion, update_confusion_matrix
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
            aux_layer=args.aux_layer
        )
        self.device = torch.device(args.local_rank)
        self.total_classes = sum(tasks.get_per_task_classes(args.dataset, args.task, args.step)) - 1
        self.new_classes = tasks.get_per_task_classes(args.dataset, args.task, args.step)[-1]
        self.old_classes = self.total_classes - self.new_classes
        self.class_weight = torch.ones(self.total_classes + 1).to(self.device)
        self.class_weight[0] = 1.0
        self.confusion = OnlineDirectedConfusion(self.total_classes+1,
            momentum=args.confusion_momentum, high_threshold=args.high_thre,
            low_threshold=args.low_thre, stage=self.step).to(self.device)
        self.pair_selector = GeometryPairSelector(self.total_classes+1,
            old_classes=self.old_classes, stage=self.step,
            refresh_interval=args.pair_refresh_interval, min_row_images=args.pair_min_row_images,
            min_pair_images=args.pair_min_pair_images, min_rate=args.pair_min_rate,
            min_updates=args.pair_min_updates, ramp_updates=args.pair_ramp_updates,
            max_stale_updates=args.pair_max_stale_updates).to(self.device)
        if args.step == 0:  # if step 0, we don't need to instance the model_old
            self.model_old = None
        else:  # instance model_old
            self.model_old = network(
                backbone=args.backbone,
                num_classes=sum(tasks.get_per_task_classes(args.dataset, args.task, args.step - 1)),
                classes_list=tasks.get_per_task_classes(args.dataset, args.task, args.step - 1),
                pretrained=args.pretrained,
                init_momentum=args.momentum,
                aux_layer=args.aux_layer
            )
            # freeze old model and set eval mode
            for par in self.model_old.parameters():
                par.requires_grad = False
            self.model_old.eval()

        param_groups = self.model.get_param_groups()
        self.optimizer = self.get_optimizer(args, param_groups)

    def get_optimizer(self, args, param_groups):

        optim = getattr(optimizer, args.optimizer)(
            params=[
                {
                    "params": param_groups[0],
                    "lr": args.lr,
                    "weight_decay": args.wt_decay,
                },
                {
                    "params": param_groups[1],
                    "lr": args.lr,
                    "weight_decay": args.wt_decay,
                },
                {
                    "params": param_groups[2],
                    "lr": args.lr * 10,
                    "weight_decay": args.wt_decay,
                },
                {
                    "params": param_groups[3],
                    "lr": args.lr * 10,
                    "weight_decay": args.wt_decay,
                },
            ],
            lr=args.lr,
            weight_decay=args.wt_decay,
            betas=args.betas,
            warmup_iter=args.warmup_iters,
            max_iter=args.max_iters,
            warmup_ratio=args.warmup_lr,
            power=args.power)

        return optim

    def load_step_ckpt(self, path):
        # generate model from path
        if osp.exists(path):
            step_checkpoint = torch.load(path, map_location="cpu", weights_only=True)
            state = step_checkpoint.get('model_state', step_checkpoint)
            state = {key.removeprefix('module.'): value for key, value in state.items()}
            # step_checkpoint1 = torch.load('/root/autodl-tmp/ToCo-type3/10-5/step2/checkpoints/baseline.pth', map_location="cpu")
            incompatible = self.model.load_state_dict(state, strict=False)
            unexpected = incompatible.unexpected_keys
            missing = [key for key in incompatible.missing_keys
                       if not key.startswith((f'decoder.conv8.{self.step}.',
                                              f'decoder.class_prototypes.{self.step}.',
                                              f'classifier.{self.step}.',
                                              f'aux_classifier.{self.step}.'))]
            if unexpected or missing:
                raise RuntimeError(f'Invalid previous checkpoint: missing={missing}, unexpected={unexpected}')
            # if self.opts.init_balanced:
            #     # implement the balanced initialization (new cls has weight of background and bias = bias_bkg - log(N+1)
            #     self.model.module.init_new_classifier(self.device)
            # Load state dict from the model state dict, that contains the old model parameters
            self.model_old.load_state_dict(state, strict=True)
            # Initialize the newly added head from the loaded background head.
            self.model.decoder.conv8.init_weights()

            logging.info(f"[!] Previous model loaded from {path}")
            # clean memory
            del step_checkpoint, state
        else:
            raise FileNotFoundError(f"Previous-step checkpoint is required: {path}")


    def validate(self, model=None, data_loader=None, args=None):
        if args.async_eval:
            if args.local_rank == 0:
                publish_evaluation(args, self.current_iteration)
            dist.barrier()
            return 0.0, 'Checkpoint queued for four-GPU asynchronous evaluation'
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
        model.eval()
        par = PAR(num_iter=10, dilations=[1, 2, 4, 8, 12, 24]).cuda()
        conf_matrix = torch.zeros((self.total_classes + 1, self.total_classes + 1), dtype=torch.int64).cuda()
        with torch.no_grad():
            for i, data in tqdm(enumerate(data_loader), total=len(data_loader), ncols=100, ascii=" >="):
                img_name, inputs, cls_label, img_box, crops = data

                inputs = inputs.to(self.device, non_blocking=True)
                inputs_denorm = imutils.denormalize_img2(inputs.clone())

                cls_label = cls_label.to(self.device, non_blocking=True)
                cls_label = cls_label[:, :self.total_classes]

                old_cls, old_cls_aux, old_segs, _x4, type_seg = self.model_old(inputs, step0=True)
                cls_label_old_pred = (old_cls > 0).long()

                cls_label_gt_new = cls_label[:, -self.new_classes:]
                cls_label = torch.cat((cls_label_old_pred, cls_label_gt_new), dim=1)
                # get local crops from uncertain regions
                cams, cams_aux = multi_scale_cam2(model, inputs=inputs, scales=args.cam_scales)

                # seg_loss & reg_loss
                valid_cam, _ = cam_to_label(cams.detach(), cls_label=cls_label, img_box=img_box, ignore_mid=True,
                                            bkg_thre=args.bkg_thre, high_thre=args.high_thre, low_thre=args.low_thre,
                                            ignore_index=args.ignore_index)
                refined_pseudo_label = refine_cams_with_bkg_v2(par, inputs_denorm, cams=valid_cam, cls_labels=cls_label,
                                                               high_thre=args.high_thre, low_thre=args.low_thre,
                                                               ignore_index=args.ignore_index, img_box=img_box, )

                inputs = F.interpolate(inputs, size=[args.crop_size, args.crop_size], mode='bilinear',
                                       align_corners=False)
                cls, segs, _, _ = model(inputs, cal_sim=True)

                resized_segs = F.interpolate(segs, size=refined_pseudo_label.shape[1:], mode='bilinear',
                                             align_corners=False)

                # preds += list(torch.argmax(resized_segs, dim=1).cpu().numpy().astype(np.int16))
                # gts += list(refined_pseudo_label.cpu().numpy().astype(np.int16))

                pred = torch.argmax(resized_segs, dim=1).to(torch.int64)
                gt = refined_pseudo_label.to(torch.int64)

                update_confusion_matrix(conf_matrix, pred, gt, self.total_classes + 1)
                if i % 100 == 0:
                    print(conf_matrix)

        model.train()
        return conf_matrix

    def get_weight(self, con_matrix, total_classes, new_classes):
        new_class_indices = list(range(total_classes - new_classes + 1, total_classes + 1))  # 新类的ID
        old_class_indices = list(range(1, total_classes - new_classes + 1))  # 旧类的ID
        similarity_matrix = []
        print(con_matrix)
        for new_class in new_class_indices:
            row = con_matrix[new_class]  # 当前新类的预测结果 [n]
            wrong_predictions = row[old_class_indices]  # [n-6]

            total_samples_of_this_class = row.sum()
            print(total_samples_of_this_class)
            # 避免除0
            if total_samples_of_this_class > 0:
                probs = wrong_predictions.float() / total_samples_of_this_class
            else:
                probs = torch.zeros_like(wrong_predictions, dtype=torch.float)

            similarity_matrix.append(probs)
        print(similarity_matrix)

        similarity_matrix = torch.stack(similarity_matrix)
        print(similarity_matrix.max(dim=1).values)
        return similarity_matrix.max(dim=1).values  # 返回每个新类的最大相似度

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
        start_iteration = 0
        if args.resume_checkpoint:
            from kd_runtime import safe_path
            resume_path = safe_path(args.resume_checkpoint)
            saved = torch.load(resume_path, map_location='cpu', weights_only=True)
            model.load_state_dict(saved['model_state'], strict=True)
            optim.load_state_dict(saved['optimizer_state'])
            start_iteration = int(saved['iteration'])
            optim.global_step = start_iteration
            if saved.get('online_confusion_state') is not None:  # Same ALD anchor semantics; preserve online history.
                self.confusion.load_state_dict(saved['online_confusion_state'],strict=True)
            if False:  # Do not re-use the stale prior pair selector.
                self.pair_selector.load_state_dict(saved['geometry_selector_state'],strict=True)
            if not 0 < start_iteration < args.max_iters:
                raise ValueError('Resume iteration outside active stage')
            logging.info('Resume student AND optimizer at iteration %d from %s; teacher remains previous stage', start_iteration, resume_path)
            del saved
        logging.info('\nOptimizer: \n%s' % optim)
        # The best prior student is a frozen self-training reference, not GT.
        from kd_runtime import safe_path, digest
        reference_path = safe_path(args.ald_reference)
        reference_saved = torch.load(reference_path, map_location='cpu', weights_only=True, mmap=True)
        self.ald_reference = network(backbone=args.backbone, num_classes=21,
            classes_list=[11,5,5], pretrained=False, init_momentum=args.momentum, aux_layer=args.aux_layer)
        self.ald_reference.load_state_dict(reference_saved['model_state'], strict=True)
        self.ald_reference = self.ald_reference.to(device).eval().requires_grad_(False)
        del reference_saved
        evidence = torch.load(safe_path(args.ald_evidence), map_location='cpu', weights_only=True)
        self.ald_states = {name: evidence['state'][i] for i,name in enumerate(evidence['names'])}
        assert len(self.ald_states) == 2145
        # Only the two spatial decoder convolutions learn. Encoder, classifier,
        # old/new segmentation heads and prototype anchors remain fixed.
        import copy
        self.sep_reference = copy.deepcopy(model.decoder).eval().requires_grad_(False)
        for name, parameter in model.named_parameters():
            parameter.requires_grad_(name in {'decoder.conv6.weight', 'decoder.conv7.weight'})
        bank = torch.load(safe_path(Path(__file__).resolve().parents[2]/'bank.pth'), weights_only=True, map_location=device)
        self.sep_centers, self.sep_available = bank['centers'], bank['available']
        optim.max_iter = 8900
        model = DistributedDataParallel(model, device_ids=[args.local_rank], find_unused_parameters=True)
        active_sampler = step0_sampler if self.step == 0 else train_sampler
        active_sampler.set_epoch(np.random.randint(args.max_iters))
        train_loader_iter = iter(train_loader) if self.step > 0 else None
        avg_meter = AverageMeter()

        train_step0_loader_iter = iter(train_step0_loader) if self.step == 0 else None
        cal_sim_loader_iter = iter(cal_sim_loader) if args.confusion_reweight and self.step > 0 else None


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

                filter_idx = labels >= self.total_classes + 1
                labels[filter_idx] = 255

                _, cls, segs, fmap, cls_aux, type_seg, new_prototypes = model(inputs, crops=None, cam_grad=True, n_iter=n_iter)
                T = 0.1
                type_seg = type_seg / T

                prototype_sep = prototype_separation_loss(new_prototypes, margin=args.proto_margin)

                segs = F.interpolate(segs, size=labels.shape[-2:], mode='bilinear', align_corners=False)
                type_seg = F.interpolate(type_seg, size=labels.shape[-2:], mode='bilinear', align_corners=False)
                cls_loss = F.multilabel_soft_margin_loss(cls, cls_label)
                cls_loss_aux = F.multilabel_soft_margin_loss(cls_aux, cls_label)

                seg_loss = get_seg_loss(segs, labels.type(torch.long), class_weight=self.class_weight, ignore_index=args.ignore_index)
                type_seg_loss = get_seg_loss(type_seg, labels.type(torch.long), class_weight=self.class_weight, ignore_index=args.ignore_index)

                loss = cls_loss + cls_loss_aux + args.w_seg * seg_loss + args.w_proto_seg * type_seg_loss + args.w_proto_sep * prototype_sep
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
                    'proto_seg': type_seg_loss,
                })
                if (n_iter + 1) % args.log_iters == 0:

                    delta, eta = cal_eta(time0, n_iter + 1, args.max_iters)
                    cur_lr = optim.param_groups[0]['lr']

                    if args.local_rank == 0:
                        logging.info(
                            "Iter: %d; Elasped: %s; ETA: %s; LR: %.3e; cls_loss: %.4f, seg_loss: %.4f, proto_seg: %.4f, proto_sep: %.6f " % (
                            n_iter + 1, delta, eta, cur_lr, avg_meter.pop('cls_loss'), avg_meter.pop('seg_loss'),
                            avg_meter.pop('proto_seg'), avg_meter.pop('proto_sep')
                            ))

                if (n_iter + 1) % args.eval_iters == 0:
                    ckpt_name = os.path.join(args.ckpt_dir, "model_iter_%d.pth" % (n_iter + 1))
                    if args.local_rank == 0:
                        logging.info('Validating...')
                        if args.save_ckpt:
                            torch.save({'model_state': self.model.state_dict(), 'iteration': n_iter + 1, 'optimizer_state': optim.state_dict(), 'online_confusion_state': self.confusion.state_dict(), 'geometry_selector_state': self.pair_selector.state_dict()}, ckpt_name)
                    val_cls_score, tab_results = self.validate(model=model, data_loader=val_loader, args=args)
                    if args.local_rank == 0:
                        logging.info("val cls score: %.6f" % (val_cls_score))
                        logging.info("\n" + tab_results)

        else:
            model_old = self.model_old.to(device)
            # loss_layer = DenseEnergyLoss(weight=1e-7, sigma_rgb=15, sigma_xy=100, scale_factor=0.5)
            ncrops = 10

            par = PAR(num_iter=10, dilations=[1, 2, 4, 8, 12, 24]).cuda()

            for n_iter in range(start_iteration, args.max_iters, args.ald_stride):
                # Actual ALD refinement updates; resume8300, finish8600.
                # Restored parameter-wise Adam counters may differ at entry.
                self.current_iteration = n_iter + args.ald_stride
                optim.global_step = n_iter
                if args.confusion_reweight and n_iter == 4000:
                    con_matrix = self.cal_sim(model=model, data_loader=cal_sim_loader_iter, args=args)
                    new_class_weight = self.get_weight(con_matrix, self.total_classes, self.new_classes)
                    new_class_weight = new_class_weight / new_class_weight.mean()
                    mean = new_class_weight.mean()
                    alpha = 0.5
                    new_class_weight_scaled = mean + alpha * (new_class_weight - mean)
                    self.class_weight[(self.old_classes + 1):] = new_class_weight_scaled

                try:
                    img_name, inputs, cls_label, img_box, crops = next(train_loader_iter)
                except StopIteration:
                    train_sampler.set_epoch(np.random.randint(args.max_iters))
                    train_loader_iter = iter(train_loader)
                    img_name, inputs, cls_label, img_box, crops = next(train_loader_iter)


                inputs = inputs.to(device, non_blocking=True)
                inputs_denorm = imutils.denormalize_img2(inputs.clone())

                # import torchvision.transforms.functional as TF

                # input_image = TF.to_pil_image(inputs_denorm[0])
                # input_image.save('input_image denorm.png')

                cls_label = cls_label.to(device, non_blocking=True)
                cls_label = cls_label[:, :self.total_classes]

                # cls_gt_label = torch.clone(cls_label)

                # cls_label old + gt_new
                with torch.no_grad():
                    old_cls, old_cls_aux, old_segs, _x4, old_prototypes = model_old(inputs, step0=True)
                    teacher_native_logits = old_segs
                cls_label_gt_new = cls_label[:, -self.new_classes:]
                with torch.no_grad():
                    _, reference_logits, _, _ = self.ald_reference(inputs)
                    ald_state = torch.stack([self.ald_states[str(name)] for name in img_name]).to(device)
                    ald_target, cls_label, ald_conflict = image_targets(ald_state, old_cls, old_cls_aux,
                        teacher_native_logits, reference_logits, cls_label_gt_new)




                # get local crops from uncertain regions
                if args.ald:
                    cams, cams_aux, cam_channel_max = multi_scale_cam2_filter(model, inputs=inputs, scales=args.cam_scales)
                    cam_filter_label = cam_high_pass_filter(cam_channel_max, cls_label, 12.5)
                else:
                    cams, cams_aux = multi_scale_cam2(model, inputs=inputs, scales=args.cam_scales)



                roi_mask = cam_to_roi_mask2(cams_aux.detach(), cls_label=cls_label, low_thre=args.low_thre,
                                            hig_thre=args.high_thre)

                local_crops, flags = crop_from_roi_neg(images=crops[2], roi_mask=roi_mask, crop_num=ncrops - 2,
                                                       crop_size=args.local_crop_size)
                roi_crops = crops[:2] + local_crops

                # Only deterministic, segmentation-independent label operations
                # move before the main forward. ROI random crops stay above.
                old_segs = F.interpolate(old_segs, size=inputs.shape[-2:], mode='bilinear', align_corners=False)
                old_pixel_label = torch.argmax(old_segs, dim=1)
                # seg_loss & reg_loss
                valid_cam, _ = cam_to_label(cams.detach(), cls_label=cls_label, img_box=img_box, ignore_mid=True,
                                            bkg_thre=args.bkg_thre, high_thre=args.high_thre, low_thre=args.low_thre,
                                            ignore_index=args.ignore_index)
                refined_pseudo_label = refine_cams_with_bkg_v2(par, inputs_denorm, cams=valid_cam, cls_labels=cls_label,
                                                               high_thre=args.high_thre, low_thre=args.low_thre,
                                                               ignore_index=args.ignore_index, img_box=img_box, )

                if args.ald:
                    refined_pseudo_label = filter_cam_pesudo_label(refined_pseudo_label, cam_filter_label, 255)

                # Past selector state, before observing this batch.
                pair_targets = self.pair_selector.update(self.confusion,self.current_iteration)
                pair_ramp = self.pair_selector.ramp(self.confusion)
                ald_fused = fuse(refined_pseudo_label, teacher_native_logits, reference_logits, ald_state, img_box)
                mixed_pseudo_label = ald_fused['labels']

                def prototype_objective(native_type_seg, current_prototypes):
                    semantic_logits = F.interpolate(native_type_seg / 0.1,
                        size=mixed_pseudo_label.shape[1:], mode='bilinear', align_corners=False)
                    semantic_loss = get_seg_loss(semantic_logits, mixed_pseudo_label.long(),
                        class_weight=self.class_weight, ignore_index=args.ignore_index)
                    geometry_loss, geometry_record = confusion_prototype_sep_loss(
                        current_prototypes, pair_targets, self.old_classes, margin=args.proto_margin)
                    protected = native_type_seg.sum() * 0.
                    protection = {'enabled': False, 'reason': 'fixed_prototypes_local_feature_SEP',
                                  'effective_geometry_weight': 0., 'semantic_gradient_used': False}
                    geometry_record['semantic_protection'] = protection
                    return semantic_loss, protected, geometry_loss, geometry_record

                from model.local_sep import separation_loss, retention_loss
                def local_objective(features, logits, encoder_features):
                    with torch.no_grad():
                        fixed_logits = self.sep_reference(encoder_features.detach())[0]
                    separation, record = separation_loss(features, mixed_pseudo_label,
                        valid_cam, ald_fused['valid'], pair_targets, self.sep_centers, self.sep_available)
                    retention = retention_loss(logits, fixed_logits, ald_fused['valid'])
                    record.update(retention=float(retention.detach()))
                    return separation, retention, record
                cls, segs, fmap, cls_aux, type_seg, new_prototypes, prototype_outputs, local_outputs = model(
                    inputs, crops=roi_crops, n_iter=n_iter, prototype_objective=prototype_objective,
                    local_objective=local_objective)
                local_sep, local_retention, local_record = local_outputs
                type_seg_loss, protected_sep, geometry_sep, geometry_stats = prototype_outputs

                prototype_kd = prototype_distillation_loss(new_prototypes, old_prototypes)
                prototype_sep = prototype_separation_loss(new_prototypes, margin=args.proto_margin)


                cls_loss = F.binary_cross_entropy_with_logits(cls, ald_target)
                cls_loss_aux = F.binary_cross_entropy_with_logits(cls_aux, ald_target)
                ald_soft = uncertain_loss(segs, reference_logits, ald_fused['unknown'],
                    ald_fused['valid'], ald_state, cls_label_gt_new, args.kd_temperature)
                ald_record = ald_stats(ald_fused, ald_state, ald_conflict, ald_soft)

                # # ctc_loss
                # ctc_loss = CTC_loss(out_s, out_t, flags)

                # Observer starts fresh at warmup; source anchors remain PAR,
                # calibrated for old presence / reference BG, never current agreement.
                ald_anchors = gate_auxiliary(refined_pseudo_label, ald_state, reference_logits, calibrate_new=False)
                self.confusion.update(segs,ald_anchors,valid_cam,img_box)
                if args.local_rank==0 and (self.current_iteration)%args.log_iters==0:
                    from kd_runtime import atomic_json
                    atomic_json(Path(args.work_dir)/'online_confusion.json',
                        {'training_iteration':self.current_iteration,**self.confusion.export()})
                pixel_kd, kd_stats = pixel_kd_loss(segs, teacher_native_logits,
                    refined_pseudo_label, valid_cam, img_box, args.kd_temperature,
                    trusted_mask=ald_fused['trusted_old'])

                segs = F.interpolate(segs, size=refined_pseudo_label.shape[1:], mode='bilinear', align_corners=False)
                seg_loss = get_seg_loss(segs, mixed_pseudo_label.type(torch.long), class_weight=self.class_weight, ignore_index=args.ignore_index)

                resized_cams_aux = F.interpolate(cams_aux, size=fmap.shape[2:], mode="bilinear", align_corners=False)
                _, pseudo_label_aux = cam_to_label(resized_cams_aux.detach(), cls_label=cls_label, img_box=img_box,
                                                   ignore_mid=True, bkg_thre=args.bkg_thre, high_thre=args.high_thre,
                                                   low_thre=args.low_thre, ignore_index=args.ignore_index)
                pseudo_label_aux = gate_auxiliary(pseudo_label_aux, ald_state, reference_logits)
                aff_mask = label_to_aff_mask(pseudo_label_aux)
                ptc_loss = get_masked_ptc_loss(fmap, aff_mask)

                # warmup
                if n_iter < args.loss_warmup_iters:
                    # Keep every returned decoder branch in the DDP backward
                    # graph while its training objective is still warming up.
                    loss = cls_loss + cls_loss_aux + args.w_ptc * ptc_loss + 0.0 * (seg_loss + type_seg_loss + prototype_kd + prototype_sep + pixel_kd + geometry_sep + protected_sep)
                else:
                    loss = cls_loss + cls_loss_aux + args.w_ptc * ptc_loss + args.w_seg * seg_loss + args.w_proto_seg * type_seg_loss + args.w_proto_kd * prototype_kd + args.w_proto_sep * prototype_sep + args.w_pixel_kd * pixel_kd + protected_sep + args.w_seg * ald_soft + .1 * local_sep + local_retention
                if not torch.isfinite(loss):
                    raise FloatingPointError(f'Nonfinite loss at step {self.step} iteration {self.current_iteration}')

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
                    'proto_seg': type_seg_loss,
                })
                optim.zero_grad()
                loss.backward()
                if n_iter == start_iteration or args.ald_smoke:
                    trainable = [p for p in self.model.parameters() if p.grad is not None]
                    assert trainable and all(torch.isfinite(p.grad).all() for p in trainable)
                    assert all(p.grad is None for p in self.ald_reference.parameters())
                    assert all(p.grad is None for p in model_old.parameters())
                if args.local_rank == 0 and (self.current_iteration) % args.log_iters == 0:
                    local_record.update(iteration=self.current_iteration, selected_pairs=pair_targets.tolist())
                    with open(osp.join(args.work_dir,'local_sep_metrics.jsonl'),'a') as handle:
                        handle.write(json.dumps(local_record,allow_nan=False)+'\n')
                    ald_record.update(iteration=self.current_iteration, global_batch=args.spg*dist.get_world_size())
                    with open(osp.join(args.work_dir,'ald_metrics.jsonl'),'a') as handle:
                        handle.write(json.dumps(ald_record,allow_nan=False)+'\n')
                    kd_stats.update(step=self.step, iteration=self.current_iteration,
                        active=n_iter >= args.loss_warmup_iters, weight=args.w_pixel_kd,
                        global_batch=args.spg*dist.get_world_size())
                    with open(osp.join(args.work_dir, 'kd_metrics.jsonl'), 'a') as handle:
                        handle.write(json.dumps(kd_stats, allow_nan=False)+'\n')
                    geometry_stats.update(step=self.step,iteration=self.current_iteration,
                        active=n_iter>=args.loss_warmup_iters,ramp=pair_ramp,
                        weight=args.w_geometry_sep,selector=self.pair_selector.export())
                    with open(osp.join(args.work_dir,'geometry_metrics.jsonl'),'a') as handle:
                        handle.write(json.dumps(geometry_stats,allow_nan=False)+'\n')
                    protection_record = dict(geometry_stats['semantic_protection'])
                    protection_record.update(step=self.step, iteration=self.current_iteration,
                        active=n_iter>=args.loss_warmup_iters,
                        weight=args.w_geometry_sep, ramp=pair_ramp)
                    with open(osp.join(args.work_dir,'semantic_guard_metrics.jsonl'),'a') as handle:
                        handle.write(json.dumps(protection_record,allow_nan=False)+'\n')
                # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optim.step()
                if (self.current_iteration) % args.log_iters == 0:

                    delta, eta = cal_eta(time0, self.current_iteration, args.max_iters)
                    cur_lr = optim.param_groups[0]['lr']

                    if args.local_rank == 0:
                        logging.info(
                            "Iter: %d; Elasped: %s; ETA: %s; LR: %.3e; cls_loss: %.4f, cls_loss_aux: %.4f, ptc_loss: %.4f, seg_loss: %.4f, proto_seg: %.4f, proto_kd: %.6f, proto_sep: %.6f" % (
                            self.current_iteration, delta, eta, cur_lr, avg_meter.pop('cls_loss'), avg_meter.pop('cls_loss_aux'),
                            avg_meter.pop('ptc_loss'), avg_meter.pop('seg_loss'), avg_meter.pop('proto_seg'),
                            avg_meter.pop('proto_kd'), avg_meter.pop('proto_sep')))

                if (self.current_iteration) % args.eval_iters == 0:
                    ckpt_name = os.path.join(args.ckpt_dir, "model_iter_%d.pth" % (self.current_iteration))
                    if args.local_rank == 0:
                        logging.info('Validating...')
                        if args.save_ckpt:
                            torch.save({'model_state': self.model.state_dict(), 'iteration': self.current_iteration, 'optimizer_state': optim.state_dict(), 'online_confusion_state': self.confusion.state_dict(), 'geometry_selector_state': self.pair_selector.state_dict()}, ckpt_name)
                    val_cls_score, tab_results = self.validate(model=model, data_loader=val_loader, args=args)
                    if args.local_rank == 0:
                        logging.info("val cls score: %.6f" % (val_cls_score))
                        logging.info("\n" + tab_results)

        if args.max_iters % args.eval_iters:
            val_cls_score, tab_results = self.validate(model=model, data_loader=val_loader, args=args)
            if args.local_rank == 0:
                logging.info('Final validation classification F1: %.6f\n%s', val_cls_score, tab_results)
        if args.ald_smoke:
            flat = torch.cat([p.detach().flatten() for p in self.model.parameters()])
            root = flat.clone(); dist.broadcast(root, 0)
            assert torch.equal(root, flat), 'DDP parameters differ across ranks'
            from kd_runtime import atomic_json
            atomic_json(Path(args.work_dir)/f'ald_smoke_rank{dist.get_rank()}.json',
                {'passed': True, 'finite_gradients': True, 'frozen_teachers_no_gradient': True,
                 'all_rank_parameters_exact': True, 'iteration': self.current_iteration})
        return True
