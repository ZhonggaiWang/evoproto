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
    get_pixel_refine_cam_loss_v2
from model.model_seg_neg import network
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from model.PAR import PAR
from utils import evaluate, imutils, optimizer
from utils.camutils import *
from utils.evaluate import compute_confusion, update_confusion_matrix
from utils.pyutils import AverageMeter, cal_eta, format_tabs, setup_logger


def kd_prototypes(new_prototypes, old_prototypes, margin_neg=-0.7):
    # 计算新原型与旧原型之间的余弦相似度矩阵 (只考虑相同类别)
    num_old = old_prototypes.shape[0]
    num_new = new_prototypes.shape[0]

    device = new_prototypes.device
    proto_sim = torch.matmul(new_prototypes, new_prototypes.t())  # [N, N]
    identity = torch.eye(num_new, device=device)

    negatives = proto_sim[identity == 0]  # 非对角线元素

    # 负样本loss: 只要小于 margin_neg 就可以
    neg_loss = F.relu(negatives - margin_neg) ** 2
    new_contrastive_loss = neg_loss.mean()

    new_prototypes_old = new_prototypes[:num_old]
    proto_old_sim = torch.matmul(new_prototypes_old, old_prototypes.t())
    diagonal_elements = proto_old_sim.diagonal()
    old_contrastive_loss = ((1.0 - diagonal_elements) ** 2).mean()
    total_loss = new_contrastive_loss + old_contrastive_loss
    return total_loss


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
            aux_layer=args.aux_layer
        )
        self.device = torch.device(args.local_rank)
        self.total_classes = sum(tasks.get_per_task_classes(args.dataset, args.task, args.step)) - 1
        self.new_classes = tasks.get_per_task_classes(args.dataset, args.task, args.step)[-1]
        self.old_classes = self.total_classes - self.new_classes
        self.class_weight = torch.ones(self.total_classes + 1).to(self.device)
        self.class_weight[0] = 0.1
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
            step_checkpoint = torch.load(path, map_location="cpu")
            self.model.load_state_dict(step_checkpoint['model_state'], strict=False)  # False for incr. classifiers
            # if self.opts.init_balanced:
            #     # implement the balanced initialization (new cls has weight of background and bias = bias_bkg - log(N+1)
            #     self.model.module.init_new_classifier(self.device)
            # Load state dict from the model state dict, that contains the old model parameters
            self.model_old.load_state_dict(step_checkpoint['model_state'], strict=True)  # Load also here old parameters

            logging.info(f"[!] Previous model loaded from {path}")
            # clean memory
            del step_checkpoint['model_state']
        else:
            logging.info(f"[!] WARNING: Unable to find of step {self.args.step - 1}! "
                             f"Do you really want to do from scratch?")

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
        cal_sim_loader_iter = iter(cal_sim_loader)
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

                _, cls, segs, fmap, cls_aux, type_seg, new_prototypes = model(inputs, crops=None, cam_grad=True,
                                                                              n_iter=n_iter)
                T = 0.1
                type_seg = type_seg / T

                proto_sim = torch.matmul(new_prototypes, new_prototypes.t())  # [N, N]
                identity = torch.eye(proto_sim.size(0), device=proto_sim.device)
                # 只取非对角线元素
                negatives = proto_sim[identity == 0]  # shape: [N*(N-1)]

                # 设置 margin，比如 -0.5
                margin = -0.7
                # 只惩罚比 margin 大的负样本，相似度太大才罚
                neg_loss = F.relu(negatives - margin) ** 2
                contrastive_loss = neg_loss.mean()

                segs = F.interpolate(segs, size=[448, 448], mode='bilinear', align_corners=False)
                type_seg = F.interpolate(type_seg, size=[448, 448], mode='bilinear', align_corners=False)
                cls_loss = F.multilabel_soft_margin_loss(cls, cls_label)
                cls_loss_aux = F.multilabel_soft_margin_loss(cls_aux, cls_label)

                seg_loss = get_seg_loss(segs, labels.type(torch.long), class_weight=self.class_weight,
                                        ignore_index=args.ignore_index)
                type_seg_loss = get_seg_loss(type_seg, labels.type(torch.long), class_weight=self.class_weight,
                                             ignore_index=args.ignore_index)

                loss = 1 * cls_loss + 1 * cls_loss_aux + args.w_seg * seg_loss + 0.1 * type_seg_loss + 0.1 * contrastive_loss

                optim.zero_grad()
                loss.backward()
                # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optim.step()
                avg_meter.add({
                    'cls_loss': cls_loss,
                    'seg_loss': seg_loss,
                    'type_seg_loss': type_seg_loss,
                    'contrastive_loss': contrastive_loss,
                })
                if (n_iter + 1) % args.log_iters == 0:

                    delta, eta = cal_eta(time0, n_iter + 1, args.max_iters)
                    cur_lr = optim.param_groups[0]['lr']

                    if args.local_rank == 0:
                        logging.info(
                            "Iter: %d; Elasped: %s; ETA: %s; LR: %.3e; cls_loss: %.4f ,seg_loss: %.4f ,type_seg_loss: %.4f, contrastive_loss: %.4f" % (
                                n_iter + 1, delta, eta, cur_lr, avg_meter.pop('cls_loss'), avg_meter.pop('seg_loss'),
                                avg_meter.pop('type_seg_loss'), avg_meter.pop('contrastive_loss'),
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
                if n_iter==4000:
                    con_matrix = self.cal_sim(model=model, data_loader=cal_sim_loader_iter, args=args)
                    new_class_weight = self.get_weight(con_matrix, self.total_classes, self.new_classes)
                    new_class_weight = new_class_weight / new_class_weight.mean()
                    mean = new_class_weight.mean()
                    alpha = 0.5
                    new_class_weight_scaled = mean + alpha * (new_class_weight - mean)
                    self.class_weight[(self.old_classes + 1):] = new_class_weight_scaled

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

                # cls_label old + gt_new
                old_cls, old_cls_aux, old_segs, _x4, type_seg = model_old(inputs, step0=True)
                cls_label_old_pred = (old_cls > 0).long()

                cls_label_gt_new = cls_label[:, -self.new_classes:]
                cls_label = torch.cat((cls_label_old_pred, cls_label_gt_new), dim=1)

                # get local crops from uncertain regions
                cams, cams_aux, cam_channel_max = multi_scale_cam2_filter(model, inputs=inputs, scales=args.cam_scales)
                cam_filter_label = cam_high_pass_filter(cam_channel_max, cls_label, 7.5)

                roi_mask = cam_to_roi_mask2(cams_aux.detach(), cls_label=cls_label, low_thre=args.low_thre,
                                            hig_thre=args.high_thre)

                local_crops, flags = crop_from_roi_neg(images=crops[2], roi_mask=roi_mask, crop_num=ncrops - 2,
                                                       crop_size=args.local_crop_size)
                roi_crops = crops[:2] + local_crops

                cls, segs, fmap, cls_aux, type_seg, new_prototypes = model(inputs, crops=roi_crops, n_iter=n_iter)
                old_cls, old_cls_aux, old_segs, _x4, old_prototypes = model_old(inputs, step0=True)
                T = 0.1
                type_seg = type_seg / T

                contrastive_loss = kd_prototypes(new_prototypes, old_prototypes)

                old_segs = F.interpolate(old_segs, size=[448, 448], mode='bilinear', align_corners=False)
                old_pixel_label = torch.argmax(old_segs, dim=1)

                cls_loss = F.multilabel_soft_margin_loss(cls, cls_label)
                cls_loss_aux = F.multilabel_soft_margin_loss(cls_aux, cls_label)

                # # ctc_loss
                # ctc_loss = CTC_loss(out_s, out_t, flags)

                # seg_loss & reg_loss
                valid_cam, _ = cam_to_label(cams.detach(), cls_label=cls_label, img_box=img_box, ignore_mid=True,
                                            bkg_thre=args.bkg_thre, high_thre=args.high_thre, low_thre=args.low_thre,
                                            ignore_index=args.ignore_index)
                refined_pseudo_label = refine_cams_with_bkg_v2(par, inputs_denorm, cams=valid_cam, cls_labels=cls_label,
                                                               high_thre=args.high_thre, low_thre=args.low_thre,
                                                               ignore_index=args.ignore_index, img_box=img_box, )

                refined_pseudo_label = filter_cam_pesudo_label(refined_pseudo_label, cam_filter_label, 255)

                mixed_pseudo_label = get_mixed_label(refined_pseudo_label, old_pixel_label,
                                                     self.total_classes, self.new_classes)

                segs = F.interpolate(segs, size=refined_pseudo_label.shape[1:], mode='bilinear', align_corners=False)
                type_seg = F.interpolate(type_seg, size=refined_pseudo_label.shape[1:], mode='bilinear',
                                         align_corners=False)
                seg_loss = get_seg_loss(segs, mixed_pseudo_label.type(torch.long), class_weight=self.class_weight,
                                        ignore_index=args.ignore_index)
                type_seg_loss = get_seg_loss(type_seg, mixed_pseudo_label.type(torch.long),
                                             class_weight=self.class_weight,
                                             ignore_index=args.ignore_index)

                resized_cams_aux = F.interpolate(cams_aux, size=fmap.shape[2:], mode="bilinear", align_corners=False)
                _, pseudo_label_aux = cam_to_label(resized_cams_aux.detach(), cls_label=cls_label, img_box=img_box,
                                                   ignore_mid=True, bkg_thre=args.bkg_thre, high_thre=args.high_thre,
                                                   low_thre=args.low_thre, ignore_index=args.ignore_index)
                aff_mask = label_to_aff_mask(pseudo_label_aux)
                ptc_loss = get_masked_ptc_loss(fmap, aff_mask)

                # warmup
                if n_iter <= 2000:
                    loss = 1.0 * cls_loss + 1.0 * cls_loss_aux + args.w_ptc * ptc_loss + 0.0 * seg_loss + 0.0 * type_seg_loss + 0.0 * contrastive_loss
                else:
                    loss = 1.0 * cls_loss + 1.0 * cls_loss_aux + args.w_ptc * ptc_loss + args.w_seg * seg_loss + 0.1 * type_seg_loss + 0.1 * contrastive_loss

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
                    'type_seg_loss': type_seg_loss,
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
                            "Iter: %d; Elasped: %s; ETA: %s; LR: %.3e; cls_loss: %.4f, cls_loss_aux: %.4f, ptc_loss: %.4f, seg_loss: %.4f, type_seg_loss: %.4f, contrastive_loss: %.4f" % (
                                n_iter + 1, delta, eta, cur_lr, avg_meter.pop('cls_loss'),
                                avg_meter.pop('cls_loss_aux'),
                                avg_meter.pop('ptc_loss'), avg_meter.pop('seg_loss'), avg_meter.pop('type_seg_loss'),
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