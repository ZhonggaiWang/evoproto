import argparse
import datetime
import logging
import os
import random
import sys
from torch import distributed

sys.path.insert(0, "/root/code/ToCo-type3")
import numpy as np
import torch
import tasks
from utils.pyutils import AverageMeter, cal_eta, format_tabs, setup_logger
from continual.Trainer_coco import Trainer
import torch.distributed as dist

torch.hub.set_dir("./pretrained")

parser = argparse.ArgumentParser()

parser.add_argument("--backbone", default='vit_base_patch16_224', type=str, help="vit_base_patch16_224")
parser.add_argument("--pooling", default='gmp', type=str, help="pooling choice for patch tokens")
parser.add_argument("--pretrained", default=True, type=bool, help="use imagenet pretrained weights")
parser.add_argument("--task", type=str, default="coco2voc", choices=tasks.get_task_list(),
                    help="Task to be executed (default: 10-5)")
parser.add_argument("--step", default=0, type=int, help="training_step")

parser.add_argument("--dataset", type=str, default='coco2voc', help='Name of dataset')
parser.add_argument("--coco_img_folder", default='./coco/images', type=str, help="dataset folder")
parser.add_argument("--coco_label_folder", default='./coco/annotations_map', type=str, help="dataset folder")
parser.add_argument("--coco_list_folder", default='./datasets/coco', type=str, help="train/val/test list file")

parser.add_argument("--voc_data_folder", default='VOCdevkit/VOC2012', type=str, help="dataset folder")
parser.add_argument("--voc_list_folder", default='./datasets/coco', type=str, help="train/val/test list file")

parser.add_argument("--num_classes", default=81, type=int, help="number of classes")
parser.add_argument("--crop_size", default=448, type=int, help="crop_size in training")
parser.add_argument("--local_crop_size", default=96, type=int, help="crop_size for local view")
parser.add_argument("--ignore_index", default=255, type=int, help="random index")

parser.add_argument("--work_dir", default="work_dir_coco_wseg", type=str, help="work_dir_coco_wseg")

parser.add_argument("--train_set", default="train", type=str, help="training split")
parser.add_argument("--coco_filter_val_set", default="val", type=str, help="validation split")

parser.add_argument("--coco_filter_val_set_step1", default="val_step1", type=str, help="validation split")

parser.add_argument("--spg", default=4, type=int, help="samples_per_gpu")
parser.add_argument("--scales", default=(0.5, 2), help="random rescale in training")

parser.add_argument("--optimizer", default='PolyWarmupAdamW', type=str, help="optimizer")
parser.add_argument("--lr", default=6e-5, type=float, help="learning rate")
parser.add_argument("--warmup_lr", default=1e-6, type=float, help="warmup_lr")
parser.add_argument("--wt_decay", default=1e-2, type=float, help="weights decay")
parser.add_argument("--betas", default=(0.9, 0.999), help="betas for Adam")
parser.add_argument("--power", default=0.9, type=float, help="poweer factor for poly scheduler")

parser.add_argument("--max_iters", default=80000, type=int, help="max training iters")
parser.add_argument("--log_iters", default=200, type=int, help=" logging iters")
parser.add_argument("--eval_iters", default=2000, type=int, help="validation iters")
parser.add_argument("--warmup_iters", default=2000, type=int, help="warmup_iters")

parser.add_argument("--high_thre", default=0.65, type=float, help="high_bkg_score")
parser.add_argument("--low_thre", default=0.25, type=float, help="low_bkg_score")
parser.add_argument("--bkg_thre", default=0.45, type=float, help="bkg_score")
parser.add_argument("--cam_scales", default=(1.0, 0.5, 1.5, 0.75), help="multi_scales for cam")

parser.add_argument("--w_ptc", default=0.2, type=float, help="w_ptc")
parser.add_argument("--w_ctc", default=0.5, type=float, help="w_ctc")
parser.add_argument("--w_seg", default=0.1, type=float, help="w_seg")
parser.add_argument("--w_reg", default=0.05, type=float, help="w_reg")
parser.add_argument("--w_kd", default=0.2, type=float, help="w_kd")
parser.add_argument("--w_cls_kd", default=0.2, type=float, help="w_cls_kd")
parser.add_argument("--w_entropy", default=0.02, type=float, help="w_entropy")
parser.add_argument("--temp", default=0.5, type=float, help="temp")
parser.add_argument("--momentum", default=0.9, type=float, help="temp")
parser.add_argument("--aux_layer", default=-3, type=int, help="aux_layer")

parser.add_argument("--seed", default=0, type=int, help="fix random seed")
parser.add_argument("--save_ckpt", action="store_true", help="save_ckpt")

parser.add_argument("--local-rank", default=1, type=int, help="local_rank")
parser.add_argument("--num_workers", default=10, type=int, help="num_workers")
parser.add_argument('--backend', default='nccl')


# os.environ['MASTER_ADDR'] = 'localhost'
# os.environ['MASTER_PORT'] = '5682'
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def save_ckpt(path, trainer):
    """ save current model
    """
    state = {
        "model_state": trainer.model.state_dict(),
    }
    path = os.path.join(path, "model_final.pth")
    torch.save(state, path)


if __name__ == "__main__":
    distributed.init_process_group(backend='nccl', init_method='env://')

    # dist.init_process_group(backend='nccl', init_method='env://', rank = 0, world_size = 1)

    args = parser.parse_args()
    args.work_dir = os.path.join('/root/autodl-tmp/ToCo-type3', args.task)
    load_ckpt_dir = os.path.join(args.work_dir, "step" + str(args.step - 1), "checkpoints")
    args.work_dir = os.path.join(args.work_dir, "step" + str(args.step))
    args.ckpt_dir = os.path.join(args.work_dir, "checkpoints")
    args.pred_dir = os.path.join(args.work_dir, "predictions")

    if args.local_rank == 0:
        os.makedirs(args.ckpt_dir, exist_ok=True)
        os.makedirs(args.pred_dir, exist_ok=True)

        setup_logger(filename=os.path.join(args.work_dir, 'train.log'))
        logging.info('Pytorch version: %s' % torch.__version__)
        logging.info("GPU type: %s" % (torch.cuda.get_device_name(0)))
        logging.info('\nargs: %s' % args)
    ## fix random seed
    setup_seed(args.seed)
    trainer = Trainer(args=args)
    if args.step > 0:
        trainer.load_step_ckpt(os.path.join(load_ckpt_dir, "model_final.pth"))
    Flag = trainer.train(args=args)
    if args.local_rank == 0 and Flag:  # save best model at the last iteration
        # best model to build incremental steps
        save_ckpt(args.ckpt_dir, trainer)
        logging.info("[!] Checkpoint saved.")