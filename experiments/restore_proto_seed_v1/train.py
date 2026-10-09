import argparse
import datetime
import logging
import os
import random
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
runtime = PROJECT_ROOT / '.runtime'
for variable, relative in {
    'TMPDIR': 'tmp', 'XDG_CACHE_HOME': 'cache', 'HF_HOME': 'cache/huggingface',
    'TORCH_HOME': 'cache/torch', 'MPLCONFIGDIR': 'cache/matplotlib',
    'CUDA_CACHE_PATH': 'cache/cuda',
}.items():
    path = runtime / relative
    if not path.resolve().is_relative_to(PROJECT_ROOT):
        raise RuntimeError(f'Cache path escapes project: {path}')
    path.mkdir(parents=True, exist_ok=True)
    os.environ[variable] = str(path)
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
sys.dont_write_bytecode = True
import json
from torch import distributed
import numpy as np
import torch
import tasks
from utils.pyutils import AverageMeter, cal_eta, format_tabs, setup_logger
from experiments.restore_proto_seed_v1.Trainer import Trainer
import torch.distributed as dist

torch.hub.set_dir(str(PROJECT_ROOT / 'pretrained'))
parser = argparse.ArgumentParser()
parser.add_argument('--seed_mode', choices=['off','correct','ignore','no_graph'], default='off')
parser.add_argument('--variant',choices=['full','without_confusion','without_proto'],default='full')

parser.add_argument("--backbone", default='vit_base_patch16_224', type=str, help="vit_base_patch16_224")
parser.add_argument("--pooling", default='gmp', type=str, help="pooling choice for patch tokens")
parser.add_argument("--pretrained", default=True, action=argparse.BooleanOptionalAction, help="use imagenet pretrained weights")

parser.add_argument("--task", type=str, default="10-5", choices=tasks.get_task_list(),
                        help="Task to be executed (default: 10-5)")
parser.add_argument("--step", default=0, type=int, help="training_step")

parser.add_argument("--dataset", type=str, default='voc', help='Name of dataset')
parser.add_argument("--data_folder", default='/ML-vePFS/infra_rd/kun/others/wzg/data/datasets/VOCdevkit/VOC2012', type=str, help="read-only VOC image root")
parser.add_argument("--list_folder", default=str(PROJECT_ROOT / 'datasets/voc'), type=str, help="train/val/test list file")
parser.add_argument("--seg_label_dir", default=str(PROJECT_ROOT / 'data/SegmentationClassAug'), type=str)
parser.add_argument("--val_label_dir", default='/ML-vePFS/infra_rd/kun/others/wzg/data/datasets/VOCdevkit/VOC2012/SegmentationClass', type=str)
parser.add_argument("--prev_checkpoint", default=None, type=str, help="explicit frozen previous-step checkpoint")
parser.add_argument("--num_classes", default=21, type=int, help="number of classes")
parser.add_argument("--crop_size", default=448, type=int, help="crop_size in training")
parser.add_argument("--local_crop_size", default=96, type=int, help="crop_size for local view")
parser.add_argument("--ignore_index", default=255, type=int, help="random index")

parser.add_argument("--work_dir", default=str(PROJECT_ROOT / 'runs/baseline'), type=str, help="experiment root within this project")

parser.add_argument("--train_set", default="train", type=str, help="training split")
parser.add_argument("--val_set", default="val", type=str, help="validation split")
parser.add_argument("--spg", default=8, type=int, help="samples_per_gpu")
parser.add_argument("--scales", default=(0.5, 2), type=float, nargs=2, help="random rescale in training")

parser.add_argument("--optimizer", default='PolyWarmupAdamW', type=str, help="optimizer")
parser.add_argument("--lr", default=2e-5, type=float, help="learning rate")
parser.add_argument("--warmup_lr", default=1e-6, type=float, help="warmup_lr")
parser.add_argument("--wt_decay", default=1e-2, type=float, help="weights decay")
parser.add_argument("--betas", default=(0.9, 0.999), type=float, nargs=2, help="betas for Adam")
parser.add_argument("--power", default=0.9, type=float, help="poweer factor for poly scheduler")

parser.add_argument("--max_iters", default=6000, type=int, help="max training iters")
parser.add_argument("--log_iters", default=50, type=int, help=" logging iters")
parser.add_argument("--eval_iters", default=2000, type=int, help="validation iters")
parser.add_argument("--warmup_iters", default=2000, type=int, help="warmup_iters")

parser.add_argument("--high_thre", default=0.7, type=float, help="high_bkg_score")
parser.add_argument("--low_thre", default=0.25, type=float, help="low_bkg_score")
parser.add_argument("--bkg_thre", default=0.5, type=float, help="bkg_score")
parser.add_argument("--cam_scales", default=(1.0, 0.5, 1.5), type=float, nargs='+', help="multi_scales for cam")

parser.add_argument("--w_ptc", default=0.2, type=float, help="w_ptc")
parser.add_argument("--w_ctc", default=0.5, type=float, help="w_ctc")
parser.add_argument("--w_seg", default=0.1, type=float, help="w_seg")
parser.add_argument("--w_reg", default=0.05, type=float, help="w_reg")
parser.add_argument("--w_proto_kd", "--w_kd", dest="w_proto_kd", default=0.1, type=float)
parser.add_argument("--w_proto_sep", default=0.1, type=float)
parser.add_argument("--w_proto_seg", default=0.1, type=float)
parser.add_argument("--proto_margin", default=0.0, type=float)
parser.add_argument("--ald", action="store_true", help="enable existing ALD; disabled for fixed baseline")
parser.add_argument("--ald_mode", default=None,
                    choices=["off", "legacy", "preserve_rejected", "preserve_background"],
                    help="ALD fusion policy; --ald alone selects legacy")
parser.add_argument("--confusion_reweight", action="store_true", help="enable existing dynamic class weighting")
parser.add_argument("--loss_warmup_iters", default=2000, type=int)
parser.add_argument("--train_limit", default=0, type=int, help="optional smoke-test image limit")
parser.add_argument("--val_limit", default=0, type=int, help="optional smoke-test image limit")

parser.add_argument("--temp", default=0.5, type=float, help="temp")
parser.add_argument("--momentum", default=0.9, type=float, help="temp")
parser.add_argument("--aux_layer", default=-3, type=int, help="aux_layer")

parser.add_argument("--seed", default=0, type=int, help="fix random seed")
parser.add_argument("--save_ckpt", action="store_true", help="save_ckpt")

parser.add_argument("--local-rank", "--local_rank", default=int(os.environ.get('LOCAL_RANK', 0)), type=int, help="local_rank")
parser.add_argument("--num_workers", default=4, type=int, help="num_workers (must be positive)")
parser.add_argument('--backend', default='nccl')

# os.environ['MASTER_ADDR'] = 'localhost'
# os.environ['MASTER_PORT'] = '5681'
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
        "iteration": trainer.current_iteration,
        "variant": trainer.args.variant,
        "seed_mode": trainer.args.seed_mode,
    }
    path = os.path.join(path, "model_final.pth")
    torch.save(state, path)

if __name__ == "__main__":
    args = parser.parse_args()
    if args.ald_mode is None:
        args.ald_mode = 'legacy' if args.ald else 'off'
    elif args.ald and args.ald_mode == 'off':
        parser.error('--ald and --ald_mode off conflict')
    args.ald = args.ald_mode != 'off'
    if args.ald or args.confusion_reweight or args.variant != 'full':
        parser.error('Seed correction requires full prototype model, ALD off, unit class weights')
    if args.max_iters < 1 or args.eval_iters < 1 or args.log_iters < 1 or args.num_workers < 1:
        parser.error('iterations and num_workers must be positive')
    if not 0 <= args.proto_margin <= 1:
        parser.error('proto_margin must be between 0 and 1')
    if min(args.w_proto_kd, args.w_proto_sep, args.w_proto_seg) < 0:
        parser.error('prototype loss weights must be nonnegative')
    experiment = Path(args.work_dir)
    if not experiment.is_absolute():
        experiment = PROJECT_ROOT / experiment
    experiment = experiment.resolve()
    if not experiment.is_relative_to(PROJECT_ROOT):
        parser.error('work_dir must remain inside this project')
    args.work_dir = str(experiment / args.task)
    load_ckpt_dir = os.path.join(args.work_dir, "step"+str(args.step - 1), "checkpoints")
    args.work_dir = os.path.join(args.work_dir, "step"+str(args.step))
    args.ckpt_dir = os.path.join(args.work_dir, "checkpoints")
    args.pred_dir = os.path.join(args.work_dir, "predictions")


    torch.cuda.set_device(args.local_rank)
    distributed.init_process_group(backend=args.backend, init_method='env://')
    torch.set_float32_matmul_precision('high')
    torch.backends.cudnn.allow_tf32 = True
    if args.local_rank == 0:
        os.makedirs(args.ckpt_dir, exist_ok=True)
        os.makedirs(args.pred_dir, exist_ok=True)

        setup_logger(filename=os.path.join(args.work_dir, 'train.log'))
        logging.info('Pytorch version: %s' % torch.__version__)
        logging.info("GPU type: %s"%(torch.cuda.get_device_name(0)))
        logging.info('\nargs: %s' % args)
        with open(os.path.join(args.work_dir, 'config.json'), 'w', encoding='utf-8') as handle:
            json.dump(vars(args), handle, indent=2)
    distributed.barrier()
    ## fix random seed
    setup_seed(args.seed)
    trainer = Trainer(args=args)
    if args.step > 0:
        trainer.load_step_ckpt(args.prev_checkpoint or os.path.join(load_ckpt_dir, "model_final.pth"))
    Flag = trainer.train(args=args)
    if args.local_rank == 0 and Flag:  # save best model at the last iteration
        # best model to build incremental steps
        save_ckpt(args.ckpt_dir, trainer)
        logging.info("[!] Checkpoint saved.")
    distributed.barrier()
    distributed.destroy_process_group()
