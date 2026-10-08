export CUDA_VISIBLE_DEVICES=0,1
task=coco2voc
#python -m torch.distributed.launch --nproc_per_node=2 --master_port=29514 scripts/coco_seg.py --step 0 --max_iters 80000 --lr 6e-5 --task ${task} --work_dir output_voc
for t in 1; do
  python -m torch.distributed.launch --nproc_per_node=2 --master_port=29516 scripts/coco_seg.py --step ${t} --max_iters 22000 --lr 2e-5 --task ${task} --work_dir output_voc
done
