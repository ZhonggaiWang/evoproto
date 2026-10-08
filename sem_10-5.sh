master_port=29519
export CUDA_VISIBLE_DEVICES=2
task=10-5
python -m torch.distributed.launch --nproc_per_node=1 --master_port=${master_port} scripts/dist_train_voc_seg_neg.py --step 0 --max_iters 20000 --lr 6e-5 --task ${task} --work_dir output_voc
for t in 1 2; do
  python -m torch.distributed.launch --nproc_per_node=1 --master_port=${master_port} scripts/dist_train_voc_seg_neg.py --step ${t} --max_iters 8000 --lr 2e-5 --task ${task} --work_dir output_voc
done
