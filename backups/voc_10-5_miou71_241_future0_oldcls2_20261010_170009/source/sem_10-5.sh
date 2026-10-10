#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
cd -- "$project_root"

export CUDA_VISIBLE_DEVICES="${GPU_ID:-0}"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export PYTHONDONTWRITEBYTECODE=1
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MPLBACKEND=Agg
temp_root="$project_root/.runtime/tmp"
export TRITON_CACHE_DIR="$project_root/.runtime/cache/triton"
mkdir -p -- "$temp_root" "$TRITON_CACHE_DIR"
# Multiprocessing socket names must fit AF_UNIX's 108-byte path limit. The
# short alias keeps temporary files inside this project even for deep snapshots.
temp_parent="$(mktemp -d /tmp/evoproto.XXXXXXXX)"
ln -s -- "$temp_root" "$temp_parent/tmp"
export TMPDIR="$temp_parent/tmp"
export EVOPROTO_TMPDIR="$TMPDIR"
trap 'rm -f -- "$temp_parent/tmp"; rmdir -- "$temp_parent"' EXIT

python_bin="${PYTHON_BIN:-/home/fangkai/miniconda3/envs/py39/bin/python}"
data_folder="${DATA_FOLDER:-/data/DatasetCollection/VOCdevkit/VOC2012}"
seg_label_dir="${SEG_LABEL_DIR:-$data_folder/SegmentationClassAug}"
val_label_dir="${VAL_LABEL_DIR:-$data_folder/SegmentationClass}"
work_dir="${WORK_DIR:-$project_root/runs/prototype_transfer_10-5_gpu${GPU_ID:-0}_$(TZ=Asia/Hong_Kong date +%Y%m%d_%H%M%S)}"
batch_size="${BATCH_SIZE:-8}"
num_workers="${NUM_WORKERS:-4}"
proto_sep_mode="${PROTO_SEP_MODE:-paper}"
proto_margin="${PROTO_MARGIN:-0.0}"
ald_mode="${ALD_MODE:-fixed}"
old_cls_threshold="${OLD_CLS_THRESHOLD:-2.0}"
ald_threshold="${ALD_THRESHOLD:-0.0}"
future_class_label="${FUTURE_CLASS_LABEL:-255}"
confusion_temperature="${CONFUSION_TEMPERATURE:-0.5}"
w_seg="${W_SEG:-0.2}"
proto_temperature="${PROTO_TEMPERATURE:-0.01}"
layer_decay="${LAYER_DECAY:-0.9}"
proto_kd_mode="${PROTO_KD_MODE:-direction}"
lr_warmup_iters="${LR_WARMUP_ITERS:-2000}"

# Every invocation starts a fresh chain; never overwrite an existing stage.
for step in 0 1 2; do
    if [[ -e "$work_dir/10-5/step$step" ]]; then
        echo "Existing stage directory: $work_dir/10-5/step$step" >&2
        exit 1
    fi
done
mkdir -p -- "$work_dir"

for step in 0 1 2; do
    if [[ "$step" == 0 ]]; then
        iterations=20000
        learning_rate=6e-5
        predecessor=()
    else
        iterations=8000
        learning_rate=2e-5
        previous="$work_dir/10-5/step$((step - 1))/checkpoints/model_final.pth"
        [[ -s "$previous" ]] || { echo "Missing predecessor: $previous" >&2; exit 1; }
        predecessor=(--prev_checkpoint "$previous")
    fi
    echo "Starting VOC 10-5 step $step: GPU $CUDA_VISIBLE_DEVICES, batch $batch_size, $iterations iterations"
    # Explicit loopback also works on hosts whose numeric name is parsed as an IP.
    master_port="${MASTER_PORT:-$("$python_bin" -B -c 'import socket; s = socket.socket(); s.bind(("127.0.0.1", 0)); print(s.getsockname()[1]); s.close()')}"
    "$python_bin" -B -m torch.distributed.run --nnodes=1 --nproc_per_node=1 \
        --master_addr=127.0.0.1 --master_port="$master_port" \
        scripts/dist_train_voc_seg_neg.py \
        --task 10-5 --step "$step" --max_iters "$iterations" --lr "$learning_rate" \
        --warmup_iters "$lr_warmup_iters" \
        --work_dir "$work_dir" --data_folder "$data_folder" \
        --seg_label_dir "$seg_label_dir" --val_label_dir "$val_label_dir" \
        --spg "$batch_size" --num_workers "$num_workers" --seed 0 --save_ckpt \
        --old_cls_threshold "$old_cls_threshold" --ald --ald_mode "$ald_mode" --ald_threshold "$ald_threshold" --confusion_reweight \
        --future_class_label "$future_class_label" \
        --loss_warmup_iters 2000 --confusion_start_iter 4000 --confusion_interval 2000 --confusion_temperature "$confusion_temperature" \
        --proto_sep_mode "$proto_sep_mode" --proto_margin "$proto_margin" \
        --w_seg "$w_seg" --proto_temperature "$proto_temperature" \
        --layer_decay "$layer_decay" --proto_kd_mode "$proto_kd_mode" --cpa_skip_background \
        "${predecessor[@]}" > "$work_dir/launcher-step$step.log" 2>&1
    [[ -s "$work_dir/10-5/step$step/checkpoints/model_final.pth" ]] || {
        echo "Step $step finished without its final checkpoint" >&2; exit 1;
    }
    echo "Completed step $step"
done
echo "VOC 10-5 chain completed: $work_dir/10-5"
