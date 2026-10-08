import os
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm

mask_dir = "/home/zhonggai/python-work-space/WSSS/Incremental WSSS/Toco_incremental/KKK47/coco/annotations/val2017"  # segmentation 标签存放目录
output_file = "/home/zhonggai/python-work-space/WSSS/Incremental WSSS/Toco_incremental/KKK47/datasets/coco/namelist.txt"

# 需要忽略的类别ID
IGNORE_CLASSES = {12, 26, 29, 30, 45, 66, 68, 69, 71, 83, 91}

valid_img_names = []

# 遍历所有 segmentation mask
mask_files = sorted(os.listdir(mask_dir))

for mask_file in tqdm(mask_files, desc="Processing masks"):
    mask_path = os.path.join(mask_dir, mask_file)

    # 读取 PNG 格式的 segmentation mask
    mask = np.array(Image.open(mask_path))

    # 检查是否包含 IGNORE_CLASSES
    if not np.isin(mask, list(IGNORE_CLASSES)).any():
        # 如果 mask 没有包含忽略类别，则记录该图片名
        img_name = mask_file.replace(".png", "")  # 假设对应的图片是 jpg 格式
        valid_img_names.append(img_name)

# 保存结果到 namelist.txt
with open(output_file, "w") as f:
    for name in valid_img_names:
        f.write(name + "\n")

print(f"共筛选出 {len(valid_img_names)} 张有效图片，已保存至 {output_file}")
