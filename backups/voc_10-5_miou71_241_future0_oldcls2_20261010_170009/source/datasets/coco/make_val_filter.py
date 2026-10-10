import os
import numpy as np
import cv2
from tqdm import tqdm

# 配置路径
ANNOTATION_DIR = "/root/code/ToCo-type3/coco/annotations_map/val2017"
OUTPUT_TXT = "/root/code/ToCo-type3/datasets/coco/val_step1_filtered.txt"

valid_names = []

# 遍历标注文件夹
# 遍历标注文件夹
for filename in tqdm(os.listdir(ANNOTATION_DIR)):
    filepath = os.path.join(ANNOTATION_DIR, filename)
    label = cv2.imread(filepath, cv2.IMREAD_UNCHANGED)  # 保留原始类别值

    # 检查是否所有类别都在 0 到 80 范围内，且不全是 0
    if np.any((label >= 1) & (label <= 80)):
        valid_names.append(filename.replace(".png", ""))


# 写入符合条件的图像编号
with open(OUTPUT_TXT, "w") as f:
    for name in valid_names:
        f.write(name + "\n")

print(f"完成，共保存 {len(valid_names)} 个图像编号到 {OUTPUT_TXT}")

