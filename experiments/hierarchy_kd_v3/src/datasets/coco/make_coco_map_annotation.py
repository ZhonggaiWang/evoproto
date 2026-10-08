import os
import numpy as np
import cv2
from tqdm import tqdm

# 配置
ANNOTATION_DIR = "/root/code/ToCo-type3/coco/annotations_my/val2017"
OUTPUT_DIR = "/root/code/ToCo-type3/coco/annotations_map/val2017"
IGNORE_CLASSES = {12, 26, 29, 30, 45, 66, 68, 69, 71, 83, 91}
VOC_CLASSES = {1, 2, 3, 4, 5, 6, 7, 9, 16, 17, 18, 19, 20, 21, 44, 62, 63, 64, 67, 72}

# 获取所有类别
ALL_CLASSES = set(range(1, 92))  # COCO 类别范围 1-91
valid_classes = sorted(ALL_CLASSES - IGNORE_CLASSES)
non_voc_classes = [c for c in valid_classes if c not in VOC_CLASSES]
voc_classes_sorted = sorted(VOC_CLASSES)

# 构建映射：
class_mapping = {0: 0}  # 背景 0 保持为 0

# 处理非VOC类别 (1-60)
for i, c in enumerate(non_voc_classes, start=1):  # 确保从 1 开始
    class_mapping[c] = i

# 处理VOC类别 (61-80)
for i, c in enumerate(voc_classes_sorted, start=61):
    class_mapping[c] = i

# 忽略类别映射为背景 0
for c in IGNORE_CLASSES:
    class_mapping[c] = 0

print(class_mapping)
# 确保输出目录存在
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 读取 name_list.txt 获取需要处理的文件名
with open("/root/code/ToCo-type3/datasets/coco/val_step1_filtered.txt", "r") as f:
    name_list = {line.strip() + ".png" for line in f} 

# 存储文件名和对应的 one-hot 标签
cls_labels_dict = {}

# 处理所有 PNG 注释文件
for filename in tqdm(os.listdir(ANNOTATION_DIR)):
    filepath = os.path.join(ANNOTATION_DIR, filename)
    output_path = os.path.join(OUTPUT_DIR, filename)

    # 读取 PNG 文件
    annotation = cv2.imread(filepath, cv2.IMREAD_UNCHANGED)
    if annotation is None:
        print(f"无法读取 {filename}")
        continue
    # 映射类别 ID
    mapped_annotation = np.vectorize(lambda x: class_mapping.get(x, 0))(annotation)

    # 获取图像中存在的类别（去除背景和255类）
    present_classes = set(mapped_annotation[mapped_annotation > 0].flatten())
    present_classes.discard(255)  # 移除 255 类

    # 创建 one-hot 向量 (长度 80)
    one_hot_label = np.zeros(80, dtype=np.uint8)
    for cls in present_classes:
        one_hot_label[cls - 1] = 1  # 标记对应类别为 1

    # 将文件名和对应的 one-hot 标签存入字典
    cls_labels_dict[filename] = one_hot_label

    # 保存映射后的图片
    # cv2.imwrite(output_path, mapped_annotation.astype(np.uint8))

# 将字典保存为 .npy 文件
np.save("/root/code/ToCo-type3/datasets/coco/val_step1/cls_labels_dict.npy", cls_labels_dict)
np.save("/root/code/ToCo-type3/datasets/coco/val/cls_labels_dict.npy", cls_labels_dict)

print("处理完成！")
