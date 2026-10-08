import os
import numpy as np
import cv2
from PIL import Image
COCO_IDX_MAPPING = {
    1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6, 8: 7, 9: 8, 10: 9, 11: 10, 
    13: 11, 14: 12, 15: 13, 16: 14, 17: 15, 18: 16, 19: 17, 20: 18, 21: 19, 22: 20, 
    23: 21, 24: 22, 25: 23, 27: 24, 28: 25, 31: 26, 32: 27, 33: 28, 34: 29, 35: 30, 
    36: 31, 37: 32, 38: 33, 39: 34, 40: 35, 41: 36, 42: 37, 43: 38, 44: 39, 46: 40, 
    47: 41, 48: 42, 49: 43, 50: 44, 51: 45, 52: 46, 53: 47, 54: 48, 55: 49, 56: 50, 
    57: 51, 58: 52, 59: 53, 60: 54, 61: 55, 62: 56, 63: 57, 64: 58, 65: 59, 67: 60, 
    70: 61, 72: 62, 73: 63, 74: 64, 75: 65, 76: 66, 77: 67, 78: 68, 79: 69, 80: 70, 
    81: 71, 82: 72, 84: 73, 85: 74, 86: 75, 87: 76, 88: 77, 89: 78, 90: 79
}

IGNORE_CLASSES = {12, 26, 29, 30, 45, 66, 68, 69, 71, 83, 91}

label_dir = "/data/fangkai/coco/annotations_my/train2017"
save_path = "/home/zhonggai/python-work-space/WSSS/Incremental WSSS/Toco_incremental/KKK47/datasets/coco/cls_labels_onehot_new.npy"

label_files = [f for f in os.listdir(label_dir) if f.endswith(".png")]

one_hot_dict = {}

for file_name in label_files:
    img_path = os.path.join(label_dir, file_name)
    img_name = os.path.splitext(file_name)[0]  # 去掉扩展名

    print(f"Processing {img_name}...")

    # 用 PIL 读取，转换为灰度
    label = Image.open(img_path).convert("P") 
    label = np.array(label)

    print(f"Unique pixel values in {img_name}: {np.unique(label)}")

    # 去除 0 和 255
    mask = (label != 0) & (label != 255)
    filtered_labels = label[mask]

    # 获取 unique 类别，并排除 IGNORE_CLASSES
    unique_classes = np.unique(filtered_labels)
    unique_classes = [cls for cls in unique_classes if cls not in IGNORE_CLASSES]

    # 进行类别 ID 映射
    mapped_classes = [COCO_IDX_MAPPING[cls] for cls in unique_classes if cls in COCO_IDX_MAPPING]

    print(f"Original IDs: {unique_classes}")
    print(f"Mapped IDs: {mapped_classes}")

    # 生成 one-hot 标签
    one_hot = np.zeros(80, dtype=np.uint8)  # 80 类（COCO 数据集）
    one_hot[mapped_classes] = 1

    # 存储
    one_hot_dict[img_name] = one_hot

# 保存为 .npy 文件
np.save(save_path, one_hot_dict)
print(f"One-hot labels saved to {save_path}")
