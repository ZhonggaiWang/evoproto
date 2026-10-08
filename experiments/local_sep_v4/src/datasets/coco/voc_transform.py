from PIL import Image
import numpy as np
import os

def adjust_labels_and_generate_onehot(image_path):
    """ 调整标签并生成one-hot向量 """
    # 加载原始分割标签
    image = Image.open(image_path)
    
    # 转换为numpy数组
    label_array = np.array(image)
    
    # 对前景类标签加60，背景类（0）和忽略类（255）保持不变
    adjusted_label_array = np.where((label_array != 0) & (label_array != 255), label_array + 60, label_array)
    adjusted_label_array = np.clip(adjusted_label_array, 0, 255)
    
    # 获取前景类（去掉背景0和忽略类255）
    unique_classes = np.unique(adjusted_label_array)
    unique_classes = unique_classes[(unique_classes != 0) & (unique_classes != 255)]
    
    # 生成one-hot向量（长度80）
    onehot_vector = np.zeros(80, dtype=np.uint8)
    for cls in unique_classes:
        onehot_vector[cls - 1] = 1  # 类别从1开始，所以索引需要 -1

    return onehot_vector

# 设置数据集路径
input_folder = '/home/zhonggai/python-work-space/WSSS/Incremental WSSS/Toco_incremental/KKK47/VOCdevkit/VOC2012/SegmentationClassAug'  # 输入标签文件夹路径
output_npy_path = '/home/zhonggai/python-work-space/WSSS/Incremental WSSS/Toco_incremental/KKK47/datasets/coco/_'  # 统一保存的one-hot npy文件路径

# 存储所有图片的 one-hot 向量
onehot_dict = {}

# 遍历数据集中的每个标签图像
for filename in os.listdir(input_folder):
    if filename.endswith('.png'):  # 只处理PNG文件
        image_path = os.path.join(input_folder, filename)
        
        # 生成one-hot编码
        onehot_vector = adjust_labels_and_generate_onehot(image_path)
        
        # 以文件名（不含扩展名）作为索引存储
        onehot_dict[filename.replace('.png', '')] = onehot_vector

# 保存到 .npy 文件
np.save(output_npy_path, onehot_dict)

print(f"One-hot vectors saved to {output_npy_path}")