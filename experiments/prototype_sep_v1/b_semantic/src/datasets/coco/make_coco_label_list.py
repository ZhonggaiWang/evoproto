import os
import numpy as np
from PIL import Image


category_mapping = {
    0: [8, 10, 11, 13, 14, 15, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42,
        43, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 65, 70, 73, 74, 75, 76, 77, 78,
        79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90],  # COCO 类别
    1: [1, 2, 3, 4, 5, 6, 7, 9, 16, 17, 18, 19, 20, 21, 44, 62, 63, 64, 67, 72, 12, 26, 29, 30, 45, 66, 68, 69, 71, 83, 91]  # VOC 类别
}

# 获取VOC类别标签
voc_categories = set(category_mapping[1])


name_list_cleaned = []

label_dir = "/data/fangkai/coco/annotations/val2017"
save_path = '/home/zhonggai/python-work-space/WSSS/Incremental WSSS/Toco_incremental/KKK47/datasets/coco'
stage = 'val'

# 获取所有 PNG 文件
label_files = [f for f in os.listdir(label_dir) if f.endswith(".png")]

# 存储 one-hot 标签
one_hot_dict = {}

# 处理每张标签图片
for file_name in label_files:
    img_path = os.path.join(label_dir, file_name)
    img_name = os.path.splitext(file_name)[0]  # 去掉扩展名

    print(f"Processing {img_name}...")

    # 用 PIL 读取，转换为灰度
    label = Image.open(img_path).convert("P") 
    label = np.array(label)
    print(label)
    print(f"Unique pixel values in {img_name}: {np.unique(label)}")
        

    # 获取图像中的VOC类别标签
    voc_labels_in_image = np.unique(label[np.isin(label, list(voc_categories))])
    
    # 检查是否包含VOC类别的标签
    if len(voc_labels_in_image) > 0:
        # 打印图像路径和类别标签
        print(f"Image: {img_name}, VOC Categories: {voc_labels_in_image}")

    
    # 如果不包含VOC类别，添加到清洗后的name_list
    if len(voc_labels_in_image) == 0:
        name_list_cleaned.append(img_name)
            


output_file = save_path + '/' +str(stage) + '_filtered.txt'
with open(output_file, 'w') as f:
    for name in name_list_cleaned:
        f.write(name + '\n')