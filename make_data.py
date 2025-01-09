import os
import json
import cv2
import numpy as np

def generate_binary_images_and_labels(images_dir, jsons_dir, output_images_dir, output_labels_dir):
    if not os.path.exists(output_images_dir):
        os.makedirs(output_images_dir)
    if not os.path.exists(output_labels_dir):
        os.makedirs(output_labels_dir)

    for json_name in os.listdir(jsons_dir):
        if not json_name.endswith('.json'):
            continue

        json_path = os.path.join(jsons_dir, json_name)
        with open(json_path, 'r') as f:
            data = json.load(f)

        image_path = os.path.join(images_dir, data['imagePath'])
        if not os.path.exists(image_path):
            continue

        img = cv2.imread(image_path)
        img_height, img_width, _ = img.shape

        # 遍历所有形状
        for i, shape in enumerate(data['shapes']):
            label = shape['label']
            if label not in ['DOOR', 'FRONT']:  # 只处理 DOOR 和 FRONT
                continue

            # 创建二值化图像
            binary_img = np.zeros((img_height, img_width), dtype=np.uint8)

            # 获取多边形点坐标
            points = np.array(shape['points'], dtype=np.int32)
            cv2.fillPoly(binary_img, [points], 255)  # 在二值化图像中绘制多边形

            # 保存二值化图像
            output_img_name = f"{json_name.replace('.json', '')}_{label}_{i}.jpg"
            output_img_path = os.path.join(output_images_dir, output_img_name)
            cv2.imwrite(output_img_path, binary_img)

            # 保存标签文件
            output_label_name = f"{json_name.replace('.json', '')}_{label}_{i}.txt"
            output_label_path = os.path.join(output_labels_dir, output_label_name)
            with open(output_label_path, 'w') as f_label:
                for point in points:
                    f_label.write(f"{point[0]} {point[1]}\n")  # 保存多边形点坐标

# Example usage
images_dir = 'path/to/images'  # 替换为你的原始图片路径
jsons_dir = 'path/to/jsons'  # 替换为你的 JSON 文件路径
output_images_dir = 'path/to/output/images'  # 替换为输出二值化图像的路径
output_labels_dir = 'path/to/output/labels'  # 替换为输出标签文件的路径

generate_binary_images_and_labels(images_dir, jsons_dir, output_images_dir, output_labels_dir)