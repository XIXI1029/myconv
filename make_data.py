import os
import cv2
import numpy as np

def normalize_to_pixel_coords(normalized_coords, img_width, img_height):
    x_center, y_center, width, height = normalized_coords
    x_center *= img_width
    y_center *= img_height
    width *= img_width
    height *= img_height
    x_min = int(x_center - width / 2)
    y_min = int(y_center - height / 2)
    x_max = int(x_center + width / 2)
    y_max = int(y_center + height / 2)
    return x_min, y_min, x_max, y_max

def generate_binary_images_and_labels(images_dir, labels_dir, output_images_dir, output_labels_dir):
    if not os.path.exists(output_images_dir):
        os.makedirs(output_images_dir)
    if not os.path.exists(output_labels_dir):
        os.makedirs(output_labels_dir)

    for img_name in os.listdir(images_dir):
        if not img_name.endswith('.jpg'):
            continue

        img_path = os.path.join(images_dir, img_name)
        label_path = os.path.join(labels_dir, img_name.replace('.jpg', '.txt'))

        if not os.path.exists(label_path):
            continue

        img = cv2.imread(img_path)
        img_height, img_width, _ = img.shape

        with open(label_path, 'r') as f:
            lines = f.readlines()

        for i, line in enumerate(lines):
            parts = line.strip().split()
            class_id = int(parts[0])
            if class_id not in [0, 2]:  # Only process DOOR (0) and FRONT (2)
                continue

            normalized_coords = list(map(float, parts[1:5]))
            x_min, y_min, x_max, y_max = normalize_to_pixel_coords(normalized_coords, img_width, img_height)

            # Create binary image
            binary_img = np.zeros((img_height, img_width), dtype=np.uint8)
            cv2.rectangle(binary_img, (x_min, y_min), (x_max, y_max), 255, -1)

            # Save binary image
            output_img_name = f"{img_name.replace('.jpg', '')}_{i}.jpg"
            output_img_path = os.path.join(output_images_dir, output_img_name)
            cv2.imwrite(output_img_path, binary_img)

            # Save label
            output_label_name = f"{img_name.replace('.jpg', '')}_{i}.txt"
            output_label_path = os.path.join(output_labels_dir, output_label_name)
            with open(output_label_path, 'w') as f_label:
                f_label.write(f"{x_min} {y_min} {x_max} {y_max}")

# Example usage
images_dir = 'path/to/images'
labels_dir = 'path/to/labels'
output_images_dir = 'path/to/output/images'
output_labels_dir = 'path/to/output/labels'

generate_binary_images_and_labels(images_dir, labels_dir, output_images_dir, output_labels_dir)