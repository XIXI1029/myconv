class QuadrilateralDataset(Dataset):
    def __init__(self, images_dir, labels_dir, transform=None):
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.transform = transform
        self.image_files = []

        # 过滤掉不是四个点的样本
        for img_name in os.listdir(images_dir):
            if not img_name.endswith('.jpg'):
                continue

            label_path = os.path.join(labels_dir, img_name.replace('.jpg', '.txt'))
            if not os.path.exists(label_path):
                continue

            # 检查点数量是否为 4
            with open(label_path, 'r') as f:
                lines = f.readlines()
            if len(lines) == 4:  # 只有四个点的样本才会被加入
                self.image_files.append(img_name)

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.images_dir, img_name)
        label_path = os.path.join(self.labels_dir, img_name.replace('.jpg', '.txt'))

        # 读取二值化图像
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        # 读取标签文件中的点坐标
        with open(label_path, 'r') as f:
            lines = f.readlines()
        points = [list(map(float, line.strip().split())) for line in lines]  # 读取所有点坐标
        points = np.array(points, dtype=np.float32)  # 转换为 numpy 数组

        # 转换为模型需要的格式（展平为一维数组）
        target = points.flatten()

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(target, dtype=torch.float32)