from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import cv2
import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim

# 定义数据集类
class QuadrilateralDataset(Dataset):
    def __init__(self, images_dir, labels_dir, transform=None):
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.transform = transform
        self.image_files = [f for f in os.listdir(images_dir) if f.endswith('.jpg')]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.images_dir, img_name)
        label_path = os.path.join(self.labels_dir, img_name.replace('.jpg', '.txt'))

        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        with open(label_path, 'r') as f:
            coords = list(map(float, f.readline().strip().split()))

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(coords, dtype=torch.float32)

# 定义卷积块
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        x = self.pool(x)
        return x

# 定义神经网络
class QuadrilateralNet(nn.Module):
    def __init__(self):
        super(QuadrilateralNet, self).__init__()
        self.conv_block1 = ConvBlock(1, 32)  # Input: 1 channel (binary image), Output: 32 channels
        self.conv_block2 = ConvBlock(32, 64)  # Input: 32 channels, Output: 64 channels
        self.fc1 = nn.Linear(64 * 64 * 64, 256)  # Adjust based on input image size
        self.fc2 = nn.Linear(256, 8)  # Output: 4 coordinates (x1, y1, x2, y2, x3, y3, x4, y4)

    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 数据路径
images_dir = 'path/to/output/images'  # 替换为你的二值化图像路径
labels_dir = 'path/to/output/labels'  # 替换为你的标签路径

# 数据预处理
transform = transforms.Compose([
    transforms.ToTensor(),
])

# 加载数据集
dataset = QuadrilateralDataset(images_dir, labels_dir, transform=transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# 初始化模型
model = QuadrilateralNet()

# 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练循环
num_epochs = 10
for epoch in range(num_epochs):
    for images, labels in dataloader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# 保存模型
torch.save(model.state_dict(), 'quadrilateral_net.pth')