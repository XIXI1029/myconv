import os
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import cv2
import numpy as np
import matplotlib.pyplot as plt

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

        # 读取二值化图像
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        # 读取标签文件中的点坐标
        with open(label_path, 'r') as f:
            lines = f.readlines()
        points = [list(map(float, line.strip().split())) for line in lines]  # 读取所有点坐标
        points = np.array(points, dtype=np.float32)  # 转换为 numpy 数组

        # 转换为模型需要的格式（例如，展平为一维数组）
        target = points.flatten()

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(target, dtype=torch.float32)

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
        self.conv_block1 = ConvBlock(1, 32)  # 输入：1 通道（二值化图像），输出：32 通道
        self.conv_block2 = ConvBlock(32, 64)  # 输入：32 通道，输出：64 通道
        self.fc1 = nn.Linear(64 * 64 * 64, 256)  # 根据输入图像大小调整
        self.fc2 = nn.Linear(256, 8)  # 输出：4 个点的坐标 (x1, y1, x2, y2, x3, y3, x4, y4)

    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = x.view(x.size(0), -1)  # 展平
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 数据路径
images_dir = 'path/to/output/images'  # 替换为二值化图像的路径
labels_dir = 'path/to/output/labels'  # 替换为标签文件的路径

# 数据预处理
transform = transforms.Compose([
    transforms.ToTensor(),
])

# 加载数据集
dataset = QuadrilateralDataset(images_dir, labels_dir, transform=transform)

# 划分训练集和验证集 (7:3)
train_size = int(0.7 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# 创建数据加载器
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# 初始化模型
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = QuadrilateralNet().to(device)

# 定义损失函数和优化器
criterion = nn.MSELoss()  # 使用均方误差损失
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练循环
num_epochs = 10
train_losses = []
val_losses = []

for epoch in range(num_epochs):
    # 训练阶段
    model.train()
    train_loss = 0.0
    for images, labels in train_dataloader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * images.size(0)
    train_loss /= len(train_dataset)
    train_losses.append(train_loss)

    # 验证阶段
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for images, labels in val_dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            val_loss += criterion(outputs, labels).item() * images.size(0)
    val_loss /= len(val_dataset)
    val_losses.append(val_loss)

    # 打印训练和验证损失
    print(f'Epoch [{epoch+1}/{num_epochs}], '
          f'Train Loss: {train_loss:.4f}, '
          f'Validation Loss: {val_loss:.4f}')

# 保存模型
torch.save(model.state_dict(), 'quadrilateral_net.pth')

# 绘制损失曲线
plt.figure(figsize=(10, 5))
plt.plot(range(1, num_epochs + 1), train_losses, label='Train Loss', marker='o')
plt.plot(range(1, num_epochs + 1), val_losses, label='Validation Loss', marker='o')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.savefig('loss_curve.png')
plt.show()