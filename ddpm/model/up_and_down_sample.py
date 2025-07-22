import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms


class DownSample(nn.Module):
    def __init__(self, in_ch):
        super(DownSample, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_ch, out_channels=in_ch, kernel_size=3, stride=2, padding=1)

    def init_weights(self):
        nn.init.xavier_uniform_(self.conv.weight).float()
        nn.init.zeros_(self.conv.bias).float()

    def forward(self, x, temb):
        return self.conv(x)


class UpSample(nn.Module):
    def __init__(self, in_ch):
        super(UpSample, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_ch, out_channels=in_ch, kernel_size=3, stride=1, padding=1)

    def init_weights(self):
        nn.init.xavier_uniform_(self.conv.weight).float()
        nn.init.zeros_(self.conv.bias).float()

    def forward(self, x, temb):
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        return self.conv(x)


if __name__ == '__main__':
    # 下载 MNIST 数据
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    dataset = torchvision.datasets.MNIST(root='./mnist', train=True, transform=transform, download=True)
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)

    # 取一张图像
    image, label = next(iter(loader))  # image.shape: [1, 1, 28, 28]

    # 初始化模块
    down = DownSample(1)
    up = UpSample(1)
    down.init_weights()
    up.init_weights()

    # 应用模块
    with torch.no_grad():
        temb = []
        x_down = down(image, temb)
        x_up = up(x_down, temb)


    # 可视化结果
    def show(img, title):
        img = img.squeeze().numpy()
        plt.imshow(img, cmap='gray')
        plt.title(title)
        plt.axis('off')


    plt.figure(figsize=(10, 3))
    plt.subplot(1, 3, 1)
    show(image, 'Original (28x28)')
    plt.subplot(1, 3, 2)
    show(x_down, 'DownSampled (14x14)')
    plt.subplot(1, 3, 3)
    show(x_up, 'UpSampled (28x28)')
    plt.tight_layout()
    plt.show()
