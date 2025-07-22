import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

from .attention import AttnBlock
from .swish import Swish


class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, tdim, dropout, num_groups=32, attn=False):
        super(ResBlock, self).__init__()

        # GroupNorm：分组归一化，有助于加快训练和稳定性；
        # Swish()：激活函数，效果类似 ReLU 但更平滑；
        # Conv2d：3×3 卷积，提取特征。
        self.block1 = nn.Sequential(
            nn.GroupNorm(num_groups, in_ch),
            Swish(),
            nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=3, stride=1, padding=1),
        )

        # 将时间嵌入向量 temb 映射到 out_ch 维度，并通过 Swish 激活。
        # 这个结果稍后会加到 block1 的输出上（会 broadcast 到特征图维度）。
        self.temb_proj = nn.Sequential(
            Swish(),
            nn.Linear(tdim, out_ch),
        )

        # 归一化、激活；
        # Dropout 提供正则化；
        # 1×1 卷积：调整通道而不改变空间维度（注意：1×1 是轻量操作，类似于通道上的投影）。
        self.block2 = nn.Sequential(
            nn.GroupNorm(num_groups, out_ch),
            Swish(),
            nn.Dropout(dropout),
            nn.Conv2d(in_channels=out_ch, out_channels=out_ch, kernel_size=1, stride=1, padding=0),
        )

        if in_ch != out_ch:
            self.shortcut = nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=1, stride=1, padding=0)
        else:
            self.shortcut = nn.Identity()

        if attn:
            self.attn = AttnBlock(out_ch)
        else:
            self.attn = nn.Identity()

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
        nn.init.xavier_uniform_(self.block2[-1].weight, gain=1e-5)

    def forward(self, x, temb):
        h = self.block1(x)

        # 这个模块将时间嵌入 temb（形状为 [B, tdim]）变换成 [B, out_ch] 的向量
        # [:, :, None, None] —— reshape 成 [B, C, 1, 1]
        # 将 [B, out_ch] 扩展为 [B, out_ch, 1, 1]，使其能够和 2D 特征图广播相加。
        # 所以如果 h 的形状是 [B, C, H, W]，这个扩展后的张量会自动在 (H, W) 维度上 broadcast。

        # 将通过全连接层（self.temb_proj）处理后的时间嵌入，作为全局调制因子加到当前的
        # 特征图 h 上，方式是每个通道增加一个常数偏置，这个偏置是从 temb 映射来的。
        h += self.temb_proj(temb)[:, :, None, None]
        h = self.block2(h)

        h = h + self.shortcut(x)
        h = self.attn(h)
        return h


if __name__ == '__main__':
    in_ch = 1
    out_ch = 32
    batch_size = 128
    tdim = 128
    dropout = 0.2
    num_groups = 1

    # 下载 MNIST 数据
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    dataset = torchvision.datasets.MNIST(root='./mnist', train=True, transform=transform, download=True)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # 取一张图像
    image, label = next(iter(loader))  # image.shape: [batch_size, 1, 28, 28]

    resblock = ResBlock(in_ch=in_ch, out_ch=out_ch, tdim=tdim, dropout=dropout, num_groups=num_groups, attn=False)

    # 时间嵌入张量 (B, tdim)
    temb = torch.randn(batch_size, tdim)  # [batch_size, tdim]

    # 运行 forward
    out = resblock(image, temb)
