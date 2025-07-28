import torch
import torch.nn as nn
import torch.nn.functional as F


class AttnBlock(nn.Module):
    def __init__(self, in_ch, num_groups=32):
        super(AttnBlock, self).__init__()
        assert in_ch % num_groups == 0
        self.group_norm = nn.GroupNorm(num_groups, in_ch)
        self.proj_q = nn.Conv2d(in_channels=in_ch, out_channels=in_ch, kernel_size=1, stride=1, padding=0)
        self.proj_k = nn.Conv2d(in_channels=in_ch, out_channels=in_ch, kernel_size=1, stride=1, padding=0)
        self.proj_v = nn.Conv2d(in_channels=in_ch, out_channels=in_ch, kernel_size=1, stride=1, padding=0)
        self.proj = nn.Conv2d(in_channels=in_ch, out_channels=in_ch, kernel_size=1, stride=1, padding=0)

    def init_weights(self):
        for m in [self.proj_q, self.proj_k, self.proj_v, self.proj]:
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)
        nn.init.xavier_uniform_(self.proj.weight, gain=1e-5)

    def forward(self, x, temb=None):
        B, C, H, W = x.shape
        h = self.group_norm(x)

        # 使用1x1卷积生成QKV，不改变空间结构
        q = self.proj_q(h)
        k = self.proj_k(h)
        v = self.proj_v(h)

        # 自注意力计算
        # q、k 被 reshape 成序列格式，每个位置作为 token，长度为 $HW$。
        # torch.bmm(q, k) 实现的是自注意力权重计算 Attention(Q,K)=Softmax(QK^T / sqrt(C))
        # 输出权重矩阵 w 是 [B, HW, HW]，即每个像素点对所有像素的注意力分数
        q = q.permute(0, 2, 3, 1).view(B, H * W, C)  # [B, HW, C]
        k = k.view(B, C, H * W)  # [B, C, HW]
        w = torch.bmm(q, k) * (C ** -0.5)  # [B, HW, HW]
        assert list(w.shape) == [B, H * W, H * W]
        w = F.softmax(w, dim=-1)

        # 应用注意力到 Value
        v = v.permute(0, 2, 3, 1).view(B, H * W, C)  # [B, HW, C]
        h = torch.bmm(w, v)  # [B, HW, C]
        assert list(h.shape) == [B, H * W, C]
        h = h.view(B, H, W, C).permute(0, 3, 1, 2)  # [B, C, H, W]
        h = self.proj(h)

        return x + h


if __name__ == '__main__':
    # 构造一个简单输入
    B, C, H, W = 1, 32, 4, 4
    torch.manual_seed(0)
    x = torch.randn(B, C, H, W)

    # 应用注意力块
    model = AttnBlock(in_ch=C, num_groups=32)
    torch.onnx.export(model, x, "attention.onnx")
