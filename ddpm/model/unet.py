import torch
import torch.nn as nn

from .resnet import ResBlock
from .swish import Swish
from .time_embedding import TimeEmbedding
from .up_and_down_sample import DownSample, UpSample


class UNet(nn.Module):
    def __init__(self, T, ch, ch_mult, attn, num_res_blocks, dropout):
        super(UNet, self).__init__()
        assert all([i < len(ch_mult) for i in attn]), 'attn index out of bound'
        tdim = ch * 4
        self.time_embedding = TimeEmbedding(T, ch, tdim)

        # head
        self.head = nn.Conv2d(3, out_channels=ch, kernel_size=3, stride=1, padding=1)

        # down sample
        self.down_blocks = nn.ModuleList()
        chs = [ch]
        now_ch = ch
        for i, mult in enumerate(ch_mult):
            out_ch = ch * mult
            for _ in range(num_res_blocks):
                self.down_blocks.append(
                    ResBlock(in_ch=now_ch, out_ch=out_ch, tdim=tdim, dropout=dropout, attn=(i in attn))
                )
                now_ch = out_ch
                chs.append(now_ch)

            if i != len(ch_mult) - 1:
                self.down_blocks.append(DownSample(now_ch))
                chs.append(now_ch)

        # middle
        self.middle_blocks = nn.ModuleList([
            ResBlock(now_ch, now_ch, tdim=tdim, dropout=dropout, attn=True),
            ResBlock(now_ch, now_ch, tdim=tdim, dropout=dropout, attn=False),
        ])

        # up sample
        self.up_blocks = nn.ModuleList()
        for i, mult in reversed(list(enumerate(ch_mult))):
            out_ch = ch * mult
            for _ in range(num_res_blocks + 1):
                self.up_blocks.append(ResBlock(
                    in_ch=chs.pop() + now_ch, out_ch=out_ch, tdim=tdim, dropout=dropout, attn=(i in attn)
                ))
                now_ch = out_ch
            if i != 0:
                self.up_blocks.append(UpSample(now_ch))

        # tail
        self.tail = nn.Sequential(
            nn.GroupNorm(32, now_ch),
            Swish(),
            nn.Conv2d(now_ch, 3, 3, stride=1, padding=1)
        )
        self.init_weights()

    def init_weights(self):
        nn.init.xavier_uniform_(self.head.weight)
        nn.init.zeros_(self.head.bias)
        nn.init.xavier_uniform_(self.tail[-1].weight, gain=1e-5)
        nn.init.zeros_(self.tail[-1].bias)

    def forward(self, x, t):
        # Timestep embedding
        temb = self.time_embedding(t)
        # Down
        h = self.head(x)
        hs = [h]
        for layer in self.down_blocks:
            h = layer(h, temb)
            hs.append(h)
        # Middle
        for layer in self.middle_blocks:
            h = layer(h, temb)
        # Up
        for layer in self.up_blocks:
            if isinstance(layer, ResBlock):
                h = torch.cat([h, hs.pop()], dim=1)
            h = layer(h, temb)
        h = self.tail(h)

        assert len(hs) == 0
        return h


if __name__ == '__main__':
    batch_size = 8
    model = UNet(
        T=1000, ch=128, ch_mult=[1, 2, 2, 2], attn=[1],
        num_res_blocks=2, dropout=0.1)
    x = torch.randn(batch_size, 3, 32, 32)
    t = torch.randint(1000, (batch_size,))
    y = model(x, t)
