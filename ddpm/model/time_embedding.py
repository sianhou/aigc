import math

import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from .swish import Swish


class TimeEmbedding(nn.Module):
    def __init__(self, T, d_model, dim):
        super(TimeEmbedding, self).__init__()
        assert d_model % 2 == 0
        emb = torch.arange(0, d_model, step=2) / d_model * math.log(10000)
        emb = torch.exp(-emb)
        pos = torch.arange(T).float()
        emb = pos[:, None] * emb[None, :]
        assert list(emb.shape) == [T, d_model // 2]
        emb = torch.stack([torch.sin(emb), torch.cos(emb)], dim=-1)
        assert list(emb.shape) == [T, d_model // 2, 2]
        emb = emb.view(T, d_model)

        self.time_embedding = nn.Sequential(
            nn.Embedding.from_pretrained(emb),
            nn.Linear(d_model, dim),
            Swish(),
            nn.Linear(dim, dim),
        )
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, t):
        emb = self.time_embedding(t)
        return emb


if __name__ == '__main__':
    T = 1000
    d_model = 64

    # plot emb
    # 构造时间编码（参考你的代码）
    emb = torch.arange(0, d_model, step=2) / d_model * math.log(10000)
    emb = torch.exp(-emb)
    pos = torch.arange(T).float()
    emb = pos[:, None] * emb[None, :]
    sin_emb = torch.sin(emb)
    cos_emb = torch.cos(emb)
    pe = torch.stack([sin_emb, cos_emb], dim=-1).view(T, d_model)

    plt.figure(figsize=(10, 6))
    plt.imshow(pe.numpy(), aspect='auto', cmap='viridis', origin='lower')
    plt.colorbar(label='Embedding Value')
    plt.title('Positional Encoding (Time Embedding) Matrix')
    plt.xlabel('Embedding Dimension')
    plt.ylabel('Time Step')
    plt.tight_layout()
    plt.show()
