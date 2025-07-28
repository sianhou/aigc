import torch
import torch.nn as nn


class EncoderBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.relu2 = nn.ReLU()

    def forward(self, x):
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        return x


# 定义一个简单的模块化网络
class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = EncoderBlock()
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(32, 16, 3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, 3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


x = torch.randn(1, 1, 64, 64)
model = UNet()

torch.onnx.export(model,
                  x,
                  'model.onnx',
                  input_names=["input"],
                  output_names=["output"],
                  opset_version=13,
                  do_constant_folding=True)
