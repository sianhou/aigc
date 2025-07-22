import matplotlib.pyplot as plt
import torch
import torch.nn as nn


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


if __name__ == '__main__':
    relu = nn.ReLU()
    swish = Swish()

    # 输入数据
    x = torch.linspace(-5, 5, 100)

    # 输出
    y_relu = relu(x)
    y_swish = swish(x)

    # 可视化
    plt.plot(x.numpy(), y_relu.numpy(), label='ReLU', linestyle='--')
    plt.plot(x.numpy(), y_swish.detach().numpy(), label='Swish')
    plt.title('Swish vs ReLU')
    plt.legend()
    plt.grid(True)
    plt.show()
