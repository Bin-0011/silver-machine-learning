import torch
import torch.nn as nn

class BasicBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()

        self.conv1 = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            padding=1,
            bias=False
        )

        self.relu = nn.ReLU()

        self.conv2 = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            padding=1,
            bias=False
        )

    def forward(self, x):
        identity = x          # 保存原始输入（shortcut）

        out = self.conv1(x)
        out = self.relu(out)

        out = self.conv2(out)

        out = out + identity  # 残差连接（逐元素相加）
        out = self.relu(out)

        return out


if __name__ == "__main__":
    # 1. 设置超参数
    batch_size = 4
    channels = 64
    height = 32
    width = 32

    # 2. 构造随机输入 (模拟特征图)
    x = torch.randn(batch_size, channels, height, width)

    # 3. 实例化 BasicBlock
    block = BasicBlock(channels)

    # 4. 前向传播
    out = block(x)

    # 5. 打印形状，观察是否保持不变
    print(f"输入 shape: {x.shape}")
    print(f"输出 shape: {out.shape}")

    # 可选：打印具体的数值，看 residual add 是否生效（取第一个样本的第一个通道的左上角 3x3）
    print("\n输入 x 的左上角 3x3（第一个样本，第一个通道）:")
    print(x[0, 0, :3, :3])
    print("\n输出 out 的左上角 3x3:")
    print(out[0, 0, :3, :3])