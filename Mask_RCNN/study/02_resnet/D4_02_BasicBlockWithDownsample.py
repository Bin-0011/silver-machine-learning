import torch
import torch.nn as nn

class BasicBlockWithDownsample(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        # 主分支第一层：可能带 stride（下采样）
        self.conv1 = nn.Conv2d(
            in_channels, 
            out_channels, 
            kernel_size=3, 
            stride=stride,  # 注意：这里可能为 2
            padding=1, 
            bias=False
        )
        self.relu = nn.ReLU()

        # 主分支第二层：保持 shape 不变
        self.conv2 = nn.Conv2d(
            out_channels, 
            out_channels, 
            kernel_size=3, 
            stride=1, 
            padding=1, 
            bias=False
        )

        # ★★★ 核心：投影 shortcut（downsample）★★★
        # 只有当 输入通道 != 输出通道 或者 步长 != 1 时，才需要激活
        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Conv2d(
                in_channels, 
                out_channels, 
                kernel_size=1,   # 1x1 改变通道
                stride=stride,   # 2 改变 H/W
                bias=False
            )

    def forward(self, x):
        identity = x  # 保存原始输入

        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)

        # ★★★ 如果 downsample 存在，先对 identity 做投影 ★★★
        if self.downsample is not None:
            identity = self.downsample(identity)

        # 此时 out 和 identity 形状绝对一致
        out = out + identity
        out = self.relu(out)

        return out

if __name__ == "__main__":
    # 1. 设置超参数
    batch_size = 4
    in_channels = 64
    out_channels = 128
    height = 32
    width = 32

    # 2. 构造随机输入 (模拟特征图)
    x = torch.randn(batch_size, in_channels, height, width)

    # 3. 实例化 BasicBlockWithDownsample
    block = BasicBlockWithDownsample(in_channels, out_channels, stride=2)

    # 4. 前向传播
    out = block(x)

    # 5. 打印形状，观察是否发生了下采样和通道变化
    print(f"输入 shape: {x.shape}")
    print(f"输出 shape: {out.shape}")