import torch
import torch.nn as nn


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.downsample = None

        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False
                ),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(identity)

        out = out + identity
        out = self.relu(out)

        return out


class TinyResNet(nn.Module):
    def __init__(self):
        super().__init__()

        # 前置卷积
        self.stem = nn.Sequential(
            nn.Conv2d(
                3,
                64,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        # Stage 1：空间尺寸保持
        self.layer1 = nn.Sequential(
            BasicBlock(64, 64, stride=1),
            BasicBlock(64, 64, stride=1)
        )

        # Stage 2：第一个 Block 下采样
        self.layer2 = nn.Sequential(
            BasicBlock(64, 128, stride=2),
            BasicBlock(128, 128, stride=1)
        )

    def forward(self, x):
        print("输入:", x.shape)

        x = self.stem(x)
        print("stem:", x.shape)

        x = self.layer1(x)
        print("layer1:", x.shape)

        x = self.layer2(x)
        print("layer2:", x.shape)

        return x


if __name__ == "__main__":
    x = torch.randn(1, 3, 32, 32)

    model = TinyResNet()

    y = model(x)

    print("最终输出:", y.shape)

'''
运行前应该先自己推：

输入
[1,3,32,32]

stem
→ [1,64,32,32]

layer1
→ [1,64,32,32]

layer2 第一个 Block
→ [1,128,16,16]

layer2 第二个 Block
→ [1,128,16,16]

最终
→ [1,128,16,16]
'''