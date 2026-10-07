import torch
import torch.nn as nn


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        # 主分支第一层
        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False
        )

        # [outchannels, stride, padding, bias] = [out_channels, stride, 1, False]

        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

        # 主分支第二层
        self.conv2 = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
        )

        # [outchannels, stride, padding, bias] = [out_channels, 1, 1, False]

        self.bn2 = nn.BatchNorm2d(out_channels)

        # Shortcut
        self.downsample = None

        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False
                ),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = x

        # 主分支
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        # Shortcut
        if self.downsample is not None:
            identity = self.downsample(identity)

        # Residual Add
        out = out + identity
        out = self.relu(out)

        return out


if __name__ == "__main__":
    x = torch.randn(4, 64, 32, 32)

    block = BasicBlock(
        in_channels=64,
        out_channels=128,
        stride=2
    )

    y = block(x)

    print("输入 shape:", x.shape)
    print("输出 shape:", y.shape)