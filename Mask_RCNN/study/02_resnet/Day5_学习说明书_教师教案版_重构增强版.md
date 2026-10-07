# Day 5：BasicBlock、Bottleneck 与 ResNet 整体结构

> **本节定位：** 从 Day 4 的 Residual Connection 继续，目标不是“记住几个 ResNet 名词”，而是开始具备**读懂 ResNet 源码、根据结构推 Shape、知道每一行为什么存在，并逐步能够自己写出 Block / Stage 源码**的能力。
>
> **本节不进入：** 完整 ResNet-50 源码、C2/C3/C4/C5、FPN。  
> 这些内容建立在本节的 Block、Stage、Channel / Spatial 两类变化已经真正理解之后。

Day 4 已经解决了一个 Residual Block 最基本的问题：

```text
主分支：
x → F(x)

Shortcut：
x → identity

两条路径 Shape 对齐后：
F(x) + identity
```

Day 5 要进一步回答：

```text
F(x) 内部到底可以怎么写？
↓
为什么有 BasicBlock 和 Bottleneck 两种常见写法？
↓
多个 Block 又怎样组成 Stage？
↓
源码里的 layer1 / layer2 / Sequential / planes / expansion 到底是什么？
```

---

# 1. 先建立 Day 5 的“源码层级图”

在读 ResNet 源码之前，先知道自己正在看哪一层。

```text
ResNet
│
├─ Stage / layer1
│  ├─ Residual Block
│  ├─ Residual Block
│  └─ ...
│
├─ Stage / layer2
│  ├─ Residual Block
│  └─ ...
│
├─ Stage / layer3
│
└─ Stage / layer4
```

再继续往一个 Residual Block 里面拆：

```text
Residual Block
│
├─ 主分支 F(x)
│  ├─ Conv
│  ├─ BN
│  ├─ ReLU
│  └─ ...
│
└─ Shortcut
   └─ identity / projection
```

所以源码阅读时不要把所有 `Conv2d` 看成一条没有层次的长代码。

应该不断问：

> **我现在看到的是整个 ResNet、一个 Stage、一个 Block，还是 Block 内部的一层？**

这会成为你以后看 Mask R-CNN Backbone 源码时非常重要的习惯。

---

# 2. 先统一 Tensor 中两类最容易混淆的变化

ResNet 中反复出现：

```text
[B, C, H, W]
```

这里后面经常会有两种完全不同的“变小”。

## 2.1 Channel 方向：改变 C

术语常见写法：

```text
Channel reduction
Channel compression
Channel projection
```

例如：

```text
[B,256,56,56]
→
[B,64,56,56]
```

这里只有：

```text
C：256 → 64
```

而：

```text
H/W：56×56 → 56×56
```

没有变化。

可以把 C 理解成：

> **每一个空间位置上，用多少个特征数来描述这个位置。**

例如固定某个 `(h,w)`：

```text
原来：
256 个特征值

降低 Channel 后：
64 个特征值
```

---

## 2.2 Spatial 方向：改变 H/W

术语常见写法：

```text
spatial downsampling
downsampling
reduce spatial resolution
```

例如：

```text
[B,64,56,56]
→
[B,64,28,28]
```

这里：

```text
C：64 → 64
```

不变，而：

```text
H/W：56×56 → 28×28
```

变小。

因此以后看到源码时，要养成两个不同问题：

```text
这层有没有改变 C？
→ 看 in_channels / out_channels

这层有没有改变 H/W？
→ 看 kernel_size / stride / padding
```

尤其在当前 ResNet 中：

> **`out_channels` 主要决定 Channel 变化；`stride=2` 是常见的 spatial downsampling 手段。**

这两个维度方向不是两套网络系统，只是同一个 Tensor 的两类维度变化。

---

# 3. BatchNorm：先学到“能读源码”的程度

真实 ResNet 中经常出现：

```text
Conv → BN → ReLU
```

这里的 BN 是 **Batch Normalization**，PyTorch 二维特征图对应：

```python
nn.BatchNorm2d(num_features)
```

例如：

```python
nn.BatchNorm2d(64)
```

这里的：

```text
64
```

表示：

> **输入 Tensor 的 Channel 数 C=64。**

输入：

```text
[B,64,H,W]
```

输出仍然：

```text
[B,64,H,W]
```

所以目前读源码时可以先建立：

```text
Conv
→ 负责提取 / 组合特征，也可能改变 C/H/W

BN
→ 对 Conv 输出做归一化和可学习调整
→ 通常不改变 Shape

ReLU
→ 引入非线性
→ 通常不改变 Shape
```

今天不要求推 BN 数学公式。

当前真正需要掌握的是：

> **看到 `BatchNorm2d(out_channels)` 时，知道它为什么紧跟在 Conv 后面，也知道它通常不会改变你正在追踪的 Shape。**

---

# 4. Residual Block 为什么会有 BasicBlock 和 Bottleneck 两种形式

这是 Day 5 最核心的概念。

首先必须明确：

> **BasicBlock 和 Bottleneck 都不是完整网络，也不是不同任务模块。它们都是“Residual Block 的两种内部实现”。**

它们具有同一个外部骨架：

```text
主分支：
x → F(x)

Shortcut：
x → identity

最后：
F(x) + identity
→ ReLU
```

真正不同的是：

```text
F(x) 内部怎么计算
```

可以先类比成两个“功能相同、内部流水线不同”的加工车间：

```text
输入 x
↓
【加工车间】
↓
F(x)

同时 x 走 Shortcut
↓
最后汇合
```

BasicBlock 和 Bottleneck 都完成“Residual Block”这项工作，但车间内部机器的排列不同。

---

# 5. BasicBlock：最直接的 Residual Block

## 5.1 先给定义

BasicBlock 的典型主分支：

```text
3×3 Conv
→ BN
→ ReLU
→ 3×3 Conv
→ BN
```

然后：

```text
+ Shortcut
→ ReLU
```

所以完整逻辑：

```text
x
├─ 主分支：
│  3×3 Conv → BN → ReLU
│  → 3×3 Conv → BN
│
└─ Shortcut：
   identity

最后：
F(x) + identity
→ ReLU
```

它常用于：

```text
ResNet-18
ResNet-34
```

可以先把 BasicBlock 定义成：

> **用两个 3×3 Conv 直接计算 F(x) 的简单 Residual Block。**

---

## 5.2 为什么是两个 3×3 Conv？

这里不要把答案背成“因为官方就是这么写”。

`3×3 Conv` 的主要作用仍然是你 Day 2/3 已经学过的：

> **读取局部空间区域，并在 Channel 上做加权组合，产生新的局部特征。**

第一层：

```text
输入特征
→ 第一次局部特征提取 / 组合
```

第二层：

```text
第一层产生的特征
→ 再做一次局部组合
→ 得到这个 Block 的 F(x)
```

所以 BasicBlock 的设计很直接：

```text
连续做两次 3×3 局部特征计算
↓
得到 F(x)
↓
和 Shortcut 相加
```

它不像 Bottleneck 那样额外设计“先压 Channel、再扩 Channel”的结构，因此更直观。

---

## 5.3 两层都是 Conv，“下采样”不是另一种层

这是当前最容易混淆的地方。

BasicBlock 不是：

```text
第一层 = Conv / Downsample 二选一
第二层 = Conv
```

而是始终：

```text
第一层 = 3×3 Conv
第二层 = 3×3 Conv
```

只是第一层的 `stride` 有时不同。

普通情况下：

```text
Conv1：stride=1
Conv2：stride=1
```

Shape 可以保持：

```text
[B,64,32,32]
→ [B,64,32,32]
→ [B,64,32,32]
```

需要进入新的空间尺度时：

```text
Conv1：stride=2
Conv2：stride=1
```

于是：

```text
[B,64,32,32]
→ [B,128,16,16]
→ [B,128,16,16]
```

因此：

> **Conv 是“层的类型”；downsampling 是“这层操作产生的尺寸效果”。**

同一个 `Conv2d`，因为 `stride=2`，可以同时承担：

```text
提取特征
+
spatial downsampling
```

---

## 5.4 为什么通常第一层 stride=2，第二层 stride=1？

假设我们想从：

```text
32×32
```

进入新的：

```text
16×16
```

空间尺度。

第一层：

```text
stride=2
32×32 → 16×16
```

已经完成了尺度切换。

第二层继续：

```text
stride=1
16×16 → 16×16
```

目的就是：

> **继续在新的 16×16 特征图上提取和组合特征，而不是再次把尺寸缩小。**

如果第二层也 `stride=2`：

```text
32×32
→ 16×16
→ 8×8
```

一个 Block 就连续跨了两个空间尺度，这通常不是当前 Stage 设计想要的结构。

所以读源码时看到：

```text
conv1 stride=2
conv2 stride=1
```

要想到：

> **这个 Block 很可能正在负责进入一个新的 Stage / spatial scale。**

---

# 6. BasicBlock 源码：把刚才的概念映射回代码

下面代码保持原样。读的时候不要一次看完整段，而是按四个区域拆：

```text
① __init__：主分支有哪些层
② __init__：Shortcut 是否需要 projection
③ forward：主分支 F(x) 怎么走
④ forward：Shortcut 如何汇合
```

```python
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

        self.bn2 = nn.BatchNorm2d(out_channels)

        # Shortcut
        self.downsample = None

        # python 中 or 在判断语句中为从左到右计算顺序，具有短路求值特性 
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

    #  4, 64, 32, 32
    #  in_channels=64,out_channels=128,kernel_size=3,stride=2,padding=1
    #  Hout1 = floor((32+2*1-3)/2)+1 = 16 kernel_size=3, stride=2,padding=1 折半
    #  out_channels=128  4, 128, 16, 16
    #  in_channels=out_channels,out_channels=out_channels,kernel_size=3,stride=1,padding=1
    #  Hout2 = floor((16+2*1-3)/1)+1 = 16 kernel_size=3,stride=1,padding=1 保持原样
    #  4, 128, 16, 16

    # 由于 stride = 2 != 1 ，因此有 downsample 下采样操作
    # in_channels=64,out_channels=128,kernel_size=1,stride=2 
    # 原始 X 4, 64, 32, 32
    # Hout3 = floor((32+2*0-1)/2)+1 = 16 kernel_size=1,stride=2,padding=0 折半
    # 4, 128, 16, 16

    block = BasicBlock(
        in_channels=64,
        out_channels=128,
        stride=2
    )

    y = block(x)

    print("输入 shape:", x.shape)
    print("输出 shape:", y.shape)
```

---

## 6.1 读 BasicBlock 源码时应该逐行识别什么

看到：

```text
self.conv1
```

先问：

```text
输入 C 是多少？
输出 C 是多少？
stride 是多少？
因此 H/W 会不会变？
```

看到：

```text
self.bn1 = BatchNorm2d(out_channels)
```

想到：

```text
BN 的 Channel 必须和上一层 Conv 输出 C 对应
Shape 通常不变
```

看到：

```text
self.conv2
```

注意：

```text
in_channels=out_channels
out_channels=out_channels
stride=1
```

这说明第二层是在：

> **当前新的 Channel 数和新的空间尺度上继续做特征计算。**

看到：

```text
if stride != 1 or in_channels != out_channels
```

不要只背判断语句，要翻译成：

```text
主分支是否改变了 H/W？
或者
主分支是否改变了 C？

只要有一个发生变化，
原始 identity 就可能无法直接和主分支相加。
```

所以需要：

```text
Projection Shortcut
```

再看到：

```text
out = out + identity
```

要立刻检查：

```text
主分支 Shape
是否等于
Shortcut Shape
```

这就是以后看源码时应该形成的反射。

---

# 7. `nn.Sequential`：它不是新的网络数学操作

BasicBlock 的 downsample 中出现：

```python
nn.Sequential(...)
```

它的定义可以先理解为：

> **PyTorch 用来把多个 `nn.Module` 按顺序连接起来的容器。**

例如：

```text
Sequential(
    Conv,
    BN
)
```

数据流就是：

```text
输入
→ Conv
→ BN
→ 输出
```

它的意义主要是**代码组织**。

如果不用 Sequential，也可以概念上写成：

```text
identity = conv(identity)
identity = bn(identity)
```

使用：

```text
self.downsample = nn.Sequential(...)
```

只是把：

```text
1×1 Conv
→ BN
```

这两个连续步骤包装成一个模块，以后只需要：

```text
identity = self.downsample(identity)
```

这对读源码非常重要：

> **看到 `Sequential` 时，不要把它当成一个新的神经网络层；先展开看里面装了哪些 Module。**

---

# 8. BasicBlock 的 Shape：把 C 与 H/W 分开追

示例：

```text
输入：
[B,64,32,32]
```

Block 参数：

```text
in_channels=64
out_channels=128
stride=2
```

先看主分支。

第一层：

```text
3×3 Conv
C：64 → 128
H/W：32×32 → 16×16
```

因此：

```text
[B,64,32,32]
→
[B,128,16,16]
```

BN、ReLU：

```text
[B,128,16,16]
→
[B,128,16,16]
```

第二层：

```text
3×3 Conv, stride=1
C：128 → 128
H/W：16×16 → 16×16
```

最终主分支：

```text
[B,128,16,16]
```

再看 Shortcut。

原始：

```text
[B,64,32,32]
```

不能直接和：

```text
[B,128,16,16]
```

相加。

所以使用：

```text
1×1 Conv, stride=2
```

同时完成：

```text
C：64 → 128
H/W：32×32 → 16×16
```

最终：

```text
Shortcut：
[B,128,16,16]

主分支：
[B,128,16,16]
```

才能 Residual Add。

这一段是以后所有 Residual Block Shape 调试的通用方法：

> **先分别追主分支和 Shortcut，最后在 Add 前检查两边 Shape。**

---

# 9. 为什么还需要 Bottleneck

BasicBlock 已经能实现 Residual Connection：

```text
3×3 → 3×3
```

那为什么还要再设计 Bottleneck？

问题出在更深、更宽的网络中。

假设某一层已经有很多 Channel，例如：

```text
C=256
```

如果一直直接在：

```text
256 channels
```

上做昂贵的 `3×3 Conv`，计算量会很大。

所以 Bottleneck 的设计思路是：

```text
先把 Channel 变少
↓
在较少 Channel 上做主要的 3×3 Conv
↓
最后再把 Channel 扩到需要的输出维度
```

这就是 Bottleneck 最核心的目的：

> **它仍然是 Residual Block，但通过改变 F(x) 的内部计算方式，让深网络中的 3×3 Conv 更经济。**

因此 BasicBlock 与 Bottleneck 不是：

```text
一个负责提特征
一个负责省成本
```

因为两者都在提取特征。

更准确地说：

```text
BasicBlock：
更直接地计算 F(x)

Bottleneck：
更经济地组织 F(x) 的计算
```

---

# 10. Bottleneck：先定义三层各自负责什么

Bottleneck 主分支：

```text
1×1 Conv
→ BN
→ ReLU
→ 3×3 Conv
→ BN
→ ReLU
→ 1×1 Conv
→ BN
```

然后：

```text
+ Shortcut
→ ReLU
```

三层 Conv 不应该只记尺寸，要记**职责**。

---

## 10.1 第一层 `1×1 Conv`：Channel projection / reduction

例如：

```text
[B,256,56,56]
→
[B,64,56,56]
```

变化：

```text
C：256 → 64
H/W：不变
```

主要目的：

> **降低 Channel 数，减少后续 3×3 Conv 的计算量。**

为什么 `1×1` 还能改变 C？

因为一个 filter 的 Shape 是：

```text
[C_in,1,1]
```

例如：

```text
[256,1,1]
```

它在某个 `(h,w)` 空间位置上虽然只看一个位置，但会读取：

```text
全部 256 个 input channels
```

一个 filter 输出一个新特征值。

64 个 filter：

```text
→ 64 个输出 Channel
```

所以：

```text
256 → 64
```

这里的 `1×1` 描述的是：

```text
空间窗口大小
```

不是 Channel 数。

---

## 10.2 第二层 `3×3 Conv`：主要的局部空间特征计算

第一层已经把：

```text
C：256 → 64
```

第二层就在：

```text
64 channels
```

上做：

```text
3×3 Conv
```

这一步负责：

> **读取周围 3×3 局部区域并进一步组合特征。**

Bottleneck 节省成本的核心就在这里：

```text
不是在 256 channels 上做 3×3
而是在 64 channels 上做 3×3
```

注意：

> `3×3 Conv` 不等于一定下采样。

如果：

```text
stride=1
```

H/W 可以完全不变。

如果：

```text
stride=2
```

才会产生 spatial downsampling。

---

## 10.3 第三层 `1×1 Conv`：扩展到最终输出 Channel

例如：

```text
[B,64,56,56]
→
[B,256,56,56]
```

主要任务：

```text
C：64 → 256
```

为什么仍然用 `1×1`？

因为这一步最主要需要完成的是：

> **Channel projection / expansion，而不是继续扩大空间邻域。**

`1×1 Conv` 可以在每个空间位置对全部输入 channels 做重新组合，并产生需要数量的输出 channels。

因此它很适合：

```text
64 → 256
```

同时让：

```text
H/W
```

在 `stride=1` 时保持不变。

这就是：

```text
1×1 → 3×3 → 1×1
```

背后的完整逻辑：

```text
先降 C
→ 让 3×3 计算变便宜
→ 做主要的局部空间计算
→ 再把 C 扩到 Block 最终输出要求
```

---

# 11. `planes` 和 `expansion`：源码变量到底在表达什么

这是看 torchvision ResNet 源码时很容易卡住的地方。

## 11.1 `planes` 不是新的深度学习理论术语

在这份 Bottleneck 代码里：

```text
planes
```

可以暂时翻译成：

> **Bottleneck 中间“窄通道”的基础 Channel 数。**

例如：

```text
planes = 64
```

表示中间两层主要工作在：

```text
C=64
```

---

## 11.2 `expansion=4` 只是一个类里的数值

代码：

```text
expansion = 4
```

不要理解成：

> Tensor 自动扩张 4 倍。

它首先只是一个 Python 数值。

后面：

```text
out_channels = planes × expansion
```

如果：

```text
planes=64
expansion=4
```

Python 先计算：

```text
out_channels=256
```

然后这个 `256` 被拿去定义：

```text
最后一个 Conv2d 的 out_channels
```

真正改变 Tensor Channel 的仍然是：

```text
Conv2d(..., out_channels=256, ...)
```

---

## 11.3 为什么最终会出现 `Conv2d(64,256,1)`

把源码变量代入：

```text
planes = 64
expansion = 4
out_channels = 64 × 4 = 256
```

第三层：

```text
in_channels = planes = 64
out_channels = 256
kernel_size = 1
```

所以就得到：

```text
Conv2d(64,256,1)
```

三个数字分别回答：

```text
64
→ 输入 Channel

256
→ 输出 Channel

1
→ 空间 kernel 为 1×1
```

而最后这个 `kernel_size=1` 的设计目的正是：

> **高效完成 Channel expansion，不额外做一次 3×3 邻域计算。**

---

# 12. Bottleneck 源码：现在再回到代码

下面代码保持原样。

读源码时分成：

```text
① expansion / out_channels 怎么算
② conv1 为什么 1×1
③ conv2 为什么 3×3
④ conv3 为什么 1×1
⑤ Shortcut 为什么可能需要 projection
⑥ forward 两条路径如何汇合
```

```python
import torch
import torch.nn as nn


class Bottleneck(nn.Module):
    expansion = 4
    # 64*4=256

    def __init__(self, in_channels, planes, stride=1):
        super().__init__()

        out_channels = planes * self.expansion
        # 256
         
        # [4, 64, 32, 32]
        # 1×1：压缩 / 调整 Channel
        self.conv1 = nn.Conv2d(
            in_channels,
            planes,
            kernel_size=1,
            stride=1,
            bias=False
        )
        self.bn1 = nn.BatchNorm2d(planes)
        # [4, 64, 32, 32]

        # 3×3：处理空间局部特征
        self.conv2 = nn.Conv2d(
            planes,
            planes,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)
        # [4, 64, 32, 32]

        # 1×1：扩展到最终输出 Channel
        self.conv3 = nn.Conv2d(
            planes,
            out_channels,
            kernel_size=1,
            stride=1,
            bias=False
        )
        self.bn3 = nn.BatchNorm2d(out_channels)
        # [4, 256, 32, 32]

        self.relu = nn.ReLU()

        # Shortcut
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
            # [4, 64, 32, 32]
            # [4, 256, 32, 32]

    def forward(self, x):
        identity = x

        # 主分支
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        # Shortcut
        if self.downsample is not None:
            identity = self.downsample(identity)

        # Residual Add
        out = out + identity
        out = self.relu(out)

        return out


if __name__ == "__main__":
    x = torch.randn(4, 64, 32, 32)

    block = Bottleneck(
        in_channels=64,
        planes=64,
        stride=1
    )

    y = block(x)

    print("输入 shape:", x.shape)
    print("输出 shape:", y.shape)
```

---

# 13. Bottleneck 的 Shape：不要只看最终 64→256

示例输入：

```text
[B,64,32,32]
```

参数：

```text
planes=64
expansion=4
stride=1
```

先计算：

```text
out_channels
= planes × expansion
= 64 × 4
= 256
```

主分支：

```text
输入
[B,64,32,32]

↓ conv1：1×1
C：64 → 64
H/W：不变

[B,64,32,32]

↓ conv2：3×3, stride=1
C：64 → 64
H/W：不变

[B,64,32,32]

↓ conv3：1×1
C：64 → 256
H/W：不变

[B,256,32,32]
```

Shortcut 原始输入还是：

```text
[B,64,32,32]
```

主分支已经：

```text
[B,256,32,32]
```

所以虽然 H/W 一样，但：

```text
C 不一样
```

仍然不能直接 Add。

于是 Shortcut 使用：

```text
1×1 Conv
64 → 256
```

得到：

```text
[B,256,32,32]
```

再相加。

所以：

> **Projection Shortcut 不只是“下采样工具”。只要 C 或 H/W 不匹配，就可能需要它。**

---

# 14. BasicBlock 与 Bottleneck 放在一起比较

现在再对比，不只看结构，还看“为什么”。

| 问题 | BasicBlock | Bottleneck |
|---|---|---|
| 都是什么 | Residual Block | Residual Block |
| 共同核心 | `F(x)+identity` | `F(x)+identity` |
| F(x) 主分支 | `3×3 → 3×3` | `1×1 → 3×3 → 1×1` |
| 设计思路 | 简单、直接做两次局部特征计算 | 先降 C，让中间 3×3 更经济，再扩 C |
| 常见网络 | ResNet-18 / 34 | ResNet-50 / 101 / 152 |
| 是否可能下采样 | 可以 | 可以 |
| 下采样主要由什么决定 | `stride` | `stride` |
| 是否有 Shortcut | 有 | 有 |

最重要的判断标准：

> **一个 Block 是不是 Residual Block，不是看它里面用了几个 Conv，而是看它是否存在主分支 + Shortcut，并在最后进行 Residual Add。**

---

# 15. 从 Block 到 Stage：源码为什么突然出现 `layer1`

一个 Residual Block 只负责一小段特征变换。

真实 ResNet 会把很多 Block 连起来。

例如：

```text
Block 1
→ Block 2
→ Block 3
```

把这样一组 Block 作为更高一层的结构单位，就得到：

> **Stage**

---

## 15.1 Stage 的定义

当前学习阶段可以定义成：

> **Stage = 一组按顺序执行、主要工作在同一 spatial scale 上的 Residual Blocks。**

这里的：

```text
same spatial scale
```

主要指：

```text
H/W 相同
```

例如：

```text
[B,512,28,28]
[B,512,28,28]
[B,512,28,28]
```

它们都工作在：

```text
28×28
```

这个空间尺度。

通常同一个 Stage 内：

```text
Block 类型相同
H/W 通常保持一致
后续 Block 的 C 通常也保持一致
```

但“同尺度”这个词最核心指的是：

```text
H/W
```

---

## 15.2 为什么 Stage 第一个 Block 经常特殊

假设上一 Stage 输出：

```text
[B,256,56,56]
```

下一 Stage 要工作在：

```text
[B,512,28,28]
```

那新 Stage 第一个 Block 必须完成“换挡”：

```text
C：
256 → 512

H/W：
56×56 → 28×28
```

所以第一个 Block 经常出现：

```text
stride=2
Projection Shortcut
```

完成新的 Shape。

后面的 Block 通常：

```text
[B,512,28,28]
→ [B,512,28,28]
→ [B,512,28,28]
```

继续在同一个空间尺度上工作。

可以用一个只帮助理解层级的类比：

```text
Stage = 一个楼层

Stage 第一个 Block
= 上楼的楼梯 / 入口
= 把上一层的 Shape 切换到这一层

后续 Blocks
= 在当前楼层继续工作
```

---

# 16. 为什么源码叫 `layer1 / layer2`，理论上却叫 Stage

PyTorch torchvision ResNet 源码常写：

```text
self.layer1
self.layer2
self.layer3
self.layer4
```

这里：

```text
layer
```

只是源码作者选择的变量命名。

它不代表：

```text
只有一层 Conv
```

例如：

```text
self.layer1
```

内部可能装：

```text
Block 1
Block 2
Block 3
```

所以架构分析时，我们通常把：

```text
layer1
```

理解成：

```text
Stage 1
```

也就是说：

```text
源码变量名：layer1
架构术语：Stage 1
```

以后读源码要接受一个事实：

> **代码变量名是实现约定，不一定等于理论术语。**

---

# 17. 为什么 Stage 常用 `nn.Sequential`

一个 Stage 的数据流通常就是：

```text
Block 1
→ Block 2
→ Block 3
```

这恰好符合：

```text
Sequential
```

“按顺序执行 Module”的语义。

所以源码可以写：

```text
layer1 = Sequential(
    Block1,
    Block2,
    Block3
)
```

然后：

```text
x = layer1(x)
```

就会依次执行三个 Block。

因此：

```text
Stage
```

是网络架构概念；

```text
nn.Sequential
```

是 PyTorch 用来实现“按顺序执行这些 Block”的代码容器。

这两个也不要混在一起。

---

# 18. TinyResNet：把 Block、Stage、Sequential 放进一份源码

现在再看 TinyResNet，就不应该只看到很多类和函数。

应该先按层级拆：

```text
TinyResNet
│
├─ stem
│  └─ Conv + BN + ReLU
│
├─ layer1 / Stage 1
│  ├─ BasicBlock
│  └─ BasicBlock
│
└─ layer2 / Stage 2
   ├─ BasicBlock ← 第一个 Block 换尺度
   └─ BasicBlock ← 保持当前尺度
```

下面代码保持原样。

```python
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
```

---

# 19. 如何用“源码阅读五问”看 TinyResNet

以后看到源码里的一个模块，固定问：

### ① 它属于哪个层级？

例如：

```text
self.layer2
```

不是单层 Conv，而是：

```text
一个 Stage
```

### ② 输入 Shape 是什么？

例如：

```text
layer2 输入：
[B,64,32,32]
```

### ③ 第一个 Block 有没有换尺度？

看到：

```text
BasicBlock(64,128,stride=2)
```

马上拆：

```text
C：64 → 128
H/W：32×32 → 16×16
```

### ④ 后面的 Block 为什么参数变了？

第二个：

```text
BasicBlock(128,128,stride=1)
```

因为它接收到的是：

```text
[B,128,16,16]
```

所以：

```text
in_channels=128
out_channels=128
```

继续保持当前 Stage 的 Shape。

### ⑤ 输出为什么能交给下一 Stage？

因为一个 Stage 输出的 Tensor：

```text
[B,C,H,W]
```

就是下一 Stage 第一个 Block 的输入。

所以整个 ResNet 本质仍然是一条逐层可追踪的数据流。

---

# 20. Day 5 最终应该形成的“源码脑图”

看到：

```text
ResNet
```

先拆成：

```text
Stage / layer
```

看到：

```text
Stage
```

再拆成：

```text
多个 Residual Blocks
```

看到：

```text
Residual Block
```

先看：

```text
它是 BasicBlock 还是 Bottleneck？
```

如果是 BasicBlock：

```text
主分支：
3×3 → 3×3
```

如果是 Bottleneck：

```text
主分支：
1×1 → 3×3 → 1×1
```

然后无论哪一种，都继续检查：

```text
主分支 F(x)
Shortcut identity
↓
Shape 是否匹配？
↓
Residual Add
```

最后再逐层检查：

```text
C 是否变化？
H/W 是否变化？
为什么？
```

这套阅读顺序比“看到一行查一行”更接近真正的源码阅读能力。

---

# 21. Day 5 当前薄弱点集中复盘

## 21.1 `1×1 Conv` 不等于下采样

```text
1×1
→ kernel 的空间大小

stride
→ 是否改变空间采样步长
```

`1×1 Conv, stride=1` 可以：

```text
只改 C
H/W 不变
```

---

## 21.2 Conv 与 Downsampling 不是同一级概念

```text
Conv
= 操作 / 层类型

Downsampling
= H/W 变小这一结果
```

一个 Conv 如果：

```text
stride=2
```

就可以产生 downsampling。

---

## 21.3 BasicBlock 与 Bottleneck 都是 Residual Block 的原因

不是因为：

```text
都能提特征
```

真正原因：

```text
都有主分支 F(x)
都有 Shortcut
最后都有 F(x)+identity
```

---

## 21.4 Bottleneck 最后的 `1×1` 为什么不是 `3×3`

因为最后一步主要任务是：

```text
Channel expansion
```

不需要再额外进行一次昂贵的 3×3 邻域计算。

因此使用：

```text
1×1 Conv
```

高效地完成：

```text
C：64 → 256
```

---

## 21.5 Stage 的“同尺度”

主要指：

```text
H/W 相同
```

同一 Stage 的后续 Block：

```text
C 通常也一致
H/W 通常也一致
```

第一个 Block 经常特殊，因为它负责：

```text
上一 Stage Shape
→
当前 Stage Shape
```

---

# 22. 本节和“会写源码”的关系

目前不要追求背出完整 ResNet。

真正需要做到的是，给你一个 Block 需求，例如：

```text
输入：
[B,64,32,32]

要求输出：
[B,128,16,16]

使用 BasicBlock
```

你应该开始能够自己推：

```text
主分支第一层：
Conv2d(64,128,3,stride=2,padding=1)

第二层：
Conv2d(128,128,3,stride=1,padding=1)

Shortcut：
因为 C 和 H/W 都变了
→ 需要 Projection
→ Conv2d(64,128,1,stride=2)

最后：
两边 [B,128,16,16]
→ Add
```

这才是 Day 5 真正的“会写源码”目标。

Bottleneck 也一样。

当给出：

```text
in_channels
planes
expansion
stride
```

你最终应该能够自己推出：

```text
conv1
conv2
conv3
downsample 是否需要
每一步 Shape
```

而不是靠记忆整段代码。

---

# Day 5 完成标准

- [ ] `BatchNorm2d(64)` 中的 64 表示什么？输入 Tensor 的 Channel 数。
- [ ] BatchNorm2d 通常会不会改变 `[B,C,H,W]` 的 Shape？不会。
- [ ] 真实 BasicBlock 为什么常写成 `Conv → BN → ReLU → Conv → BN`？因为就是卷积整理激活这样循环，最后因为要 Shortcut 再激活所以最后那次卷积一般只带着BN整理，没有重复激活。
- [ ] `nn.Sequential(...)` 是做什么的？按顺序把多个层装在一起。
- [ ] BasicBlock 的主分支通常由哪两层卷积组成？两层3×3卷积，第一层可能会下采样，后面一层保持形状。
BasicBlock 有两个 3×3 Conv。通常第二个负责保持当前 Shape；当需要进入新的空间尺度时，第一个 Conv 可以用 stride=2 完成下采样。
- [ ] Bottleneck 的主分支为什么是 `1×1 → 3×3 → 1×1`？压缩 Channel，提取特征，把 Channel 扩回去
- [ ] Bottleneck 第一层 1×1 Conv 为什么常用于压缩 Channel？压缩 Channel是因为要简化后续3*3Conv的计算量
- [ ] Bottleneck 最后一层 1×1 Conv 为什么要把 Channel 扩回去？因为第一层 1×1 Conv压缩了 Channel，最后一层用来复原
把 Channel 扩展回最终输出维度。
- [ ] `expansion=4` 表示什么？out_channel=planes*expansion
- [ ] 当 `planes=64`、`expansion=4` 时，Bottleneck 最终输出 C 是多少？out_C=64*4=256
- [ ] `[B,64,32,32]` 经过示例 Bottleneck 后为什么会变成 `[B,256,32,32]`？因为第三层 1×1 Conv 的 out_channels 被设置成了 planes × expansion = 256。
- [ ] 为什么这个 Bottleneck 的 Shortcut 也需要 1×1 Conv？因为Shortcut是维度对齐，大小不变(不对)
Projection Shortcut 的目的是 Shape 对齐，它既可能改变 C，也可能在 stride=2 时改变 H/W。
- [ ] BasicBlock 与 Bottleneck 相同的核心是什么？两者都是 Residual Block，都有主分支 F(x) 和 Shortcut，最后执行 F(x)+identity。
- [ ] BasicBlock 与 Bottleneck 最主要的结构区别是什么？BasicBlock 直接用两个 3×3 Conv；Bottleneck 用 1×1 → 3×3 → 1×1，先压缩 Channel，再做较便宜的 3×3 计算，最后扩展 Channel。
- [ ] Stage 是什么？由由多个Block组成的layer
- [ ] `layer1 / layer2 / layer3 / layer4` 为什么不能简单理解成四个卷积层？因为一个layer由多个Block组成，一个Block可能有卷积层、BN层、激活函数等等
- [ ] 一个 Stage 内多个 Block 通常有什么共同特点？
- [ ] 为什么进入下一个 Stage 时常出现 H/W 变小、C 变大？H/W 看 stride，C 看 out_channels，因为stride=2，out_channels=planes*expansion>in_channels
- [ ] 能否解释 `ResNet → Stage → Block → Conv/BN/ReLU` 这四层结构？包含关系，由大到小，前面的由后面的部分组成

---

- [ ] **Channel reduction** 和 **spatial downsampling** 分别改变 `[B,C,H,W]` 中哪些维度？分别是降低C和降低H/W
- [ ] Bottleneck 第一层 `1×1 Conv` 最主要是为什么降低 C？需要降低C是因为简化后续3*3Conv的计算量
- [ ] BasicBlock 为什么两层都是 `3×3 Conv`，而第一层有时会 `stride=2`？BasicBlock 是一种简单直接的残差块。 两个 3×3 Conv 的作用是：用来连续提取和组合局部特征；当这个 Block 负责进入新的 Stage 时，第一层 Conv 可以设置 stride=2 完成一次空间下采样，第二层通常继续在新的 H/W 上处理特征。
- [ ] BasicBlock 与 Bottleneck 为什么都是 Residual Block？因为都能提取特征？都是经过了卷积+BN+激活+shortcut那个add+激活？正确答案：BasicBlock 和 Bottleneck 都属于 Residual Block，因为它们都有一条主分支计算 F(x)，同时有一条 Shortcut 传递 identity，最后执行 F(x)+identity。两者区别只是主分支 F(x) 内部怎么计算。
- [ ] `planes=64, expansion=4` 怎样一步步变成 `Conv2d(64,256,1)`？in_channel是planes=64，out_channel是planes×expansion=64×4=256，kernel_size为什么是1不清楚
- [ ] Stage 的“同尺度”主要指什么？第一个 Block 为什么经常特殊？Stage 的“同尺度”主要指 Block 的 H/W 相同，C 一般也相同。第一个 Block 经常负责把上一 Stage 的 Shape 切换到当前 Stage 的 Shape。

如果这 6 个能独立解释，Day 5 目前最严重的概念混淆就基本解决了。

你现在最应该改掉的两条旧记忆是：

> **`1×1 Conv` 不等于下采样。1×1 描述空间 kernel 大小；是否下采样主要看 stride。**

以及：

> **BasicBlock 和 Bottleneck 不是两个不同功能模块，而是实现同一种 Residual Block 角色的两种内部结构：前者简单直接，后者为深网络控制计算成本。**


# Day 5 最终必须牢牢记住

```text
真实 ResNet Block 常见：
Conv → BN → ReLU
```

```text
BasicBlock：
3×3 → 3×3
```

```text
Bottleneck：
1×1 → 3×3 → 1×1
```

```text
Bottleneck 的 1×1：
前面压缩 / 调整 Channel
后面扩展到最终输出 Channel
```

```text
经典 Bottleneck：
expansion = 4

planes=64
→ 输出 C=256
```

```text
很多 Block
→ 一个 Stage

多个 Stage
→ ResNet
```

```text
ResNet
→ Stage
→ Residual Block
→ Conv / BN / ReLU
```

最关键的一句话：

> **Day 4 学的是“一个残差块怎么把两条路相加”，Day 5 要进一步看懂“残差块有哪些形式，以及很多残差块怎样按 Stage 组织成一整个 ResNet”。**
