# Day 5：BasicBlock、Bottleneck 与 ResNet 整体结构

Day 4 已经把残差块最重要的数据流理清楚了：

`输入 x → 主分支 F(x)`，同时 `x → Shortcut`，两条路径 Shape 对齐后执行 `F(x) + identity`。

你也已经知道，当主分支不改变 Shape 时，可以直接使用 `identity = x`；当 C 或 H/W 改变时，需要在 Shortcut 上加入 `1×1 Conv` 做 Projection，让两边重新对齐。

Day 5 在这个基础上继续向真正的 ResNet 靠近。今天不急着追完整 ResNet-50，也不进入 FPN，只解决三件事：

- 真实 ResNet Block 为什么常出现 `BatchNorm`
- `BasicBlock` 和 `Bottleneck` 到底有什么区别
- 很多个 Block 怎样组成一整个 ResNet

今天的主线可以先记成：

`Residual Block → BasicBlock / Bottleneck → 多个 Block → 一个 Stage → 多个 Stage → ResNet`

---

## 1. 从 Day 4 的 BasicBlock 继续

Day 4 写过的简化版 BasicBlock 是：

`x → Conv3×3 → ReLU → Conv3×3 → + Shortcut → ReLU`

这已经足够帮助我们理解 Residual Connection。

但打开真正的 ResNet 代码时，经常会看到：

`Conv → BatchNorm → ReLU → Conv → BatchNorm → Add → ReLU`

这里第一次出现的新词是：

**BatchNorm，完整名称 Batch Normalization，中文常叫“批归一化”。**

Day 5 暂时不推它的完整数学公式，只先理解它在网络代码中的位置和作用。

---

## 2. BatchNorm 今天先理解到什么程度

PyTorch 中二维卷积后常使用：

```python
nn.BatchNorm2d(num_features)
```

例如：

```python
nn.BatchNorm2d(64)
```

这里的 `64` 对应输入 Tensor 的 Channel 数。

如果输入：

`[B,64,H,W]`

经过：

```python
nn.BatchNorm2d(64)
```

输出仍然是：

`[B,64,H,W]`

所以今天先形成两个认识：

1. `BatchNorm2d` **通常不改变 Shape**
2. 它通常接在 Conv 后，对卷积输出做归一化和可学习的缩放/平移，使训练过程更稳定

因此真实 ResNet 中经常看到：

`Conv → BN → ReLU`

而不是只写：

`Conv → ReLU`

今天先把 BN 当成“卷积后面的训练稳定器”即可，完整数学细节以后单独学习。

---

## 3. 加上 BatchNorm 后的 BasicBlock

真实 ResNet-18 / ResNet-34 使用的基础残差单元就是 **BasicBlock**。

它的主分支大致是：

`3×3 Conv → BN → ReLU → 3×3 Conv → BN`

然后与 Shortcut 相加：

`F(x) + identity → ReLU`

完整可运行代码：

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

这里又出现了一个常用函数：

```python
nn.Sequential(...)
```

它的作用是：

> 按顺序把多个层装在一起。

例如：

```python
nn.Sequential(
    Conv2d(...),
    BatchNorm2d(...)
)
```

可以理解成：

`输入 → Conv2d → BatchNorm2d → 输出`

所以这里的 `self.downsample` 不再只有一个 Conv，而是：

`1×1 Conv → BN`

---

## 4. BasicBlock 的 Shape 仍然怎么推

假设：

```text
输入：
[B,64,32,32]
```

使用：

```python
BasicBlock(
    in_channels=64,
    out_channels=128,
    stride=2
)
```

主分支：

`[B,64,32,32] → Conv3×3,s=2 → [B,128,16,16] → BN → [B,128,16,16] → ReLU → [B,128,16,16] → Conv3×3 → [B,128,16,16] → BN → [B,128,16,16]`

Shortcut：

`[B,64,32,32] → 1×1 Conv,s=2 → [B,128,16,16] → BN → [B,128,16,16]`

最后：

`[B,128,16,16] + [B,128,16,16] → [B,128,16,16]`

因此加入 BatchNorm 后，**Residual Add 的 Shape 逻辑完全没有改变**。

---

## 5. BasicBlock 为什么还不够

BasicBlock 的主分支是：

`3×3 Conv → 3×3 Conv`

例如输入、输出都采用较大的 Channel 数时，两层 3×3 Conv 的计算量会比较大。

当 ResNet 继续做得更深，例如 ResNet-50、101、152 时，常改用另一种残差块：

**Bottleneck Block（瓶颈块）**

这里的“瓶颈”不是说网络出故障，而是说：

> 中间先把 Channel 压小，在较窄的特征空间里完成主要的 3×3 计算，再把 Channel 扩大回来。

---

## 6. Bottleneck 的三层结构

Bottleneck 主分支通常是：

`1×1 Conv → 3×3 Conv → 1×1 Conv`

例如：

`256 channels → 64 → 64 → 256`

三层分别可以这样理解。

### 第一层 1×1 Conv：压缩 Channel

`256 → 64`

目的：

> 先把 Channel 数变少，降低后面 3×3 Conv 的计算量。

### 第二层 3×3 Conv：处理局部空间特征

`64 → 64`

这一层继续负责我们熟悉的局部 H/W 特征组合。

### 第三层 1×1 Conv：恢复 / 扩展 Channel

`64 → 256`

把较窄的中间特征重新投影到更高维的输出空间。

所以 Bottleneck 的核心形状可以写成：

`宽 → 窄 → 窄 → 宽`

---

## 7. `expansion` 是什么

在 PyTorch ResNet 的 Bottleneck 代码中，经常看到：

```python
expansion = 4
```

这是今天必须认识的新词。

例如我们把 Bottleneck 中间的基础 Channel 设为：

```text
planes = 64
```

那么最终输出 Channel：

```text
64 × 4 = 256
```

所以：

```text
中间 Channel = 64
最终输出 Channel = 256
```

可以先记：

> **Bottleneck 的 `expansion=4` 表示最终输出 Channel 是中间基础 Channel 的 4 倍。**

这也是为什么看真实 ResNet-50 代码时，经常会突然看到：

`64 → 256`

而不是一直保持 64。

---

## 8. 完整 Bottleneck 小实验

下面先写一个适合学习 Shape 的简化 Bottleneck。

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

因为：



```text
planes = 64
expansion = 4
```

最终：

```text
out_channels = 64×4 = 256
```

所以：

`[4,64,32,32] → Bottleneck → [4,256,32,32]`

Shortcut 也必须通过 `1×1 Conv`：

`[4,64,32,32] → [4,256,32,32]`

才能与主分支相加。

---

## 9. Bottleneck 的 Shape 一步一步推

输入：

```text
[B,64,32,32]
```

第一层：

```text
1×1 Conv
64 → 64
```

得到：

```text
[B,64,32,32]
```

第二层：

```text
3×3 Conv
64 → 64
```

得到：

```text
[B,64,32,32]
```

第三层：

```text
1×1 Conv
64 → 256
```

得到：

```text
[B,256,32,32]
```

Shortcut：

```text
[B,64,32,32]
→ 1×1 Conv
→ [B,256,32,32]
```

最后：

```text
[B,256,32,32]
+
[B,256,32,32]

→
[B,256,32,32]
```

---

## 10. BasicBlock 和 Bottleneck 放在一起看

| | BasicBlock | Bottleneck |
|---|---|---|
| 主要使用 | ResNet-18 / 34 | ResNet-50 / 101 / 152 |
| 主分支 | `3×3 → 3×3` | `1×1 → 3×3 → 1×1` |
| 1×1 Conv | 主要用于 Shortcut Projection | 主分支本身就大量使用 |
| expansion | 通常可看作 1 | 经典实现为 4 |
| 特点 | 结构直观 | 更适合构建很深的网络 |

不要把它们理解成两种完全不同的网络思想。

它们都遵守同一个核心：

`主分支 F(x) + Shortcut`

区别只是：

> **F(x) 内部怎么计算。**

---

## 11. 从一个 Block 到一整个 ResNet

现在已经认识：

```text
BasicBlock
Bottleneck
```

下一步不是把它们只用一次，而是连续堆很多个。

例如可以粗略想成：

```text
输入
→ 前置 Conv
→ Stage 1
→ Stage 2
→ Stage 3
→ Stage 4
→ 分类头
```

这里的新词是：

**Stage**

可以先理解成：

> **一组输出空间尺度相同、结构相似的 Residual Blocks。**

例如：

```text
Stage 1
Block → Block → Block

Stage 2
Block → Block → Block → Block
```

一个 Stage 里面会重复多个残差块。

进入下一个 Stage 时，通常会发生：

```text
H/W 变小
C 变大
```

例如非常粗略地理解：

```text
空间：
56×56
→ 28×28
→ 14×14
→ 7×7

Channel：
较少
→ 更多
→ 更多
→ 更多
```

这正是下一课理解 ResNet-50 的 `C2 / C3 / C4 / C5` 所需要的基础。

---

## 12. `layer1 / layer2 / layer3 / layer4` 是什么

看 PyTorch ResNet 源码时，经常会遇到：

```python
self.layer1
self.layer2
self.layer3
self.layer4
```

这里的 `layer` 不代表“只有一层 Conv”。

它通常表示：

> **一个 Stage，也就是一组连续的 Residual Blocks。**

例如概念上：

```text
layer1 = Block + Block + Block
layer2 = Block + Block + Block + Block
```

所以以后看到：

```python
self.layer2 = ...
```

不要立刻理解成“第二个卷积层”。

先想到：

> **这是 ResNet 的第二组 Residual Blocks。**

---

## 13. 一个最小 ResNet 骨架实验

今天不用实现官方 ResNet-50，只做一个 TinyResNet，目的是观察：

> 多个 BasicBlock 怎样组成多个 Stage。

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

运行前应该先自己推：

```text
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
```

这就是“很多 Block 组成 Stage，很多 Stage 再组成 ResNet”的最小版本。

---

## 14. Day 5 今天真正要形成的整体关系

今天不要把知识记成一堆孤立名词。

应该串成：

`Residual Connection` 是总思想。

在这个思想下，可以设计不同的残差块：

```text
Residual Block
├─ BasicBlock
└─ Bottleneck
```

很多同类 Block 再组成：

```text
Stage / layer
```

多个 Stage 继续串联：

```text
ResNet
```

所以整体层级是：

```text
ResNet
→ Stage
→ Residual Block
→ Conv / BN / ReLU
```

以后阅读源码时，要从这个层级往下拆，而不是把所有 `Conv2d` 看成一条没有层次的长代码。

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
