# Day 5 总结复习手册：BasicBlock、Bottleneck、Stage 与 ResNet

> 复习目标：不用重新通读整篇教案，也能快速恢复 Day 5 的知识框架，并能够继续完成 **看源码 → 推 Shape → 判断结构 → 反推代码**。

---

# 1. Day 5 总知识链

```text
Residual Connection
↓
Residual Block
├─ BasicBlock
└─ Bottleneck
↓
多个 Residual Blocks
↓
Stage / layer
↓
多个 Stage
↓
ResNet
```

同时始终追踪两类 Shape 变化：

```text
Channel 方向：
C

Spatial 方向：
H / W
```

---

# 2. 核心概念速查表

| 概念 | 是什么 | 看到什么时想到它 |
|---|---|---|
| Channel | Tensor 的 `C` 维 | 每个空间位置有多少维特征 |
| Spatial dimensions | `H/W` | 特征图的空间大小 |
| Channel reduction | 降低 `C` | `256 → 64` |
| Spatial downsampling | 降低 `H/W` | `56×56 → 28×28` |
| Residual Block | 主分支 + Shortcut + Add | `F(x)+identity` |
| BasicBlock | 一种 Residual Block | `3×3 → 3×3` |
| Bottleneck | 一种 Residual Block | `1×1 → 3×3 → 1×1` |
| Projection Shortcut | 调整 Shortcut Shape | `1×1 Conv` |
| BatchNorm2d | Conv 后常见归一化层 | Shape 通常不变 |
| `planes` | Bottleneck 中间基础 Channel | 可先理解成 `middle_channels` |
| `expansion` | 最终 C 相对 planes 的倍数 | 经典 Bottleneck 常为 4 |
| Stage | 一组 Residual Blocks | 同一 spatial scale |
| `layer1~4` | torchvision 源码变量名 | 架构上通常对应 Stage |
| `nn.Sequential` | PyTorch 顺序容器 | 依次执行多个 Module |

---

# 3. 最容易混淆的两类变化

## 3.1 Channel reduction

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

常见来源：

```text
out_channels 改变
```

记忆：

> **Channel reduction = 改 C。**

---

## 3.2 Spatial downsampling

例如：

```text
[B,64,56,56]
→
[B,64,28,28]
```

变化：

```text
C：不变
H/W：56×56 → 28×28
```

常见来源：

```text
stride=2
```

记忆：

> **Spatial downsampling = 改 H/W。**

---

## 3.3 不要混成一件事

```text
C ↓
≠
H/W ↓
```

源码阅读时固定问：

```text
C 有没有变？
→ 看 in_channels / out_channels

H/W 有没有变？
→ 看 stride / kernel_size / padding
```

---

# 4. `1×1 Conv` 快速复习

```python
Conv2d(256,64,kernel_size=1,stride=1)
```

输入：

```text
[B,256,H,W]
```

输出：

```text
[B,64,H,W]
```

单个 filter：

```text
[256,1,1]
```

所以 `1×1` 只表示：

```text
空间上看一个位置
```

但会读取：

```text
全部 256 个输入 channels
```

64 个 filter：

```text
→ 64 个输出 channels
```

结论：

> **1×1 Conv 非常适合做 Channel projection。**

不要错误记成：

> `1×1 Conv = 下采样`

是否下采样主要看：

```text
stride
```

---

# 5. BasicBlock

## 5.1 定义

BasicBlock 是：

> **用两个 3×3 Conv 直接计算 F(x) 的 Residual Block。**

主分支：

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

---

## 5.2 为什么两层都是 3×3 Conv

第一层：

```text
第一次局部特征提取 / 组合
```

第二层：

```text
继续组合前一层产生的局部特征
```

最后得到：

```text
F(x)
```

BasicBlock 没有 Bottleneck 的：

```text
先压 C
→ 再扩 C
```

结构，因此更直接。

---

## 5.3 第一层为什么有时 `stride=2`

因为某些 BasicBlock 负责：

> **进入一个新的 Stage / spatial scale。**

例如：

```text
[B,64,32,32]
→
[B,128,16,16]
```

第一层：

```text
3×3 Conv
stride=2

32×32 → 16×16
```

第二层：

```text
stride=1

16×16 → 16×16
```

记忆：

> **两层都是 Conv；第一层必要时兼顾一次下采样。**

---

# 6. BasicBlock 完整代码

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

# 7. BasicBlock Shape 母线

例：

```text
输入：
[B,64,32,32]

参数：
in_channels=64
out_channels=128
stride=2
```

主分支：

```text
[B,64,32,32]

↓ Conv3×3, stride=2
[B,128,16,16]

↓ BN
[B,128,16,16]

↓ ReLU
[B,128,16,16]

↓ Conv3×3, stride=1
[B,128,16,16]

↓ BN
[B,128,16,16]
```

Shortcut：

```text
[B,64,32,32]

↓ 1×1 Conv, stride=2
[B,128,16,16]

↓ BN
[B,128,16,16]
```

Add：

```text
[B,128,16,16]
+
[B,128,16,16]
→
[B,128,16,16]
```

---

# 8. Bottleneck

## 8.1 定义

Bottleneck 是：

> **为了让深网络中的 3×3 Conv 更经济而设计的 Residual Block。**

主分支：

```text
1×1
→ 3×3
→ 1×1
```

完整：

```text
1×1 Conv → BN → ReLU
→ 3×3 Conv → BN → ReLU
→ 1×1 Conv → BN
→ Add
→ ReLU
```

---

## 8.2 三层职责

### 第一层 `1×1`

```text
主要任务：
降低 / 调整 C
```

例如：

```text
256 → 64
```

目的：

> **减少后续 3×3 Conv 的计算量。**

### 第二层 `3×3`

```text
主要任务：
局部空间特征计算
```

在较低 Channel 上做：

```text
64 → 64
```

比直接：

```text
256 → 256
```

便宜很多。

### 第三层 `1×1`

```text
主要任务：
把 C 扩展到最终输出维度
```

例如：

```text
64 → 256
```

之所以仍用 `1×1`：

> **这里只需要高效完成 Channel expansion，不需要再做一次昂贵的 3×3 邻域计算。**

---

# 9. Bottleneck 完整代码

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

# 10. `planes` / `expansion` 速查

经典：

```python
expansion = 4
```

假设：

```text
planes=64
```

则：

```text
out_channels
=
planes × expansion
=
64 × 4
=
256
```

所以：

```python
Conv2d(
    planes,
    out_channels,
    kernel_size=1
)
```

实际就是：

```python
Conv2d(
    64,
    256,
    kernel_size=1
)
```

这里：

```text
64
→ 输入 C

256
→ 输出 C

1
→ kernel_size=1
```

注意：

> **`expansion=4` 本身不会自动改变 Tensor。它只是参与 Python 运算，真正改变 C 的还是 Conv2d 的 `out_channels=256`。**

---

# 11. Bottleneck Shape 母线

例：

```text
输入：
[B,64,32,32]

planes=64
expansion=4
stride=1
```

先算：

```text
out_channels=256
```

主分支：

```text
[B,64,32,32]

↓ 1×1 Conv
64 → 64
[B,64,32,32]

↓ 3×3 Conv
64 → 64
[B,64,32,32]

↓ 1×1 Conv
64 → 256
[B,256,32,32]
```

Shortcut：

```text
原始：
[B,64,32,32]
```

由于：

```text
C：64 ≠ 256
```

需要 Projection：

```text
1×1 Conv
64 → 256
```

得到：

```text
[B,256,32,32]
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

# 12. BasicBlock vs Bottleneck

| 问题 | BasicBlock | Bottleneck |
|---|---|---|
| 都是什么 | Residual Block | Residual Block |
| 共同核心 | `F(x)+identity` | `F(x)+identity` |
| 主分支 | `3×3 → 3×3` | `1×1 → 3×3 → 1×1` |
| 设计风格 | 简单直接 | 更经济 |
| 是否提取特征 | 是 | 是 |
| 是否可能下采样 | 是 | 是 |
| 下采样主要看 | `stride` | `stride` |
| 常见网络 | ResNet-18/34 | ResNet-50/101/152 |

一句话：

> **BasicBlock 与 Bottleneck 的架构角色相同，区别在 F(x) 内部怎么计算。**

---

# 13. Stage

## 13.1 定义

Stage：

> **一组按顺序执行、主要工作在同一 spatial scale 上的 Residual Blocks。**

例如：

```text
Stage 2
├─ Block 1
├─ Block 2
└─ Block 3
```

---

## 13.2 “同尺度”是什么意思

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

都属于：

```text
28×28 spatial scale
```

通常：

```text
H/W 一致
C 也通常一致
Block 类型相同
```

但“同尺度”这个词本身主要指：

```text
H/W
```

---

## 13.3 为什么 Stage 第一个 Block 经常特殊

上一 Stage：

```text
[B,256,56,56]
```

当前 Stage：

```text
[B,512,28,28]
```

第一个 Block 要完成：

```text
C：
256 → 512

H/W：
56×56 → 28×28
```

因此常见：

```text
stride=2
Projection Shortcut
```

后续 Block：

```text
[B,512,28,28]
→
[B,512,28,28]
```

所以：

> **第一个 Block 负责进入新尺度；后续 Block 在这个尺度继续工作。**

---

# 14. `layer1~4` 与 Stage

源码：

```python
self.layer1
self.layer2
self.layer3
self.layer4
```

不要理解成：

```text
4 个 Conv 层
```

它们通常表示：

```text
4 组 Residual Blocks
```

因此架构上可理解为：

```text
layer1 ≈ Stage 1
layer2 ≈ Stage 2
layer3 ≈ Stage 3
layer4 ≈ Stage 4
```

重要：

> **代码变量名 ≠ 必须严格等于理论术语。**

---

# 15. `nn.Sequential`

定义：

> **PyTorch 用来按顺序执行多个 Module 的容器。**

例如：

```python
nn.Sequential(
    block1,
    block2,
    block3
)
```

相当于：

```text
x → block1 → block2 → block3
```

它不是新的数学层。

所以：

```text
Stage
= 架构概念

Sequential
= 代码组织工具
```

---

# 16. TinyResNet 完整代码

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

# 17. TinyResNet 层级图

```text
TinyResNet
│
├─ stem
│  └─ Conv + BN + ReLU
│
├─ layer1 / Stage 1
│  ├─ BasicBlock(64,64,s=1)
│  └─ BasicBlock(64,64,s=1)
│
└─ layer2 / Stage 2
   ├─ BasicBlock(64,128,s=2)
   └─ BasicBlock(128,128,s=1)
```

Shape：

```text
输入
[1,3,32,32]

↓ stem
[1,64,32,32]

↓ layer1
[1,64,32,32]

↓ layer2 Block1
[1,128,16,16]

↓ layer2 Block2
[1,128,16,16]
```

---

# 18. 源码阅读固定五问

以后看到一个模块，固定问：

1. **它属于哪个层级？**
   - ResNet？
   - Stage？
   - Block？
   - Block 内某一层？

2. **输入 Shape 是什么？**

3. **这层改变 C 吗？**
   - 看 `in_channels / out_channels`

4. **这层改变 H/W 吗？**
   - 看 `stride / kernel_size / padding`

5. **下一步为什么能接上？**
   - 特别检查 Residual Add 前两条路径 Shape

---

# 19. 母题

## 母题 1：区分 C 变化和 H/W 变化

给定：

```text
[B,256,56,56]
→
[B,64,56,56]
```

问：

```text
发生了什么？
```

答：

```text
Channel reduction
C：256 → 64
H/W 不变
```

训练点：

> **看到 Shape 先逐维比较。**

---

## 母题 2：BasicBlock 进入新 Stage

给定：

```text
输入：
[B,64,32,32]

要求输出：
[B,128,16,16]
```

推：

```text
Conv1：
64→128
stride=2

Conv2：
128→128
stride=1

Shortcut：
64→128
stride=2
```

训练点：

> **从输入/输出 Shape 反推 Block 参数。**

---

## 母题 3：为什么 BasicBlock 第二层不用 stride=2

因为第一层已经：

```text
32×32 → 16×16
```

第二层再 `stride=2` 会：

```text
16×16 → 8×8
```

训练点：

> **一个 Stage 边界通常只完成一次尺度切换。**

---

## 母题 4：Bottleneck 三层职责

给定：

```text
256 → 64 → 64 → 256
```

回答：

```text
第一层 1×1：
降 C

中间 3×3：
较低 C 上做局部特征计算

最后 1×1：
扩 C
```

训练点：

> **不只认 kernel，还要知道设计目的。**

---

## 母题 5：`planes=64, expansion=4`

推：

```text
out_channels
= 64×4
= 256
```

所以最后：

```text
Conv2d(64,256,1)
```

训练点：

> **区分 Python 变量计算和真正改变 Tensor 的 Conv。**

---

## 母题 6：什么时候 Shortcut 需要 Projection

主分支：

```text
[B,256,32,32]
```

identity：

```text
[B,64,32,32]
```

C 不同。

所以：

```text
1×1 Conv
64→256
```

训练点：

> **Add 前检查完整 Shape，不只检查 H/W。**

---

## 母题 7：Stage 第一个 Block 为什么特殊

上一 Stage：

```text
[B,256,56,56]
```

当前 Stage：

```text
[B,512,28,28]
```

第一个 Block：

```text
负责 C 变化
+
负责 H/W 下采样
+
Shortcut 对齐
```

后续 Block：

```text
Shape 基本保持
```

训练点：

> **理解 Stage 边界。**

---

# 20. 错题本

## 错题 1：把 `1×1 Conv` 当成下采样

错误：

```text
1×1 → H/W 变小
```

正确：

```text
1×1
→ kernel 大小

stride=2
→ 常见下采样原因
```

记忆：

> **kernel 看多大区域，stride 看走多远。**

---

## 错题 2：把 Conv 和 downsampling 当成两种平行层

错误：

```text
Conv 或 Downsample
```

正确：

```text
Conv
可以因为 stride=2
产生 downsampling 效果
```

---

## 错题 3：认为 BasicBlock 与 Bottleneck 都是 Residual Block，因为都能提特征

错误。

普通 CNN 也能提特征。

正确定义：

```text
主分支 F(x)
+
Shortcut identity
+
Residual Add
```

---

## 错题 4：认为 Bottleneck 一定压缩 H/W

错误。

Bottleneck 的核心：

```text
压 C
→ 低 C 上做 3×3
→ 扩 C
```

H/W 是否变化另看：

```text
stride
```

---

## 错题 5：`expansion=4` 会自动改变 Tensor

错误。

真正过程：

```text
expansion=4
↓
Python 算出 out_channels
↓
Conv2d 使用这个 out_channels
↓
Tensor 的 C 才真正改变
```

---

## 错题 6：Stage 就是一层

错误。

```text
Stage
→ 多个 Residual Blocks
```

源码中的：

```text
layer1
```

往往是一个 Stage，不是一层 Conv。

---

# 21. 重点掌握清单

## 必须能解释

- [ ] Channel reduction
- [ ] Spatial downsampling
- [ ] `1×1 Conv`
- [ ] BasicBlock
- [ ] Bottleneck
- [ ] Residual Block
- [ ] Projection Shortcut
- [ ] `planes`
- [ ] `expansion`
- [ ] Stage
- [ ] `layer1~4`
- [ ] `nn.Sequential`

## 必须能推 Shape

- [ ] BasicBlock：`64,32×32 → 128,16×16`
- [ ] Bottleneck：`64 → 64 → 64 → 256`
- [ ] Shortcut Projection
- [ ] Stage 第一个 Block
- [ ] Stage 后续 Block

## 必须能看源码判断

- [ ] `stride=2` → H/W 可能变小
- [ ] `out_channels` 改变 → C 改变
- [ ] `Conv2d(...,1)` → 1×1 Channel projection
- [ ] `expansion=4` → 参与计算 final C
- [ ] `nn.Sequential` → 顺序容器
- [ ] `self.layer2` → 通常是 Stage，不是一层 Conv
- [ ] `out + identity` → Residual Add，先检查 Shape

## 必须开始会反推代码

给定：

```text
输入 Shape
目标输出 Shape
Block 类型
stride
```

能够判断：

```text
conv1 怎么写
conv2 怎么写
conv3 是否存在
Shortcut 是否需要 projection
每一步 Shape 是什么
```

---

# 22. 触发式记忆

看到：

```text
C: 256 → 64
```

想到：

> Channel reduction。

看到：

```text
56×56 → 28×28
```

想到：

> Spatial downsampling。

看到：

```python
Conv2d(..., kernel_size=1)
```

想到：

> Channel projection / 1×1，不等于一定下采样。

看到：

```python
stride=2
```

想到：

> H/W 可能减小。

看到：

```text
3×3 → 3×3
```

想到：

> BasicBlock。

看到：

```text
1×1 → 3×3 → 1×1
```

想到：

> Bottleneck。

看到：

```python
expansion = 4
```

想到：

> `final C = planes × 4`。

看到：

```python
self.layer2
```

想到：

> 一个 Stage / 一组 Blocks。

看到：

```python
nn.Sequential(...)
```

想到：

> 顺序执行里面的 Module。

---

# Day 5 最终必须牢牢记住

```text
Residual Block
=
主分支 F(x)
+
Shortcut identity
+
Residual Add
```

```text
BasicBlock
=
3×3 → 3×3
```

```text
Bottleneck
=
1×1 → 3×3 → 1×1

先降 C
→ 低 C 上做 3×3
→ 再扩 C
```

```text
Channel reduction
→ 改 C

Spatial downsampling
→ 改 H/W
```

```text
Stage
→ 一组 Residual Blocks
→ H/W 通常处于同一 spatial scale
```

```text
ResNet
→ Stage
→ Residual Block
→ Conv / BN / ReLU
```

最关键的一句话：

> **看 ResNet 源码时，不要逐行死记；先判断当前代码属于 ResNet、Stage、Block 还是 Block 内部层，再分别追踪 C 和 H/W，最后检查主分支与 Shortcut 能否在 Residual Add 前完成 Shape 对齐。**
