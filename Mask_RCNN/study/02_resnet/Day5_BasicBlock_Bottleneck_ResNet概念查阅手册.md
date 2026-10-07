# Day 5 查阅手册：BasicBlock、Bottleneck、Stage 与 ResNet 代码约定

> 用途：快速查阅 Day 5 中最容易混淆的概念、术语、Shape 变化和源码命名。  
> 核心目标：分清 **Channel 变化、空间下采样、BasicBlock、Bottleneck、Stage、layer、Sequential、planes、expansion**。

---

# 1. 先统一 Tensor 的两类“尺寸”

CNN 中常见 Tensor：

```text
[B, C, H, W]
```

这 4 个维度里，实际经常分成两组来讨论：

| 维度 | 常用术语 | 含义 |
|---|---|---|
| `B` | Batch dimension | 一次处理多少个样本 |
| `C` | Channel dimension / feature dimension | 每个空间位置用多少维特征描述 |
| `H, W` | Spatial dimensions | 特征图的空间高和宽 |

## 1.1 Channel 方向

例如：

```text
[B,256,56,56]
→
[B,64,56,56]
```

变化的是：

```text
C：256 → 64
H/W：不变
```

常见说法：

- reduce channels
- channel reduction
- channel compression
- channel projection
- feature dimension reduction

可以统一理解为：

> **改变每个空间位置的特征维度。**

## 1.2 Spatial 方向

例如：

```text
[B,64,56,56]
→
[B,64,28,28]
```

变化的是：

```text
C：不变
H/W：56×56 → 28×28
```

常见说法：

- downsampling
- spatial downsampling
- reduce spatial resolution
- spatial resolution decreases

可以统一理解为：

> **让特征图在空间上变小。**

## 1.3 两者不是“两套系统”

它们只是同一个 Tensor 的不同维度方向：

```text
[B, C, H, W]
    ↑  ↑  ↑
    │  └──┴── Spatial
    └──────── Channel
```

以后看到“压缩 Channel”，想到 `C ↓`；看到“下采样”，想到 `H/W ↓`。

最重要：

> **Channel reduction 和 spatial downsampling 不是一回事。**

---

# 2. `1×1 Conv` 到底在干什么

`1×1` 只表示：

> **在 H/W 空间上每次只看 1×1 的位置。**

它并不表示只看一个 Channel。

例如：

```python
nn.Conv2d(
    in_channels=256,
    out_channels=64,
    kernel_size=1,
    stride=1
)
```

单个 filter 的 Shape：

```text
[256,1,1]
```

所以一个 filter 在某个空间位置会读取全部 256 个输入 channels，做加权求和，得到 1 个输出值。64 个 filter 就会得到 64 个输出值，因此：

```text
[B,256,56,56]
→
[B,64,56,56]
```

这里：

```text
C：256 → 64
H/W：56×56 → 56×56
```

结论：

> **1×1 Conv 的主要作用之一是 Channel 映射。是否改变 H/W，要看 stride 等参数。**

---

# 3. Bottleneck 第一层 `1×1 Conv` 的主要目的

Bottleneck 的经典结构：

```text
1×1 Conv
→ 3×3 Conv
→ 1×1 Conv
```

它的核心设计动机是：

> **先把 Channel 压小，在较低 Channel 上执行昂贵的 3×3 Conv，再把 Channel 扩展回最终输出维度。**

例如：

```text
输入：
[B,256,56,56]
```

第一层：

```text
1×1 Conv
256 → 64
```

得到：

```text
[B,64,56,56]
```

第二层：

```text
3×3 Conv
64 → 64
```

得到：

```text
[B,64,56,56]
```

第三层：

```text
1×1 Conv
64 → 256
```

得到：

```text
[B,256,56,56]
```

所以经典 Bottleneck 是：

```text
宽 → 窄 → 窄 → 宽
256 → 64 → 64 → 256
```

注意：

> **这个过程中 H/W 可以完全不变。**

---

# 4. Bottleneck 不是“压 Channel + 压 H/W + 扩 Channel”

错误理解：

```text
1×1
→ 压 Channel

3×3
→ 压 H/W

1×1
→ 扩 Channel
```

这不准确。

正确理解：

| 层 | 主要作用 |
|---|---|
| 第一层 `1×1` | 降低 / 调整 C |
| 中间 `3×3` | 在较低 C 上进行主要局部空间特征计算 |
| 最后一层 `1×1` | 扩展到最终输出 C |

H/W 是否变小：

> **主要看 stride，而不是看 kernel_size 是 1×1 还是 3×3。**

---

# 5. Bottleneck 什么时候会下采样

如果某一层 `stride=2`，才会让 H/W 减小。

例如：

```text
[B,256,56,56]
→
[B,256,28,28]
```

这就是 spatial downsampling。

PyTorch torchvision 的常见 ResNet Bottleneck 实现中，`stride=2` 通常放在中间的 `3×3 Conv` 上。

所以可以这样记：

```text
out_channels
→ 主要决定 C

stride
→ 主要决定 H/W 是否缩小
```

---

# 6. BasicBlock 到底是什么

BasicBlock 是一种最直接的 Residual Block。

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

所以完整结构：

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

BasicBlock 常用于：

```text
ResNet-18
ResNet-34
```

可以把它理解成：

> **简单、直接的 Residual Block。**

---

# 7. BasicBlock 两层都是 Conv

BasicBlock 的两层主要都是：

```text
3×3 Conv
```

不是：

```text
一层 Conv
+
一层 Downsampling
```

Downsampling 只是某一层 Conv 的 `stride=2` 带来的结果。

## 7.1 普通情况

```text
Conv1：stride=1
Conv2：stride=1
```

例如：

```text
[B,64,32,32]
→ [B,64,32,32]
→ [B,64,32,32]
```

## 7.2 进入新 Stage 时

第一层可能：

```text
Conv1：stride=2
```

于是：

```text
[B,64,32,32]
→
[B,128,16,16]
```

第二层通常：

```text
Conv2：stride=1
```

保持：

```text
[B,128,16,16]
```

所以：

> **第一层必要时负责切换到新的空间尺度，第二层在新尺度上继续提取特征。**

---

# 8. 为什么只在第一层做一次下采样

假设输入是 `32×32`。如果两层都 `stride=2`：

```text
32×32
→ 16×16
→ 8×8
```

一个 Block 内连续缩小两次，会下降得过快。

通常只需要：

```text
32×32
→ 16×16
```

完成一次尺度切换。

所以：

```text
第一层：stride=2
第二层：stride=1
```

是常见设计。

---

# 9. BasicBlock 和 Bottleneck 为什么都叫 Residual Block

因为两者的真正共同核心不是“都由 Conv + BN + ReLU 组成”，而是：

```text
主分支计算 F(x)

Shortcut 传递 identity

最后执行：
F(x) + identity
```

也就是：

```text
Residual Block
├─ 主分支 F(x)
├─ Shortcut
└─ Add
```

BasicBlock 与 Bottleneck 只是：

> **F(x) 内部的计算方式不同。**

---

# 10. BasicBlock 与 Bottleneck 的最大结构区别

## BasicBlock

```text
3×3 Conv
→ BN
→ ReLU
→ 3×3 Conv
→ BN
→ Shortcut Add
→ ReLU
```

## Bottleneck

```text
1×1 Conv
→ BN
→ ReLU
→ 3×3 Conv
→ BN
→ ReLU
→ 1×1 Conv
→ BN
→ Shortcut Add
→ ReLU
```

对比：

| | BasicBlock | Bottleneck |
|---|---|---|
| Block 类型 | Residual Block | Residual Block |
| 主分支层数 | 2 个 Conv | 3 个 Conv |
| 主分支 kernel | `3×3 → 3×3` | `1×1 → 3×3 → 1×1` |
| 是否有 Shortcut | 有 | 有 |
| 最后是否 `F(x)+identity` | 是 | 是 |
| 常见 ResNet | 18 / 34 | 50 / 101 / 152 |
| 设计特点 | 简单直接 | 更适合深网络，计算更经济 |

最重要：

> **它们不是“功能不同”，而是承担相同架构角色的两种内部实现。**

---

# 11. 为什么要设计 Bottleneck

如果高 Channel 上直接做：

```text
256 → 256 的 3×3 Conv
```

单个空间位置大约需要：

```text
256 × 256 × 3 × 3
= 589,824
```

如果先压到 64：

```text
64 → 64 的 3×3 Conv
```

则：

```text
64 × 64 × 3 × 3
= 36,864
```

所以 Bottleneck 的核心思路是：

> **把昂贵的 3×3 Conv 放到较低 Channel 的空间里做。**

这使它更适合堆很多层，形成：

```text
ResNet-50
ResNet-101
ResNet-152
```

---

# 12. `planes` 是什么

在 torchvision ResNet 源码里：

```python
planes
```

只是一个变量名，不是必须背的数学术语。

在 Bottleneck 中，可以先理解成：

> **中间那条“窄路”的基础 Channel 数。**

例如：

```python
planes = 64
```

那么：

```text
conv1：
? → 64

conv2：
64 → 64

conv3：
64 → 256
```

---

# 13. `expansion=4` 是什么

经典 Bottleneck：

```python
expansion = 4
```

表示最终输出 Channel：

```text
out_channels
=
planes × expansion
```

例如：

```text
planes = 64
expansion = 4
```

则：

```text
out_channels
=
64 × 4
=
256
```

---

# 14. `out_channels = planes * expansion` 的代码怎么读

原代码：

```python
class Bottleneck(nn.Module):
    expansion = 4
```

后面：

```python
out_channels = planes * self.expansion
```

假设：

```python
planes = 64
```

Python 实际计算：

```text
out_channels
=
64 × 4
=
256
```

于是：

```python
self.conv3 = nn.Conv2d(
    planes,
    out_channels,
    kernel_size=1
)
```

等价于：

```python
self.conv3 = nn.Conv2d(
    64,
    256,
    kernel_size=1
)
```

真正把 Channel 改成 256 的：

> **是这层 `Conv2d(64,256,1)`。**

`expansion=4` 本身只是参与计算 `out_channels` 的一个 Python 数值。

---

# 15. `planes` 换成更直观的名字

如果暂时不喜欢 `planes`，可以脑内翻译：

```text
planes
≈ middle_channels
```

而：

```text
planes * expansion
≈ final_out_channels
```

例如：

```python
middle_channels = 64
final_out_channels = 256
```

就比：

```python
planes = 64
expansion = 4
```

更容易直观理解。

---

# 16. Stage 是什么

Stage 可以理解成：

> **一组按顺序堆叠的 Residual Blocks。**

例如：

```text
Stage 2
├─ Block 1
├─ Block 2
└─ Block 3
```

## 16.1 “同尺度”是什么意思

在 CNN 中说：

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

这些 Tensor 的空间尺度都是：

```text
28×28
```

## 16.2 一个 Stage 内通常还有这些特点

通常：

1. 使用相同类型的 Block
2. 后续 Block 的输出 C 通常一致
3. 后续 Block 的 H/W 通常一致

但要注意：

> **“同尺度”这个词本身主要指 H/W 相同。**

---

# 17. 为什么 Stage 第一个 Block 经常特殊

上一 Stage：

```text
[B,256,56,56]
```

下一 Stage 想变成：

```text
[B,512,28,28]
```

那么新 Stage 的第一个 Block 要完成：

```text
C：
256 → 512

H/W：
56×56 → 28×28
```

所以它常使用：

```text
stride=2
+ Projection Shortcut
```

后续 Blocks：

```text
[B,512,28,28]
→
[B,512,28,28]
→
[B,512,28,28]
```

继续保持当前尺度。

类比：

> **第一个 Block 像进入新楼层的楼梯；后面的 Block 都在这个楼层里工作。**

---

# 18. 为什么源码叫 `layer1`，概念上却叫 Stage

PyTorch torchvision ResNet 代码里常见：

```python
self.layer1
self.layer2
self.layer3
self.layer4
```

但每个 `layerX` 内部通常包含多个 Residual Blocks，所以从架构层级来看，它更像一个 Stage。

也就是说：

```text
代码变量名：
layer1

架构概念：
Stage 1
```

这只是实现命名习惯。

可以记：

> **代码变量名不一定等于理论术语。**

---

# 19. `nn.Sequential(...)` 是什么

`nn.Sequential` 是 PyTorch 的一个容器，用来：

> **把多个 Module 按给定顺序串起来。**

例如：

```python
self.stem = nn.Sequential(
    nn.Conv2d(...),
    nn.BatchNorm2d(...),
    nn.ReLU()
)
```

执行：

```python
x = self.stem(x)
```

等价于：

```python
x = conv(x)
x = bn(x)
x = relu(x)
```

所以：

> **Sequential 本身不是新的数学操作，只是“按顺序执行这些层”的代码组织工具。**

---

# 20. 为什么 Stage 也适合用 Sequential

Stage 中的数据通常就是：

```text
Block 1
→ Block 2
→ Block 3
→ ...
```

所以：

```python
self.layer1 = nn.Sequential(
    block1,
    block2,
    block3
)
```

执行：

```python
x = self.layer1(x)
```

等价于：

```python
x = block1(x)
x = block2(x)
x = block3(x)
```

因此 `Sequential` 很适合表达：

> **多个 Block 按顺序组成一个 Stage。**

---

# 21. ResNet 的层级关系

最终应该建立这棵树：

```text
ResNet
│
├─ Stage 1
│  ├─ Block 1
│  ├─ Block 2
│  └─ ...
│
├─ Stage 2
│  ├─ Block 1
│  ├─ Block 2
│  └─ ...
│
├─ Stage 3
│  └─ ...
│
└─ Stage 4
   └─ ...
```

Block 有两种经典实现：

```text
Residual Block
├─ BasicBlock
│  └─ 3×3 → 3×3
│
└─ Bottleneck
   └─ 1×1 → 3×3 → 1×1
```

每个 Block 内部：

```text
主分支 F(x)
+
Shortcut
```

---

# 22. 一张表把所有概念放在一起

| 概念 | 所属层级 | 核心作用 |
|---|---|---|
| Conv | Block 内部 | 提取 / 组合特征 |
| BN | Block 内部 | 归一化并稳定训练 |
| ReLU | Block 内部 | 引入非线性 |
| Shortcut | Block 内部 | 旁路传递 identity |
| BasicBlock | Block 类型 | `3×3 → 3×3` |
| Bottleneck | Block 类型 | `1×1 → 3×3 → 1×1` |
| Stage | ResNet 中间层级 | 一组 Residual Blocks |
| `layer1...4` | 代码变量名 | 通常对应四个 Stage |
| `Sequential` | PyTorch 容器 | 顺序执行多个 Module |
| `planes` | 源码变量 | Bottleneck 中间基础 C |
| `expansion` | 源码常量 | 最终 C 相对 planes 的倍数 |

---

# 23. 两组 Shape 例子

## 23.1 BasicBlock：进入新 Stage

输入：

```text
[B,64,32,32]
```

主分支：

```text
3×3 Conv, stride=2
64 → 128
→ [B,128,16,16]

3×3 Conv, stride=1
128 → 128
→ [B,128,16,16]
```

Shortcut：

```text
1×1 Conv, stride=2
64 → 128
→ [B,128,16,16]
```

Add：

```text
[B,128,16,16]
+
[B,128,16,16]
→
[B,128,16,16]
```

## 23.2 Bottleneck：不下采样

输入：

```text
[B,256,56,56]
```

主分支：

```text
1×1 Conv
256 → 64
→ [B,64,56,56]

3×3 Conv
64 → 64
→ [B,64,56,56]

1×1 Conv
64 → 256
→ [B,256,56,56]
```

Shortcut：

```text
identity = x
→ [B,256,56,56]
```

Add：

```text
[B,256,56,56]
+
[B,256,56,56]
→
[B,256,56,56]
```

---

# 24. 高频误区

## 误区 1：`1×1 Conv` 就是下采样

错误。`1×1` 只描述 kernel 的空间大小；是否下采样主要看 `stride`。

## 误区 2：Channel reduction 就是 H/W 变小

错误。`C ↓` 和 `H/W ↓` 是两个不同维度方向的变化。

## 误区 3：BasicBlock 第一层是“下采样层”

错误。它首先是 `3×3 Conv`，只是必要时 `stride=2`，于是产生下采样效果。

## 误区 4：Bottleneck 的 3×3 一定负责下采样

不一定。只有 `stride=2` 时 H/W 才会下降。

## 误区 5：`expansion=4` 会自动把 Tensor 放大 4 倍

错误。它只是一个 Python 数值，参与 `out_channels = planes × expansion`。真正改变 C 的仍然是 `Conv2d(..., out_channels=...)`。

## 误区 6：Stage 就等于一层 Conv

错误。Stage 通常由多个 Residual Blocks 组成。

---

# 25. 触发式记忆

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
Conv2d(256,64,1)
```

想到：

> 1×1 Channel projection。

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

> 代码名叫 layer，架构上通常对应一个 Stage。

看到：

```python
nn.Sequential(...)
```

想到：

> 按顺序执行里面的 Module。

---

# 26. 最终必须牢牢记住

```text
Channel reduction
→ 改 C

Spatial downsampling
→ 改 H/W
```

```text
BasicBlock
→ 3×3 → 3×3
→ 简单直接
```

```text
Bottleneck
→ 1×1 → 3×3 → 1×1
→ 先降 C
→ 在较低 C 上做昂贵 3×3
→ 再扩 C
```

```text
BasicBlock 和 Bottleneck
共同核心：
F(x) + identity
```

```text
Stage
→ 一组 Residual Blocks
→ 同一 Stage 内 H/W 通常保持同一空间尺度
```

```text
layer1 / layer2 / layer3 / layer4
→ torchvision 源码变量名
→ 架构上通常对应 Stage
```

```text
nn.Sequential
→ PyTorch 顺序容器
→ 不增加新的数学操作
```

最关键的一句话：

> **BasicBlock 和 Bottleneck 都是在实现同一个 Residual Block 角色；BasicBlock 用两个 3×3 Conv 直接计算 F(x)，Bottleneck 为了让深网络更经济，先用 1×1 Conv 降低 Channel，再在较窄 Channel 上做 3×3 Conv，最后用 1×1 Conv 扩展回最终 Channel。是否下采样主要看 stride，而不是看 1×1 或 3×3。**
