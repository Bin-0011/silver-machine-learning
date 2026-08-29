# Day 4 总结：Residual Connection + BasicBlock

## 1. 核心概念表

| 概念 | 是什么 | 看到什么时想到它 |
|---|---|---|
| Residual | `F(x)+x` 的残差结构 | 主分支 + Shortcut |
| `H(x)` | block 最终希望得到的映射 | `H(x)=F(x)+x` |
| `F(x)` | 主分支学习的变化量 | Conv 后得到的 `out` |
| Shortcut | x 绕过中间层的旁路 | 输入直接连到加法位置 |
| identity | Shortcut 上保存的 x | `identity=x` |
| Residual Add | 两条路径逐元素相加 | `out=out+identity` |
| Identity Shortcut | x 原样通过 Shortcut | Shape 已经一样 |
| Projection Shortcut | Shortcut 先调整 Shape | 常见 1×1 Conv |
| downsample | Shortcut 上常见的 Shape 调整模块名 | `self.downsample(...)` |
| 1×1 Conv | 用 1×1 空间窗口做通道映射 | 常用于 Projection |

---

## 2. Residual 核心关系

普通卷积块：

```text
x → Conv → ReLU → Conv → H(x)
```

Residual Block：

```text
主分支：
x → Conv → ReLU → Conv → F(x)

Shortcut：
x → identity

最后：
F(x)+identity
```

公式：

```text
F(x)=H(x)-x
H(x)=F(x)+x
```

可以先把 `F(x)` 理解成：

> 原输入 x 基础上还需要补上的变化。

---

## 3. Shortcut / identity / downsample 对照

| 名词 | 本质 |
|---|---|
| Shortcut | 路径 |
| identity | Shortcut 上保存的 Tensor |
| downsample | 必要时放在 Shortcut 上的 Shape 调整模块 |
| Projection | 对 Shortcut 做可学习映射 |

最重要：

> **Shortcut 是路，identity 是路上的数据，downsample/projection 是必要时放在路上的变换模块。**

---

## 4. Shape 一样时：BasicBlock

核心代码：

```python
import torch
import torch.nn as nn


class BasicBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()

        self.conv1 = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
        )

        self.relu = nn.ReLU()

        self.conv2 = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
        )

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)

        out = out + identity
        out = self.relu(out)

        return out


if __name__ == "__main__":
    x = torch.randn(4,64,32,32)

    block = BasicBlock(64)

    out = block(x)

    print("输入:", x.shape)
    print("输出:", out.shape)
```

Shape：

| 位置 | Shape |
|---|---|
| 输入 x | `[B,64,32,32]` |
| Conv1 | `[B,64,32,32]` |
| ReLU | `[B,64,32,32]` |
| Conv2 | `[B,64,32,32]` |
| identity | `[B,64,32,32]` |
| Add | `[B,64,32,32]` |
| 输出 | `[B,64,32,32]` |

结论：

> Shape 一样时，Shortcut 直接使用 `identity=x`。

---

## 5. Shape 不一样时：Projection Shortcut

主分支：

```text
[B,64,32,32]
→ Conv2d(64,128,3,stride=2,padding=1)
→ [B,128,16,16]
```

Shortcut 原始 x：

```text
[B,64,32,32]
```

两边 Shape 不一致，所以 Shortcut 加：

```python
nn.Conv2d(
    64,
    128,
    kernel_size=1,
    stride=2,
    bias=False
)
```

得到：

```text
[B,64,32,32]
→ [B,128,16,16]
```

这样：

```text
[B,128,16,16]
+
[B,128,16,16]
→
[B,128,16,16]
```

---

## 6. 1×1 Conv 在 Shortcut 上做了什么

| 参数 | 作用 |
|---|---|
| `in_channels=64` | 读取 64 个输入 channels |
| `out_channels=128` | 输出 C=128 |
| `kernel_size=1` | 使用 1×1 空间窗口 |
| `stride=2` | H/W 从 32×32 变成 16×16 |

所以：

```text
[B,64,32,32]
→ 1×1 Conv, stride=2
→ [B,128,16,16]
```

---

## 7. 带 downsample 的完整代码

```python
import torch
import torch.nn as nn


class BasicBlockWithDownsample(nn.Module):
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

        self.relu = nn.ReLU()

        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
        )

        self.downsample = None

        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
                stride=stride,
                bias=False
            )

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)

        if self.downsample is not None:
            identity = self.downsample(identity)

        out = out + identity
        out = self.relu(out)

        return out


if __name__ == "__main__":
    x = torch.randn(4,64,32,32)

    block = BasicBlockWithDownsample(
        in_channels=64,
        out_channels=128,
        stride=2
    )

    out = block(x)

    print("输入:", x.shape)
    print("输出:", out.shape)
```

---

## 8. Shape 速查表

| 路径/位置 | Shape |
|---|---|
| 输入 x | `[B,64,32,32]` |
| 主分支 Conv1 | `[B,128,16,16]` |
| 主分支 ReLU | `[B,128,16,16]` |
| 主分支 Conv2 | `[B,128,16,16]` |
| Shortcut 原始 identity | `[B,64,32,32]` |
| Shortcut 1×1 Conv | `[B,128,16,16]` |
| Add | `[B,128,16,16]` |
| 最终输出 | `[B,128,16,16]` |

---

## 9. 两种 Shortcut 对比

| | Identity Shortcut | Projection Shortcut |
|---|---|---|
| 使用条件 | Shape 已一致 | Shape 不一致 |
| identity | `identity=x` | `identity=projection(x)` |
| 是否有额外 Conv | 否 | 常见 1×1 Conv |
| 是否可改 C | 否 | 可以 |
| 是否可改 H/W | 否 | 配合 stride 可以 |

---

# 10. 母题

## 母题 1：Shape 相同时怎么判断

给定：

```text
输入 [B,64,32,32]
Conv1: 64→64, k=3,s=1,p=1
Conv2: 64→64, k=3,s=1,p=1
```

主分支最终：

```text
[B,64,32,32]
```

Shortcut：

```text
[B,64,32,32]
```

所以：

```text
Identity Shortcut
identity=x
```

这道母题训练：

> **看到主分支 Shape 没变，先判断是否可以直接 identity shortcut。**

---

## 母题 2：Shape 不一样时怎么处理

给定：

```text
x=[B,64,32,32]
主分支输出=[B,128,16,16]
```

Shortcut 需要调整为：

```text
[B,128,16,16]
```

常见：

```python
Conv2d(64,128,1,stride=2)
```

这道母题训练：

> **看到 Residual Add 两边 Shape 不一样，先想 Projection Shortcut。**

---

## 母题 3：1×1 Conv 为什么能改 C

给定：

```python
Conv2d(64,128,kernel_size=1)
```

单个 filter：

```text
[64,1,1]
```

filter 数量：

```text
128
```

所以输出：

```text
C=128
```

这道母题训练：

> **看到 `out_channels`，想到输出 channel 数；kernel=1 不代表只能看 1 个 channel。**

---

## 母题 4：`out + identity` 是什么运算

给定：

```text
out =
0.1 0.2
0.3 0.4

identity =
1.0 2.0
3.0 4.0
```

得到：

```text
1.1 2.2
3.3 4.4
```

这道母题训练：

> **Residual Add 是逐元素加法，不是矩阵乘法。**

---

## 母题 5：什么时候触发 downsample

代码：

```python
if stride != 1 or in_channels != out_channels:
    ...
```

如果：

```text
stride=1
in_channels=64
out_channels=64
```

不需要 downsample。

如果：

```text
stride=2
```

或者：

```text
64→128
```

就需要调整 Shortcut。

这道母题训练：

> **检查 C/H/W 是否会发生变化。**

---

# 11. 错题本

## 错题 1：把 Shortcut 和 identity 当成同一个概念

**错误理解：**

> Shortcut 就是 identity。

**正确理解：**

```text
Shortcut = 路
identity = 路上的 Tensor
```

**记忆：**

> **路和路上的数据不是一回事。**

---

## 错题 2：把 Residual 只理解成“x 绕过去”

**正确理解：**

Residual Block 的核心是：

```text
F(x)+identity
```

Shortcut 只是其中一条路径。

**记忆：**

> **Residual 看“两条路最后相加”，不是只看旁路。**

---

## 错题 3：认为 `identity=x` 会重新计算一份新特征

**正确理解：**

`identity=x` 在当前代码里只是保存原输入，方便后面参与 Shortcut。

**记忆：**

> **identity 保存，不负责提取特征。**

---

## 错题 4：看到 `out+identity` 不检查 Shape

**正确理解：**

Residual Add 前必须检查主分支和 Shortcut 的 Shape 是否对应。

**记忆：**

> **看到加法，先看 Shape。**

---

## 错题 5：把 downsample 当成第三条独立路径

**正确理解：**

downsample 通常就在 Shortcut 上。

关系：

```text
Shortcut
├─ 直接通过
└─ 先经过 downsample/projection
```

---

## 错题 6：认为 1×1 Conv 不能改变 Channel

**正确理解：**

1×1 只表示空间窗口 1×1。

`out_channels` 仍然可以让：

```text
64→128
```

---

## 错题 7：认为 Shortcut 一路跳到最终 logits

**正确理解：**

当前 Shortcut 只绕过当前 Residual Block 中的中间层，最后仍然回到这个 block 的输出。

**记忆：**

> **Shortcut 跳的是当前 block，不是整张网络。**

---

# 12. 重点掌握清单

## 必须会解释

- [ ] `H(x)` 与 `F(x)`
- [ ] Residual Connection
- [ ] Shortcut
- [ ] identity
- [ ] Identity Shortcut
- [ ] Projection Shortcut
- [ ] downsample
- [ ] 1×1 Conv 在 Shortcut 中的作用

## 必须会手推

- [ ] `[B,64,32,32] → BasicBlock → ?`
- [ ] `[B,64,32,32] → Conv(64→128,s=2) → ?`
- [ ] Shortcut 应该调整到什么 Shape
- [ ] `Conv2d(64,128,1,stride=2)` 的输出 Shape
- [ ] 两个小矩阵的 Residual Add

## 必须会看代码判断

- [ ] `identity=x` → 保存原输入
- [ ] `out=out+identity` → Residual Add
- [ ] `self.downsample is not None` → Shortcut 要做 Shape 调整
- [ ] `kernel_size=1` → 1×1 Projection Conv
- [ ] `stride=2` → H/W 变小
- [ ] `in_channels != out_channels` → C 不一致，需要处理

---

# 13. 触发式记忆

看到：

```python
identity = x
```

想到：

> Shortcut 保存原输入。

看到：

```python
out = out + identity
```

想到：

> Residual Add，先检查 Shape。

看到：

```python
if self.downsample is not None:
```

想到：

> Shortcut 的 Shape 需要调整。

看到：

```python
Conv2d(64,128,1,stride=2)
```

想到：

> Projection：C 64→128，H/W 32→16。

看到：

```text
[B,64,32,32]
→ [B,128,16,16]
```

想到：

> 主分支发生了 C/H/W 变化，Shortcut 也必须对齐。

---

# Day 4 最终必须牢牢记住

```text
Residual：
H(x)=F(x)+x
```

```text
主分支：
x → Conv → ReLU → Conv → F(x)
```

```text
Shortcut：
x → identity
```

```text
Shape 一样：
Identity Shortcut
```

```text
Shape 不一样：
Projection Shortcut
→ 常用 1×1 Conv
```

最关键的一句话：

> **主分支学变化，Shortcut 留原输入；Shape 一样直接加，Shape 不一样先投影对齐再加。**
