# Day 4：Residual Connection + BasicBlock

Day 3 已经把最小 CNN 串起来了：

`图片 → Conv → ReLU → Pooling → Conv → ReLU → Pooling → Flatten → Linear → logits`

到这里，我们已经能够让网络一层层提取和组合特征。Day 4 开始进入 ResNet 的基础结构，但今天不学习完整 ResNet，只学习它最核心的一个小单元：**Residual Connection（残差连接）和 BasicBlock**。

> Residual Connection 是一种网络连接结构，它通过 Shortcut 保留原输入 x，再把主分支学到的 F(x) 与 x 相加；
它背后的设计思想叫 Residual Learning。

这一课的主线是：

`普通卷积块 → 保存原输入 x → 主分支学习 F(x) → F(x)+x → Shape 对不上时做 Projection Shortcut`

---

## 1. 从普通卷积块到残差块

普通卷积块可以写成：

`x → Conv → ReLU → Conv → 输出`

假设我们希望这几层最后得到理想结果 `H(x)`。

普通做法可以理解为：

> 让中间这些卷积层直接学习完整的 `H(x)`。

ResNet 换了一种表达方式。它先把原输入 `x` 保留下来，让主分支学习：

`F(x)=H(x)-x`

于是：

`H(x)=F(x)+x`

这里第一次出现两个符号：

- `H(x)`：我们希望这个 block 最终得到的目标映射
- `F(x)`：目标结果和原输入之间的差，也就是需要补上的变化

所以 Residual（残差）可以先理解成：

> **原来的 x 已经有一部分信息，主分支只学习“还需要补多少”。**

最终：

`输出 = 原输入 x + 主分支学到的 F(x)`

---

## 2. 先用纯 Tensor 看懂 `F(x)+x`

最小实验：

```python
import torch

x = torch.tensor([
    [1., 2., 3.]
])

fx = torch.tensor([
    [0.1, -0.2, 0.5]
])

y = x + fx

print("x =", x)
print("F(x) =", fx)
print("y =", y)
```

手算：

```text
[1.0, 2.0, 3.0]
+
[0.1,-0.2,0.5]
=
[1.1,1.8,3.5]
```

所以 Residual Add 最基础的动作就是：

> **两个 Tensor 对应位置逐元素相加。**

这不是矩阵乘法，也不是拼接。

---

## 3. Shortcut、identity 和 Residual Add

这三个词必须放在一起理解。

### Shortcut

Shortcut 是：

> **原输入 x 绕过中间卷积，直接走到最后相加位置的那条旁路。**

它是一条“路径”。

### identity

代码里：

```python
identity = x
```

表示把原输入 `x` 保存下来，准备让它沿 Shortcut 走到最后。

所以：

```text
Shortcut = 路
identity = 路上保存的原输入 Tensor
```

### Residual Add

主分支算出：

```text
out = F(x)
```

Shortcut 保留：

```text
identity = x
```

最后：

```python
out = out + identity
```

就是：

`F(x)+x`

---

## 4. Shape 一样时：最简单 BasicBlock

先看最简单情况：主分支不会改变 C/H/W。

完整代码：

```python
import torch
import torch.nn as nn


class BasicBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
        )

        self.relu = nn.ReLU()

        self.conv2 = nn.Conv2d(
            in_channels=channels,
            out_channels=channels,
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
    x = torch.randn(4, 64, 32, 32)

    block = BasicBlock(channels=64)

    out = block(x)

    print("输入 shape:", x.shape)
    print("输出 shape:", out.shape)
```

这里第一次出现的几个 PyTorch 词可以继续这样理解：

| 写法 | 当前含义 |
|---|---|
| `nn.Module` | PyTorch 神经网络模块的基础类 |
| `__init__()` | 定义这个 block 里有哪些层 |
| `forward()` | 定义 Tensor 按什么顺序流过这些层 |
| `self.conv1` | 当前 block 自己拥有的第一层卷积 |
| `block(x)` | 把 x 输入 block，执行 forward |

---

## 5. 最简单 BasicBlock 的 Shape

输入：

```text
[B,64,32,32]
```

主分支：

```text
Conv1
[B,64,32,32] → [B,64,32,32]

ReLU
[B,64,32,32] → [B,64,32,32]

Conv2
[B,64,32,32] → [B,64,32,32]
```

Shortcut：

```text
identity = x
→ [B,64,32,32]
```

最后：

```text
out      [B,64,32,32]
identity [B,64,32,32]

→ out + identity
→ [B,64,32,32]
```

这里两边 Shape 一样，所以 Shortcut 不需要额外处理。

这种情况叫：

**Identity Shortcut**

可以理解成：

> x 原样绕过去。

---

## 6. Residual Add 前为什么要检查 Shape

最简单情况下：

```text
[B,64,32,32]
+
[B,64,32,32]
```

可以逐元素对应相加。

但如果主分支变成：

```text
[B,128,16,16]
```

而 Shortcut 还是：

```text
[B,64,32,32]
```

两边的 Channel、H、W 都对不上，就不能直接按照这个残差结构逐元素相加。

因此看到：

```python
out = out + identity
```

要逐渐形成一个条件反射：

> **先检查主分支和 Shortcut 的 Shape 是否对应。**

---

## 7. Shape 不一样时：Projection Shortcut

假设输入：

```text
[B,64,32,32]
```

主分支第一层改成：

```python
nn.Conv2d(
    in_channels=64,
    out_channels=128,
    kernel_size=3,
    stride=2,
    padding=1,
    bias=False
)
```

那么主分支会得到：

```text
[B,64,32,32]
→
[B,128,16,16]
```

但原始 `identity=x` 仍然是：

```text
[B,64,32,32]
```

这时 Shortcut 也必须把 Shape 调整成：

```text
[B,128,16,16]
```

常见做法是在 Shortcut 上加：

```python
nn.Conv2d(
    in_channels=64,
    out_channels=128,
    kernel_size=1,
    stride=2,
    bias=False
)
```

这条经过变换的 Shortcut 叫：

**Projection Shortcut（投影 Shortcut）**

---

## 8. 为什么用 1×1 Conv

这正好接回 Day 2。

```python
Conv2d(
    64,
    128,
    kernel_size=1,
    stride=2
)
```

单个 filter：

```text
[64,1,1]
```

`out_channels=128`：

```text
C：64 → 128
```

`stride=2`：

```text
H/W：32 → 16
```

所以：

```text
[B,64,32,32]
→ 1×1 Conv, stride=2
→ [B,128,16,16]
```

这样 Shortcut 就能和主分支对齐。

---

## 9. `downsample` 是什么

在很多 ResNet 风格代码中，会看到：

```python
self.downsample
```

它通常表示：

> **Shortcut 路径上的 Shape 调整模块。**

例如：

```python
self.downsample = nn.Conv2d(
    in_channels,
    out_channels,
    kernel_size=1,
    stride=stride,
    bias=False
)
```

如果主分支和 Shortcut 本来就一样，则：

```python
self.downsample = None
```

如果：

```text
stride != 1
```

或者：

```text
in_channels != out_channels
```

就需要对 identity 做调整。

---

## 10. 带 downsample 的 BasicBlock

完整可运行代码：

```python
import torch
import torch.nn as nn


class BasicBlockWithDownsample(nn.Module):
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

        # Shortcut 默认不需要变换
        self.downsample = None

        # C 或 H/W 对不上时，给 Shortcut 加 Projection
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=stride,
                bias=False
            )

    def forward(self, x):
        identity = x

        # 主分支
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)

        # Shortcut
        if self.downsample is not None:
            identity = self.downsample(identity)

        # 两条路径汇合
        out = out + identity
        out = self.relu(out)

        return out


if __name__ == "__main__":
    x = torch.randn(4, 64, 32, 32)

    block = BasicBlockWithDownsample(
        in_channels=64,
        out_channels=128,
        stride=2
    )

    out = block(x)

    print("输入 shape:", x.shape)
    print("输出 shape:", out.shape)
```

---

## 11. 带 downsample 的 Shape 推导

输入：

```text
[B,64,32,32]
```

主分支：

```text
Conv1(64→128, stride=2)
→ [B,128,16,16]

ReLU
→ [B,128,16,16]

Conv2(128→128, stride=1)
→ [B,128,16,16]
```

Shortcut：

```text
identity = x
→ [B,64,32,32]

1×1 Conv(64→128, stride=2)
→ [B,128,16,16]
```

最后：

```text
主分支     [B,128,16,16]
Shortcut   [B,128,16,16]

→ out + identity
→ [B,128,16,16]
```

---

## 12. 两种 Shortcut 放在一起看

| 类型 | 使用条件 | Shortcut 做什么 |
|---|---|---|
| Identity Shortcut | Shape 已经一样 | `identity=x` |
| Projection Shortcut | Shape 不一样 | `identity=projection(x)` |

关系：

```text
Shortcut
├─ Identity Shortcut
└─ Projection Shortcut
```

不要把 `downsample` 和 Shortcut 当成两个平行概念。

更准确地说：

> **Shortcut 是路径；downsample/projection 是必要时放在 Shortcut 上的 Shape 调整模块。**

---

## 13. Day 4 完整数据流

最简单情况：

```text
x
├─ 主分支：Conv → ReLU → Conv → F(x)
└─ Shortcut：identity=x

F(x)+identity
→ ReLU
→ 输出
```

Shape 改变时：

```text
x
├─ 主分支：Conv(stride=2) → ReLU → Conv → F(x)
└─ Shortcut：1×1 Conv(stride=2) → identity

两边 Shape 对齐
→ F(x)+identity
→ ReLU
→ 输出
```

---

# Day 4 完成标准

- [ ] `H(x)` 和 `F(x)` 分别表示什么？
- [ ] 为什么可以把 Residual 理解成“需要补上的变化”？
- [ ] `identity = x` 做了什么？
- [ ] Shortcut 是路径还是 Tensor？
- [ ] Residual Add 是什么运算？
- [ ] 为什么 `out + identity` 前要检查 Shape？
- [ ] `[B,64,32,32] + [B,64,32,32]` 输出什么 Shape？
- [ ] Shape 一样时为什么可以直接用 Identity Shortcut？
- [ ] `[B,128,16,16]` 为什么不能直接和 `[B,64,32,32]` 按当前残差结构相加？
- [ ] Projection Shortcut 是干什么的？
- [ ] 1×1 Conv 为什么可以改变 Channel？
- [ ] `stride=2` 为什么能让 H/W 变小？
- [ ] `self.downsample` 通常位于哪条路径？
- [ ] `stride != 1 or in_channels != out_channels` 为什么会触发 downsample？
- [ ] 能否不运行代码推导 BasicBlock 的完整 Shape？
- [ ] 能否不运行代码推导 BasicBlockWithDownsample 的完整 Shape？

# Day 4 最终必须牢牢记住

```text
Residual：
H(x)=F(x)+x
```

```text
Shortcut：
x 绕过中间卷积的旁路
```

```text
identity：
Shortcut 上保存的原输入 Tensor
```

```text
Shape 一样：
identity = x
→ Identity Shortcut
```

```text
Shape 不一样：
identity = projection(x)
→ Projection Shortcut
```

```text
常见 Projection：
1×1 Conv
必要时配合 stride=2
```

最关键的一句话：

> **Residual Block 的核心是两条路：主分支学习 F(x)，Shortcut 保留或调整原输入；两边 Shape 对齐后逐元素相加，再继续向后传。**
