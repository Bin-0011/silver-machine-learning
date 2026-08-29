# Day 2 总结：Conv2d

## 1. 核心概念表

| 概念 | 是什么 | 看到什么时想到它 |
|---|---|---|
| `Conv2d` | 二维卷积层 | 局部窗口滑动、乘加 |
| `in_channels` | 输入通道数 | 决定每个 filter 有多厚 |
| `out_channels` | 输出通道数 | 决定有多少个 filter |
| `kernel_size` | H/W 上的局部窗口大小 | `3` → 3×3 |
| filter | 产生一个输出 channel 的一组权重 | `[C_in,K_h,K_w]` |
| `weight` | 整层全部 filter | `[C_out,C_in,K_h,K_w]` |
| `stride` | 每次滑动多少格 | 越大，输出 H/W 越小 |
| `padding` | 输入边缘补多少圈 | 控制边缘与输出尺寸 |
| `bias` | 每个输出 channel 的额外偏置 | `bias.shape=[C_out]` |
| feature map | 一个 filter 滑完整张输入得到的空间响应图 | 1 filter → 1 feature map |
| `requires_grad` | Tensor 是否需要梯度 | 参数是否参与梯度计算 |
| `grad_fn` | Tensor 由什么可求导运算产生 | `ConvolutionBackward0` |
| `torch.no_grad()` | 当前作用域不记录 autograd 图 | 推理、手动改权重 |

---

## 2. 三个最容易混淆的对象

| 对象 | 例子 | 含义 |
|---|---|---|
| 卷积层配置 | `Conv2d(3,64,3)` | 输入 3，输出 64，kernel 3×3 |
| 单个 filter | `[3,3,3]` | `[C_in,K_h,K_w]` |
| 整层 weight | `[64,3,3,3]` | `[C_out,C_in,K_h,K_w]` |

最关键关系：

```text
Conv2d(3,64,3)
→ 64 个 filter
→ 每个 filter [3,3,3]
→ weight [64,3,3,3]
```

---

## 3. 一个输出值怎么计算

单通道例子：

```text
输入 patch：

1   2   3
6   7   8
11 12 13
```

全 1 filter：

```text
1 1 1
1 1 1
1 1 1
```

计算：

```text
1×1 + 2×1 + 3×1
+ 6×1 + 7×1 + 8×1
+ 11×1 + 12×1 + 13×1
= 63
```

所以：

```text
一个空间位置
→ 局部 patch × filter
→ 逐元素相乘
→ 全部求和
→ + bias（如果有）
→ 一个输出值
```

---

## 4. filter 如何生成 feature map

输入 5×5，kernel 3×3，stride=1，padding=0。

窗口依次滑动：

```text
位置 1 → 63
位置 2 → 72
位置 3 → 81
下一行 → 108 ...
```

最终：

```text
63   72   81
108 117  126
153 162  171
```

因此：

```text
一个 filter
在所有 H/W 位置重复计算
→ 一张 feature map
```

---

## 5. `stride` 与 `padding`

| 参数 | 控制什么 | 典型效果 |
|---|---|---|
| `stride=1` | 每次移动 1 格 | 采样位置较密 |
| `stride=2` | 每次移动 2 格 | 常用于下采样 |
| `padding=0` | 不补边 | H/W 容易缩小 |
| `padding=1` | 外围补 1 圈 | `k=3,s=1` 时常保持 H/W |

经典组合：

```text
k=3, s=1, p=1
→ H/W 通常保持不变

k=3, s=2, p=1
→ H/W 常近似减半
```

---

## 6. 输出尺寸公式

Day 2 常用简化版：

```text
H_out = floor((H_in + 2P - K) / S) + 1
W_out = floor((W_in + 2P - K) / S) + 1
```

例如：

```text
H_in=32
K=3
P=1
S=2
```

得到：

```text
H_out=16
```

所以：

```text
[1,3,32,32]
→ Conv2d(3,16,3,stride=2,padding=1)
→ [1,16,16,16]
```

---

## 7. 多输入通道

RGB 输入：

```text
[B,3,H,W]
```

单个 filter：

```text
[3,3,3]
```

一个空间位置的计算：

```text
R patch × R 权重
+
G patch × G 权重
+
B patch × B 权重
→ 全部求和
→ 一个数
```

所以：

> **一个 filter 会读取全部输入 channels，但一个位置只产生一个输出值。**

---

## 8. 为什么 `out_channels=64` 会得到 64 个 channels

```python
Conv2d(3,64,3)
```

表示：

```text
64 个不同 filter
每个 filter [3,3,3]
```

因此：

```text
Filter 1  → Feature Map 1
Filter 2  → Feature Map 2
...
Filter 64 → Feature Map 64
```

输出：

```text
[B,64,H_out,W_out]
```

---

## 9. `bias`

```python
Conv2d(
    3,
    64,
    3,
    bias=True
)
```

则：

```text
bias.shape = [64]
```

原因：

```text
每个输出 channel
→ 对应 1 个 bias
```

---

## 10. `requires_grad` / `grad_fn` / `no_grad`

| 概念 | 作用 |
|---|---|
| `requires_grad=True` | 这个 Tensor 需要梯度 |
| `grad_fn` | 记录这个 Tensor 由什么运算产生 |
| `torch.no_grad()` | 当前作用域不建立 autograd 计算图 |

正常：

```python
y = conv(x)
```

常见：

```text
conv.weight.requires_grad = True
y.requires_grad = True
y.grad_fn = ConvolutionBackward0
```

而：

```python
with torch.no_grad():
    y = conv(x)
```

通常：

```text
y.requires_grad = False
y.grad_fn = None
```

但：

```text
conv.weight.requires_grad
```

仍然是：

```text
True
```

---

## 11. 核心完整代码

```python
import torch


x = torch.tensor(
    [[[
        [1., 2., 3., 4., 5.],
        [6., 7., 8., 9., 10.],
        [11., 12., 13., 14., 15.],
        [16., 17., 18., 19., 20.],
        [21., 22., 23., 24., 25.]
    ]]]
)

conv = torch.nn.Conv2d(
    in_channels=1,
    out_channels=1,
    kernel_size=3,
    stride=1,
    padding=0,
    bias=False
)

with torch.no_grad():
    conv.weight[:] = torch.tensor(
        [[[
            [1., 1., 1.],
            [1., 1., 1.],
            [1., 1., 1.]
        ]]]
    )

y = conv(x)

print("x.shape =", x.shape)
print("weight.shape =", conv.weight.shape)
print("y =", y)
print("y.shape =", y.shape)

print("\n===== Autograd =====")
print("conv.weight.requires_grad =", conv.weight.requires_grad)
print("y.requires_grad =", y.requires_grad)
print("y.grad_fn =", y.grad_fn)

with torch.no_grad():
    y_no_grad = conv(x)

print("y_no_grad.requires_grad =", y_no_grad.requires_grad)
print("y_no_grad.grad_fn =", y_no_grad.grad_fn)

print("\n===== 参数实验 =====")
print(
    "p=0,s=1:",
    torch.nn.Conv2d(1,1,3,stride=1,padding=0)(x).shape
)

print(
    "p=1,s=1:",
    torch.nn.Conv2d(1,1,3,stride=1,padding=1)(x).shape
)

print(
    "p=1,s=2:",
    torch.nn.Conv2d(1,1,3,stride=2,padding=1)(x).shape
)

conv_rgb = torch.nn.Conv2d(
    in_channels=3,
    out_channels=64,
    kernel_size=3,
    bias=True
)

print("\nRGB Conv:")
print("weight.shape =", conv_rgb.weight.shape)
print("bias.shape =", conv_rgb.bias.shape)
```

---

# 12. 母题

## 母题 1：看到 `Conv2d(3,64,3)`，能否立刻拆开？

要求：

```text
单个 filter.shape = ?
weight.shape = ?
```

推导：

```text
C_in=3
C_out=64
kernel=3×3

单个 filter
= [3,3,3]

整层 weight
= [64,3,3,3]
```

这道母题训练：

> **看到 `Conv2d(C_in,C_out,K)`，立刻想到“C_out 个 `[C_in,K,K]` filter”。**

---

## 母题 2：一个输出值怎么手算？

给定：

```text
patch:
1 2 3
6 7 8
11 12 13

filter:
1 1 1
1 1 1
1 1 1
```

答案：

```text
63
```

这道母题训练：

> **卷积一个位置的本质是“局部 patch × filter → 逐元素乘 → 求和”。**

---

## 母题 3：输出 H/W 怎么推

给定：

```python
Conv2d(
    3,
    16,
    kernel_size=3,
    stride=2,
    padding=1
)
```

输入：

```text
[1,3,32,32]
```

推导：

```text
H_out
= floor((32+2-3)/2)+1
= 16
```

所以：

```text
输出 = [1,16,16,16]
```

这道母题训练：

> **看到 stride/padding/kernel 时，先手推 H/W，再看代码。**

---

## 母题 4：为什么一个 filter 要“厚到覆盖所有 C_in”

给定：

```text
输入 [B,16,H,W]
Conv2d(16,64,3)
```

单个 filter：

```text
[16,3,3]
```

原因：

```text
一个输出值
需要综合同一空间位置附近的全部 16 个输入 channels
```

这道母题训练：

> **看到 `in_channels=16`，立刻想到单个 filter 第一维也是 16。**

---

## 母题 5：64 个 filter 为什么输出 64 个 channels

```text
1 个 filter
→ 1 张 feature map

64 个 filter
→ 64 张 feature maps
→ 输出 C=64
```

这道母题训练：

> **看到 `out_channels`，想到 filter 数量和输出 feature map 数量。**

---

## 母题 6：`no_grad()` 会不会把参数永久冻结？

给定：

```python
with torch.no_grad():
    y = conv(x)
```

结论：

```text
y.requires_grad=False
y.grad_fn=None

但：
conv.weight.requires_grad
仍然可能是 True
```

这道母题训练：

> **区分“参数长期是否需要梯度”和“当前这次运算是否被记录”。**

---

# 13. 错题本

## 错题 1：把 `Conv2d(3,64,3)` 当成单个 filter shape

**错误理解：**

```text
filter = [3,64,3]
```

**正确理解：**

```text
Conv2d(3,64,3)
= 整层配置

单个 filter
= [3,3,3]

weight
= [64,3,3,3]
```

**记忆：**

> **层配置看参数；filter 看 `[C_in,K_h,K_w]`；整层 weight 再在最前面加 `C_out`。**

---

## 错题 2：认为 `out_channels=64` 表示一个 filter 有 64 层

**正确理解：**

```text
out_channels=64
→ 一共有 64 个 filter
```

不是一个 filter 有 64 层。

**记忆：**

> **C_out 数 filter；C_in 定厚度。**

---

## 错题 3：认为 `kernel_size=3` 表示只处理 3 个数字

**正确理解：**

```text
kernel_size=3
→ H/W 上的 3×3 局部窗口
```

如果输入有多个 channel，这个 filter 还会覆盖全部 C_in。

---

## 错题 4：把卷积当成普通矩阵乘法

**正确理解：**

一个空间位置上：

```text
patch 与 filter
逐元素相乘
→ 所有乘积求和
```

不是线性代数里两个二维矩阵直接做普通矩阵乘法。

---

## 错题 5：认为 `padding` 只是“让图片变大”

**正确理解：**

padding 的直接动作是：

```text
卷积前在输入边缘补像素
```

它的主要目的之一是：

```text
让 kernel 能覆盖边缘
并控制输出 H/W
```

---

## 错题 6：认为 `stride=2` 只影响窗口移动，不影响输出尺寸

**正确理解：**

stride 越大：

```text
可计算的位置越少
→ 输出 H/W 越小
```

所以常用于下采样。

---

## 错题 7：认为 `torch.no_grad()` 会把 `requires_grad=True` 永久改成 False

**正确理解：**

`no_grad()` 只控制当前作用域里的运算是否记录计算图。

参数本身：

```text
requires_grad
```

不会因此永久变化。

---

## 错题 8：认为 `grad_fn=None` 一定表示 Tensor 不能训练

**正确理解：**

例如：

```text
conv.weight.requires_grad=True
conv.weight.grad_fn=None
```

完全正常，因为它是 leaf tensor，不是由其他运算产生的。

---

## 错题 9：位置参数可以随便调换顺序

**错误：**

```python
Conv2d(64,3,3)
```

不能当成：

```python
Conv2d(3,64,3)
```

**正确理解：**

位置参数前三位固定：

```text
in_channels
out_channels
kernel_size
```

---

# 14. 重点掌握清单

## 必须会解释

- [ ] `Conv2d(C_in,C_out,K)` 的三个核心参数
- [ ] 单个 filter 与整层 weight 的区别
- [ ] stride 和 padding 的作用
- [ ] 一个 filter 为什么覆盖全部输入 channels
- [ ] 为什么一个 filter 产生一张 feature map
- [ ] 为什么 `out_channels` 决定输出 channels
- [ ] `requires_grad / grad_fn / no_grad` 的区别

## 必须会手算

- [ ] 5×5 输入 + 3×3 全 1 filter 的第一个输出值
- [ ] 第二个输出值为什么从 63 变成 72
- [ ] `H_out/W_out`
- [ ] `Conv2d(3,16,3,s=2,p=1)` 的输出 shape

## 必须会看代码判断

- [ ] `Conv2d(3,64,3)` → 单个 filter `[3,3,3]`
- [ ] `conv.weight.shape=[64,3,3,3]` → 每维含义
- [ ] `bias.shape=[64]` → 为什么
- [ ] `stride=2` → H/W 通常减小
- [ ] `padding=1,k=3,s=1` → H/W 通常保持
- [ ] `with torch.no_grad()` → 当前运算不记录图

---

# 15. 触发式记忆

看到：

```python
Conv2d(3,64,3)
```

想到：

> 3 输入 channels，64 个 filter，kernel 3×3。

看到：

```text
[64,3,3,3]
```

想到：

> `[C_out,C_in,K_h,K_w]`。

看到：

```python
stride=2
```

想到：

> 窗口跳着走，输出 H/W 通常变小。

看到：

```python
padding=1
```

想到：

> 边缘补 1 圈，控制边缘与输出尺寸。

看到：

```python
conv.weight[:]
```

想到：

> 选中整个 weight Tensor 的所有元素。

看到：

```python
with torch.no_grad():
```

想到：

> 当前作用域不建立 autograd 计算图。

看到：

```text
ConvolutionBackward0
```

想到：

> 这个 Tensor 是由可求导卷积操作产生的。

---

# Day 2 最终必须牢牢记住

```text
输入：
[B,C_in,H,W]
```

```text
单个 filter：
[C_in,K_h,K_w]
```

```text
全部 weight：
[C_out,C_in,K_h,K_w]
```

```text
一个 filter
在一个位置
→ 一个数

一个 filter
在 H/W 上滑动
→ 一张 feature map

C_out 个 filter
→ C_out 张 feature maps
→ 输出 [B,C_out,H_out,W_out]
```

```text
requires_grad
→ Tensor 是否需要梯度

torch.no_grad()
→ 当前这次运算是否记录 autograd 图
```

最关键的一句话：

> **Conv2d 的核心就是：让 C_out 个可学习 filter 在输入 H/W 上滑动；每到一个位置，都读取全部 C_in 通道的局部 patch，与权重逐元素相乘并求和，最终形成新的 feature maps。**
