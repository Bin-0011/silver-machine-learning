# Day 2：真正搞懂 `Conv2d` 是怎么计算的

Day 1 已经解决了“图片进入 PyTorch 后是什么样子”的问题：我们知道了图片常见的 `HWC`、PyTorch 常用的 `BCHW`，也能通过索引判断哪些维度会保留、哪些会消失。现在可以继续向前一步：**既然图片已经变成 Tensor，CNN 到底怎样从这些数字里提取出新的特征？**

Day 2 只学习普通二维卷积 `Conv2d`。今天先不进入 ReLU、Pooling、BatchNorm、ResNet 或 FPN，而是把卷积最底层的计算过程真正看清楚。

---

## 1. 从“图片 Tensor”继续向前：卷积要解决什么

Day 1 的输入可以写成：

```text
[B, C, H, W]
```

例如 RGB 图片：

```text
[B, 3, H, W]
```

这些数目前还是原始输入。CNN 接下来需要做的是：**在图片的局部区域里寻找有用的模式，并把局部信息重新计算成新的 feature map。**

`Conv2d` 就是完成这件事的基础层。

可以先把它理解成：

> **拿一个小窗口在输入的 H/W 平面上滑动，每到一个位置，就把局部数据与 filter 权重逐元素相乘并求和，得到一个新的输出值。**

今天所有知识都从这句话展开。

---

## 2. 先把输入缩小到可以手算

真实图片太大，不适合第一次理解卷积。先使用一个 shape 为 `[1,1,5,5]` 的小 Tensor：

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

print(x.shape)
print(x)
```

这里：

```text
B = 1
C = 1
H = 5
W = 5
```

暂时把 B 和 C 放在一边，只看中间的 5×5：

```text
 1   2   3   4   5
 6   7   8   9  10
11  12  13  14  15
16  17  18  19  20
21  22  23  24  25
```

这个小矩阵是今天所有手算的基础。

---

## 3. `Conv2d` 的几个参数分别控制什么

先定义最简单的一层卷积：

```python
conv = torch.nn.Conv2d(
    in_channels=1,
    out_channels=1,
    kernel_size=3,
    stride=1,
    padding=0,
    bias=False
)
```

第一次出现的术语先这样理解：

| 参数 | 当前值 | 当前只需要理解成 |
|---|---:|---|
| `in_channels` | 1 | 输入有几个通道 |
| `out_channels` | 1 | 输出有几个通道，也决定有几个 filter |
| `kernel_size` | 3 | 每次在 H/W 上看 3×3 的局部区域 |
| `stride` | 1 | 小窗口每次移动 1 格 |
| `padding` | 0 | 输入边缘不补 0 |
| `bias` | False | 输出计算后不额外加偏置 |

这时：

```python
print(conv.weight.shape)
```

得到：

```text
[1,1,3,3]
```

PyTorch 中卷积权重的排列顺序是：

```text
[C_out, C_in, K_h, K_w]
```

---

## 4. `Conv2d`、单个 filter、整层 weight 必须分开

这是 Day 2 最容易混淆的地方。

例如：

```python
Conv2d(3, 64, 3)
```

这表示的是**整个卷积层配置**：

```text
输入通道 = 3
输出通道 = 64
kernel = 3×3
```

它不是单个 filter 的 shape。

### 单个 filter

如果输入有 3 个通道，kernel 是 3×3，那么一个 filter 必须同时覆盖全部输入通道，因此：

```text
单个 filter
=
[C_in, K_h, K_w]
=
[3,3,3]
```

### 整层 weight

因为 `out_channels=64`，所以一共有 64 个这样的 filter：

```text
conv.weight.shape
=
[C_out, C_in, K_h, K_w]
=
[64,3,3,3]
```

关系可以压成：

```text
Conv2d(3,64,3)
→ 64 个 filter
→ 每个 filter [3,3,3]
→ 整层 weight [64,3,3,3]
```

最重要的两句话：

> **`out_channels` 决定“有多少个 filter”。**

> **`in_channels` 决定“每个 filter 有多厚”。**

---

## 5. `kernel_size=3` 到底表示什么

```python
kernel_size=3
```

在二维卷积中默认等于：

```text
kernel_size=(3,3)
```

也就是：

```text
K_h=3
K_w=3
```

它表示每次在 H/W 空间上看一个 3×3 的局部区域，而不是“只看 3 个数”。

如果想要非正方形 kernel，例如 5×3，需要写：

```python
kernel_size=(5,3)
```

此时：

```text
单个 filter = [C_in,5,3]
```

---

## 6. 位置参数和关键字参数

推荐刚学习时先写完整：

```python
torch.nn.Conv2d(
    in_channels=3,
    out_channels=64,
    kernel_size=3
)
```

这里使用了关键字参数，因此参数名已经明确，书写顺序可以调整。

工程代码中常看到：

```python
torch.nn.Conv2d(3,64,3)
```

这使用的是位置参数，前三个位置固定为：

```text
第 1 个：in_channels
第 2 个：out_channels
第 3 个：kernel_size
```

所以：

```python
Conv2d(3,64,3)
```

和：

```python
Conv2d(64,3,3)
```

是两层完全不同的卷积。

---

## 7. 手动把 filter 设置成全 1

默认卷积权重是随机初始化的，不方便手算，所以先把 3×3 filter 全部改成 1：

```python
with torch.no_grad():
    conv.weight[:] = torch.tensor(
        [[[
            [1., 1., 1.],
            [1., 1., 1.],
            [1., 1., 1.]
        ]]]
    )
```

这里有两个新点。

### `conv.weight[:]`

表示选中 `conv.weight` 的全部元素，然后原地修改它们的数值。

### `torch.no_grad()`

这里表示：

> 这次只是手动设置参数值，不把这次赋值记录进 autograd 计算图。

它不会把：

```text
conv.weight.requires_grad
```

永久改成 `False`。

---

## 8. 一个输出值到底是怎么算出来的

现在计算：

```python
y = conv(x)
```

卷积核第一次覆盖输入左上角 3×3：

```text
输入 patch：

1   2   3
6   7   8
11 12 13
```

filter：

```text
1 1 1
1 1 1
1 1 1
```

做的不是矩阵乘法，而是：

```text
对应位置逐元素相乘
→ 把所有乘积求和
```

计算：

```text
1×1 + 2×1 + 3×1
+ 6×1 + 7×1 + 8×1
+ 11×1 + 12×1 + 13×1
= 63
```

所以输出左上角：

```text
y[0,0,0,0] = 63
```

这就是卷积最基础的计算单元：

> **一个 filter 在一个空间位置，最终只产生一个数。**

---

## 9. filter 是怎样滑出整张 feature map 的

因为：

```text
stride=1
```

所以窗口每次移动一格。

第一位置：

```text
1   2   3
6   7   8
11 12 13
```

得到：

```text
63
```

向右移动一格：

```text
2   3   4
7   8   9
12 13 14
```

得到：

```text
72
```

再向右：

```text
3   4   5
8   9  10
13 14 15
```

得到：

```text
81
```

到最右边后回到左侧，再向下移动一格。最终输出：

```text
63   72   81
108 117  126
153 162  171
```

shape：

```text
[1,1,3,3]
```

所以：

> **一个 filter 在整个 H/W 上滑动，会产生一张 feature map。**

---

## 10. `stride`：决定窗口每次走多远

`stride` 控制：

> 卷积核每次在 H/W 上移动多少格。

`stride=1`：

```text
0 → 1 → 2 → 3 → ...
```

`stride=2`：

```text
0 → 2 → 4 → ...
```

stride 越大，窗口能落下的位置越少，因此输出 H/W 通常越小。

所以在 CNN 中：

```text
stride=2
```

经常用于下采样。

---

## 11. `padding`：决定边缘是否补像素

`padding=0` 表示不补边。

`padding=1` 可以理解为在输入外围补一圈 0。

例如：

```text
原始：
1 2 3
4 5 6
7 8 9
```

补一圈 0 后概念上变成：

```text
0 0 0 0 0
0 1 2 3 0
0 4 5 6 0
0 7 8 9 0
0 0 0 0 0
```

padding 的一个重要作用是让 kernel 能覆盖到边缘位置，同时帮助控制输出尺寸。

经典组合：

```text
kernel_size=3
stride=1
padding=1
```

通常可以保持 H/W 不变。

---

## 12. 输出 H/W 怎么算

Day 2 暂时使用 `dilation=1`，可以用简化公式：

```text
H_out = floor((H_in + 2P - K) / S) + 1
W_out = floor((W_in + 2P - K) / S) + 1
```

其中：

```text
K = kernel_size
P = padding
S = stride
```

### 示例 A：5×5 → 3×3

```text
H_in=5
K=3
P=0
S=1
```

所以：

```text
H_out = floor((5-3)/1)+1 = 3
```

### 示例 B：32×32 → 16×16

```text
H_in=32
K=3
P=1
S=2
```

所以：

```text
H_out = floor((32+2-3)/2)+1
      = floor(31/2)+1
      = 15+1
      = 16
```

于是：

```text
32×32 → 16×16
```

---

## 13. 从单通道进入多通道

前面的例子只有一个输入通道。现在换成 RGB：

```python
x = torch.randn(1,3,5,5)

conv = torch.nn.Conv2d(
    in_channels=3,
    out_channels=1,
    kernel_size=3,
    bias=False
)
```

此时：

```text
conv.weight.shape = [1,3,3,3]
```

只有 1 个 filter，但这个 filter 的 shape 是：

```text
[3,3,3]
```

因为它必须同时覆盖 R、G、B 三个输入通道。

在一个空间位置上：

```text
R patch × R 权重
+
G patch × G 权重
+
B patch × B 权重
```

所有结果继续求和，最后仍然只得到：

```text
一个数
```

所以：

> **一个 filter 会读取全部输入 channels，但在一个空间位置只产生一个输出值。**

---

## 14. 为什么 64 个 filter 会得到 64 个输出 channels

如果：

```python
conv = torch.nn.Conv2d(
    in_channels=3,
    out_channels=64,
    kernel_size=3
)
```

那就有：

```text
64 个不同 filter
```

每个 filter 都是：

```text
[3,3,3]
```

每个 filter 扫完整张输入后都会得到一张 feature map：

```text
Filter 1  → Feature Map 1
Filter 2  → Feature Map 2
...
Filter 64 → Feature Map 64
```

因此：

```text
64 个 filter
→ 64 张 feature maps
→ 输出 C=64
```

输入：

```text
[B,3,H,W]
```

输出：

```text
[B,64,H_out,W_out]
```

---

## 15. `bias`：每个输出 channel 一个偏置

例如：

```python
Conv2d(
    3,
    64,
    kernel_size=3,
    bias=True
)
```

那么：

```text
conv.bias.shape = [64]
```

因为每一个输出 channel 对应一个 bias。

单个输出值可以理解成：

```text
局部输入与 filter 逐元素乘积求和
+ 该输出 channel 对应的 bias
```

---

## 16. `requires_grad`、`grad_fn` 和 `torch.no_grad()`

这一部分不是卷积几何本身，但你的 Day 2 代码已经实际遇到了，所以一起整理清楚。

### `requires_grad`

```python
conv.weight.requires_grad
```

默认是：

```text
True
```

表示这个参数需要梯度，将来可以被训练更新。

### `grad_fn`

正常 forward：

```python
y = conv(x)
```

由于参与运算的 `conv.weight` 需要梯度，PyTorch 会记录这次卷积操作。

所以：

```text
y.requires_grad = True
y.grad_fn = <ConvolutionBackward0 ...>
```

`grad_fn` 可以理解成：

> 这个 Tensor 是通过哪一种可求导运算产生的。

### 为什么 `conv.weight.grad_fn=None`

`conv.weight` 是模型参数本身，是一个 leaf tensor，不是由其他 Tensor 算出来的，所以：

```text
requires_grad=True
grad_fn=None
```

并不矛盾。

### `torch.no_grad()`

如果：

```python
with torch.no_grad():
    y = conv(x)
```

这一次 forward 不建立 autograd 计算图，所以通常：

```text
y.requires_grad=False
y.grad_fn=None
```

但：

```text
conv.weight.requires_grad
```

仍然是：

```text
True
```

最重要的区别：

> **`requires_grad` 决定 Tensor 是否需要梯度；`torch.no_grad()` 决定当前这次运算是否被 autograd 记录。**

---

## 17. 为什么卷积能提取特征

今天手算的全 1 filter 只是为了让结果容易验证，它本质上是一个局部求和器。

真正训练时，filter 里的权重是可学习参数，例如：

```text
0.23
-0.51
0.07
...
```

训练过程中：

```text
输入
→ Conv
→ 预测
→ Loss
→ 反向传播
→ 更新 filter 权重
```

不同 filter 会逐渐对不同局部模式产生不同响应。

入门阶段可以粗略理解成：

```text
某些 filter 对边缘敏感
某些 filter 对方向变化敏感
某些 filter 对纹理敏感
某些 filter 对颜色组合敏感
```

因此卷积的特征提取能力不是“卷积这个动作自动拥有的魔法”，而是：

> **局部乘加机制 + 训练过程中逐渐学到的 filter 权重。**

---

## 18. 今日完整练习代码

```python
import torch


# ============================================================
# 1. 创建 5×5 单通道输入
# ============================================================

x = torch.tensor(
    [[[
        [1., 2., 3., 4., 5.],
        [6., 7., 8., 9., 10.],
        [11., 12., 13., 14., 15.],
        [16., 17., 18., 19., 20.],
        [21., 22., 23., 24., 25.]
    ]]]
)

print("x.shape =", x.shape)
print("x =", x)


# ============================================================
# 2. 创建最简单 Conv2d
# ============================================================

conv = torch.nn.Conv2d(
    in_channels=1,
    out_channels=1,
    kernel_size=3,
    stride=1,
    padding=0,
    bias=False
)

print("\nconv.weight.shape =", conv.weight.shape)


# ============================================================
# 3. 手动设置全 1 filter
# ============================================================

with torch.no_grad():
    conv.weight[:] = torch.tensor(
        [[[
            [1., 1., 1.],
            [1., 1., 1.],
            [1., 1., 1.]
        ]]]
    )


# ============================================================
# 4. 正常 forward
# ============================================================

y = conv(x)

print("\ny =", y)
print("y.shape =", y.shape)


# ============================================================
# 5. Autograd 信息
# ============================================================

print("\n===== Autograd =====")
print("x.requires_grad =", x.requires_grad)
print("conv.weight.requires_grad =", conv.weight.requires_grad)
print("y.requires_grad =", y.requires_grad)
print("x.grad_fn =", x.grad_fn)
print("conv.weight.grad_fn =", conv.weight.grad_fn)
print("y.grad_fn =", y.grad_fn)


# ============================================================
# 6. 正常 forward 与 no_grad forward 对比
# ============================================================

y1 = conv(x)

with torch.no_grad():
    y2 = conv(x)

print("\n===== 正常 forward vs no_grad =====")
print("y1.requires_grad =", y1.requires_grad)
print("y1.grad_fn =", y1.grad_fn)
print("y2.requires_grad =", y2.requires_grad)
print("y2.grad_fn =", y2.grad_fn)


# ============================================================
# 7. padding / stride 实验
# ============================================================

conv_p0 = torch.nn.Conv2d(
    1, 1,
    kernel_size=3,
    stride=1,
    padding=0
)

conv_p1 = torch.nn.Conv2d(
    1, 1,
    kernel_size=3,
    stride=1,
    padding=1
)

conv_s2 = torch.nn.Conv2d(
    1, 1,
    kernel_size=3,
    stride=2,
    padding=1
)

print("\n===== padding / stride =====")
print("padding=0, stride=1:", conv_p0(x).shape)
print("padding=1, stride=1:", conv_p1(x).shape)
print("padding=1, stride=2:", conv_s2(x).shape)


# ============================================================
# 8. 多输入通道
# ============================================================

x_rgb = torch.randn(1, 3, 5, 5)

conv_rgb_1 = torch.nn.Conv2d(
    in_channels=3,
    out_channels=1,
    kernel_size=3,
    bias=False
)

print("\n===== in_channels=3 =====")
print("conv_rgb_1.weight.shape =", conv_rgb_1.weight.shape)


# ============================================================
# 9. 64 输出通道
# ============================================================

conv_rgb_64 = torch.nn.Conv2d(
    in_channels=3,
    out_channels=64,
    kernel_size=3,
    bias=True
)

print("\n===== out_channels=64 =====")
print("weight.shape =", conv_rgb_64.weight.shape)
print("bias.shape =", conv_rgb_64.bias.shape)
```

---

# Day 2 完成标准

- [ ] `Conv2d(3,64,3)` 的 3、64、3 分别是什么？
- [ ] `Conv2d(c_in,c_out,kernel_size)`、单个 filter、整层 weight 有什么区别？
- [ ] `conv.weight.shape=[64,3,3,3]` 四维分别是什么？
- [ ] 单个 filter 为什么是 `[C_in,K_h,K_w]`？
- [ ] 一个 filter 在一个空间位置怎样算出一个值？
- [ ] 为什么一个 filter 产生一张 feature map？
- [ ] 为什么 64 个 filter 产生 64 个 output channels？
- [ ] `kernel_size=3` 为什么表示 3×3？
- [ ] 非正方形 5×3 kernel 应该怎么写？
- [ ] 位置参数和关键字参数有什么区别？
- [ ] `stride` 控制什么？
- [ ] `padding` 控制什么？
- [ ] 如何推导 `H_out/W_out`？
- [ ] 为什么 `k=3,p=1,s=1` 可以保持 H/W？
- [ ] 为什么 `stride=2` 常用于下采样？
- [ ] `bias.shape` 为什么是 `[out_channels]`？
- [ ] `conv.weight[:]` 是什么？
- [ ] `requires_grad` 与 `torch.no_grad()` 有什么区别？
- [ ] `grad_fn=<ConvolutionBackward0>` 表示什么？
- [ ] 为什么卷积能够学习局部特征？

# Day 2 最终必须牢牢记住

```text
Conv2d 输入：
[B, C_in, H, W]
```

```text
Conv2d weight：
[C_out, C_in, K_h, K_w]
```

```text
单个 filter：
[C_in, K_h, K_w]
```

```text
一个 filter + 一个空间位置
→ 局部 patch 与 filter 逐元素相乘并求和
→ 一个数
```

```text
一个 filter 在 H/W 上滑动
→ 一张 feature map
```

```text
C_out 个 filters
→ C_out 张 feature maps
→ C_out 个输出通道
```

最关键的一句话：

> **卷积不是把图片“神奇地变成特征”。它只是不断执行“局部取值 → 权重相乘 → 求和 → 滑动”。真正的特征提取能力来自训练过程中逐渐学到的卷积权重。**
