# Day 2：真正搞懂 `Conv2d` 是怎么计算的

> 对应示例代码：`01_conv2d_basic.py`  
> 本节定位：只学习 **普通二维卷积 Conv2d** 的核心机制，不进入 BatchNorm、ReLU、Pooling、反向传播公式推导、ResNet、FPN。

Day 2 只解决一个核心问题：

> **`Conv2d` 到底怎样把输入 Tensor 计算成 feature map？**

---

## 1. Day 2 学习目标

学完这一节，应该能够不看资料回答：

1. `nn.Conv2d(3, 64, 3)` 中三个数字分别表示什么？
2. `Conv2d(c_in, c_out, kernel_size)` 和 **单个 filter 的 shape** 有什么区别？
3. `conv.weight.shape` 为什么是 `[c_out, c_in, kH, kW]`？
4. 一个 filter 为什么必须覆盖全部输入通道？
5. 一个卷积位置如何计算出一个输出值？
6. `stride`、`padding`、`kernel_size` 各自控制什么？
7. 如何计算卷积后的 `H_out`、`W_out`？
8. 为什么 `stride=2` 常用于下采样？
9. 64 个 filter 为什么产生 64 个输出通道？
10. `bias` 是什么？
11. `conv.weight[:]` 是什么？
12. `requires_grad`、`grad_fn`、`torch.no_grad()` 分别是什么？
13. 为什么训练 forward 不能随便放进 `torch.no_grad()`？
14. 为什么卷积可以学习“边缘、纹理、局部结构”等特征？

---

# 2. 先从最小例子开始：`1×1×5×5`

本节先不用真实图片，而使用一个可以手算的输入：

```python
x = torch.tensor(
    [[[
        [1., 2., 3., 4., 5.],
        [6., 7., 8., 9., 10.],
        [11., 12., 13., 14., 15.],
        [16., 17., 18., 19., 20.],
        [21., 22., 23., 24., 25.]
    ]]]
)
```

其 shape：

```text
[B, C, H, W]
[1, 1, 5, 5]
```

即：

```text
B = 1：1 张输入
C = 1：单通道
H = 5：高度 5
W = 5：宽度 5
```

为了理解卷积，先把 Batch 和 Channel 暂时放到一边，只看中间的 5×5：

```text
 1   2   3   4   5
 6   7   8   9  10
11  12  13  14  15
16  17  18  19  20
21  22  23  24  25
```

---

# 3. 定义一个最简单的 `Conv2d`

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

含义：

| 参数 | 当前值 | 含义 |
|---|---:|---|
| `in_channels` | 1 | 输入有 1 个通道 |
| `out_channels` | 1 | 输出 1 个通道，也就是使用 1 个 filter |
| `kernel_size` | 3 | 空间窗口为 3×3 |
| `stride` | 1 | 每次滑动 1 格 |
| `padding` | 0 | 输入外围不补 0 |
| `bias` | False | 不添加偏置项 |

---

# 4. 最容易混淆的三件事：Conv2d、单个 filter、全部 weight

这是 Day 2 必须彻底区分的地方。

## 4.1 `Conv2d(c_in, c_out, kernel_size)` 是“卷积层配置”

例如：

```python
torch.nn.Conv2d(3, 64, 3)
```

表示：

```text
输入通道数 c_in = 3
输出通道数 c_out = 64
kernel_size = 3×3
```

它描述的是 **整个卷积层**，不是单个 filter 的 shape。

---

## 4.2 单个 filter 的 shape

如果：

```text
c_in = 3
kernel_size = 3×3
```

那么 **单个 filter** 必须同时覆盖全部 3 个输入通道，因此：

```text
单个 filter.shape
= [c_in, kH, kW]
= [3, 3, 3]
```

也就是：

```text
R 通道上的 3×3 权重
G 通道上的 3×3 权重
B 通道上的 3×3 权重
```

三部分共同组成 **一个** filter。

因此不要把：

```text
Conv2d(3,64,3)
```

误解为：

```text
单个 filter = [3,64,3]
```

这是错误的。

---

## 4.3 整层 `conv.weight.shape`

如果：

```python
conv = torch.nn.Conv2d(
    in_channels=3,
    out_channels=64,
    kernel_size=3
)
```

那么共有：

```text
64 个 filter
```

每个 filter：

```text
[3, 3, 3]
```

所以整层权重：

```text
conv.weight.shape
=
[c_out, c_in, kH, kW]
=
[64, 3, 3, 3]
```

可以记成：

```text
conv.weight.shape
        ↓
[多少个 filter,
 每个 filter 覆盖多少输入通道,
 kernel 高度,
 kernel 宽度]
```

---

## 4.4 三者终极对照

| 概念 | 表达方式 | 例子 |
|---|---|---|
| 卷积层配置 | `Conv2d(c_in,c_out,kernel_size)` | `Conv2d(3,64,3)` |
| 单个 filter | `[c_in,kH,kW]` | `[3,3,3]` |
| 整层权重 | `[c_out,c_in,kH,kW]` | `[64,3,3,3]` |

**必须记住：**

> `out_channels` 决定“有多少个 filter”；  
> `in_channels` 决定“每个 filter 有多厚”。

---

# 5. `kernel_size` 到底是什么？

```python
kernel_size=3
```

PyTorch 默认解释为：

```text
kernel_size = (3, 3)
```

即：

```text
kH = 3
kW = 3
```

它表示卷积核每次在 H/W 空间上看一个：

```text
3×3
```

局部区域。

---

## 易错提示 1：非正方形 kernel

如果 kernel 不是正方形，例如你想使用：

```text
5×3
```

不能只写一个数字。

应该写：

```python
kernel_size=(5, 3)
```

表示：

```text
kH = 5
kW = 3
```

而：

```python
kernel_size=3
```

默认就是：

```text
(3,3)
```

同样，很多二维参数都支持：

```text
一个整数
```

或者：

```text
(height方向, width方向)
```

这样的二元组。

---

# 6. 参数名和参数位置

完整写法：

```python
torch.nn.Conv2d(
    in_channels=3,
    out_channels=64,
    kernel_size=3
)
```

这里使用了 **关键字参数（keyword arguments）**。

因此即使写成：

```python
torch.nn.Conv2d(
    out_channels=64,
    kernel_size=3,
    in_channels=3
)
```

仍然可以正确识别，因为参数名已经明确指出每个值属于谁。

---

## 易错提示 2：位置参数不能乱

工程代码中经常为了简洁写：

```python
torch.nn.Conv2d(3, 64, 3)
```

这时使用的是 **位置参数（positional arguments）**。

前三个位置固定是：

```text
第 1 个：in_channels
第 2 个：out_channels
第 3 个：kernel_size
```

因此：

```python
Conv2d(3, 64, 3)
```

等价于：

```python
Conv2d(
    in_channels=3,
    out_channels=64,
    kernel_size=3
)
```

但：

```python
Conv2d(64, 3, 3)
```

就完全是另一个卷积层了，它表示：

```text
输入 64 通道
输出 3 通道
kernel = 3×3
```

所以：

> **有参数名时，顺序可以调整；没有参数名、只按位置写时，顺序绝对不能乱。**

---

# 7. 手工把卷积核设置成全 1

为了能够手算，我们不使用随机初始化，而是把权重改成：

```text
1 1 1
1 1 1
1 1 1
```

代码：

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

---

# 8. `conv.weight[:]` 是什么？

`conv.weight` 是这个卷积层的参数 Tensor。

当前：

```text
conv.weight.shape = [1,1,3,3]
```

而：

```python
conv.weight[:]
```

表示：

> 选中 `conv.weight` 的全部元素。

所以：

```python
conv.weight[:] = new_weight
```

就是：

> 在原来的参数 Tensor 内部，把全部数值原地改成 `new_weight` 的数值。

这里的关键不是“重新创建一个新的卷积层”，而是：

```text
conv 还是原来的 conv
conv.weight 还是这个模块注册的参数
只是参数内部数值被改了
```

---

# 9. 为什么手动赋值要放进 `torch.no_grad()`？

`conv.weight` 默认是模型可训练参数：

```text
conv.weight.requires_grad = True
```

PyTorch 会保护这种需要梯度的 leaf tensor，不允许你在正常梯度追踪环境下随意原地修改。

所以手动初始化时写：

```python
with torch.no_grad():
    conv.weight[:] = ...
```

意思是：

> 这次只是人为设置参数值，不把这次修改当成模型训练计算的一部分。

特别注意：

```text
torch.no_grad()
```

**不会把**

```text
conv.weight.requires_grad
```

从 `True` 永久改成 `False`。

修改完成后：

```text
conv.weight.requires_grad
仍然是 True
```

---

# 10. 手算第一个输出值：63

输入左上角 3×3：

```text
1   2   3
6   7   8
11 12  13
```

filter：

```text
1 1 1
1 1 1
1 1 1
```

对应元素相乘：

```text
1×1  +  2×1  +  3×1
6×1  +  7×1  +  8×1
11×1 + 12×1 + 13×1
```

全部求和：

```text
1+2+3+6+7+8+11+12+13
= 63
```

因此：

```text
y[0,0,0,0] = 63
```

---

# 11. 卷积的一个输出值，本质上是什么？

对于单通道：

```text
一个输出值
=
输入局部区域
与
filter
逐元素相乘
然后全部求和
再加 bias（如果 bias=True）
```

数学上可以粗略写成：

```text
output
=
Σ(input_patch × weight) + bias
```

注意：

> 这里是**逐元素相乘后求和**，不是普通线性代数里的矩阵乘法。

---

# 12. 为什么第二个值是 72？

由于：

```text
stride = 1
```

卷积核向右移动 1 格。

第二个窗口：

```text
2   3   4
7   8   9
12 13  14
```

全 1 filter 下：

```text
2+3+4+7+8+9+12+13+14
= 72
```

由于这个特殊输入的每一个对应元素都比前一个窗口大 1，而窗口内有 9 个元素：

```text
63 + 9×1
= 72
```

再向右移动：

```text
72 + 9
= 81
```

---

# 13. 为什么下一行第一个值是 108？

第一行最后一个窗口已经到达最右侧。

接下来：

```text
回到最左边
向下移动 1 格
```

新窗口：

```text
6   7   8
11 12 13
16 17 18
```

与最开始相比，每一个数都增加了 5。

一共有 9 个元素，所以：

```text
63 + 9×5
= 108
```

最终：

```text
63   72   81
108 117  126
153 162  171
```

因此：

```text
y.shape = [1,1,3,3]
```

---

# 14. 卷积核是怎么“滑动”的？

输入 5×5，kernel 3×3。

第一位置：

```text
[1  2  3]
[6  7  8]
[11 12 13]
```

向右一格：

```text
[2  3  4]
[7  8  9]
[12 13 14]
```

再向右：

```text
[3  4  5]
[8  9 10]
[13 14 15]
```

此时无法继续右移，因为 kernel 会超出输入范围。

于是回到左侧，向下移动：

```text
[6  7  8]
[11 12 13]
[16 17 18]
```

如此继续。

---

# 15. `stride` 是什么？

`stride` 表示：

> 卷积核每次在 H/W 方向移动多少格。

## `stride=1`

```text
0 → 1 → 2 → 3 → ...
```

每次移动一格。

## `stride=2`

```text
0 → 2 → 4 → ...
```

每次跳两格。

因此 stride 越大：

```text
卷积位置越少
↓
输出 H/W 越小
↓
计算量通常越低
```

所以：

```text
stride=2
```

经常被用于 **下采样**。

---

# 16. `padding` 是什么？

`padding` 表示：

> 在卷积前，输入边缘额外补多少圈像素。

默认补 0。

例如输入：

```text
1 2 3
4 5 6
7 8 9
```

如果：

```text
padding=1
```

概念上变成：

```text
0 0 0 0 0
0 1 2 3 0
0 4 5 6 0
0 7 8 9 0
0 0 0 0 0
```

它的作用之一是：

> 让卷积核能够以原图边缘位置为中心进行计算，并控制输出尺寸。

---

# 17. 三个核心实验：padding 与 stride

## 实验 A

```python
Conv2d(
    1, 1,
    kernel_size=3,
    stride=1,
    padding=0
)
```

输入：

```text
5×5
```

输出：

```text
3×3
```

---

## 实验 B

```python
Conv2d(
    1, 1,
    kernel_size=3,
    stride=1,
    padding=1
)
```

输入：

```text
5×5
```

输出：

```text
5×5
```

经典规律：

```text
kernel_size=3
stride=1
padding=1
```

可以保持 H/W 不变。

---

## 实验 C

```python
Conv2d(
    1, 1,
    kernel_size=3,
    stride=2,
    padding=1
)
```

输入：

```text
5×5
```

输出 H/W 会进一步变小。

这里的关键不是死记某个输出，而是：

> `stride=2` 使采样位置变稀疏，因此产生下采样效果。

---

# 18. 输出尺寸公式

更完整的单维公式：

\[
H_{out}
=
\left\lfloor
\frac{
H_{in}
+2P
-D(K-1)
-1
}{S}
+1
\right\rfloor
\]

宽度同理：

\[
W_{out}
=
\left\lfloor
\frac{
W_{in}
+2P
-D(K-1)
-1
}{S}
+1
\right\rfloor
\]

其中：

```text
H_in / W_in = 输入高宽
K           = kernel size
P           = padding
S           = stride
D           = dilation
```

Day 2 暂时使用：

```text
dilation = 1
```

所以可以简化为：

\[
H_{out}
=
\left\lfloor
\frac{H_{in}+2P-K}{S}
\right\rfloor
+1
\]

---

## 示例 1

```text
H_in = 5
K = 3
P = 0
S = 1
```

得到：

```text
H_out
= floor((5-3)/1)+1
= 3
```

---

## 示例 2

```text
H_in = 32
K = 3
P = 1
S = 2
```

得到：

```text
H_out
= floor((32+2-3)/2)+1
= floor(31/2)+1
= 15+1
= 16
```

因此：

```text
32×32
↓
16×16
```

这就是很多 CNN / ResNet 中常见的空间尺寸减半。

---

# 19. 多输入通道：一个 filter 如何计算？

现在输入：

```python
x = torch.randn(1, 3, 5, 5)
```

shape：

```text
[1,3,5,5]
```

卷积：

```python
conv = torch.nn.Conv2d(
    in_channels=3,
    out_channels=1,
    kernel_size=3,
    bias=False
)
```

此时：

```text
conv.weight.shape
= [1,3,3,3]
```

只有 1 个 filter，但这个 filter 的 shape 是：

```text
[3,3,3]
```

---

# 20. 为什么一个 filter 必须覆盖全部输入通道？

假设输入是 RGB：

```text
R → 3×3 patch
G → 3×3 patch
B → 3×3 patch
```

一个 filter 同样有：

```text
R 权重 → 3×3
G 权重 → 3×3
B 权重 → 3×3
```

计算过程：

```text
R patch × R weights
+
G patch × G weights
+
B patch × B weights
```

所有乘积继续求和：

```text
↓
一个数
```

因此：

> 一个 filter 在一个空间位置只产生 **一个输出值**。

它之所以必须有 `c_in` 层，是因为它需要把所有输入通道的信息综合起来。

---

# 21. `out_channels=64` 为什么得到 64 个 feature maps？

例如：

```python
conv = torch.nn.Conv2d(
    in_channels=3,
    out_channels=64,
    kernel_size=3
)
```

这里：

```text
64 个不同 filter
```

每个 filter：

```text
[3,3,3]
```

每个 filter 在整张输入上滑动：

```text
Filter 1  → Feature Map 1
Filter 2  → Feature Map 2
...
Filter 64 → Feature Map 64
```

因此：

```text
64 个 filter
→ 64 个 feature maps
→ 输出 C = 64
```

所以输入：

```text
[B,3,H,W]
```

经过该卷积：

```text
[B,64,H_out,W_out]
```

---

# 22. 一个空间位置上的含义发生了什么变化？

输入 RGB 图像某位置：

```text
x[b,:,h,w]
```

是：

```text
[R,G,B]
```

3 维描述。

卷积后：

```text
y[b,:,h,w]
```

可能是：

```text
[f1,f2,f3,...,f64]
```

64 维特征描述。

所以：

> CNN 并不是把图片变成“64 种颜色”，而是在每个空间位置学习 64 个特征响应。

---

# 23. `bias` 是什么？

例如：

```python
Conv2d(
    3,
    64,
    kernel_size=3,
    bias=True
)
```

每个输出通道对应一个 bias。

因此：

```text
conv.bias.shape
= [64]
```

单个输出值可以粗略理解为：

```text
output
=
局部输入与 filter 的逐元素乘积求和
+
该输出通道对应的 bias
```

所以：

```text
out_channels = 64
↓
64 个 filter
↓
64 个 bias
```

如果：

```python
bias=False
```

则没有这个加法项。

---

# 24. `requires_grad`、`grad_fn`、`no_grad()` 的区别

这是你当前代码中另一个非常重要的知识点。

## 24.1 `requires_grad`

例如：

```python
conv.weight.requires_grad
```

默认：

```text
True
```

表示：

> 这个 Tensor 是需要梯度的可训练参数。

可以把它理解为 **Tensor 的长期属性**。

---

## 24.2 `grad_fn`

正常 forward：

```python
y = conv(x)
```

由于 `conv.weight.requires_grad=True`，PyTorch 会记录这次卷积计算。

所以：

```text
y.requires_grad = True
```

并且：

```text
y.grad_fn = <ConvolutionBackward0 ...>
```

`grad_fn` 可以理解为：

> 这个 Tensor 是通过哪一种可求导运算产生的？

例如：

```text
MulBackward0
AddBackward0
ReluBackward0
ConvolutionBackward0
```

这些节点共同组成 autograd 的计算图。

---

## 24.3 为什么 `conv.weight.grad_fn = None`？

因为：

```text
conv.weight
```

本身就是模型参数，是一个需要优化的 **leaf tensor**。

它不是由其他 Tensor 运算生成的。

因此：

```text
conv.weight.requires_grad = True
conv.weight.grad_fn = None
```

两者并不矛盾。

---

## 24.4 `torch.no_grad()`

```python
with torch.no_grad():
    y = conv(x)
```

意思是：

> 当前代码块中的运算不建立 autograd 计算图。

因此通常：

```text
y.requires_grad = False
y.grad_fn = None
```

但注意：

```text
conv.weight.requires_grad
仍然 = True
```

所以：

> `no_grad()` 不会永久修改参数的 `requires_grad`。

---

# 25. `requires_grad` 和 `no_grad()` 终极对比

| 问题 | `requires_grad` | `torch.no_grad()` |
|---|---|---|
| 属于谁？ | Tensor / Parameter 的属性 | 当前运算环境 |
| 控制什么？ | 这个 Tensor 是否需要梯度 | 当前代码块是否记录计算图 |
| 是否长期？ | 是，直到手动修改 | 否，只在作用域内 |
| 常见用途 | 冻结/解冻参数 | 推理、验证、手动改权重 |

一句话：

> **`requires_grad` 决定“这个 Tensor 是否需要梯度”；`torch.no_grad()` 决定“当前这次运算是否被 autograd 记录”。**

---

# 26. 三个真实项目场景

## 场景 1：验证 / 推理

```python
model.eval()

with torch.no_grad():
    outputs = model(images)
```

目的：

```text
不反向传播
不需要计算图
减少内存开销
```

注意：

```text
model.eval()
```

和：

```text
torch.no_grad()
```

不是同一件事。

- `model.eval()`：改变 Dropout、BatchNorm 等层的行为；
- `torch.no_grad()`：关闭当前 autograd 记录。

---

## 场景 2：迁移学习冻结参数

```python
for param in model.parameters():
    param.requires_grad = False
```

目的：

```text
这些参数不再需要梯度
```

这是长期冻结参数。

---

## 场景 3：手动修改权重

```python
with torch.no_grad():
    conv.weight[:] = ...
```

目的：

```text
参数以后仍然可以训练
但这一次手动赋值不要进入计算图
```

---

# 27. 为什么训练 forward 不能放进 `no_grad()`？

训练需要：

```text
forward
↓
prediction
↓
loss
↓
loss.backward()
↓
gradient
↓
optimizer.step()
```

如果：

```python
with torch.no_grad():
    y = conv(x)
```

这次卷积没有建立计算图。

之后就无法通过这次 `y` 沿着：

```text
loss
→ y
→ conv
→ weight
```

计算 `conv.weight` 的梯度。

因此：

> 训练 forward 正常情况下不能使用 `torch.no_grad()`。

---

# 28. 为什么卷积可以“提取特征”？

当前全 1 filter：

```text
1 1 1
1 1 1
1 1 1
```

只是一个便于手算的局部求和器。

真正 CNN 中，filter 里的权重是可学习参数：

```text
0.23
-0.51
0.07
...
```

训练过程：

```text
输入
↓
卷积
↓
预测
↓
Loss
↓
反向传播
↓
修改 filter 权重
```

最终，不同 filter 会学习对不同局部模式产生不同响应。

入门阶段可以粗略理解成：

```text
Filter 1 → 某类边缘
Filter 2 → 某种方向变化
Filter 3 → 某种纹理
Filter 4 → 某种颜色组合
...
```

但要注意：

> 深层网络中的 feature channel 往往不能简单对应一个人类可命名概念。

---

# 29. PyTorch 的 `Conv2d` 实际做的是 cross-correlation

严格从数学定义来说，PyTorch 的 `Conv2d` 实现更接近 **cross-correlation（互相关）**：

```text
它不会在计算前把 kernel 翻转 180°
```

但在深度学习中大家仍习惯称它为：

```text
convolution / 卷积
```

对于神经网络训练来说，这通常不造成实际问题，因为 kernel 权重本身就是通过训练学习得到的。

Day 2 只需要知道这个区别，不需要深入数学证明。

---

# 30. PyTorch 与 TensorFlow/Keras 权重维度顺序

PyTorch：

```text
[c_out, c_in, kH, kW]
```

例如：

```text
[64,3,3,3]
```

而 TensorFlow/Keras 常见卷积权重顺序是：

```text
[kH, kW, c_in, c_out]
```

例如：

```text
[3,3,3,64]
```

因此以后看不同框架代码：

> 不要只凭 shape 猜含义，先确认框架的权重排列规则。

---

# 31. 当前示例代码中的实验结构

建议把 `01_conv2d_basic.py` 看成这些实验：

```text
实验 1
单通道 5×5 输入
+ 3×3 全 1 kernel
→ 验证手算结果

实验 2
正常 forward
→ 查看 requires_grad / grad_fn

实验 3
no_grad forward
→ 对比计算图是否建立

实验 4
requires_grad_(False)
→ 理解参数冻结

实验 5
修改 padding / stride
→ 观察 H/W 如何变化

实验 6
in_channels=3
→ 查看单个 filter 的厚度

实验 7
out_channels=64
→ 查看整层 weight 和 bias shape
```

---

# 32. Day 2 练习题

假设：

```python
x = torch.randn(1, 3, 32, 32)
```

## 题 1

```python
conv = torch.nn.Conv2d(
    3,
    16,
    kernel_size=3,
    stride=1,
    padding=0
)
```

回答：

```text
单个 filter.shape = ?
conv.weight.shape = ?
conv.bias.shape = ?
输出 shape = ?
```

答案：

```text
单个 filter = [3,3,3]

conv.weight
= [16,3,3,3]

conv.bias
= [16]

H_out = W_out
= (32-3)/1 + 1
= 30

输出
= [1,16,30,30]
```

---

## 题 2

```python
conv = torch.nn.Conv2d(
    3,
    16,
    kernel_size=3,
    stride=1,
    padding=1
)
```

答案：

```text
weight = [16,3,3,3]

H_out = W_out
= 32

输出 = [1,16,32,32]
```

经典组合：

```text
kernel=3
stride=1
padding=1
```

保持空间尺寸不变。

---

## 题 3

```python
conv = torch.nn.Conv2d(
    3,
    16,
    kernel_size=3,
    stride=2,
    padding=1
)
```

答案：

```text
weight = [16,3,3,3]

H_out = W_out
= floor((32+2-3)/2)+1
= floor(31/2)+1
= 15+1
= 16

输出 = [1,16,16,16]
```

这就是：

```text
空间尺寸减半
通道数增加
```

在 ResNet 等 CNN 中非常常见。

---

## 题 4

```python
conv = torch.nn.Conv2d(
    16,
    64,
    kernel_size=3
)
```

答案：

```text
单个 filter.shape
= [16,3,3]

全部 weight.shape
= [64,16,3,3]

bias.shape
= [64]
```

如果输入：

```text
[1,16,32,32]
```

且使用默认：

```text
stride=1
padding=0
```

则输出：

```text
[1,64,30,30]
```

---

## 题 5：非正方形 kernel

```python
conv = torch.nn.Conv2d(
    3,
    8,
    kernel_size=(5,3)
)
```

回答：

```text
单个 filter.shape = ?
conv.weight.shape = ?
```

答案：

```text
单个 filter
= [3,5,3]

conv.weight
= [8,3,5,3]
```

---

# 33. Day 2 易错点清单

## 易错点 1

```text
kernel_size=3
```

不是：

```text
只看 3 个数
```

而是：

```text
3×3 空间窗口
```

---

## 易错点 2

```text
Conv2d(3,64,3)
```

不是单个 filter 的 shape。

而是：

```text
整个卷积层配置
```

单个 filter 是：

```text
[3,3,3]
```

整层 weight 是：

```text
[64,3,3,3]
```

---

## 易错点 3

```text
out_channels=64
```

不是：

```text
每个 filter 有 64 层
```

而是：

```text
一共有 64 个 filter
```

---

## 易错点 4

```text
in_channels=3
```

决定：

```text
每个 filter 必须覆盖 3 个输入通道
```

---

## 易错点 5

```text
torch.no_grad()
```

不会永久把：

```text
requires_grad=True
```

变成：

```text
False
```

---

## 易错点 6

```text
y.grad_fn=None
```

如果是在 `no_grad()` 下 forward，不是“计算图存在但没显示”，而是：

```text
这次运算确实没有建立 autograd 计算图
```

---

## 易错点 7

使用关键字参数：

```python
Conv2d(
    out_channels=64,
    in_channels=3,
    kernel_size=3
)
```

顺序可以调整。

但位置参数：

```python
Conv2d(3,64,3)
```

顺序固定为：

```text
in_channels
out_channels
kernel_size
```

---

# 34. Day 2 核心逻辑链

把今天所有内容压缩成下面这条链：

```text
输入 Tensor
[B, C_in, H, W]

        ↓

Conv2d(
    C_in,
    C_out,
    kernel_size
)

        ↓

共有 C_out 个 filter

每个 filter：
[C_in, kH, kW]

整层 weight：
[C_out, C_in, kH, kW]

        ↓

每个 filter
在输入 H/W 上滑动

        ↓

每个位置：
局部 patch
×
filter
逐元素相乘并求和
+ bias

        ↓

一个 filter
生成一张 feature map

        ↓

C_out 个 filter
生成 C_out 张 feature map

        ↓

输出 Tensor
[B, C_out, H_out, W_out]
```

这就是普通 `Conv2d` 的核心数据流。

---

# 35. Day 2 完成标准

如果下面的问题能不看笔记回答，Day 2 就可以结束：

- [ ] `Conv2d(3,64,3)` 三个数字各是什么？
- [ ] 关键字参数和位置参数有什么区别？
- [ ] `kernel_size=3` 为什么等于 `(3,3)`？
- [ ] 非正方形 5×3 kernel 怎么写？
- [ ] `Conv2d(3,64,3)`、单个 filter `[3,3,3]`、整层 weight `[64,3,3,3]` 有什么区别？
- [ ] 为什么 filter 必须覆盖所有输入通道？
- [ ] 一个卷积位置怎样得到一个输出值？
- [ ] 5×5 输入、3×3 kernel、stride=1、padding=0 为什么得到 3×3？
- [ ] stride 为什么会影响输出尺寸？
- [ ] padding 为什么可以帮助保留空间尺寸？
- [ ] `bias.shape` 为什么等于 `[out_channels]`？
- [ ] `conv.weight[:]` 是什么？
- [ ] `requires_grad` 和 `torch.no_grad()` 有什么区别？
- [ ] `grad_fn=<ConvolutionBackward0>` 表示什么？
- [ ] 为什么正常训练 forward 不能包在 `no_grad()` 中？
- [ ] 为什么一个 filter 生成一个 feature map？
- [ ] 为什么 64 个 filter 生成 64 个输出 channel？
- [ ] 能否根据输入和 Conv2d 参数，在运行代码前先推导 `weight.shape` 和输出 shape？

---

# 36. 下一步：Day 3

Day 2 解决的是：

```text
线性卷积
```

下一步 Day 3 应该进入：

```text
Conv2d
↓
ReLU
↓
Pooling / 下采样
↓
多个卷积层串联
↓
一个最小 CNN
```

真正开始回答：

> 为什么只有卷积还不够？  
> 为什么需要非线性激活？  
> 为什么 CNN 会逐层从低级局部特征变成高级语义特征？

这会为后面的 Residual Block 和 ResNet 打地基。
