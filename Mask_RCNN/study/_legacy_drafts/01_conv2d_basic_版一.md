# Day 2：真正搞懂 `Conv2d` 是怎么计算的

> 对应示例代码：`01_conv2d_basic.py`  
> 前置知识：Day 1 的 `Tensor / BCHW / channel / shape / 索引`

---

## 0. Day 2 的核心目标

Day 1 已经知道：

```text
图片 → Tensor → [B, C, H, W]
```

Day 2 只解决一个核心问题：

> **`Conv2d` 到底怎样把输入 Tensor 计算成 feature map？**

学完后应能解释：`in_channels`、`out_channels`、`kernel_size`、`stride`、`padding`、`bias`、`conv.weight.shape`、一个输出值怎么计算、输出 H/W 怎么推导、多通道和多 filter 怎么工作，以及 `grad_fn / no_grad / requires_grad` 的区别。

---

# 1. 从最简单的输入开始

为了能手算，先不用真实 RGB 图片，而使用：

```text
[B, C, H, W] = [1, 1, 5, 5]
```

输入矩阵：

```text
1   2   3   4   5
6   7   8   9  10
11 12  13  14  15
16 17  18  19  20
21 22  23  24  25
```

代码见 `01_conv2d_basic.py`。当前 `x.shape = [1,1,5,5]`：1 张图、1 个通道、高 5、宽 5。

---

# 2. 定义第一个卷积层

```python
conv = torch.nn.Conv2d(
    in_channels=1,
    out_channels=1,
    kernel_size=3,
    stride=1,
    padding=0,
    bias=False,
)
```

| 参数 | 当前值 | 含义 |
|---|---:|---|
| `in_channels` | 1 | 输入通道数 |
| `out_channels` | 1 | 输出 feature channel 数 |
| `kernel_size` | 3 | 空间窗口大小 3×3 |
| `stride` | 1 | 每次移动 1 格 |
| `padding` | 0 | 输入边缘不补值 |
| `bias` | False | 暂时不加偏置 |

---

# 3. `conv.weight.shape`

PyTorch 中普通 `Conv2d` 权重排列：

```text
[C_out, C_in, K_h, K_w]
```

当前：

```text
conv.weight.shape = [1,1,3,3]
```

也就是：1 个 filter；每个 filter 覆盖 1 个输入通道；空间大小 3×3。

如果以后：

```python
Conv2d(3, 64, kernel_size=3)
```

则：

```text
conv.weight.shape = [64,3,3,3]
```

单个 filter 为：

```text
[3,3,3] = [C_in,K_h,K_w]
```

---

# 4. 为什么手工把 kernel 设成全 1？

默认卷积权重是随机初始化的，不便于手算，所以教学时改成：

```text
1 1 1
1 1 1
1 1 1
```

代码：

```python
with torch.no_grad():
    conv.weight[:] = ...
```

## `conv.weight[:]` 是什么？

`[:]` 表示对整个权重 Tensor 做全范围切片。

```python
conv.weight[:] = new_weight
```

表示：**原地修改原 Parameter 里的全部数值**。

## 为什么用 `torch.no_grad()`？

`conv.weight` 默认 `requires_grad=True`，它是训练参数。这里人工赋值只是初始化/调试，不希望这次赋值进入 autograd 计算图，因此暂时关闭梯度记录。

但：

```text
torch.no_grad() 不会把 conv.weight.requires_grad 永久改成 False
```

---

# 5. 一个输出值是怎样算出来的？

执行：

```python
y = conv(x)
```

输出左上角对应输入左上 3×3：

```text
输入局部区域          kernel

1   2   3            1 1 1
6   7   8            1 1 1
11 12 13            1 1 1
```

计算方式是：

> **对应位置逐元素相乘，然后全部求和。**

```text
1×1 + 2×1 + 3×1
+ 6×1 + 7×1 + 8×1
+ 11×1 + 12×1 + 13×1
= 63
```

所以 `y` 左上角是 63。

> 这不是普通矩阵乘法，而是局部逐元素乘 + 求和。

---

# 6. kernel 是怎么滑动出整张 feature map 的？

因为 `stride=1`，每次移动 1 格。

第一格：

```text
1   2   3
6   7   8
11 12 13
→ 63
```

向右 1 格：

```text
2   3   4
7   8   9
12 13 14
→ 72
```

因为相比上一窗口，9 个位置的值都增加 1，而 kernel 全是 1，所以总和增加 `9×1=9`：

```text
63 + 9 = 72
```

再向右：

```text
72 + 9 = 81
```

当前行到最右端后，回到左侧并向下移动 1 格。新窗口相对最初窗口每个值都增加 5，所以：

```text
63 + 9×5 = 108
```

最终：

```text
63   72   81
108 117 126
153 162 171
```

因此：

```text
y.shape = [1,1,3,3]
```

---

# 7. Feature Map 是什么？

当前只有 1 个 filter，因此生成 1 张输出特征图：

```text
63   72   81
108 117 126
153 162 171
```

可以理解为：

> **同一个 filter 在输入不同空间位置上的响应集合。**

当前全 1 kernel 只是局部求和器；真正训练后的 kernel 会变成各种可学习权重，从而对边缘、纹理、颜色组合、局部形状等模式产生不同响应。

---

# 8. `stride` 是什么？

`stride` 表示卷积窗口每次在 H/W 方向移动多少格。

```text
stride=1：0 → 1 → 2 → 3 ...
stride=2：0 → 2 → 4 ...
```

步长越大，合法落点越少，输出 H/W 通常越小，因此 `stride=2` 常用于下采样。

---

# 9. `padding` 是什么？

`padding` 表示卷积前在输入边缘增加额外像素。常见 `padding=1` 是在四周补一圈 0。

## `padding=0`

```text
5×5 输入 + 3×3 kernel + stride=1
→ 3×3 输出
```

因为 kernel 只能放在完全位于当前有效输入区域内的位置。

## `padding=1`

原 5×5 周围补一圈：

```text
0  0  0  0  0  0  0
0  1  2  3  4  5  0
0  6  7  8  9 10  0
0 11 12 13 14 15  0
0 16 17 18 19 20  0
0 21 22 23 24 25  0
0  0  0  0  0  0  0
```

于是 `k=3,s=1,p=1` 时：

```text
5×5 → 5×5
```

注意：

> “kernel 必须完全落在输入内部”应理解为**完全落在当前有效输入（包括 padding 后的区域）内**。

---

# 10. 输出尺寸公式

一般形式：

```text
H_out = floor((H_in + 2P - D(K-1) - 1) / S + 1)
```

宽度同理。

Day 2 暂时 `dilation=1`，可简化为：

```text
H_out = floor((H_in + 2P - K) / S) + 1
```

当前：

```text
H_in=5, K=3, P=0, S=1
H_out=(5-3)/1+1=3
```

所以：

```text
[1,1,5,5] → [1,1,3,3]
```

---

# 11. 三个尺寸实验

| 配置 | 输入 | 输出 |
|---|---|---|
| `k=3,s=1,p=0` | 5×5 | 3×3 |
| `k=3,s=1,p=1` | 5×5 | 5×5 |
| `k=3,s=2,p=1` | 5×5 | 3×3 |

第三个：

```text
floor((5-3+2)/2)+1
= floor(4/2)+1
= 3
```

如果是 32×32：

```text
k=3,s=2,p=1
32×32 → 16×16
```

这就是后面 ResNet 中常见的“空间减半”。

> 注意：若 `5×5, k=3, s=2, p=0`，合法横向起点是 `0、2`，只有 **2 个**，不是 3 个。

---

# 12. `kernel_size=3` 不等于只有 9 个权重

`kernel_size=3` 只表示空间尺寸是 3×3。

如果输入有 3 个通道：

```text
C_in = 3
```

单个普通 Conv2d filter 是：

```text
[3,3,3] = [C_in,K_h,K_w]
```

也就是它同时覆盖 R/G/B 三个通道。

---

# 13. 多输入通道怎样产生一个输出值？

假设输入有 R/G/B 三个通道。一个 filter 也对应三组 3×3 权重：

```text
R 局部 3×3 × R 权重 3×3
+
G 局部 3×3 × G 权重 3×3
+
B 局部 3×3 × B 权重 3×3
↓
全部求和
↓
一个输出值
```

数学结构：

```text
y = Σ_c Σ_i Σ_j x[c,i,j] * w[c,i,j] + b
```

因此：

> **一个普通 Conv2d filter 在某个空间位置，会同时查看全部输入通道，最后只产生一个数。**

---

# 14. 为什么 `out_channels=64` 会产生 64 个输出通道？

```python
Conv2d(3, 64, kernel_size=3)
```

等价理解：

```text
Filter 1  [3,3,3] → Feature Map 1
Filter 2  [3,3,3] → Feature Map 2
...
Filter 64 [3,3,3] → Feature Map 64
```

所以：

```text
64 个 filter
→ 64 张 feature maps
→ 64 个 output channels
```

即：

```text
[B,3,H,W] → [B,64,H_out,W_out]
```

---

# 15. `bias` 是什么？

如果：

```python
Conv2d(3, 64, kernel_size=3, bias=True)
```

则：

```text
conv.bias.shape = [64]
```

因为每个输出通道有一个 bias。

```text
一个输出值 = 局部输入×权重全部求和 + 当前输出通道的 bias
```

注意：bias 不是“每个像素一个”，而是“每个输出通道一个”。

---

# 16. 为什么卷积能提取特征？

全 1 kernel 只是局部求和。训练后，kernel 可能变成正负不同的权重，例如：

```text
-1  0  1
-1  0  1
-1  0  1
```

它可能对某种左右强度变化产生较大响应。

真实 CNN：

```text
随机初始化 kernel
↓
forward
↓
loss
↓
backward
↓
更新 weight
↓
逐渐学到对任务有用的局部模式
```

所以卷积并不是“天然知道边缘是什么”，而是：

> **通过训练，让大量共享的局部数字模板逐渐变成有用的特征探测器。**

---

# 17. CNN 卷积的两个重要性质

## 局部连接

一个输出位置只查看 kernel 覆盖的局部区域，而不是一次读取整张图。

## 权重共享

同一个 filter 在整张 H/W 上反复使用同一套权重。

```text
同一个 kernel
↓
左→右
↓
上→下
↓
重复使用
```

这也是 CNN 能高效处理图像的重要原因。

---

# 18. 严格来说，PyTorch `Conv2d` 做的是互相关

数学上，严格的“卷积”会先把 kernel 旋转 180°；PyTorch `Conv2d` 前向计算不会做这个翻转，因此更接近：

```text
cross-correlation（互相关）
```

但深度学习里仍习惯称它为 convolution / Conv2d。因为 filter 是可学习的，这个命名差异不影响网络学习能力。

Day 2 知道这一点即可，不必深究。

---

# 19. `grad_fn=<ConvolutionBackward0>` 是什么？

正常执行：

```python
y = conv(x)
```

虽然：

```text
x.requires_grad = False
```

但：

```text
conv.weight.requires_grad = True
```

所以这次运算涉及可训练参数，PyTorch 会建立 autograd 计算关系。

因此通常：

```text
y.requires_grad = True
y.grad_fn = <ConvolutionBackward0>
```

含义：

> `y` 是通过可求导的卷积操作产生的，PyTorch 保存了未来反向传播需要的信息。

---

# 20. 为什么 `conv.weight.grad_fn = None`？

`conv.weight` 是模型直接持有的 Parameter，不是由其他可求导操作算出来的。它属于 leaf tensor。

所以可以同时：

```text
requires_grad = True
grad_fn = None
```

---

# 21. `requires_grad` 和 `torch.no_grad()` 的区别

`requires_grad`：

> 这个 Tensor 本身是否需要梯度。

`torch.no_grad()`：

> 当前代码块里的运算是否建立 autograd 计算图。

例如：

```python
with torch.no_grad():
    y = conv(x)
```

这时通常：

```text
conv.weight.requires_grad = True   # 参数本身没被改
y.requires_grad = False
y.grad_fn = None
```

这不是“记录了但没显示”，而是真的没有为这次 forward 建立用于反向传播的 autograd 图。

而：

```python
conv.weight.requires_grad_(False)
```

才是真正冻结这个参数。

| 情况 | 参数 requires_grad | 当前 forward 建图？ | 常见用途 |
|---|---:|---:|---|
| 正常训练 | True | 是 | 训练 |
| `torch.no_grad()` | 仍可为 True | 否 | 推理 |
| `requires_grad_(False)` | False | 取决于其他输入/参数 | 冻结参数 |

一句话：

> **`requires_grad` 决定“Tensor 是否需要梯度”；`torch.no_grad()` 决定“当前运算是否被 autograd 记录”。**

---

# 22. Shape 练习答案

假设：

```python
x = torch.randn(1, 3, 32, 32)
```

## 题 1

```python
Conv2d(3,16,kernel_size=3,stride=1,padding=0)
```

```text
weight = [16,3,3,3]
output = [1,16,30,30]
```

## 题 2

```python
Conv2d(3,16,kernel_size=3,stride=1,padding=1)
```

```text
weight = [16,3,3,3]
output = [1,16,32,32]
```

规律：`k=3,p=1,s=1` 时 H/W 保持不变。

## 题 3

```python
Conv2d(3,16,kernel_size=3,stride=2,padding=1)
```

```text
weight = [16,3,3,3]
output = [1,16,16,16]
```

即：

```text
空间：32×32 → 16×16
通道：3 → 16
```

这是 ResNet 等网络中的常见模式。

## 题 4

```python
Conv2d(16,64,kernel_size=3)
```

```text
单个 filter = [16,3,3]
全部 weight = [64,16,3,3]
```

若输入 `[1,16,32,32]`，默认 `stride=1,padding=0`：

```text
output = [1,64,30,30]
```

---

# 23. PyTorch 与 TensorFlow/Keras 的 weight 排列

PyTorch 常见：

```text
[out_channels, in_channels, kH, kW]
```

TensorFlow/Keras 常见：

```text
[kH, kW, in_channels, out_channels]
```

所以看不同框架源码时，先确认维度约定，不要只看数字猜语义。

---

# 24. Day 2 概念总图

```text
输入 Tensor
[B, C_in, H, W]
        │
        ▼
      Conv2d
        │
        ├─ C_in：单个 filter 深度
        ├─ kernel_size：局部窗口
        ├─ stride：移动距离
        ├─ padding：边缘扩展
        ├─ C_out：filter 数量
        └─ bias：每个输出通道一个
        │
        ▼
输出 Feature Maps
[B, C_out, H_out, W_out]
```

---

# 25. 从“一个值”到“整层输出”的三层关系

## 第一层：一个 output value

```text
局部输入 × 一个 filter
→ 一个数
```

## 第二层：一张 feature map

```text
同一个 filter 在 H/W 上滑动
→ 一张 feature map
```

## 第三层：多个 output channels

```text
多个不同 filter
→ 多张 feature maps
→ C_out 个输出通道
```

把这三层关系掌握，`Conv2d` 就基本不再是黑盒。

---

# 26. Day 2 完成标准

- [ ] `Conv2d(3,64,3)` 的 3、64、3 分别是什么？
- [ ] `conv.weight.shape=[64,3,3,3]` 四维分别是什么？
- [ ] 单个 filter 为什么是 `[C_in,K_h,K_w]`？
- [ ] 一个 filter 在一个位置怎样算出一个值？
- [ ] 为什么一个 filter 产生一张 feature map？
- [ ] 为什么 64 个 filter 产生 64 个 output channels？
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

---

# 27. Day 2 最终必须牢牢记住

```text
Conv2d 输入：
[B, C_in, H, W]
```

```text
Conv2d weight：
[C_out, C_in, K_h, K_w]
```

```text
一个 filter：
[C_in, K_h, K_w]
```

```text
一个 filter + 一个空间位置
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

---

# 28. 下一课：Day 3

Day 2 已经得到：

```text
输入 → Conv2d → Feature Map
```

下一步要回答：

> **卷积本身只是线性加权求和，如果不断堆卷积，为什么神经网络还能学习复杂模式？**

因此 Day 3 学：

```text
Conv2d
↓
ReLU
↓
Pooling / stride downsampling
↓
Flatten
↓
Linear
↓
一个最小 CNN
```

Day 3 的核心问题：

> **为什么必须加入非线性激活函数 ReLU？**
