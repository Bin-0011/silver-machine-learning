Day 2 我建议你只学一件核心事情：

# Day 2：真正搞懂 `Conv2d` 是怎么计算的

Day 1 你已经解决了：

```text
图片 → Tensor
HWC → BCHW
索引怎么改变 shape
RGB 3 通道 ≠ CNN feature channels
```

Day 2 要继续往前一步：

> **一个卷积核到底怎么从输入 Tensor 算出 feature map？**

这是 CNN 最核心的一课。

---

# 一、Day 2 的学习目标

今天学完后，你应该能回答：

1. `nn.Conv2d(3, 64, kernel_size=3)` 每个参数是什么意思？
2. 为什么输入 3 通道，卷积核也必须覆盖 3 个通道？
3. 一个 `[3,3,3]` filter 怎么算出 **一个输出值**？
4. 64 个 filter 为什么产生 64 个输出通道？
5. `stride` 是什么？
6. `padding` 是什么？
7. `kernel_size` 是什么？
8. 输出的 H、W 怎么算？
9. 为什么 `stride=2` 会让特征图变小？
10. 卷积为什么能“提取特征”？

---

# 二、今天要新建的文件

建议：

```text
D:\Workspace\Machinelearning\Mask_RCNN\study\01_cnn\
```

里面新建：

```text
01_conv2d_basic.py
```

以及笔记：

```text
01_conv2d_basic.md
```

今天还是保持：

> **代码和笔记分开。**

---

# 三、第一部分：先别用 640×640，改成超小矩阵

今天最重要的原则：

> **不要一上来用真实图片。**

因为：

```text
640 × 640 × 3
```

太大了，你根本看不清卷积怎么算。

我们先用：

```text
1 × 1 × 5 × 5
```

也就是：

```text
B=1
C=1
H=5
W=5
```

例如：

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

shape 是：

```text
[1, 1, 5, 5]
```

---

# 四、第二部分：自己定义一个 3×3 卷积核

先只用：

```text
1 输入通道
1 输出通道
3×3 kernel
```

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

这时候：

```text
conv.weight.shape
```

应该是：

```text
[1, 1, 3, 3]
```

解释：

```text
1 个输出通道
1 个输入通道
3 × 3 卷积核
```

---

# 五、第三部分：手工指定卷积核

不要让它随机初始化。

比如设成：

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

这段代码是在**手动强行覆盖**一个卷积层（`conv`）的权重（`weight`），把这个卷积核变成**一个指定的 3×3 全1矩阵**。

### 1. 核心操作：`conv.weight[:] = ...`
- `conv.weight` 是卷积层的权重参数（也就是卷积核里面的那些数字）。
- `[:]` 是**原地赋值**（不改变内存地址，只替换里面的数值）。这确保 PyTorch 仍然把这个张量当作模型参数来管理，而不是把它变成一个普通的局部变量。

### 2. 被赋值的数字：`torch.tensor([[[[1., ... ]]]])`
我们数一下括号，这个张量的形状是 **`[1, 1, 3, 3]`**。
这意味着当前的 `conv` 必须是：
- `in_channels=1`（输入是单通道，比如灰度图）
- `out_channels=1`（输出也是单通道）
- `kernel_size=3`（3×3 的卷积核）

这个 3×3 矩阵长这样：
```text
[ [1, 1, 1],
  [1, 1, 1],
  [1, 1, 1] ]
```
也就是卷积核里的 **9 个数字全是 1**。

### 3. 为什么要包上 `with torch.no_grad():`？
这是**最关键**的一步！

- 默认情况下，`conv.weight` 是 `requires_grad=True`（需要计算梯度的）。
- 如果你直接用 `conv.weight[:] = ...` 赋值，PyTorch 会报一个**“就地修改”的警告/错误**，因为它会打断自动求导的追踪链条，导致反向传播无法计算梯度。
- `with torch.no_grad():` 的作用就是**临时关闭梯度追踪**。相当于告诉 PyTorch：“我现在是在手动调试/初始化，不是在训练，你不要记录这次赋值操作，不要管梯度了。”

---

### 这个操作带来的实际效果（数学意义）

假设输入一张图片，经过这个卷积层后，**输出的 Feature Map 上每一个点的数值，就等于输入图片中对应 3×3 区域里 **9 个像素值的总和**。

- 如果输入是 `[[a,b,c],[d,e,f],[g,h,i]]`，输出就是 `a+b+c+d+e+f+g+h+i`。
- 如果想让输出变成**平均值**（模糊效果），通常会再除以 9，但这里没有除，所以它就是纯粹的“局部求和”或“亮度累加器”。

### 什么时候会写这种代码？
一般只有这 3 种情况会这么干：

1. **自定义初始化**：不想用默认的随机初始化（Kaiming/Xavier），想指定一个固定的核（比如做边缘检测的 Sobel 算子，或者做模糊的平均核）。
2. **调试验证**：为了验证卷积计算的底层逻辑，用全 1 的核，可以心算出结果，方便检查代码有没有写错。
3. **迁移学习/冻结**：在加载预训练模型后，想强行修改某一层的权重，但又不想破坏这个参数在优化器里的注册状态。

**一句话总结**：这段代码就是在“不讲道理”地把卷积核硬生生改成 9 个 1，用来做局部区域像素求和运算，同时告诉 PyTorch“别管梯度，我这是手动硬调”。

---

然后：

```python
y = conv(x)

print(y)
print(y.shape)
```

---
`grad_fn=<ConvolutionBackward0>` 是 **PyTorch 自动求导系统 autograd 留下的“计算历史标记”**。

你这段：

```python
y = conv(x)
```

不是简单地算出一个普通 Tensor。

因为 `conv.weight` 默认：

```python
requires_grad = True
```

也就是说：

> 这个卷积层的权重是可训练参数，后面需要根据 loss 计算梯度、更新权重。

所以 PyTorch 会记住：

```text
x
 ↓
Conv2d
 ↓
y
```

并在 `y` 上记录：

```text
grad_fn=<ConvolutionBackward0>
```

意思就是：

> `y` 是由一次卷积操作计算得到的；如果以后调用反向传播，PyTorch 知道应该通过这个卷积操作往回计算梯度。

---

## 1. `grad_fn` 是什么？

可以理解成：

```text
grad_fn
=
这个 Tensor 是通过什么运算生成的？
```

你的输出：

```text
tensor(..., grad_fn=<ConvolutionBackward0>)
```

表示：

```text
这个 Tensor
    ↑
由卷积操作产生
    ↑
PyTorch 已经记录了反向传播规则
```

---

## 2. 为什么 `x` 没有 `grad_fn`？

你创建：

```python
x = torch.tensor(...)
```

默认：

```python
x.requires_grad == False
```

所以：

```python
print(x.requires_grad)
```

会得到：

```text
False
```

而卷积层：

```python
print(conv.weight.requires_grad)
```

通常会得到：

```text
True
```

因为卷积核权重是模型要学习的参数。

所以即使 `x` 本身不需要梯度：

```text
x.requires_grad = False
```

只要运算里参与了：

```text
conv.weight
```

这种需要梯度的 Tensor，

PyTorch 就必须记录这次计算。

因此：

```text
y.requires_grad
```

会变成：

```text
True
```

---

# 3. 你现在可以直接加这个实验

在代码最后加入：

```python
print("\n===== 自动求导信息 =====")

print("x.requires_grad =", x.requires_grad)
print("conv.weight.requires_grad =", conv.weight.requires_grad)
print("y.requires_grad =", y.requires_grad)

print("x.grad_fn =", x.grad_fn)
print("conv.weight.grad_fn =", conv.weight.grad_fn)
print("y.grad_fn =", y.grad_fn)
```

你大概率会看到类似：

```text
x.requires_grad = False

conv.weight.requires_grad = True

y.requires_grad = True
```

以及：

```text
x.grad_fn = None

conv.weight.grad_fn = None

y.grad_fn = <ConvolutionBackward0 ...>
```

这里又出现了一个值得理解的问题：

> 为什么 `conv.weight.requires_grad=True`，但 `conv.weight.grad_fn=None`？

因为：

```text
conv.weight
```

不是由其他 Tensor 算出来的。

它本身就是一个需要优化的**叶子 Tensor（leaf tensor）**。

所以：

```text
requires_grad=True
```

但：

```text
grad_fn=None
```

---

# 4. 可以把 Tensor 分成两类

## 第一类：叶子 Tensor

比如：

```python
conv.weight
```

它是模型参数本身。

可以理解成：

```text
训练要修改的“原始变量”
```

因此：

```text
requires_grad = True
grad_fn = None
```

---

## 第二类：计算产生的 Tensor

比如：

```python
y = conv(x)
```

`y` 是经过计算产生的。

所以：

```text
requires_grad = True
grad_fn = ConvolutionBackward0
```

因为 PyTorch 需要知道：

> y 是怎么算来的？

将来才能反向传播。

---

# 5. `ConvolutionBackward0` 到底是什么？

名字拆开：

```text
Convolution
+
Backward
```

意思就是：

> 卷积对应的反向传播函数。

你现在做的是正向传播：

```text
Forward

x
↓
Conv
↓
y
```

以后训练时，会有：

```text
y
↓
Loss
↓
loss.backward()
↓
计算梯度
```

PyTorch 就沿着保存好的计算图往回走：

```text
Loss
 ↑
y
 ↑
Conv
 ↑
weight
```

这时候：

```text
ConvolutionBackward0
```

就负责回答：

> 如果最终 loss 变了一点，那么卷积核 `weight` 应该往哪个方向、改多少？

---

# 6. 一个非常简化的例子

先不看卷积。

例如：

```python
a = torch.tensor(2.0, requires_grad=True)

b = a * 3

print(b)
```

会看到类似：

```text
tensor(6., grad_fn=<MulBackward0>)
```

因为：

```text
b = a × 3
```

所以 PyTorch 记录：

```text
MulBackward0
```

表示：

> b 是通过乘法产生的。

如果：

```python
c = b + 5
```

那么：

```text
c
```

可能有：

```text
AddBackward0
```

所以不同操作对应不同的 `grad_fn`。

例如你以后可能看到：

```text
AddBackward0
MulBackward0
ReluBackward0
ConvolutionBackward0
```

它们本质都是：

> **计算图中的一个节点，告诉 autograd 怎样做反向传播。**

---

# 7. 为什么你用了 `torch.no_grad()`，结果 y 还有 `grad_fn`？

这点特别容易误解。

你代码里：

```python
with torch.no_grad():
    conv.weight[:] = torch.tensor(...)
```

这里 `torch.no_grad()` **只包住了权重赋值操作**：

```text
给 conv.weight 写入全 1
```

它的目的是：

> 不要让“手动修改权重”这件事进入计算图。

但是：

```python
y = conv(x)
```

已经在：

```python
with torch.no_grad():
```

外面了。

所以卷积运算本身还是正常记录梯度。

也就是：

```text
with torch.no_grad():
    修改 weight      ← 不记录

y = conv(x)           ← 重新开始正常记录
```

因此 `y` 依然有：

```text
grad_fn=<ConvolutionBackward0>
```

---

# 8. 如果不想看到 `grad_fn` 怎么办？

例如只做推理：

```python
with torch.no_grad():
    y = conv(x)

print(y)
```

这时通常：

```text
y.requires_grad = False
y.grad_fn = None
```

因为你告诉 PyTorch：

> 我现在不训练，只计算结果，不需要建立反向传播计算图。

不过在你当前学习阶段，**我反而建议保留 grad_fn**。

因为它刚好可以让你开始理解：

```text
正向传播
→ 计算图
→ loss
→ backward
→ gradient
→ 更新 weight
```

这就是 Day 3/后续训练 CNN 时的核心。

---

# 9. 你现在可以这样理解整个代码

你的代码实际上已经做了：

```text
输入 x
[1,1,5,5]

        ↓

Conv2d

卷积核 weight
[1,1,3,3]

        ↓

输出 y
[1,1,3,3]

        ↓

PyTorch 同时记录：
“y 是由卷积算出来的”

        ↓

grad_fn =
ConvolutionBackward0
```

而你手算得到：

```text
63  72  81
108 117 126
153 162 171
```

是**前向传播结果**。

`grad_fn` 则是在为将来的：

```python
loss.backward()
```

做准备。

---

## 你现在最值得记进 Day 2 笔记的一句话

> **`grad_fn` 表示 Tensor 是通过哪一种可求导运算生成的。PyTorch 通过这些 `grad_fn` 连接成计算图，使 `loss.backward()` 能够沿计算图反向计算模型参数的梯度。**

另外，你这次的卷积计算结果本身是正确的：

```text
5×5 输入
3×3 kernel
stride=1
padding=0

↓

3×3 输出
```

而左上角的：

```text
63
```

也正是左上 3×3 区域所有元素求和得到的。
对，这里有两个概念要分开：

1. `conv.weight[:]` 是在做什么？
2. `torch.no_grad()` 到底控制的是 **“requires_grad 属性”**，还是 **“是否记录这次运算”**？

这两个很容易混在一起。

---

# 1. `conv.weight[:]` 是什么？

先看：

```python
conv.weight
```

它是卷积层的权重 Tensor。

你这个卷积：

```python
conv = torch.nn.Conv2d(
    in_channels=1,
    out_channels=1,
    kernel_size=3,
    bias=False
)
```

所以：

```python
conv.weight.shape
```

是：

```text
[1, 1, 3, 3]
```

---

## `[:]` 表示什么？

```python
conv.weight[:]
```

表示：

> 对 `conv.weight` 的所有元素进行切片，也就是“整个 Tensor 全部选中”。

类似于普通 Python：

```python
a = [1, 2, 3]

a[:]   # 整个列表
```

所以：

```python
conv.weight[:] = ...
```

可以理解成：

> 把 `conv.weight` 里面的所有元素都替换成右边给出的值。

你的代码：

```python
conv.weight[:] = torch.tensor(
    [[[
        [1., 1., 1.],
        [1., 1., 1.],
        [1., 1., 1.]
    ]]]
)
```

就是：

```text
把卷积核原来随机初始化的权重

↓

全部改成 1
```

---

# 2. 为什么不能直接随便改 `conv.weight`？

因为：

```python
conv.weight.requires_grad
```

默认是：

```text
True
```

也就是说它是一个**需要训练的模型参数**。

PyTorch 对这种叶子 Tensor 的原地修改比较严格。

例如直接：

```python
conv.weight[:] = 1
```

在启用梯度跟踪时可能报错，因为你正在对：

```text
requires_grad=True 的 leaf tensor
```

做原地修改。

所以一般这样写：

```python
with torch.no_grad():
    conv.weight[:] = ...
```

意思是：

> 这次我是在人工初始化参数，不是模型训练计算，请不要记录这次赋值。

---

# 3. `torch.no_grad()` 不会把 `requires_grad=True` 改成 False

这是你现在最需要纠正的地方。

例如：

```python
print(conv.weight.requires_grad)
```

原本：

```text
True
```

然后：

```python
with torch.no_grad():
    conv.weight[:] = 1
```

之后再：

```python
print(conv.weight.requires_grad)
```

依然是：

```text
True
```

所以：

> `torch.no_grad()` **不会修改 Tensor 本身的 `requires_grad` 属性。**

它只是在它的作用域内告诉 PyTorch：

> **不要为这些运算建立 autograd 计算图。**

---

# 4. 所以要区分两个问题

## 问题 A：这个 Tensor 本身想不想要梯度？

由：

```python
requires_grad
```

决定。

例如：

```python
conv.weight.requires_grad == True
```

说明：

> 这个参数将来可以参与梯度计算。

---

## 问题 B：当前这一次运算要不要被记录？

由当前是否处于：

```python
torch.no_grad()
```

环境决定。

所以可以记成：

```text
requires_grad
    ↓
Tensor 自身属性
“我是不是可训练的？”
```

而：

```text
torch.no_grad()
    ↓
运算环境
“这一次操作要不要建立计算图？”
```

两者不是一回事。

---

# 5. 回到你的第二个问题

你问：

> `torch.no_grad()` 不仅要包括 `conv.weight[:] = ...`，还要包括 `y = conv(x)` 才能不让 `requires_grad=True` 吗？

这里更准确地说：

如果：

```python
with torch.no_grad():
    y = conv(x)
```

那么产生的 `y` 通常会：

```python
y.requires_grad == False
```

并且：

```python
y.grad_fn == None
```

因为这一次卷积操作没有被 autograd 记录。

---

但是：

```python
conv.weight.requires_grad
```

仍然还是：

```text
True
```

这非常重要。

也就是：

```python
print(conv.weight.requires_grad)
```

依然：

```text
True
```

但：

```python
with torch.no_grad():
    y = conv(x)
```

得到：

```text
y.requires_grad = False
y.grad_fn = None
```

---

# 6. 画成一个图最好理解

正常情况：

```python
conv.weight
requires_grad=True
       │
       │
x ─── Conv ───→ y
                │
                ├─ requires_grad=True
                └─ grad_fn=ConvolutionBackward0
```

因为卷积操作被记录了。

---

如果这样：

```python
with torch.no_grad():
    y = conv(x)
```

那么：

```text
conv.weight
requires_grad=True
       │
       │
x ─── Conv ───→ y
     no_grad    │
                ├─ requires_grad=False
                └─ grad_fn=None
```

注意：

> `conv.weight` 本身仍然是可训练参数。

只是：

> **这一次 forward 没有留下计算图。**

---

# 7. 那是不是“其实记录了，只是没显示”？

不是。

这是你问题里最关键的一点。

如果：

```python
with torch.no_grad():
    y = conv(x)
```

那么并不是：

> 记录了，但不显示。

而是真的：

> **没有建立用于 autograd 反向传播的计算图。**

所以：

```python
y.grad_fn
```

是：

```text
None
```

并且你没法通过这个 `y` 反向传播到 `conv.weight`。

---

# 8. 可以自己做一个非常好的实验

建议你直接追加：

```python
print("\n===== 实验1：正常 forward =====")

y1 = conv(x)

print("conv.weight.requires_grad =", conv.weight.requires_grad)
print("y1.requires_grad =", y1.requires_grad)
print("y1.grad_fn =", y1.grad_fn)


print("\n===== 实验2：no_grad forward =====")

with torch.no_grad():
    y2 = conv(x)

print("conv.weight.requires_grad =", conv.weight.requires_grad)
print("y2.requires_grad =", y2.requires_grad)
print("y2.grad_fn =", y2.grad_fn)
```

你会看到类似：

```text
===== 实验1 =====

conv.weight.requires_grad = True
y1.requires_grad = True
y1.grad_fn = <ConvolutionBackward0 ...>
```

然后：

```text
===== 实验2 =====

conv.weight.requires_grad = True
y2.requires_grad = False
y2.grad_fn = None
```

这个实验非常值得你亲自跑。

---

# 9. 再进一步：`requires_grad=False` 又是什么？

如果你主动：

```python
conv.weight.requires_grad_(False)
```

这才是真的把权重冻结。

然后：

```python
print(conv.weight.requires_grad)
```

变成：

```text
False
```

这和：

```python
torch.no_grad()
```

完全不同。

---

## 两者对比

| 操作                             | 修改参数 `requires_grad`？ | 当前运算记录计算图？ | 常见用途 |
| ------------------------------ | --------------------: | ---------: | ---- |
| 正常 forward                     |                     否 |          是 | 训练   |
| `torch.no_grad()`              |                     否 |          否 | 推理   |
| `weight.requires_grad_(False)` |                     是 |    视其他输入而定 | 冻结参数 |

这张表建议记进笔记。

---

# 10. 为什么训练时不能把 forward 放进 `no_grad()`？

因为训练需要：

```text
forward
↓
prediction
↓
loss
↓
backward
↓
gradient
↓
update weight
```

如果你：

```python
with torch.no_grad():
    y = conv(x)
```

那么：

```text
Conv
```

这一步没有被记录。

以后：

```python
loss.backward()
```

就没有这条路径：

```text
loss
→ y
→ conv
→ weight
```

所以无法通过这个 forward 给卷积核算梯度。

---

# 11. 为什么推理反而要用 `no_grad()`？

推理时：

```text
图片
↓
模型
↓
结果
```

不需要：

```text
loss.backward()
```

所以没必要保存：

```text
每一步中间值
计算图
反向传播信息
```

因此：

```python
with torch.no_grad():
```

可以：

* 节省显存
* 减少内存占用
* 提高一些运行效率

现在 PyTorch 推理还经常看到：

```python
with torch.inference_mode():
```

概念上也属于：

> 我现在只推理，不反向传播。

---

# 12. 最后把你现在三个东西串起来

### `conv.weight`

模型参数：

```text
shape = [1,1,3,3]
requires_grad = True
```

---

### `conv.weight[:]`

意思：

```text
选择 weight 中全部元素
```

所以：

```python
conv.weight[:] = new_weight
```

就是：

> 原地修改整个卷积核。

---

### `torch.no_grad()`

不是：

> 把参数变成 `requires_grad=False`

而是：

> **在当前代码块里暂停 autograd 对运算的记录。**

所以：

```python
with torch.no_grad():
    conv.weight[:] = ...
```

只是安全地人工设置权重。

而：

```python
y = conv(x)
```

在外面，所以这次卷积正常记录：

```text
y.requires_grad=True

y.grad_fn=ConvolutionBackward0
```

如果写成：

```python
with torch.no_grad():
    conv.weight[:] = ...
    y = conv(x)
```

那么：

```text
conv.weight.requires_grad
仍然 = True

但是：

y.requires_grad
= False

y.grad_fn
= None
```

所以你的理解可以最终压缩成一句：

> **`requires_grad` 决定“这个 Tensor 是否需要梯度”；`torch.no_grad()` 决定“当前这段运算是否被 autograd 记录”。**

---

好的，抛开枯燥的定义，我们直接进入**真实项目**的 3 个核心场景。看完这些场景，你不仅分得清，甚至以后写代码时会下意识地知道“这里该用谁”。

---

### 场景一：模型验证 / 测试（推理阶段）—— 用 `torch.no_grad()`

**你在干嘛**：模型训练完一个 Epoch（一轮），你想看看它在“测试集”上表现怎么样。此时你**绝对不想**更新模型参数，只想算个准确率。

- **如果不用 `no_grad()`**：PyTorch 依然会兢兢业业地记录所有中间计算步骤（计算图），显存占用极大，且前向传播速度变慢。
- **应用代码**：
  ```python
  model.eval()  # 切换到评估模式
  with torch.no_grad():  # 临时关门，别记账！
      for images, labels in test_loader:
          outputs = model(images)  # 只算结果
          loss = criterion(outputs, labels)  # 算损失（但不反向传播）
          # 计算准确率...
  ```
- **此时 `requires_grad` 的状态**：模型权重（如 `conv.weight`）的 `requires_grad` 依然是 `True`（因为它们本来就是可训练的参数），但外层的 `no_grad()` 强行让这次前向传播**不生成计算图**。这就像员工挂着工牌（True），但财务室今天不开门（no_grad），不记账。

---

### 场景二：迁移学习 / 冻结预训练模型 —— 用 `requires_grad=False`

**你在干嘛**：你要做“猫狗分类”，手里没有多少数据，于是下载了一个在 1000 类 ImageNet 上训练好的 ResNet。你想**冻住**前面所有的卷积层（它们已经会提取通用特征了），**只训练**最后一层全连接层（让它学会区分猫和狗）。

- **应用代码**：
  ```python
  # 加载预训练模型
  model = torchvision.models.resnet18(pretrained=True)
  
  # 冻结所有卷积层
  for param in model.parameters():
      param.requires_grad = False  # 永久摘掉它们的“训练工牌”
  
  # 只把最后的全连接层替换成可训练的
  model.fc = nn.Linear(512, 2)  # 新层的 requires_grad 默认是 True
  ```
- **此时 `no_grad()` 在哪？** 不需要！在反向传播（`loss.backward()`）时，PyTorch 看到那些层的 `requires_grad=False`，**直接跳过它们**，不计算它们的梯度。优化器（Optimizer）在更新参数时，也只会更新 `requires_grad=True` 的那一层。
- **本质区别**：`requires_grad=False` 是把人的“工牌”没收了，从此公司永远不给它算工资（梯度）；`no_grad()` 是工资照常算（工牌还在），但这次公司放假，不结算。

---

### 场景三：手动修改权重 / 初始化（你昨天踩的坑）—— 用 `torch.no_grad()`

**你在干嘛**：你想做个实验，把卷积核强行改成全 1 矩阵，验证手算结果对不对（就是你昨天写的代码）。此时 `conv.weight` 是个模型参数，默认 `requires_grad=True`。

- **应用代码**：
  ```python
  # 此时 conv.weight.requires_grad == True
  with torch.no_grad():
      conv.weight[:] = torch.tensor([[[[1., 1., 1.], ...]]])
  ```
- **为什么不用 `requires_grad=False`？** 因为你只是**暂时**修改一下数值，改了之后**依然要**继续训练这个模型。如果你把 `requires_grad` 改成 `False`，那后续训练时这个卷积核就永远不更新了，这不是你要的。
- **本质区别**：`requires_grad` 是你和 PyTorch 签的“长期合同”（我要训练这个参数）。`no_grad()` 是允许你在这个长期合同下，偶尔有一次“暗箱操作”，不让 PyTorch 把这次赋值记录在案，防止报错。

---

### 终极对比表（珍藏版）

| 现实场景 | 用什么？ | 为什么？ |
| :--- | :--- | :--- |
| **模型推理（Validation/Test）** | `with torch.no_grad():` | 省显存、加速。虽然参数要学，但这次别记计算图。 |
| **冻结预训练主干网络** | 设置 `param.requires_grad=False` | 参数不想学了。优化器直接忽略它，反向传播不经过它。 |
| **只训练新增的全连接层** | 旧层 `False`，新层 `True` | 旧层工牌没收，新层工牌挂上。 |
| **调试时强行修改权重数值** | `with torch.no_grad():` | 参数还想学（保持 True），但这次修改不让 Autograd 监控。 |
| **打印/保存模型参数** | `with torch.no_grad():` | 查看数值时不需要梯度，省去不必要的计算开销。 |

---

### 一句话帮你刻进 DNA

- **看到 `requires_grad`**，想的是 **“这个参数在训练过程中该不该被优化器更新？”**（长期身份）
- **看到 `no_grad()`**，想的是 **“这次前向计算我需不需要反向传播？”**（短期行动）

以后写代码时，只要脑子里闪过“这次算完我要不要 `backward()`”？——如果要，别用 `no_grad()`；如果不要（比如只在验证集上跑一下），立刻把代码包进 `with torch.no_grad():` 里。而 `requires_grad`，只在定义优化器或冻结网络时去设置它。这样你就永远不会用混了！

---

**为什么第二个值是 72，以及卷积核是怎样从左往右滑动形成整张 3×3 feature map 的**。
由于步长为1, 且卷积核没有到最后一个位置, 因此滑动窗口向右平移一个单位, 每个数都加一, 相乘后求和应该加 9, 因此是 63+9=72, 又因为 float32, 所以输出的应该是 72.
步长为1, 卷积核没有到最后一个位置, 因此滑动窗口再向右平移一个单位, 现在是 72+9=81, 所以输出的应该是 81.
这次卷积核到达最后一个位置, 因此滑动窗口从头开始, 向下平移一个单位, 每个数都加五, 相乘后求和应该加 9*5=45, 因此是 63+45=108,所以输出的应该是 108.
依此类推,最后得到
[63.,72.,81.],
[108.,117.,126.],
[153.,162.,171.]
这样一个 shape 为 [1,1,3,3] 的前向传播结果

---

# 六、你必须手算第一个输出值

输入左上角 3×3：

```text
1  2  3
6  7  8
11 12 13
```

卷积核：

```text
1 1 1
1 1 1
1 1 1
```

对应元素相乘再求和：

```text
1×1 + 2×1 + 3×1
+ 6×1 + 7×1 + 8×1
+ 11×1 + 12×1 + 13×1
```

结果：

```text
63
```

所以输出特征图左上角应该是：

```text
63
```

这一步非常关键。

你要真正理解：

> **卷积输出的一个数字，是输入局部区域和卷积核做“对应元素相乘 + 求和”得到的。**

---
这个问题问得特别好！你贴出的这段代码和注释，其实就是**深度学习最核心的底层数学运算——卷积（Convolution）**。

我用“人话”给你彻底拆解，保证你以后看到卷积就知道它在干嘛。

### 1. 一句话概括：它在算什么？
**它在计算“输入图片左上角这个 3×3 小方块”和“我们定义的 3×3 卷积核”之间的“相似度加权和”。**

因为我们的卷积核全是 `1`，所以计算简化成了：**把这个 3×3 小方块里的 9 个数字全部加起来**。结果 `63` 就是这个 3×3 区域的“像素强度总和”。

---

### 2. 分步拆解：计算机具体是怎么算出 `63` 的？

计算机不是“一眼”看出左上角的，它严格按照以下 **3 个步骤** 操作：

**第一步：对齐（覆盖）**
卷积核（3×3）像“盖章”一样，盖在输入图片（5×5）的最左上角。此时，核上的 9 个格子，正好对应输入图片左上角的 9 个格子。

```text
输入图片区域 (3x3)         卷积核 (3x3)
[1,  2,  3]                [1, 1, 1]
[6,  7,  8]    对应相乘    [1, 1, 1]
[11, 12, 13]               [1, 1, 1]
```

**第二步：对应元素相乘（按位乘，不是矩阵乘）**
把位置相同的两个数字相乘：

- 1×1 = 1
- 2×1 = 2
- 3×1 = 3
- 6×1 = 6
- ... 以此类推。
得到 9 个临时的结果。

**第三步：全部加起来（求和）**
把这 9 个相乘的结果全部加在一起：

```text
1 + 2 + 3 + 6 + 7 + 8 + 11 + 12 + 13 = 63
```

**最终**：计算机把 `63` 这个数字，填在输出特征图（`y`）的左上角（`[0,0]` 位置）。

---

### 3. 那输出结果 `y` 里的其他数字（72, 81 等）是哪来的？
卷积核不会只算左上角。因为它设置了 `stride=1`（步长为1），算完左上角后，这个“3×3的章”会**往右平移一格**。

- **算第二个数**：核盖住 `(2,3,4 / 7,8,9 / 12,13,14)`，全乘1再相加 = `2+3+4+7+8+9+12+13+14 = **72**`。
- **算第三个数**：再右移，盖住 `(3,4,5 / 8,9,10 / 13,14,15)`，相加 = `3+4+5+8+9+10+13+14+15 = **81**`。

第一行算完（3个数），核回到最左边，**往下平移一格**，开始算第二行（`6,7,8 / 11,12,13 / 16,17,18` 得到 `108`...）。
因为输入是 5×5，核是 3×3，步长 1，所以输出是 (5-3+1) × (5-3+1) = **3×3**。这就是你最后 `y.shape` 是 `[1,1,3,3]` 的原因。

---

### 4. 加入你未来的视角（这到底有什么用？）

- **此时此刻（全1核）**：它就是个“滑动求和器”，用来统计局部区域的亮度总和。
- **未来真正训练时（随机核）**：里面的数字不再是全 1，而是 `0.23, -0.56, 0.11...` 这种**可学习的参数**。这时，卷积核就像一个“特征探测器”。
  - 如果某个 3×3 区域里有一条“斜线”，与核里的“斜线”权重相乘求和后，会得到一个很大的数字；
  - 如果没有斜线，求和结果接近 0。
  
**所以，这段代码本质是在告诉你：卷积神经网络（CNN）提取特征，靠的就是“拿着一个小的数字模板（卷积核），在整张图上到处滑动，每滑到一个地方，就把覆盖住的数字对应相乘再求和，得到一个响应值”。** 

记住你手算的这个 `63`，它就是深度学习大厦最底层的“砖块”。后面的池化、激活函数，全都是基于这个 `63` 继续操作的。

---

# 七、卷积到底在“滑动”什么？

3×3 kernel 会在 5×5 输入上滑动。

第一次：

```text
[1  2  3]
[6  7  8]
[11 12 13]
```

第二次往右移动：

```text
[2  3  4]
[7  8  9]
[12 13 14]
```

再往右：

```text
[3  4  5]
[8  9 10]
[13 14 15]
```

然后下一行。

所以：

```text
5×5 输入
3×3 kernel
stride=1
padding=0
```

最终得到：

```text
3×3 输出
```

---

这四个参数，就是决定**“卷积核（小窗口）如何在输入图片上滑动”**的四大规则。你把它们想象成**“盖章”**或者**“滑动探照灯”**，就特别好懂了。

我结合你刚才算出的 **5×5 输入 → 3×3 输出**，把每个词彻底讲透：

---

### 1. 5×5 输入（Input）
- **意思**：原始的图片（或特征图）的高（H）和宽（W）都是 5。也就是你代码里那个 5 行 5 列的矩阵。
- **作用**：它是被“扫描”的对象。卷积核要在它上面到处游走。

### 2. 3×3 核（Kernel / Filter）
- **意思**：卷积核的尺寸是 3 行 3 列。它就像一个小探照灯，一次只能看到输入图片上很小的一个局部（9 个格子）。
- **作用**：卷积核在哪个位置，它就只计算那 9 个格子（对应元素相乘求和）。算完后，这个 3×3 的窗口必须**移动**，去覆盖下一个 9 个格子。

### 3. 步长 Stride = 1（核心）
- **意思**：窗口**每次平移几个格子**。
- **具体到你这里**：Stride=1，意味着窗口算完左上角 `(1,2,3 / 6,7,8 / 11,12,13)` 后，**向右平移 1 个格子**，去盖住 `(2,3,4 / 7,8,9 / 12,13,14)`。
- **直观感受**：如果 Stride=2，窗口算完左上角后，会**直接跳 2 个格子**，跳到 `(3,4,5 / 8,9,10 / 13,14,15)`，中间那一块就被跳过了。

### 4. 填充 Padding = 0（关键）
- **意思**：在进行卷积计算**之前**，在输入图片的**最外圈补上几圈 0**。
- **具体到你这里**：Padding=0，表示**不加一圈 0**，直接用原始的 5×5 进行计算。

---

### 为什么 5×5 输入 + 3×3 核 + Stride=1 + Padding=0，输出是 3×3？

这其实是“小学数学”的排队问题。你可以把卷积核想象成你的“脚”，踩在格子上：

- **第一步**：核的左上角对准输入图片的左上角（位置 0）。
- **第二步**：因为 Stride=1，核一步一步往右挪，直到核的**最右边**碰到输入图片的**最右边**。
- 我们来数一数核能放几个位置：
  - **水平方向**：输入宽 5，核宽 3。步长 1 的情况下，核可以放在第 **0、1、2** 列（从0开始数）。
    - 放 0 列时，覆盖 1,2,3。
    - 放 1 列时，覆盖 2,3,4。
    - 放 2 列时，覆盖 3,4,5。
    - 再往右放 3 列，就覆盖 4,5,6 了，但 6 超出了图片，所以不允许。
  - 所以，水平方向只能放 **3 个位置**。

- **同理**：垂直方向也只能放 **3 个位置**。

**最终**：横向 3 个位置 × 纵向 3 个位置 = 一个 **3×3** 的输出特征图。

---

### 记住这个万能公式（计算输出尺寸）
以后不用数格子，直接用这个公式：

\[
\text{输出尺寸} = \frac{\text{输入尺寸} - \text{核尺寸} + 2 \times \text{Padding}}{\text{Stride}} + 1
\]

把你今天的数字代进去：
\[
\text{输出} = \frac{5 - 3 + 2 \times 0}{1} + 1 = 2 + 1 = 3
\]
**算出来就是 3！**

---

### 一句话总结今天的“四大金刚”：
- **5×5** 是地图；
- **3×3** 是放大镜的大小；
- **Stride=1** 是放大镜每次挪动 1 厘米；
- **Padding=0** 是地图边缘没有额外的白边，放大镜挪到边缘正好悬空，再往外就掉下去了，所以只能放 3×3 个位置。

输出尺寸变小（5 变成了 3），就是因为**没有 Padding（不补边）**，导致边缘的信息被“吞掉”了。后面你会学到，为了不让图片越卷越小，我们通常会设置 `padding=1` 来保持尺寸不变。

---

# 八、输出尺寸公式

今天必须记住：

[
H_{out}
=======

\left\lfloor
\frac{H_{in}+2P-K}{S}
\right\rfloor
+1
]

宽度同理：

[
W_{out}
=======

\left\lfloor
\frac{W_{in}+2P-K}{S}
\right\rfloor
+1
]

其中：

```text
H_in = 输入高度
W_in = 输入宽度
K = kernel_size
P = padding
S = stride
```

例如：

```text
H_in = 5
K = 3
P = 0
S = 1
```

所以：

```text
H_out
= (5 - 3) / 1 + 1
= 3
```

输出：

```text
3×3
```

---

# 九、然后做 3 个实验

这是 Day 2 最重要的实践部分。

## 实验 1：`padding=0`

```python
Conv2d(
    1,
    1,
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

你要理解：

> 边缘没法让 3×3 kernel 完整覆盖，所以尺寸变小。

---

## 实验 2：`padding=1`

改成：

```python
padding=1
```

输入：

```text
5×5
```

输出：

```text
5×5
```

为什么？

因为原图外围补了一圈：

```text
0 0 0 0 0 0 0
0 1 2 3 4 5 0
0 6 7 8 9 10 0
...
0 0 0 0 0 0 0
```

这样 kernel 就可以覆盖边缘。

---

## 实验 3：`stride=2`

```python
stride=2
```

含义：

> 每次不是移动 1 格，而是移动 2 格。

所以：

```text
kernel
↓
跳着滑
```

输出尺寸会进一步变小。

---

三个实验揭示了卷积神经网络里**“尺寸变化”的全部底层逻辑**。把卷积核想象成一个**“只能完整落地的方框”**。

### 核心前提（铁律）
**卷积核必须完完全全地待在输入图片内部，不能有一丝一毫悬空在外面。**（除非用 Padding 补边，那是人为把外面扩大了）。

---

#### 实验 1：`padding=0`，为什么 `5×5` 变成 `3×3`？

- **原因**：方框（3×3）要在原始地图（5×5）里落地。
- **落地范围**：方框的左上角（锚点）能放在哪里？
  - 最左上角：放在 (0, 0) 位置，框住 1,2,3 / 6,7,8 / 11,12,13。
  - 往右挪：放在 (1, 0)，框住 2,3,4 / 7,8,9 / 12,13,14。
  - 再往右挪：放在 (2, 0)，框住 3,4,5 / 8,9,10 / 13,14,15。
  - **再往右挪到 (3, 0)**：框的右边会伸到第 6 列去，但地图只有第 5 列，悬空了！**违规！**
- **结论**：横着只能放 **3** 个位置（索引 0,1,2）。竖着同理也只能放 3 个。所以输出是 **3×3**。
- **本质**：边缘像素（1,2,3...）只被框住了一两次，而中心像素（7,8,13）被框住的次数多，**边缘信息被“挤压”和“丢失”了**，所以尺寸变小。

---

#### 实验 2：`padding=1`，为什么 `5×5` 还是 `5×5`？

- **操作**：在执行卷积**之前**，系统偷偷在原始地图的**最外围贴了一圈“假像素”**（数值为 0）。
- **新地图**：原本 5×5，上下左右各贴一圈，变成了 **7×7** 的人工地图。
- **落地规则**：3×3 的方框现在要在 **7×7** 的地图里落地。
  - 横着数一数，在 7 的长度里，3 的方框能放几个位置？答案是 **5** 个（索引 0,1,2,3,4）。
- **结论**：因为补了边，方框的**正中心**现在可以盖到原来地图的**真正边缘**（比如原图的左上角 1）。输出尺寸变成了 **5×5**。
- **本质**：Padding 不是为了让图片变大，而是**为了让卷积核的“中心”能够到达边缘像素**，从而保留边缘信息，让尺寸不缩减。

---

#### 实验 3：`stride=2`，为什么尺寸会进一步变小？

- **改变**：Stride 决定了方框**每次跨几步**。
  - Stride=1：方框落地位置是 0, 1, 2, 3...（一步一步走）。
  - Stride=2：方框落地位置是 **0, 2, 4...**（跳着走，隔一个落一次脚）。
- **场景**：假设没有 Padding（5×5 原始地图）。
  - 第一个落脚点：索引 0（框住 1,2,3）。
  - 隔一个格，第二个落脚点：索引 **2**（框住 3,4,5）。
  - 再隔一个格，索引 4 还能落脚吗？方框右边会伸出去，**违规！**
- **结论**：横着只能落 **3** 次脚（索引 0 和 2，以及另一个维度类似），所以输出尺寸变小。如果 Stride=3，只能落地 1 次（索引 0），输出变成 1×1。
- **本质**：Stride 相当于**“降采样”**，跳过的位置直接不参与计算，输出网格自然就稀疏了。

---

### 终极结论（一个顺口溜帮你记住）

- **Padding（补边）**：负责**“保尺寸”**。补得越多，外围越大，核能落脚的地方越多，输出就越大。
- **Stride（步长）**：负责**“降密度”**。步子越大，落脚点越少，输出就越小。

用公式验证一下你的三个实验（输出尺寸 = (输入 - 核 + 2×Padding) / Stride + 1）：

- **实验 1**：(5 - 3 + 0) / 1 + 1 = **3** ✅
- **实验 2**：(5 - 3 + 2) / 1 + 1 = **5** ✅
- **实验 3**（假设 padding=0）：(5 - 3 + 0) / 2 + 1 = 2 / 2 + 1 = 1 + 1 = **2**。（5变2，确实进一步变小了）✅

你亲手推完这三个实验，CNN 的底层几何逻辑你就彻底焊死在脑子里了！

---

# 十、Day 2 必须真正理解 kernel_size

比如：

```python
kernel_size=3
```

意思不是：

> 只看 3 个数字。

而是：

```text
在 H/W 空间上看 3×3 的局部区域。
```

如果输入有 3 个 channel：

```text
RGB
```

那么单个 filter 实际上是：

```text
3 × 3 × 3
```

可以理解成：

```text
R 通道：3×3
G 通道：3×3
B 通道：3×3
```

这三个通道一起参与计算。

---

# 十一、Day 2 第二个核心实验：3 输入通道

当你把单通道卷积搞懂以后，再做：

```python
x = torch.randn(1, 3, 5, 5)

conv = torch.nn.Conv2d(
    in_channels=3,
    out_channels=1,
    kernel_size=3,
    bias=False
)
```

此时：

```python
print(conv.weight.shape)
```

应该是：

```text
[1, 3, 3, 3]
```

也就是说：

```text
一个 filter
=
3 个输入通道
×
3×3 空间窗口
```

---

# 十二、一个输出值怎么从 RGB 三通道产生？

假设输入局部区域是：

```text
R:
3×3

G:
3×3

B:
3×3
```

filter 也有：

```text
R 权重:
3×3

G 权重:
3×3

B 权重:
3×3
```

分别计算：

```text
R 局部区域 × R 权重
+
G 局部区域 × G 权重
+
B 局部区域 × B 权重
```

然后把所有结果相加：

```text
→ 一个数字
```

所以：

> 一个 filter 在一个空间位置最终只输出一个数。

---

# 十三、那 64 个输出通道呢？

如果：

```python
Conv2d(
    in_channels=3,
    out_channels=64,
    kernel_size=3
)
```

实际上有：

```text
64 个不同的 filter
```

每个 filter：

```text
[3, 3, 3]
```

每一个都会产生一张 feature map。

所以：

```text
64 个 filter
→ 64 张 feature map
→ 64 个输出 channel
```

这就和 Day 1 连起来了。

---

# 十四、今天要理解 bias

例如：

```python
Conv2d(
    3,
    64,
    kernel_size=3,
    bias=True
)
```

每一个输出通道还有一个 bias：

```text
64 个输出通道
→ 64 个 bias
```

完整可以理解成：

```text
输出值
=
局部输入 × 权重
全部求和
+
bias
```

所以：

```python
conv.bias.shape
```

一般是：

```text
[64]
```

Day 2 先理解概念即可。

---

# 十五、为什么卷积能“提取特征”？

这是非常核心的问题。

假设一个卷积核学成：

```text
-1  0  1
-1  0  1
-1  0  1
```

它对：

```text
左边暗
右边亮
```

这种变化可能响应较强。

所以它可以对：

```text
垂直边缘
```

敏感。

另外一个 kernel 可能学习：

```text
水平边缘
```

另一个：

```text
纹理
```

另一个：

```text
颜色组合
```

所以卷积层本质上是在：

> **使用大量可学习的局部模板扫描整张图。**

注意：

这些 kernel 一开始通常是随机的。

训练时通过：

```text
Loss
↓
反向传播
↓
更新 kernel 权重
```

逐渐学出对任务有用的模式。

---

# 十六、今天代码建议拆成 5 个实验区

你的 `01_conv2d_basic.py` 可以按这个结构写：

```text
1. 创建 5×5 单通道输入
2. 创建 3×3 卷积核
3. 手工设置 weight
4. 验证一次卷积结果
5. 比较 kernel / stride / padding
```

然后追加：

```text
6. 三输入通道实验
7. 64 输出通道实验
8. 查看 conv.weight.shape
9. 查看 conv.bias.shape
```

---

# 十七、Day 2 今天不要深入的东西

先不要碰：

```text
反向传播推导
梯度
BatchNorm
ReLU
Pooling
dilation
groups
depthwise convolution
ResNet
FPN
```

这些之后再加。

今天一定把：

```text
普通卷积
```

吃透。

---

# 十八、Day 2 练习题

公式：H_out = floor((H_in + 2*padding - dilation*(kernel_size-1) - 1) / stride + 1)
（为简单起见，假设膨胀为1，使用简化公式：(H_in - kernel_size + 2*padding) / stride + 1）。

假设：

```python
x = torch.randn(1, 3, 32, 32)
```

### 题 1

```python
conv = nn.Conv2d(
    3,
    16,
    kernel_size=3,
    stride=1,
    padding=0
)
```
TensorFlow/Keras 里的卷积权重顺序通常是 [kH, kW, in_c, out_c]，跟 PyTorch 正好反过来。
所以以后你如果看不同框架的代码，记得先看一眼文档。

问：

```text
conv.weight.shape = ?
[out_c, in_c, kH, kW]
[16, 3, 3, 3]

输出 shape = ?
[batch, out_c, H, W]
原x.shape是[1, 3, 32, 32] batch=1
输出尺寸 = (输入尺寸 - kernel_size + 2*padding)/stride+1 = (32-3+0)/1+1 = 30
[1, 16, 30, 30]
```

---

```python
x = torch.randn(1, 3, 32, 32)
```

### 题 2

```python
conv = nn.Conv2d(
    3,
    16,
    kernel_size=3,
    stride=1,
    padding=1
)
```

输出：

```text
conv.weight.shape = ?
[out_c, in_c, kH, kW]
[16, 3, 3, 3]

输出 shape = ?
[batch, out_c, H, W]
x.shape = [1, 3, 32, 32]
H = W = (in_H - kernel_size + 2*padding)/stride+1
= (32-3+2)/1+1 = 32
[1, 16, 32, 32]
```

规律（这在设计网络时天天用）：

> **当 `kernel_size=3` 且 `padding=1`，配合 `stride=1` 时，输入和输出的 H/W 尺寸保持不变。**

- 输入 32 → 输出 32。
- 输入 224 → 输出 224。
- 输入 512 → 输出 512。

这叫作 **"Same Padding"（保持尺寸不变）**。因为 3×3 的核，如果不补边会缩小 2 个像素（32→30），而在外围补上 1 圈 0，正好把缩小的部分抵消掉，尺寸就保住了。

---

```python
x = torch.randn(1, 3, 32, 32)
```

### 题 3

```python
conv = nn.Conv2d(
    3,
    16,
    kernel_size=3,
    stride=2,
    padding=1
)
```

输出：

```text
conv.weight.shape = ?
[16, 3, 3, 3]
输出 shape = ?
out_H = floor((32-3+2*1)/2)+1 = 16
[1, 16, 16, 16]
```

---

### 1. `conv.weight.shape = [16, 3, 3, 3]` ✅
没毛病，16 个核，每个核厚度 3，平面 3×3。

### 2. 输出 `[1, 16, 16, 16]` ✅
你的计算过程完全严谨：

\[
\text{out\_H} = \left\lfloor \frac{32 - 3 + 2 \times 1}{2} \right\rfloor + 1
= \left\lfloor \frac{31}{2} \right\rfloor + 1
= 15 + 1 = 16
\]

这里你**特别注意到了 `floor`（向下取整）**，这是一个非常容易踩坑的细节，你主动用上了，说明你对公式的理解已经非常扎实。

---

### 这个配置在实际工程中意味着什么？（恭喜你摸到了“下采样”的门槛）

`stride=2` 配合 `padding=1`，再加上 `kernel_size=3`，这是深度学习网络中最经典、最常用的**“特征图减半”组合**。

- **空间尺寸**：32 × 32 → 16 × 16（高宽直接减半，计算量骤降为原来的 1/4）。
- **通道数**：3 → 16（特征深度增加了）。
- **实际应用**：像 ResNet（残差网络）这类经典架构，就是靠这种组合来逐层缩小特征图尺寸，同时增加通道数，从而提取更高层次的语义特征。

---

### 一个小细节（PyTorch 的默认行为）
你这里用了 `floor`（向下取整）是完全正确的。PyTorch 的 `Conv2d` 在输出尺寸不是整数时，默认采用 **向下取整（floor）**，**而不是**四舍五入。

- 比如这里 31/2 = 15.5，向下取整就是 15，再加 1 得 16。
- 如果算出来是 15.1，同样会被舍弃小数部分，只取 15。

---

### 题 4

```python
conv = nn.Conv2d(
    16,
    64,
    kernel_size=3
)
```

问：

```text
单个 filter 的 shape = ?
[c_in, kH, kW]
[16, 3, 3]

全部 weight 的 shape = ?
conv.weight.shape
[c_out, c_in, kH, kW]
[64, 16, 3, 3]
```

**完全正确！给你满分！** ✅

这道题你不仅答对了，而且逻辑链条非常清晰，说明你已经完全吃透了 `[out_c, in_c, kH, kW]` 的排列逻辑。

---

### 1. 单个 filter 的 shape = `[16, 3, 3]` ✅
- 因为这一层的输入通道数 `in_channels=16`，所以**每一个**卷积核的“厚度”必须是 16。
- 空间尺寸 `kernel_size=3`，所以高和宽都是 3。
- **物理意义**：一个滤波器此时已经不是“平面小方块”了，而是一个 `16×3×3` 的**立体小魔方**。它在 16 层（深度）上同时滑动。

### 2. 全部 weight 的 shape = `[64, 16, 3, 3]` ✅
- 这一层有 `out_channels=64` 个这样的“立体小魔方”。
- 把 64 个小魔方摞在一起，就是 `[64, 16, 3, 3]`。

---

### 悄悄补充一个“隐藏细节”（虽然你没问，但以后一定会遇到）
你在题目里写 `kernel_size=3`，没有写 `padding` 和 `stride`。  
在 PyTorch 中，如果你不写，它们有**默认值**：

- `stride` 默认是 **1**
- `padding` 默认是 **0**

所以，如果这道题要你顺便算输出形状（假设输入还是 32×32），默认情况下就是：
\[
\text{out} = (32 - 3 + 0)/1 + 1 = 30
\]
输出就是 `[1, 64, 30, 30]`。

不过既然题目只问了 weight shape，你给的答案已经无可挑剔。你已经掌握了根据 `in_channels` 和 `out_channels` 瞬间口算权重大小和单个核形状的能力，这在以后看预训练模型（如 ResNet50）的参数量时至关重要！继续保持这个节奏。

---

# 十九、Day 2 完成标准

今天结束前，你应该能不用看笔记回答：

* [ ] 卷积核是什么？
* [ ] `kernel_size=3` 是什么意思？
* [ ] 一个 filter 为什么必须覆盖所有输入 channel？
* [ ] 一个卷积位置是怎么计算出一个数字的？
* [ ] 为什么 64 个 filter 会得到 64 个 feature maps？
* [ ] `stride` 是什么？
* [ ] `padding` 是什么？
* [ ] 怎么计算输出的 H/W？
* [ ] `conv.weight.shape=[64,3,3,3]` 每个维度是什么？
* [ ] 为什么卷积可以提取局部特征？

如果这些都能讲清楚，Day 2 就通过。

---

## Day 2 和后续学习的关系

你现在的路线会是：

```text
Day 1
Tensor / BCHW / Channel
        ↓
Day 2
Conv2d 真正怎么计算
        ↓
Day 3
ReLU + Pooling + Simple CNN
        ↓
Day 4
Residual Block
        ↓
Day 5+
ResNet
```

**我建议 Day 2 至少亲手手算一次 5×5 输入和 3×3 卷积核的第一个输出值。**

如果连这个都能自己算出来，后面看 ResNet、FPN、Mask R-CNN 时，`Conv2d` 就不会再是黑盒了。
