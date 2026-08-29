Day 4 最适合进入：

# Day 4：Residual Connection + BasicBlock

因为你 Day 3 已经把这条链串起来了：

```text
Conv
↓
ReLU
↓
Pooling
↓
Flatten
↓
Linear
```

下一步就该解决一个新问题：

> **CNN 继续堆深以后，会发生什么？为什么 ResNet 要发明“残差连接”？**

这一步是你后面理解：

```text
ResNet
↓
C2 / C3 / C4 / C5
↓
FPN
↓
Mask R-CNN Backbone
```

的必经之路。

---

# Day 4 最终目标

你今天不需要把完整 ResNet50 写出来。

只需要真正搞懂：

```text
普通堆叠
x
↓
Conv
↓
ReLU
↓
Conv
↓
输出 F(x)
```

和：

```text
残差结构
x ───────────────┐
↓                │
Conv             │
↓                │
ReLU             │
↓                │
Conv             │
↓                │
F(x)             │
↓                │
+  ←───────────── x
↓
ReLU
```

核心公式：

$$
y = F(x) + x
$$

---

# Day 4 的核心问题

今天围绕 6 个问题学：

1. 为什么网络不能无限堆很多 Conv？
2. 什么叫 Residual / 残差？
3. 为什么是 `F(x) + x`？
4. `x` 为什么能直接绕过卷积？
5. 两个 Tensor 相加时 shape 必须满足什么？
6. 当 shape 不一样时，shortcut 怎么处理？

---

# 1. 第一部分：先理解“普通网络变深的问题”

你可以先画：

```text
x
↓
Conv
↓
ReLU
↓
Conv
↓
ReLU
↓
Conv
↓
...
```

直觉上似乎：

> 层数越多，能力越强。

但真实情况不是简单的“越深越好”。

训练很深的普通网络时，会遇到：

```text
梯度传播困难
优化困难
网络退化
```

你 Day 4 暂时不用深入梯度数学。

只先理解：

> **更深的网络不代表一定更容易训练。**

---

# 2. 第二部分：什么叫 Residual？

假设原本希望某几层直接学习：

$$
H(x)
$$

ResNet 换一个思路：

不直接逼这几层学：

$$
H(x)
$$

而是让它们学习：

$$
F(x)=H(x)-x
$$

于是：

$$
H(x)=F(x)+x
$$

这就是：

```text
Residual
=
残差
=
目标结果与原输入之间的差
```

---

# 3. 最重要的直觉

你可以把它理解成：

普通网络：

```text
请你从头生成一个完整的新结果 H(x)
```

ResNet：

```text
原来的 x 我先保留

你只需要学习：
“我应该在 x 的基础上改多少”

也就是 F(x)
```

所以：

```text
最终结果
=
原来的 x
+
需要修正的部分
```

即：

$$
y=x+F(x)
$$

---

# 4. 为什么这叫“shortcut”？

因为：

```text
x
```

没有经过中间复杂卷积，而是：

```text
直接绕过去
```

所以叫：

```text
shortcut connection
skip connection
```

你可以先把：

```text
Residual Connection
Skip Connection
Shortcut
```

视作高度相关的概念。

---

# 5. 第一个必须手写的小实验

建议新建：

```text
study/
└── 02_resnet/
    ├── 01_residual_basic.py
    └── 01_residual_basic.md
```

先不用 Conv。

只做：

```python
import torch

x = torch.tensor([
    [1., 2., 3.]
])

fx = torch.tensor([
    [0.1, -0.2, 0.5]
])

y = x + fx

print(y)
```

先手算：

```text
x:
[1, 2, 3]

F(x):
[0.1, -0.2, 0.5]

y:
?
```

应该得到：

```text
[1.1, 1.8, 3.5]
```

先把：

```text
y = x + F(x)
```

变成非常具体的数字。

---

# 6. Tensor 相加最重要的限制

假设：

```python
y = fx + x
```

那：

```text
fx.shape
```

和：

```text
x.shape
```

必须能进行逐元素相加。

最理想：

```text
x  = [B,64,32,32]
fx = [B,64,32,32]
```

那么：

```text
[B,64,32,32]
+
[B,64,32,32]

↓

[B,64,32,32]
```

所以你 Day 4 要开始形成一个新反射：

> **看到 residual add，先检查 shape 能不能对上。**

---

# 7. 手写最简单 ResidualBlock

今天可以先写：

```python
class BasicBlock(nn.Module):

    def __init__(self, channels):
        super().__init__()

        self.conv1 = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            padding=1,
            bias=False
        )

        self.relu = nn.ReLU()

        self.conv2 = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
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
```

---

# 8. 这段代码今天要逐行审

比如：

```python
identity = x
```

你必须知道：

```text
identity
不是新计算出来的特征

而是：
把原输入 x 保存下来
```

然后：

```python
out = self.conv1(x)
```

假设：

```text
x:
[B,64,32,32]
```

因为：

```python
Conv2d(64,64,3,padding=1)
```

输出：

```text
[B,64,32,32]
```

第二个 Conv 后仍然：

```text
[B,64,32,32]
```

于是：

```python
out = out + identity
```

就是：

```text
[B,64,32,32]
+
[B,64,32,32]
```

可以直接逐元素相加。

---

# 9. 今天最关键的 shape 链

你要能不运行直接推：

```text
输入：
[B,64,32,32]

↓

conv1
[B,64,32,32]

↓

ReLU
[B,64,32,32]

↓

conv2
[B,64,32,32]

↓

+ identity
[B,64,32,32]

↓

ReLU
[B,64,32,32]
```

Residual Block 在这种最简单情况下：

> **整体 shape 不变。**

---

# 10. 第二个问题：如果 shape 不一样怎么办？

这才是 Day 4 后半段重点。

例如：

```text
输入 x:
[B,64,32,32]
```

主分支：

```python
Conv2d(
    64,
    128,
    kernel_size=3,
    stride=2,
    padding=1
)
```

输出：
H_out = floor((H_in - kernel_size + 2*padding)/stride)+1 = floor((32-3+2)/2)+1 = 16

```text
[B,128,16,16]
```

但 shortcut 里的原始 x 还是：

```text
[B,64,32,32]
```

这时：

```text
[B,128,16,16]
+
[B,64,32,32]
```

显然不能直接加。

---

# 11. 解决办法：Projection Shortcut

使用：

```python
nn.Conv2d(
    64,
    128,
    kernel_size=1,
    stride=2
)
```
它的几何意义是：步长为 2 意味着卷积核每次跳过 1 个像素进行滑动（即只取输入特征图的第 0、2、4... 个位置）。因为 1×1 卷积核只覆盖一个像素，所以它本质上是**对输入进行“隔点采样”（下采样）**，恰好把 32×32 压缩成了 16×16，和主分支的尺寸完全一致。

把 shortcut：

```text
[B,64,32,32]
```

变成：
H_out = floor((H_in - kernel_size + 2*padding)/stride)+1 = floor((32-1+0)/2)+1 = 16

```text
[B,128,16,16]
```

于是：

```text
主分支：
[B,128,16,16]

shortcut：
[B,128,16,16]

↓

相加
```

这就是你以后在 ResNet 代码里经常看到的：

```text
downsample
projection
shortcut conv
1×1 conv
```

---

# 12. 为什么是 1×1 Conv？

这正好连接 Day 2 的知识。

```python
Conv2d(64,128,kernel_size=1,stride=2)
```

单个 filter：

```text
[64,1,1]
```

它可以：

```text
改变 C
64 → 128
```

同时：

```text
stride=2
```

又可以：

```text
H/W
32 → 16
```

所以：

```text
[B,64,32,32]

↓

1×1 Conv, stride=2

↓

[B,128,16,16]
```

正好把 shortcut 调整到与主分支一致。

---

# Day 4 推荐任务顺序

今天按这个顺序：

```text
任务 1
理解普通深层 CNN 为什么训练困难

↓

任务 2
理解 y = F(x) + x

↓

任务 3
手算一个纯 Tensor residual add

↓

任务 4
理解 identity / shortcut

↓

任务 5
手写 shape 不变的 BasicBlock

↓

任务 6
逐行审查 BasicBlock

↓

任务 7
检查 residual add 前两边 shape

↓

任务 8
学习 shape 不同时的 1×1 projection shortcut

↓

任务 9
手推：
[B,64,32,32]
→
[B,128,16,16]

↓

任务 10
最后自己画完整 BasicBlock 数据流
```

---

# Day 4 验收题

完成后你应该能不看笔记回答：

1. Residual Connection 的公式是什么？
2. `F(x)` 表示什么？
3. `identity = x` 为什么要保存？
4. 为什么 `out + identity` 前必须检查 shape？
5. `[B,64,32,32] + [B,64,32,32]` 输出什么 shape？
6. 为什么 `[B,128,16,16]` 不能直接和 `[B,64,32,32]` 相加？
7. 1×1 Conv 为什么能改变 channel？
8. `stride=2` 为什么能让 H/W 减半？
9. projection shortcut 是干什么的？
10. BasicBlock 的完整数据流是什么？

---

# Day 4 暂时不要深入

今天先不要直接碰：

```text
ResNet-50 Bottleneck
BatchNorm 数学细节
完整 ResNet
C2/C3/C4/C5
FPN
```

尤其不要现在直接抄完整 `torchvision.models.resnet50`。

先把：

```text
x
↓
F(x)
↓
F(x) + x
```

这件事彻底吃透。

---

## Day 4 结束后的衔接

如果 Day 4 通过，下一步就非常自然：

```text
Day 5
BasicBlock / Bottleneck 对比
↓
ResNet 整体结构

Day 6
真正追踪 ResNet-50
↓
C2 / C3 / C4 / C5

Day 7+
FPN
↓
C2-C5 → P2-P5
```

而这时你就开始真正进入 Mask R-CNN 的 Backbone 主干了。
