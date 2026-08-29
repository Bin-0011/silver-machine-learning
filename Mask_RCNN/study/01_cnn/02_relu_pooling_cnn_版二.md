# Day 3：ReLU + Pooling + 最小 CNN

> 学习目标：把 Day 1 的 Tensor / shape 与 Day 2 的 Conv2d 串起来，真正理解一个最小 CNN 的完整数据流：
>
> **Image → Conv → ReLU → Pool → Conv → ReLU → Pool → Flatten → Linear → Logits**

---

# 0. Day 3 总目标

今天不追求训练准确率，也不进入 ResNet。

Day 3 要完成的是：

```text
图片
[B, 3, H, W]

↓ Conv2d

局部特征
[B, C1, H1, W1]

↓ ReLU

加入非线性
shape 不变

↓ Pooling

空间尺寸缩小
[B, C1, H2, W2]

↓ 再一次 Conv + ReLU + Pool

更复杂特征
[B, C2, H3, W3]

↓ Flatten

[B, N]

↓ Linear

[B, num_classes]

↓ logits

每个类别一个原始分数
```

今天最重要的能力不是背 API，而是：

> **看到每一层时，能提前预测输入 shape、输出 shape、这一层在做什么，以及为什么下一层需要它。**

---

# 1. ReLU

## 1.1 基本形式

```python
import torch
import torch.nn as nn

x = torch.tensor([
    [-2., -1., 0., 1., 2.]
])

relu = nn.ReLU()

y = relu(x)

print("x:", x)
print("y:", y)
print("x.shape:", x.shape)
print("y.shape:", y.shape)
```

ReLU：

\[
\operatorname{ReLU}(x)=\max(0,x)
\]

也就是：

\[
\operatorname{ReLU}(x)=
\begin{cases}
0,& x<0\\
x,& x\ge 0
\end{cases}
\]

因此：

```text
输入：
[-2, -1, 0, 1, 2]

输出：
[ 0,  0, 0, 1, 2]
```

---

## 1.2 ReLU 会不会改变 shape？

一般不会。

```text
[B, C, H, W]

↓ ReLU

[B, C, H, W]
```

它只是对 Tensor 中每一个元素独立执行：

```text
x < 0  → 0
x ≥ 0  → x
```

所以数值会变，但维度通常不变。

---

# 2. 为什么 Conv 后面要加 ReLU？

这是 Day 3 的第一个核心问题。

Conv 的核心运算是：

```text
乘法
+
加法
```

可以抽象成：

\[
y=wx+b
\]

如果连续堆很多线性变换：

\[
y_1=w_1x+b_1
\]

\[
y_2=w_2y_1+b_2
\]

代入：

\[
y_2=w_2(w_1x+b_1)+b_2
\]

整理：

\[
y_2=(w_2w_1)x+(w_2b_1+b_2)
\]

最后仍然可以写成：

\[
y=ax+c
\]

所以：

```text
Linear
↓
Linear
↓
Linear
```

本质仍然可以压缩成：

```text
一个更大的 Linear
```

如果 CNN 中只有卷积而完全没有非线性激活，那么模型的表达能力会受到严重限制。

---

# 3. 什么是“非线性表达能力”？

“表达能力”可以理解为：

> 模型能够表示多复杂的“输入 → 输出”关系。

线性关系只能使用一套固定规则：

```text
输入变化
↓
输出按照固定线性关系变化
```

ReLU 加入以后：

```text
输入落在一个区域
→ 使用一种关系

输入进入另一个区域
→ 使用另一种关系
```

例如：

\[
z=wx+b
\]

经过 ReLU：

\[
y=\max(0,wx+b)
\]

就分成：

```text
wx+b < 0
→ 输出 0

wx+b ≥ 0
→ 输出 wx+b
```

因此它是一个**分段线性函数**。

虽然每一段仍是线性的，但整体已经不再是一个单一线性函数。

---

## 3.1 折纸类比

没有 ReLU：

```text
旋转
拉伸
压缩
平移
```

无论重复多少次，整体仍然属于线性/仿射变换。

ReLU 相当于允许网络：

> 在某些位置“折一下”。

于是：

```text
Linear
↓
ReLU
↓
Linear
↓
ReLU
↓
Linear
```

网络就可以形成越来越复杂的分段结构。

---

# 4. 什么叫“激活 / 不激活”？

假设某个卷积核得到：

```text
Conv 输出：

 2.3   -1.2   0.7
-0.5    3.1  -2.0
 1.4    0.2  -0.8
```

经过 ReLU：

```text
2.3   0   0.7
0    3.1  0
1.4  0.2  0
```

于是可以粗略说：

```text
> 0
→ 激活

≤ 0
→ 被 ReLU 压成 0
→ 不激活
```

这里的“激活”并不是网络真的在执行 if 判断，而是帮助理解：

> 某个 feature 在某个位置是否产生正响应。

---

# 5. 什么是“激活模式”？

“激活模式”不是一个数字，而是：

> **一张 feature map 上，哪些位置有正响应、哪些位置为 0 的空间分布。**

例如：

```text
0  3  0  0
0  5  0  0
0  4  0  0
0  2  0  0
```

看起来像一条竖向响应区域。

另一个 channel 可能是：

```text
0  0  0  0
4  5  3  2
0  0  0  0
0  0  0  0
```

像一条横向响应区域。

更准确地说：

```text
Conv
先产生不同位置的正负响应

↓

ReLU
把负值压为 0

↓

“哪里有响应 / 哪里没有响应”
更加明显
```

ReLU 并不是凭空制造特征，它只是把卷积产生的响应进一步非线性化。

---

# 6. 从“简单特征”到“高级语义”是怎么来的？

关键不是某一层突然理解了“眼睛”或“豆粒”。

真正发生的是：

> **后一层卷积会把前一层的多个 feature map 当作新的输入 channels，再组合这些已有特征。**

---

## 6.1 第一层看到的是 RGB

原始输入：

```text
[B, 3, H, W]
```

也就是：

```text
R
G
B
```

第一层卷积可能输出：

```text
[B, 64, H, W]
```

可以粗略理解成学习出了很多不同的响应模式：

```text
某些 channel 更响应竖边缘
某些 channel 更响应横边缘
某些 channel 更响应斜边缘
某些 channel 更响应纹理
某些 channel 更响应颜色变化
...
```

注意：

> 真实 CNN 中的 channel 不一定能被如此干净地命名，这只是帮助理解。

---

## 6.2 第二层看到的不是 RGB，而是第一层 feature maps

第二层输入：

```text
[B, 64, H, W]
```

第二层一个 filter 的 shape 可能是：

```text
[64, 3, 3]
```

它会同时读取前一层全部 64 个 channels。

因此它可以学习：

```text
竖边缘响应
+
横边缘响应
+
特定空间位置关系

↓

角点 / 局部轮廓
```

继续往后：

```text
边缘
↓
局部形状
↓
更复杂局部结构
↓
物体部件
↓
更完整的物体相关模式
```

---

## 6.3 感受野为什么越来越大？

第一层一个 `3×3` 卷积直接看原图的：

```text
3×3
```

第二层的 `3×3` 卷积看的是：

```text
第一层 feature map 的 3×3
```

而第一层的每一个位置本身已经来自原图一个局部区域。

因此层数越深，一个深层位置能间接利用的原图区域越大。

这就是：

> **感受野逐渐增大。**

所以可以形成：

```text
浅层
→ 边缘 / 纹理

中层
→ 局部轮廓 / 形状

更深层
→ 部件 / 更高级组合
```

---

# 7. Max Pooling

先学习：

```python
nn.MaxPool2d
```

示例：

```python
x = torch.tensor([
    [[[
        1., 2., 3., 4.,
    ],
       [
        5., 6., 7., 8.,
    ],
       [
        9., 10., 11., 12.,
    ],
       [
        13., 14., 15., 16.
    ]]]
])

pool = nn.MaxPool2d(
    kernel_size=2,
    stride=2
)

y = pool(x)
```

输入：

```text
1   2   3   4
5   6   7   8
9  10  11  12
13 14  15  16
```

输出：

```text
6   8
14 16
```

---

# 8. `kernel_size` 与 `stride`

## 8.1 `kernel_size=2`

表示：

> 每次池化窗口看多大的区域。

```python
kernel_size=2
```

等价于：

```python
kernel_size=(2, 2)
```

因此每次看：

```text
2 × 2
```

---

## 8.2 `stride=2`

表示：

> 池化窗口每次移动多少格。

因此：

```python
MaxPool2d(
    kernel_size=2,
    stride=2
)
```

可以理解成：

> 拿一个 `2×2` 的窗口，每次移动 2 格，在窗口中保留最大值。

---

# 9. Conv 和 Pool 的底层数学差异

## 9.1 Conv

卷积是：

```text
局部 patch
×
weight

↓

对应位置相乘

↓

全部求和

↓

+ bias
```

单通道情况下：

```text
9 个输入数字
+
9 个权重

↓

加权融合成 1 个新数字
```

多通道情况下：

```text
C_in × kH × kW
```

中的所有值都会参与一个输出值的计算。

所以 Conv 会：

> **跨 channel 融合信息。**

---

## 9.2 MaxPool

MaxPool 是：

```text
局部窗口

↓

比较大小

↓

留下最大值
```

它没有可学习 weight。

也不会把不同 channels 混合。

所以：

> **Pooling 是每个 channel 独立地在 H/W 平面做空间压缩。**

---

# 10. Pooling 为什么通常不改变 C？

假设输入：

```text
[B, 32, 64, 64]
```

意味着：

```text
32 张 64×64 的 feature map
```

MaxPool 会分别处理：

```text
Channel 1：
64×64 → 32×32

Channel 2：
64×64 → 32×32

...

Channel 32：
64×64 → 32×32
```

所以：

```text
[B, 32, 64, 64]

↓

MaxPool2d(2,2)

↓

[B, 32, 32, 32]
```

变化：

```text
H
W
```

不变：

```text
B
C
```

---

## 10.1 当前阶段可以记住

对于普通：

```python
nn.MaxPool2d
nn.AvgPool2d
nn.AdaptiveMaxPool2d
nn.AdaptiveAvgPool2d
```

都可以先理解成：

```text
[B, C, H, W]

↓

[B, C, H', W']
```

也就是：

> 普通 spatial pooling 不负责改变 channel 数。

---

## 10.2 一个更严谨的口诀

不要记成：

> Conv2d 一定改 C，Pooling 一定把 H/W 减半。

更准确的是：

```text
Conv2d：
C_out 由 out_channels 决定
H/W 由 kernel / stride / padding / dilation 决定

Pooling：
C 通常保持不变
H/W 由 kernel / stride / padding 等决定
```

---

# 11. 为什么要下采样？

假设 feature map 一直保持：

```text
64×64
↓
64×64
↓
64×64
↓
64×64
```

计算量和显存开销都会很大。

通过 Pooling 或 stride convolution：

```text
224
↓
112
↓
56
↓
28
↓
14
↓
7
```

空间尺寸逐渐减小。

与此同时，CNN 常把 channel 增加：

```text
3
↓
64
↓
128
↓
256
↓
512
```

可以粗略理解：

> **空间越来越粗，但每个位置的特征描述越来越丰富。**

这正是未来理解：

```text
ResNet C2 / C3 / C4 / C5
```

以及：

```text
FPN P2 / P3 / P4 / P5
```

的重要基础。

---

# 12. 第一组 Conv → ReLU → Pool

```python
x = torch.randn(1, 3, 32, 32)

conv = nn.Conv2d(
    in_channels=3,
    out_channels=8,
    kernel_size=3,
    stride=1,
    padding=1
)

relu = nn.ReLU()

pool = nn.MaxPool2d(
    kernel_size=2,
    stride=2
)
```

先推 shape：

```text
输入
[1,3,32,32]

↓ Conv2d(3,8,3,stride=1,padding=1)

[1,8,32,32]

↓ ReLU

[1,8,32,32]

↓ MaxPool2d(2,2)

[1,8,16,16]
```

卷积输出 H：

\[
H_{out}
=
\left\lfloor
\frac{H_{in}+2P-K}{S}
\right\rfloor
+1
\]

这里：

\[
\frac{32+2\times1-3}{1}+1=32
\]

所以空间尺寸保持：

```text
32×32
```

---

# 13. 第二组 Conv → ReLU → Pool

```python
conv2 = nn.Conv2d(
    8,
    16,
    kernel_size=3,
    stride=1,
    padding=1
)

relu2 = nn.ReLU()

pool2 = nn.MaxPool2d(2, 2)
```

shape：

```text
[1,8,16,16]

↓ Conv2d(8,16,3,padding=1)

[1,16,16,16]

↓ ReLU

[1,16,16,16]

↓ Pool

[1,16,8,8]
```

此时你已经得到：

```text
输入图片
↓
第一层局部特征
↓
第二层更复杂特征
```

---

# 14. Flatten

假设 CNN 最后输出：

```text
[B,16,8,8]
```

使用：

```python
x = torch.flatten(x, start_dim=1)
```

意思是：

> 从第 1 维开始展开，保留第 0 维 Batch。

所以：

```text
[B,16,8,8]

↓

[B,16×8×8]

↓

[B,1024]
```

例如：

```text
[4,16,8,8]

↓

[4,1024]
```

---

## 14.1 Flatten 不会打乱数据

Flatten 不是：

```text
随机打乱
```

而是：

```text
把多个维度重新整理成一个连续特征维
```

---

# 15. 为什么 CNN 后面常接 Flatten？

CNN 输出：

```text
[B,C,H,W]
```

表示：

> 每个样本的特征仍然分布在 channel 与空间位置中。

传统 CNN 分类器常希望把：

```text
[C,H,W]
```

整体整理成一个特征向量：

```text
[features]
```

于是：

```text
[B,C,H,W]

↓

Flatten

↓

[B,C×H×W]
```

再交给全连接层综合这些特征。

---

# 16. 重要修正：Linear 不要求整个 Tensor 必须是二维

不要记：

> `nn.Linear` 只能接 `[B, features]` 二维 Tensor。

更准确的是：

> **`nn.Linear(in_features, out_features)` 要求输入 Tensor 的最后一维等于 `in_features`。**

例如：

```python
fc = nn.Linear(20, 10)
```

可以输入：

```text
[4,20]
```

输出：

```text
[4,10]
```

也可以输入：

```text
[4,7,20]
```

输出：

```text
[4,7,10]
```

因为 Linear 只对最后一个维度做线性变换。

但在传统 CNN 分类器中，我们通常希望：

> 把一整张 feature map 的 `[C,H,W]` 综合成一个特征向量。

所以经常：

```text
[B,C,H,W]
↓
Flatten
[B,features]
↓
Linear
[B,num_classes]
```

---

# 17. Linear 层

```python
fc = nn.Linear(
    16 * 8 * 8,
    10
)
```

也就是：

```python
nn.Linear(
    in_features=1024,
    out_features=10
)
```

---

## 17.1 数学本质

Linear 做：

\[
y=xW^T+b
\]

PyTorch 中：

```text
weight.shape
=
[out_features, in_features]
```

因此：

```text
W:
[10,1024]

bias:
[10]
```

输入：

```text
[B,1024]
```

输出：

```text
[B,10]
```

---

## 17.2 `10` 到底是什么意思？

严格来说：

```text
10
=
out_features
=
输出特征数量
```

在当前“10 类分类器”的最后一层：

```text
out_features
=
num_classes
```

所以才可以解释为：

```text
10 个类别
```

注意：

```python
nn.Linear(1024, 256)
```

这里的：

```text
256
```

不代表 256 个类别。

它只是 256 个输出特征。

---

# 18. Logits

分类模型最后：

```text
[B,10]
```

例如：

```text
[
  1.2,
 -0.5,
  3.1,
  ...
]
```

这些值叫：

> **logits：模型对每个类别产生的原始分数。**

特点：

```text
可以为正
可以为负
不要求和为 1
不是概率
```

当前阶段先知道：

```text
Linear
↓
logits
```

即可。

Softmax / CrossEntropy 留到后续学习。

---

# 19. 最小 CNN 完整数据流

假设输入：

```text
[B,3,32,32]
```

网络：

```python
nn.Conv2d(3, 8, 3, padding=1)
nn.ReLU()
nn.MaxPool2d(2, 2)

nn.Conv2d(8, 16, 3, padding=1)
nn.ReLU()
nn.MaxPool2d(2, 2)

Flatten

nn.Linear(16 * 8 * 8, 10)
```

shape：

```text
输入
[B,3,32,32]

↓

Conv1
[B,8,32,32]

↓

ReLU
[B,8,32,32]

↓

Pool1
[B,8,16,16]

↓

Conv2
[B,16,16,16]

↓

ReLU
[B,16,16,16]

↓

Pool2
[B,16,8,8]

↓

Flatten
[B,1024]

↓

Linear
[B,10]

↓

Logits
```

---

# 20. 各层职责总结

| 层 | 核心作用 | C | H/W |
|---|---|---|---|
| Conv2d | 提取、组合局部特征 | 由 `out_channels` 决定 | 可能改变 |
| ReLU | 引入非线性 | 不变 | 不变 |
| MaxPool2d | 空间下采样 | 通常不变 | 通常减小 |
| Flatten | 把 `[C,H,W]` 整理为特征向量 | 被合并 | 被合并 |
| Linear | 对特征做全连接映射 | 不再使用 CNN 中的 C 概念 | 不再使用 H/W |
| Logits | 分类原始分数 | - | - |

---

# 21. 一个最重要的数据流理解

```text
原始图片
[B,3,H,W]

↓

Conv
“这里有哪些局部模式？”

↓

ReLU
加入非线性，
形成不同位置的激活模式

↓

Pool
“保留较强空间响应，
同时降低空间尺寸”

↓

下一层 Conv
组合上一层多个 channels 的特征

↓

越来越深
感受野增大
特征组合越来越复杂

↓

Flatten
把空间特征整理成一个整体特征向量

↓

Linear
综合这些特征，
映射成类别输出

↓

Logits
```

---

# 22. Day 3 验收题

## 1. `ReLU(-3)` 等于多少？

```text
0
```

---

## 2. ReLU 会不会改变 `[B,C,H,W]` 的 shape？

```text
通常不会。
```

---

## 3. 为什么 CNN 不能只有很多层 Conv 而没有激活函数？

因为如果中间完全没有非线性：

```text
多层线性变换
```

仍然可以合并成：

```text
一个更大的线性变换
```

模型无法有效表达复杂非线性关系。

---

## 4. `MaxPool2d(2,2)` 做什么？

更完整回答：

> 使用一个 `2×2` 窗口，每次移动 2 格，对每个 channel 独立地在窗口内取最大值。

---

## 5. `[1,32,64,64]` 经过 `MaxPool2d(2,2)` 后是多少？

```text
[1,32,32,32]
```

---

## 6. Pooling 为什么一般不改变 channel？

因为普通 MaxPool：

> 对每个 channel 分别、独立地只在 H/W 平面执行池化，不跨 channel 融合信息。

---

## 7. `[1,16,8,8]`：

```python
torch.flatten(x, start_dim=1)
```

后是多少？

```text
[1,1024]
```

因为：

```text
16×8×8
=
1024
```

---

## 8. 为什么 Linear 前通常要 Flatten？

不是因为 Linear “必须只能接二维”。

而是：

> 传统 CNN 分类器希望把 `[C,H,W]` 的整张 feature map 整理成一个一维特征向量，再交给全连接层综合判断。

---

## 9. `Linear(1024,10)` 的 `10` 表示什么？

严格说：

```text
10
=
out_features
```

如果这是分类器最后一层，并且任务有 10 类：

```text
out_features
=
num_classes
```

所以输出 10 个类别 logits。

---

## 10. 从图片到 logits 的完整数据流是什么？

```text
图片
↓
Tensor
[B,C,H,W]

↓
Conv2d
局部提取、组合特征

↓
ReLU
引入非线性

↓
Pooling
降低空间尺寸

↓
多层 Conv + ReLU + Pool

↓
高级 feature map

↓
Flatten
[B,C,H,W]
→
[B,C×H×W]

↓
Linear

↓
[B,num_classes]

↓
logits
```

---

# 23. Day 3 常见误区

## 误区 1

```text
ReLU 在“提取特征”
```

更准确：

```text
Conv
→ 负责加权组合、提取特征

ReLU
→ 负责引入非线性
```

---

## 误区 2

```text
Pooling 会混合多个 channels
```

错误。

普通 MaxPool：

```text
每个 channel
独立池化
```

不跨 channel 混合。

---

## 误区 3

```text
MaxPool2d(2,2)
永远 H/W 减半
```

只有在常见的：

```text
kernel_size=2
stride=2
padding=0
```

并且尺寸合适时，才正好减半。

---

## 误区 4

```text
Conv2d 一定只改 C
```

错误。

Conv2d：

```text
C
由 out_channels 决定

H/W
由 kernel_size / stride / padding / dilation 决定
```

所以 Conv 也可以改变空间尺寸。

---

## 误区 5

```text
Linear 必须输入二维 Tensor
```

错误。

Linear 真正要求：

```text
输入最后一维
=
in_features
```

传统 CNN 使用 Flatten 是为了把一张 feature map 整理成一个整体特征向量。

---

## 误区 6

```text
Linear(1024,10)
中的 10 永远表示 10 个类别
```

错误。

严格来说：

```text
10
=
out_features
```

只有它作为分类器最后一层时，才通常等于类别数。

---

# 24. Day 3 当前代码对应关系

现有代码已经完成：

```text
1. ReLU 最小实验

2. MaxPool2d 最小实验

3. Conv → ReLU → Pool

4. Flatten

5. Linear

6. 输出最终 [B,10]
```

当前代码的主流程：

```text
输入：
[1,3,32,32]

↓

Conv2d(3,8,3,padding=1)
[1,8,32,32]

↓

ReLU
[1,8,32,32]

↓

MaxPool2d(2,2)
[1,8,16,16]

↓

Flatten
[1,2048]

因为：
8×16×16 = 2048

↓

Linear(2048,10)

↓

[1,10]
```

注意：

> 当前 `.py` 示例只用了 **一组 Conv → ReLU → Pool**，因此 Flatten 后是 `8×16×16 = 2048`。

如果改成两组：

```text
Conv(3→8)
↓
Pool
↓
Conv(8→16)
↓
Pool
```

最后才会得到：

```text
[1,16,8,8]
→
[1,1024]
```

这两个示例不要混淆。

---

# 25. Day 3 最终需要形成的脑内模型

看到：

```python
nn.Conv2d(3, 8, 3, padding=1)
```

脑中想到：

```text
输入 3 channels
↓
8 个 filter
↓
输出 8 channels
```

看到：

```python
nn.ReLU()
```

想到：

```text
负值 → 0
正值 → 保留
shape 不变
加入非线性
```

看到：

```python
nn.MaxPool2d(2,2)
```

想到：

```text
2×2 窗口
每次走 2 格
每个 channel 独立取最大值
H/W 约减半
C 不变
```

看到：

```python
torch.flatten(x, start_dim=1)
```

想到：

```text
保留 Batch
C/H/W 展开
[B,C,H,W]
→
[B,C×H×W]
```

看到：

```python
nn.Linear(1024,10)
```

想到：

```text
最后一个输入维度 1024
↓
线性映射
↓
输出 10 维
```

如果这是分类器最后一层：

```text
10 维
=
10 个类别 logits
```

---

# 26. Day 3 最终验收标准

给你：

```text
一张 32×32 RGB 图片
+
一个简单 CNN
```

你能够在**不运行代码**的情况下：

1. 写出每层输入 shape；
2. 写出每层输出 shape；
3. 解释 C 为什么变或不变；
4. 解释 H/W 为什么变或不变；
5. 解释 Conv、ReLU、Pool、Flatten、Linear 各自职责；
6. 从图片一路讲到 logits；
7. 不把“激活”“池化”“Flatten”“Linear”混成同一类操作。

如果这些可以独立完成：

> **Day 3 通过。**

---

# 27. 与后续 Mask R-CNN 的连接

Day 3 并不是孤立内容。

后面会直接连接：

```text
Day 3
普通 CNN 数据流

↓

Day 4
Residual Connection / BasicBlock

↓

ResNet

↓

C2 / C3 / C4 / C5

↓

FPN

↓

P2 / P3 / P4 / P5

↓

RPN

↓

RoIAlign

↓

Detection Head / Mask Head
```

以后看到 Mask R-CNN 的 Backbone，本质仍然是在重复今天已经建立的核心思想：

```text
输入 Tensor
↓
不断提取特征
↓
改变 C
↓
改变 H/W
↓
形成多尺度、多层次 feature maps
```

所以 Day 3 真正要掌握的不是几个 API，而是：

> **CNN 中 Tensor 的数据、shape 和特征语义是怎样一层层变化的。**
