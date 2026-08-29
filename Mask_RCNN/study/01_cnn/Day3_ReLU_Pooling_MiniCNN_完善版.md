# Day 3：ReLU + Pooling + Flatten + Linear + MiniCNN

> 学习目标：在 Day 1 Tensor、Day 2 Conv2d 的基础上，第一次完整串起一个最小 CNN 的数据流：
>
> **图片 → Conv → ReLU → Pool → Conv → ReLU → Pool → Flatten → Linear → Logits**

---

# 0. Day 3 最终目标

今天不是只记 API，而是要做到：

1. 知道 ReLU 为什么必须加入 CNN；
2. 理解“激活 / 不激活”“激活模式”“非线性表达能力”；
3. 理解 Pooling 的窗口、步长以及为什么普通池化不改变 Channel；
4. 能手推每一层的 `[B, C, H, W]`；
5. 理解 Flatten 为什么连接 CNN 与分类头；
6. 掌握 `torch.flatten()`、`x.view()`、`x.size()`、`reshape()` 的关系；
7. 理解 `Linear(in_features, out_features)`；
8. 能自己写出一个最小 `MiniCNN`；
9. 不运行代码也能推导整个数据流。

---

# 1. Day 3 总数据流

```text
图片
↓
Tensor
[B, 3, H, W]

↓ Conv2d

局部特征
[B, C1, H1, W1]

↓ ReLU

加入非线性
shape 通常不变

↓ Pooling

压缩空间尺寸
[B, C1, H2, W2]

↓ Conv + ReLU + Pool

组合更复杂的特征
[B, C2, H3, W3]

↓ Flatten

[B, C2 × H3 × W3]

↓ Linear

[B, num_classes]

↓ logits

每个类别对应一个原始分数
```

---

# 2. ReLU

## 2.1 最小实验

```python
import torch
import torch.nn as nn

x = torch.tensor([
    [-2., -1., 0., 1., 2.]
])

relu = nn.ReLU()
y = relu(x)

print(x)
print(y)
```

输出：

```text
输入：
[-2, -1, 0, 1, 2]

ReLU：
[0, 0, 0, 1, 2]
```

公式：

\[
\operatorname{ReLU}(x)=\max(0,x)
\]

即：

```text
x < 0  → 0
x ≥ 0  → x
```

ReLU 通常：

```text
[B,C,H,W]
↓
ReLU
↓
[B,C,H,W]
```

**不改变 shape，只改变数值。**

---

# 3. 为什么 Conv 后要加 ReLU？

Conv 的核心计算是：

```text
乘法
+
加法
```

可以抽象成：

\[
y=Wx+b
\]

如果连续堆很多线性变换：

\[
y_1=W_1x+b_1
\]

\[
y_2=W_2y_1+b_2
\]

代入：

\[
y_2=W_2(W_1x+b_1)+b_2
\]

整理后仍然可以写成：

\[
y_2=Wx+b
\]

所以：

```text
Conv
↓
Conv
↓
Conv
```

如果中间没有非线性激活，从表达能力上仍然可以等价成一个更大的线性变换。

---

## 3.1 什么是“非线性表达能力”？

“表达能力”问的是：

> **这个模型能够表示多复杂的“输入 → 输出”关系？**

线性关系只能按照一套固定规则变化。

ReLU：

\[
\operatorname{ReLU}(x)=
\begin{cases}
0,&x<0\\
x,&x\ge0
\end{cases}
\]

会根据输入所在区域使用不同规则：

```text
x < 0
→ 输出 0

x ≥ 0
→ 输出 x
```

所以它整体不是一个单独的线性函数，而是一个**分段线性函数（piecewise linear）**。

因此：

```text
Conv
↓
ReLU
↓
Conv
↓
ReLU
```

无法再简单合并成一个 Conv / Linear。

---

## 3.2 折纸类比

没有 ReLU：

```text
平移
旋转
拉伸
压缩
```

做很多次，本质上仍然只是对一个平面的线性变换。

ReLU 相当于：

> **允许把这个平面沿某些边界“折一下”。**

```text
Linear
↓
ReLU  ← 折一次
↓
Linear
↓
ReLU  ← 再折一次
```

越来越多的折叠，使网络能够表示越来越复杂的函数。

---

# 4. 什么叫“这里激活、那里不激活”？

假设一个 filter 扫描图片后得到：

```text
Conv 输出：

 2.3  -1.2   0.7
-0.5   3.1  -2.0
 1.4   0.2  -0.8
```

经过 ReLU：

```text
2.3   0    0.7
0     3.1  0
1.4   0.2  0
```

因此：

```text
> 0
→ 激活

≤ 0
→ ReLU 后变成 0
→ 不激活
```

---

## 4.1 什么是“激活模式”？

激活模式不是一个单独数值，而是：

> **整张 feature map 上，哪些位置有响应、哪些位置变成 0 的空间分布。**

例如：

```text
0  3  0  0
0  5  0  0
0  4  0  0
0  2  0  0
```

可能对应某种竖直结构。

另一张：

```text
0  0  0  0
4  5  3  2
0  0  0  0
0  0  0  0
```

可能对应某种水平结构。

注意：

> **不是 ReLU 凭空创造了特征。**

更准确地说：

```text
Conv
先产生不同位置上的响应

↓

ReLU
将负值截断为 0

↓

正响应的位置更加明确
```

---

# 5. 简单特征为什么能变成高级语义？

核心原因：

> **后一层卷积会把前一层所有 feature channels 当作输入，再组合这些已经提取出的特征。**

假设第一层：

```text
输入：
[B,3,H,W]

↓

Conv2d(3,64,...)

↓

输出：
[B,64,H,W]
```

这 64 个 channel 可以粗略想象为：

```text
Channel 1 → 某种竖边缘响应
Channel 2 → 某种横边缘响应
Channel 3 → 某种斜边缘响应
Channel 4 → 某种纹理响应
...
```

第二层的单个 filter：

```text
[64, 3, 3]
```

会同时读取这 64 个 channel。

所以第二层可以学习：

```text
竖边缘
+
横边缘
+
特定空间位置关系

→ 某种角点 / 局部轮廓
```

继续堆叠：

```text
简单边缘、纹理
↓
局部形状
↓
更复杂的局部结构
↓
物体部件
↓
更高级的物体相关语义
```

这不是代码里写：

```python
if eye:
    ...
```

而是训练过程中，大量 filter 自动学到：

> **哪些低级特征组合，对最终任务最有用。**

---

# 6. 感受野：为什么深层能看到更大的结构？

第一层 `3×3 Conv`：

```text
一个输出位置
直接看原图 3×3
```

第二层：

```text
一个位置
看上一层 3×3

而上一层每个点
又来自原图的局部区域
```

因此随着层数增加：

```text
浅层
→ 看小范围

深层
→ 间接利用越来越大的原图区域
```

所以网络才有机会从：

```text
边缘
→ 局部轮廓
→ 物体部件
→ 更完整结构
```

---

# 7. MaxPool2d

## 7.1 示例

```python
x = torch.tensor([
    [[[
        1.,  2.,  3.,  4.,
    ], [
        5.,  6.,  7.,  8.,
    ], [
        9., 10., 11., 12.,
    ], [
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

---

## 7.2 `kernel_size=2`

表示：

> **每次池化窗口看多大的空间区域。**

`2` 等价于：

```python
kernel_size=(2, 2)
```

所以窗口大小：

```text
2 × 2
```

---

## 7.3 `stride=2`

表示：

> **池化窗口每次移动多少格。**

`stride=2`：

```text
每次横向 / 纵向移动 2 个像素
```

因此：

```text
1 2     3 4
5 6     7 8

9 10    11 12
13 14   15 16
```

分别取最大值：

```text
6   8
14 16
```

因此：

```text
[1,1,4,4]
↓ MaxPool2d(2,2)
[1,1,2,2]
```

---

# 8. Conv 和 Pooling 的数学区别

## Conv

假设单通道输入：

```text
1 2 3
5 6 7
9 10 11
```

卷积核：

```text
0 1 0
1 1 1
0 1 0
```

计算：

```text
对应位置相乘
↓
全部相加
↓
+ bias
```

即：

> **Conv 做“加权乘加”。**

多通道输入时，一个 filter：

```text
[C_in, kH, kW]
```

会同时读取所有输入 channels，并跨 channel 求和。

---

## MaxPool

```text
2×2 窗口
↓
直接选择最大值
```

没有可学习权重，也不会跨 channel 加权融合。

即：

> **MaxPool 做“局部比较并保留最大值”。**

---

# 9. 为什么普通 Pooling 不改变 C？

输入：

```text
[B,C,H,W]
```

普通 `MaxPool2d` / `AvgPool2d` 会：

> **对每个 channel 独立地在 H/W 平面上做池化。**

例如：

```text
[B,32,64,64]
```

相当于有 32 张 `64×64` feature map。

```text
Channel 1:
64×64 → 32×32

Channel 2:
64×64 → 32×32

...

Channel 32:
64×64 → 32×32
```

所以：

```text
[B,32,64,64]
↓ MaxPool2d(2,2)
[B,32,32,32]
```

C 还是 32。

---

## 9.1 为什么 Conv 能改变 C？

```python
nn.Conv2d(
    in_channels=3,
    out_channels=64,
    kernel_size=3
)
```

有：

```text
64 个 filter
```

每个 filter 产生：

```text
1 张 feature map
```

所以：

```text
64 个 filter
→ 64 个输出 channels
```

因此：

> **Conv2d 的 `out_channels` 决定输出 C。**

---

## 9.2 一个便于 Day 3 记忆的口诀

```text
普通 Conv：
经常负责提取 / 组合特征，并可改变 C

普通 Pooling：
主要负责压缩 H/W，C 通常不变
```

注意不要把它背成绝对规则：

- Conv 是否改变 H/W，还取决于 `kernel_size / stride / padding / dilation`；
- Conv 也可以让 C 不变，例如 `Conv2d(64,64,3,padding=1)`；
- 标准空间 Pooling 一般不改变 C。

---

# 10. 为什么需要下采样？

如果所有 feature map 一直维持很大的 H/W：

```text
64×64
↓
64×64
↓
64×64
↓
64×64
```

计算量和显存开销都会持续很大。

传统 CNN 常逐渐：

```text
空间尺寸 H/W：
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

同时经常：

```text
C：
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

> **空间越来越粗，但每个空间位置上的特征描述越来越丰富。**

这也是后续理解：

```text
ResNet C2 / C3 / C4 / C5
```

以及：

```text
FPN P2 / P3 / P4 / P5
```

的重要基础。

---

# 11. 第一组 Conv + ReLU + Pool

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

Shape：

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

---

# 12. 第二组 Conv + ReLU + Pool

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

数据流：

```text
[1,8,16,16]

↓ Conv2d(8,16,3,padding=1)

[1,16,16,16]

↓ ReLU

[1,16,16,16]

↓ MaxPool2d(2,2)

[1,16,8,8]
```

---

# 13. Flatten

现在：

```text
x.shape = [B,16,8,8]
```

分类头希望把每张图片的所有特征整理成一个一维特征向量：

```text
[B,16,8,8]

↓

[B,16×8×8]

↓

[B,1024]
```

PyTorch：

```python
x = torch.flatten(x, start_dim=1)
```

其中：

```text
start_dim=1
```

表示：

> 从第 1 维开始一直展平到最后，保留第 0 维 Batch。

例如：

```text
[4,16,8,8]
↓
[4,1024]
```

---

# 14. `x.size()` —— 查看 Tensor 每一维大小

```python
x = torch.randn(4,16,8,8)
```

```python
print(x.size())
```

得到：

```text
torch.Size([4,16,8,8])
```

指定维度：

```python
x.size(0)   # 4
x.size(1)   # 16
x.size(2)   # 8
x.size(3)   # 8
```

对于 BCHW：

```text
dim=0 → B
dim=1 → C
dim=2 → H
dim=3 → W
```

而：

```python
x.size(0)
```

与：

```python
x.shape[0]
```

在这里含义相同。

---

# 15. `x.view()` —— 重塑 Tensor 形状

例如：

```python
x = torch.randn(4,16,8,8)
```

总元素数：

```text
4 × 16 × 8 × 8
= 4096
```

可以：

```python
y = x.view(4, 1024)
```

得到：

```text
[4,1024]
```

`view()` 的核心要求：

> **变形前后元素总数必须一致。**

---

# 16. `-1`：让 PyTorch 自动推导这一维

```python
x.view(4, -1)
```

已知：

```text
总元素 = 4096
第一维 = 4
```

所以：

```text
-1 = 4096 / 4 = 1024
```

结果：

```text
[4,1024]
```

其他例子：

| 写法 | 结果 |
|---|---|
| `x.view(4,-1)` | `[4,1024]` |
| `x.view(-1,64)` | `[64,64]` |
| `x.view(-1,16,8)` | `[32,16,8]` |

注意：

> 一个 `view()` 里最多只能有一个 `-1`。

例如：

```python
x.view(-1, -1)
```

无法确定两个维度分别多大，因此会报错。

---

# 17. `x.view(x.size(0), -1)`

这是经典 CNN 代码：

```python
x = x.view(x.size(0), -1)
```

假设：

```text
x.shape = [4,16,8,8]
```

拆解：

### 第一步

```python
x.size(0)
```

得到：

```text
4
```

### 第二步

```text
-1
```

自动计算：

```text
16 × 8 × 8
= 1024
```

### 第三步

等价于：

```python
x = x.view(4, 1024)
```

最终：

```text
[4,16,8,8]
↓
[4,1024]
```

一句话：

> **保留 Batch 维，其余维度全部压成一个 feature 维。**

---

# 18. `view` 和 `flatten` 是不是完全一样？

在这个 CNN 分类头场景：

```python
torch.flatten(x, start_dim=1)
```

和：

```python
x.view(x.size(0), -1)
```

**目标 shape 和作用通常相同：**

```text
[B,C,H,W]
→
[B,C×H×W]
```

但严格来说，它们不是任何情况下都“完全一样”的 API。

区别：

| 写法 | 主要用途 | 特点 |
|---|---|---|
| `torch.flatten(x, start_dim=1)` | 专门展平 | 语义最清晰 |
| `x.view(...)` | 通用 reshape | 要求满足 view 的内存布局条件 |
| `x.reshape(...)` | 通用 reshape | 必要时可创建拷贝 |

因此在 Day 3 推荐：

```python
torch.flatten(x, start_dim=1)
```

因为：

> 一眼就知道这里是在做 Flatten。

---

# 19. `view()` 的连续内存问题

某些操作例如：

```python
transpose
permute
```

可能让 Tensor 的逻辑维度顺序改变，但底层内存布局不满足 `view()` 的要求。

这时直接：

```python
x.view(...)
```

可能报错。

常见解决办法：

### 方法 1：使用 flatten

```python
x = torch.flatten(x, start_dim=1)
```

### 方法 2：contiguous 后再 view

```python
x = x.contiguous()
x = x.view(x.size(0), -1)
```

### 方法 3：reshape

```python
x = x.reshape(x.size(0), -1)
```

`reshape()` 会尽量返回 view；如果不能安全共享原内存，则可能创建新的连续结果。

---

# 20. 一个容易误解的点：`view()` 不负责“交换维度”

`view()` 的主要职责是：

> **按照现有元素顺序重新解释 shape。**

如果需要：

```text
HWC → CHW
```

应该使用：

```python
permute
```

例如：

```python
x = x.permute(2,0,1)
```

所以不要把：

```text
view
```

理解成：

> “万能的维度交换工具”。

它是**重塑 shape**，不是随意改变轴顺序。

---

# 21. Linear

定义：

```python
nn.Linear(
    in_features,
    out_features
)
```

例如：

```python
self.fc = nn.Linear(
    16 * 8 * 8,
    10
)
```

即：

```text
Linear(1024,10)
```

---

## 21.1 数学本质

\[
y=xW^T+b
\]

其中：

```text
x:
[..., in_features]

W:
[out_features, in_features]

b:
[out_features]
```

---

## 21.2 参数

```python
nn.Linear(1024,10)
```

表示：

```text
in_features = 1024
out_features = 10
```

在当前 10 分类任务的最后一层：

```text
10
=
10 个类别对应的 10 个输出分数
```

但注意：

> `out_features` 本质是“输出特征数”，不是永远等于类别数。

例如隐藏层：

```python
nn.Linear(1024,256)
```

这里的 `256` 只是 256 个输出特征，不代表 256 类。

---

# 22. Linear 并不是“只能接受二维 Tensor”

这是 Day 3 必须纠正的一个概念。

例如：

```python
layer = nn.Linear(20, 10)
```

可以输入：

```text
[4,20]
```

得到：

```text
[4,10]
```

也可以输入：

```text
[4,7,20]
```

得到：

```text
[4,7,10]
```

PyTorch `Linear` 真正要求的是：

> **输入最后一维必须等于 `in_features`。**

所以 CNN 分类器中 Flatten 的原因不是：

```text
Linear 法律规定输入必须二维
```

而是：

> **我们希望把每张图片的 `[C,H,W]` 特征整体整理成一个一维 feature vector，再让分类 Linear 综合这些特征。**

即：

```text
[B,C,H,W]

↓ Flatten

[B,C×H×W]

↓ Linear(C×H×W, num_classes)

[B,num_classes]
```

---

# 23. Linear 的 weight / bias shape

对于：

```python
fc = nn.Linear(1024,10)
```

有：

```text
fc.weight.shape
=
[10,1024]
```

以及：

```text
fc.bias.shape
=
[10]
```

输入：

```text
[B,1024]
```

内部：

```text
[B,1024]
×
[1024,10]

↓

[B,10]
```

---

# 24. Logits

分类器最终：

```python
logits = self.fc(x)
```

如果：

```text
logits.shape = [B,10]
```

表示每张图片得到 10 个原始类别分数。

例如：

```text
[
  1.2,
 -0.5,
  3.1,
  ...
]
```

这些值：

- 可以是正数；
- 可以是负数；
- 不要求和为 1；
- 还不是“概率”。

Day 3 暂时记：

> **logits = 模型最后直接输出的原始类别分数。**

---

# 25. `fc` 为什么叫 `fc`？

代码：

```python
self.fc = nn.Linear(...)
```

这里：

```text
fc
=
fully connected
=
全连接层
```

它只是一个**变量名 / 属性名**。

你完全可以写：

```python
self.linear = nn.Linear(...)
```

也可以写：

```python
self.classifier = nn.Linear(...)
```

这些都可以。

常见习惯：

```text
fc
→ 常表示 fully connected

classifier
→ 常表示整个分类头

linear
→ 直接按层类型命名
```

为什么不直接叫：

```python
self.logits
```

因为：

```text
self.fc
```

是一个**层 / 模块**；

而：

```text
logits
```

通常是这个层计算之后得到的**Tensor 结果**。

所以：

```python
self.fc = nn.Linear(...)
...
logits = self.fc(x)
```

语义非常清晰：

```text
fc
=
计算模块

logits
=
计算结果
```

---

# 26. 完整 MiniCNN

```python
import torch
import torch.nn as nn


class MiniCNN(nn.Module):

    def __init__(self):
        super().__init__()

        # ---------- 第一组：提取较浅层特征 ----------
        # [B,3,32,32] -> [B,8,32,32]
        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=8,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
        )

        self.relu1 = nn.ReLU()

        # [B,8,32,32] -> [B,8,16,16]
        self.pooling1 = nn.MaxPool2d(
            kernel_size=2,
            stride=2
        )

        # ---------- 第二组：组合更复杂特征 ----------
        # [B,8,16,16] -> [B,16,16,16]
        self.conv2 = nn.Conv2d(
            in_channels=8,
            out_channels=16,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
        )

        self.relu2 = nn.ReLU()

        # [B,16,16,16] -> [B,16,8,8]
        self.pooling2 = nn.MaxPool2d(
            kernel_size=2,
            stride=2
        )

        # ---------- 分类头 ----------
        # [B,1024] -> [B,10]
        self.fc = nn.Linear(
            in_features=16 * 8 * 8,
            out_features=10
        )

    def forward(self, x):

        print("提取第一组特征:")

        x = self.conv1(x)
        print("conv1:", x.shape)

        x = self.relu1(x)
        print("relu1:", x.shape)

        x = self.pooling1(x)
        print("pool1:", x.shape)

        print("提取第二组特征:")

        x = self.conv2(x)
        print("conv2:", x.shape)

        x = self.relu2(x)
        print("relu2:", x.shape)

        x = self.pooling2(x)
        print("pool2:", x.shape)

        print("Flatten:")

        x = torch.flatten(x, start_dim=1)
        print("flatten:", x.shape)

        print("分类头:")

        logits = self.fc(x)
        print("logits:", logits.shape)

        return logits


if __name__ == "__main__":

    dummy_input = torch.randn(
        1,
        3,
        32,
        32
    )

    model = MiniCNN()

    output = model(dummy_input)

    print("输入 Shape:", dummy_input.shape)
    print("输出 Shape:", output.shape)
```

---

# 27. MiniCNN 完整 Shape 推导

不运行代码，应该直接推：

```text
输入
[B,3,32,32]

↓

Conv2d(3,8,3,padding=1)
[B,8,32,32]

↓

ReLU
[B,8,32,32]

↓

MaxPool2d(2,2)
[B,8,16,16]

↓

Conv2d(8,16,3,padding=1)
[B,16,16,16]

↓

ReLU
[B,16,16,16]

↓

MaxPool2d(2,2)
[B,16,8,8]

↓

Flatten(start_dim=1)
[B,1024]

↓

Linear(1024,10)
[B,10]

↓

logits
```

---

# 28. 每一层的职责

| 层 | 核心职责 | 常见 shape 变化 |
|---|---|---|
| Conv2d | 局部加权组合、提取特征 | C 可变，H/W 也可能变 |
| ReLU | 引入非线性 | 通常不变 |
| Pooling | 压缩空间尺寸 | H/W 变小，C 通常不变 |
| Flatten | 把每张图的特征铺成向量 | `[B,C,H,W] → [B,N]` |
| Linear | 综合 feature vector 并映射到输出 | 最后一维改变 |
| logits | 最终原始分类分数 | `[B,num_classes]` |

---

# 29. Day 3 最终核心逻辑链

```text
图片 Tensor
[B,C,H,W]

↓

Conv
在局部窗口里：
输入 × weight
跨空间 / channel 求和
产生新的 feature maps

↓

ReLU
负值截断为 0
加入非线性
形成不同的响应 / 激活模式

↓

Pooling
每个 channel 独立压缩 H/W
降低空间尺寸和计算量

↓

继续 Conv + ReLU
后一层组合前一层多个 feature channels

↓

特征逐渐由：
简单模式
→ 局部结构
→ 更复杂特征

↓

Flatten
把每张图片的 [C,H,W]
整理成 feature vector

↓

Linear
综合全部特征

↓

logits
每个类别一个原始分数
```

---

# 30. Day 3 验收题

## 1. `ReLU(-3)` 等于多少？

```text
0
```

---

## 2. ReLU 会改变 `[B,C,H,W]` shape 吗？

通常：

```text
不会
```

只改变数值。

---

## 3. 为什么 CNN 不能只有很多 Conv 而没有激活函数？

因为连续线性变换仍然可以合并成一个更大的线性变换。

ReLU 为网络引入非线性，使不同层无法简单合并，从而能够表示复杂关系。

---

## 4. `MaxPool2d(2,2)` 做什么？

```text
使用 2×2 空间窗口
每次移动 2 格
对每个 channel 独立取最大值
```

---

## 5. `[1,32,64,64]` 经过 `MaxPool2d(2,2)`？

```text
[1,32,32,32]
```

---

## 6. 为什么普通 Pooling 不改变 channels？

因为：

```text
每个 channel 独立做 H/W 池化
```

不同 channel 不会被 MaxPool 混合。

---

## 7. `[1,16,8,8]` flatten 后？

```text
16 × 8 × 8
=
1024
```

所以：

```text
[1,1024]
```

---

## 8. 为什么 CNN 分类头前通常 Flatten？

不是因为 `Linear` 绝对只能处理二维 Tensor。

而是因为我们希望：

```text
每张图片的 [C,H,W]
↓
整理成一个完整 feature vector
↓
让分类 Linear 综合这些特征
```

---

## 9. `Linear(1024,10)` 的 `10` 是什么？

本质：

```text
out_features = 10
```

如果它是 10 分类模型的最后一层：

```text
10 个输出
=
10 个类别的 logits
```

---

## 10. 图片到 logits 的完整数据流？

```text
图片
↓
Tensor [B,C,H,W]
↓
Conv
↓
ReLU
↓
Pooling
↓
更多 Conv / ReLU / Pool
↓
feature map
↓
Flatten
↓
feature vector
↓
Linear
↓
logits
```

---

# 31. Day 3 完成标准

看到：

```python
nn.Conv2d(3, 8, 3, padding=1)
nn.ReLU()
nn.MaxPool2d(2, 2)

nn.Conv2d(8, 16, 3, padding=1)
nn.ReLU()
nn.MaxPool2d(2, 2)

nn.Flatten()
nn.Linear(16 * 8 * 8, 10)
```

能够不运行代码直接写出：

```text
[B,3,32,32]
↓
[B,8,32,32]
↓
[B,8,32,32]
↓
[B,8,16,16]
↓
[B,16,16,16]
↓
[B,16,16,16]
↓
[B,16,8,8]
↓
[B,1024]
↓
[B,10]
```

并且能解释：

```text
每一层输入是什么？
每一层做什么？
每一层输出什么？
为什么下一层需要这个结果？
```

即可认为 Day 3 已通过。

---

# 32. Day 3 暂时不要深入

暂时不扩展：

```text
BatchNorm
Dropout
Softmax 数学细节
CrossEntropy 数学推导
Optimizer
完整训练循环
Residual Connection
ResNet
FPN
Anchor
RPN
RoIAlign
```

下一阶段再进入：

```text
Day 4
Residual Connection / BasicBlock
```

---

# 33. 一页速记版

```text
Conv
= 局部加权组合
= 提取 / 组合特征
= C 可变，H/W 也可能变

ReLU
= max(0,x)
= 不改变 shape
= 引入非线性

激活
= ReLU 后仍 > 0 的响应

激活模式
= feature map 上不同位置“有响应 / 无响应”的空间分布

Pooling
= 每个 channel 独立压缩空间
= 普通池化通常 C 不变
= H/W 变小

Flatten
= [B,C,H,W]
→ [B,C×H×W]

x.size(0)
= Batch 大小

x.view(x.size(0), -1)
≈ CNN 分类头里常见的 Flatten 写法

-1
= 自动推导这一维

Linear(1024,10)
= 1024 个输入特征
→ 10 个输出特征

分类最后一层：
10 个输出
→ 10 个类别 logits

logits
= 原始类别分数
= 不是概率
```
