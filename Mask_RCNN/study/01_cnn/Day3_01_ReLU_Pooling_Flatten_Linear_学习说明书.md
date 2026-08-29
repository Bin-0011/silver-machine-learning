# Day 3-1：ReLU + Pooling + Flatten + Linear

Day 1 已经学会看图片 Tensor 的 `[B,C,H,W]`，Day 2 又把 `Conv2d` 的局部乘加、filter、channel、stride 和 padding 串了起来。

现在已经能够用 Conv 产生 feature map，但一个真正的 CNN 还需要解决三个后续问题：

`Conv 得到响应 → 怎样加入非线性 → 怎样压缩空间尺寸 → 怎样把 feature map 整理成最终输出`

Day 3-1 就沿着这条链学习四个组件：

`ReLU → Pooling → Flatten → Linear`

这一部分先不写完整 `MiniCNN` 类，只把每一个组件本身和它们之间的连接关系学清楚。

---

## 1. ReLU：让网络不只是连续做线性变换

Day 2 的 Conv 本质是局部加权乘加，可以抽象成 `y=Wx+b`。

如果连续堆很多只有线性计算的层，例如：

`Linear → Linear → Linear`

它们仍然可以合并成一个更大的线性变换。

例如：

`y1=w1x+b1`

`y2=w2y1+b2=w2(w1x+b1)+b2=(w2w1)x+(w2b1+b2)`

最后仍然是 `ax+c` 这种形式。

因此 CNN 需要在 Conv 之间加入非线性。Day 3 先学习最常见的：

```python
nn.ReLU()
```

ReLU 全称是 Rectified Linear Unit，中文常叫“修正线性单元”。现在只需要记住：

`ReLU(x)=max(0,x)`

也就是：

```text
x < 0 → 0
x ≥ 0 → x
```

### 最小可运行实验

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

结果：

```text
[-2,-1,0,1,2]
→ ReLU
[0,0,0,1,2]
```

Shape：

`[1,5] → ReLU → [1,5]`

所以 ReLU 的第一条规律是：

> **改变数值，通常不改变 Shape。**

---

## 2. “非线性”具体是什么意思

线性函数 `y=x` 是一条直线。

ReLU 使用两段不同规则：

```text
x < 0 → y=0
x ≥ 0 → y=x
```

它属于 **分段线性（piecewise linear）**：每一段可以是线性的，但整个函数不能用一条统一直线表示。

因此：

`Conv → ReLU → Conv → ReLU`

不能再简单压缩成一个单独的线性变换。

可以先用“折纸”理解：

```text
只有 Linear
→ 只能平移、拉伸、旋转

加入 ReLU
→ 可以把空间沿某些边界“折一下”
```

多层线性变换与 ReLU 交替后，网络就能表达更复杂的输入—输出关系。

---

## 3. 激活和激活模式

假设某个 filter 产生：

```text
 2.3  -1.2   0.7
-0.5   3.1  -2.0
 1.4   0.2  -0.8
```

ReLU 后：

```text
2.3   0    0.7
0     3.1  0
1.4   0.2  0
```

入门阶段可以把：

```text
ReLU 后 > 0 → 有激活
ReLU 后 = 0 → 没有正响应
```

**激活模式**就是整张 feature map 上“哪里有响应、哪里为 0”的空间分布。

这里要分清：

> **Conv 先产生响应，ReLU 再截断负响应。ReLU 不会凭空创造特征。**

后一层 Conv 会继续读取前一层所有 feature channels，所以特征可以逐层组合：

`边缘 / 纹理 → 局部形状 → 更复杂结构 → 更高级语义`

这仍然是在延续 Day 2 的多通道卷积机制。

---

## 4. Pooling：把空间尺寸压缩下来

随着网络加深，如果 feature map 的 H/W 始终很大，计算量和内存开销会持续增加。

因此 CNN 常使用 **下采样（downsampling）**：

> 让 H/W 变小。

Day 3 先学习：

```python
nn.MaxPool2d
```

MaxPool2d 不像 Conv 那样做“权重乘加”，而是在一个局部窗口里直接保留最大值。

### 完整可运行实验

```python
import torch
import torch.nn as nn

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

print("x:")
print(x)
print("y:")
print(y)
print("x.shape:", x.shape)
print("y.shape:", y.shape)
```

输入：

```text
1   2   3   4
5   6   7   8
9  10  11  12
13 14  15  16
```

`kernel_size=2`：每次看 2×2。

`stride=2`：窗口每次移动 2 格。

四个 2×2 区域分别取最大值：

```text
6   8
14 16
```

所以：

`[1,1,4,4] → MaxPool2d(2,2) → [1,1,2,2]`

---

## 5. 为什么普通 MaxPool 通常不改变 C

输入：

`[B,32,64,64]`

表示每个样本有 32 张 `64×64` feature maps。

MaxPool2d 会对每个 channel **独立**池化：

```text
Channel 1：64×64 → 32×32
Channel 2：64×64 → 32×32
...
Channel 32：64×64 → 32×32
```

于是：

`[B,32,64,64] → MaxPool2d(2,2) → [B,32,32,32]`

所以：

- B 不变
- C 不变
- H/W 变小

这里不要说“Pooling 不碰 C”，更准确的是：

> **Pooling 不跨 channel 混合，而是每个 channel 分开处理。**

---

## 6. Conv 与 Pooling 放在一起区分

| 模块 | 核心计算 | Channel | H/W |
|---|---|---|---|
| Conv2d | 局部加权乘加 | 可改变 | 由参数决定 |
| MaxPool2d | 局部比较取最大 | 通常不变 | 通常减小 |

例如：

```text
[B,3,32,32]
→ Conv2d(3,8,3,padding=1)
[B,8,32,32]
→ ReLU
[B,8,32,32]
→ MaxPool2d(2,2)
[B,8,16,16]
```

这时三层职责已经很清楚：

`Conv → 提取/组合特征`

`ReLU → 引入非线性`

`Pooling → 压缩 H/W`

---

## 7. Flatten：从 feature map 变成 feature vector

经过卷积与池化后，可能得到：

`[B,16,8,8]`

当前分类头希望把每张图片的 `C/H/W` 特征整体整理成一个 feature vector：

`[B,16,8,8] → [B,16×8×8] → [B,1024]`

PyTorch：

```python
x = torch.flatten(x, start_dim=1)
```

`start_dim=1` 表示：

> 第 0 维 Batch 保留，从第 1 维开始一直展平到最后。

所以：

```text
[4,16,8,8]
→
[4,1024]
```

Flatten 不会随机打乱元素，它只是改变 Shape 的组织方式。

---

## 8. `size()`、`view()`、`reshape()` 与 Flatten

假设：

```python
x = torch.randn(4,16,8,8)
```

### `x.size()`

查看 Shape：

```python
print(x.size())
print(x.size(0))
print(x.size(1))
```

对于 BCHW：

```text
x.size(0) → B
x.size(1) → C
x.size(2) → H
x.size(3) → W
```

### `view()`

可以重新解释 Shape：

```python
y = x.view(4,1024)
```

经典 CNN 写法：

```python
y = x.view(x.size(0), -1)
```

其中 `-1` 表示让 PyTorch 自动计算这一维。

`[4,16,8,8] → [4,1024]`

### `reshape()`

同样用于重塑 Shape。与 `view()` 相比，当内存布局不适合直接 view 时，`reshape()` 必要时可以创建新的连续结果。

### `flatten()`

当前任务就是“展平”时：

```python
torch.flatten(x,start_dim=1)
```

语义最直接。

### `view()` 与 `permute()` 不一样

`view()`：重塑 Shape。

`permute()`：改变轴顺序，例如 `HWC → CHW`。

---

## 9. Linear：把特征映射成输出

定义：

```python
nn.Linear(in_features, out_features)
```

例如：

```python
fc = nn.Linear(
    in_features=16 * 8 * 8,
    out_features=10
)
```

就是：

`Linear(1024,10)`

数学形式：

`y=xW^T+b`

其中：

```text
weight.shape = [10,1024]
bias.shape   = [10]
```

在当前分类场景：

`[B,1024] → Linear(1024,10) → [B,10]`

这里：

- `1024` = 输入最后一维的特征数
- `10` = `out_features`

因为它正好是分类器最后一层，所以这里的 10 被设计成 10 个类别对应的输出数。

---

## 10. Linear 并不是只能接受二维 Tensor

这是 Day 3 要专门纠正的一点。

```python
layer = nn.Linear(20,10)
```

可以处理：

```text
[4,20] → [4,10]
```

也可以处理：

```text
[4,7,20] → [4,7,10]
```

真正要求的是：

> **输入最后一维必须等于 `in_features`。**

CNN 分类头前常用 Flatten，是因为我们希望把每张图的 `[C,H,W]` 整体整理成一个 feature vector，再让当前的 Linear 一次综合这些特征。

---

## 11. logits：分类器最后的原始分数

执行：

```python
logits = fc(x)
```

如果：

```text
logits.shape=[B,10]
```

表示每张图片得到 10 个原始输出分数。

logits：

- 可以为正
- 可以为负
- 不要求和为 1
- 还不是概率

Day 3 先记：

> **logits = 模型最后直接输出的原始类别分数。**

---

## 12. 四个组件第一次串起来

完整可运行代码：

```python
import torch
import torch.nn as nn


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

fc = nn.Linear(
    in_features=8 * 16 * 16,
    out_features=10
)

print("输入:", x.shape)

x = conv(x)
print("Conv:", x.shape)

x = relu(x)
print("ReLU:", x.shape)

x = pool(x)
print("Pooling:", x.shape)

x = torch.flatten(x, start_dim=1)
print("Flatten:", x.shape)

logits = fc(x)
print("Linear / logits:", logits.shape)
```

Shape：

```text
[1,3,32,32]
→ Conv
[1,8,32,32]
→ ReLU
[1,8,32,32]
→ Pool
[1,8,16,16]
→ Flatten
[1,2048]
→ Linear
[1,10]
```

到这里先停住。

下一部分再把这些组件组织成一个真正的 `MiniCNN(nn.Module)`。

---

# Day 3-1 完成标准

- [ ] `ReLU(-3)` 等于多少？
- [ ] ReLU 为什么通常不改变 Shape？
- [ ] 为什么连续线性层仍可以合并成一个线性变换？
- [ ] ReLU 为什么能给网络引入非线性？
- [ ] 什么叫激活和激活模式？
- [ ] Conv 与 ReLU 的职责有什么区别？
- [ ] `MaxPool2d(2,2)` 中两个 2 分别表示什么？
- [ ] 能手算 4×4 输入经过 `MaxPool2d(2,2)` 的结果。
- [ ] 为什么普通 MaxPool 通常不改变 C？
- [ ] Conv 与 Pooling 在计算方式上有什么区别？
- [ ] `[B,16,8,8]` Flatten 后为什么是 `[B,1024]`？
- [ ] `start_dim=1` 为什么能保留 Batch？
- [ ] `x.size(0)` 表示什么？
- [ ] `x.view(x.size(0),-1)` 做了什么？
- [ ] `view()`、`reshape()`、`flatten()`、`permute()` 分别负责什么？
- [ ] `Linear(1024,10)` 中 1024 和 10 分别是什么？
- [ ] 为什么不能说 Linear 只能处理二维 Tensor？
- [ ] `fc.weight.shape` 和 `fc.bias.shape` 分别是什么？
- [ ] logits 是什么？
- [ ] 能否不运行代码推导 `Conv → ReLU → Pool → Flatten → Linear` 的完整 Shape？

# Day 3-1 最终必须牢牢记住

```text
ReLU
= max(0,x)
= 引入非线性
= Shape 通常不变
```

```text
MaxPool2d
= 每个 channel 独立取局部最大值
= C 通常不变
= H/W 通常减小
```

```text
Flatten
[B,C,H,W]
→ [B,C×H×W]
```

```text
Linear(in_features,out_features)
→ 处理输入最后一维
→ 输出最后一维变成 out_features
```

```text
logits
= 分类器最后直接输出的原始分数
```

最关键的一句话：

> **Conv 得到 feature map 后，ReLU 让网络获得非线性，Pooling 压缩 H/W，Flatten 把空间特征整理成 feature vector，Linear 再把这些特征映射成最终输出。**
