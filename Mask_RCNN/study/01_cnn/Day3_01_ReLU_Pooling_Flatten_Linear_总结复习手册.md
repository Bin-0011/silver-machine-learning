# Day 3-1 总结：ReLU + Pooling + Flatten + Linear

## 1. 核心概念表

| 概念 | 是什么 | 看到什么时想到它 |
|---|---|---|
| ReLU | `max(0,x)` | 负值变 0，Shape 通常不变 |
| 非线性 | 整体不能压成一个单独线性关系 | `Conv → ReLU → Conv` |
| 激活 | ReLU 后保留的正响应 | feature map 中非 0 正值 |
| 激活模式 | 响应在空间上的分布 | 哪些位置有响应 |
| MaxPool2d | 局部取最大值 | `MaxPool2d(2,2)` |
| downsampling | H/W 变小 | Pooling、stride=2 |
| Flatten | 合并多个维度 | `[B,C,H,W] → [B,N]` |
| `size()` | 查看某一维大小 | `x.size(0)` |
| `view()` | 重塑 Shape | `x.view(B,-1)` |
| `reshape()` | 通用重塑 Shape | view 不方便时 |
| `permute()` | 改变轴顺序 | `HWC → CHW` |
| Linear | `y=xW^T+b` | `nn.Linear(a,b)` |
| logits | 模型最终原始分数 | 最后一层 Linear 输出 |

---

## 2. ReLU 速查

```text
ReLU(x)=max(0,x)
```

```text
[-2,-1,0,1,2]
→
[0,0,0,1,2]
```

核心：

`Conv → 产生响应`

`ReLU → 截断负值，引入非线性`

注意：

> ReLU 不负责凭空创造特征。

---

## 3. Pooling 速查

```python
nn.MaxPool2d(
    kernel_size=2,
    stride=2
)
```

```text
1   2   3   4
5   6   7   8
9  10  11  12
13 14  15  16

→

6   8
14 16
```

当前典型 Shape：

`[B,C,H,W] → [B,C,H/2,W/2]`

---

## 4. Conv 与 Pooling 对比

| | Conv2d | MaxPool2d |
|---|---|---|
| 计算 | 局部加权乘加 | 局部比较取最大 |
| 可学习权重 | 有 | 无 |
| 跨 channel | 会组合 | 不混合 |
| C | 可改变 | 通常不变 |
| H/W | 看参数 | 通常减小 |

---

## 5. Flatten / view / reshape / permute

| 写法 | 主要作用 |
|---|---|
| `torch.flatten(x,start_dim=1)` | 从指定维开始展平 |
| `x.view(...)` | 在允许的内存布局下重塑 Shape |
| `x.reshape(...)` | 通用重塑 Shape |
| `x.permute(...)` | 改变轴顺序 |

典型：

```text
[B,16,8,8]
→ flatten(start_dim=1)
[B,1024]
```

---

## 6. Linear 速查

```python
nn.Linear(1024,10)
```

```text
in_features=1024
out_features=10
weight.shape=[10,1024]
bias.shape=[10]
```

输入输出：

```text
[B,1024]
→
[B,10]
```

注意：

> Linear 处理输入最后一维，不是只能接受二维 Tensor。

---

## 7. 核心完整代码

```python
import torch
import torch.nn as nn


x = torch.randn(1,3,32,32)

conv = nn.Conv2d(
    3, 8,
    kernel_size=3,
    stride=1,
    padding=1
)

relu = nn.ReLU()
pool = nn.MaxPool2d(2,2)
fc = nn.Linear(8 * 16 * 16, 10)

print("输入:", x.shape)

x = conv(x)
print("Conv:", x.shape)

x = relu(x)
print("ReLU:", x.shape)

x = pool(x)
print("Pool:", x.shape)

x = torch.flatten(x,start_dim=1)
print("Flatten:", x.shape)

logits = fc(x)
print("logits:", logits.shape)
```

---

# 8. 母题

## 母题 1：ReLU 数值与 Shape

```text
[-3,-1,0,2,5]
→ ReLU
[0,0,0,2,5]
```

训练：

> **看到 ReLU，想到“数值变，Shape 通常不变”。**

---

## 母题 2：MaxPool2d 手算

```text
4×4
→ MaxPool2d(2,2)
→ 2×2
```

训练：

> **kernel 决定看多大，stride 决定走多远。**

---

## 母题 3：Pooling 后的 Channel

```text
[B,32,64,64]
→ MaxPool2d(2,2)
→ [B,32,32,32]
```

训练：

> **普通 Pool 每个 channel 独立处理，C 通常保持。**

---

## 母题 4：Flatten

```text
[B,16,8,8]
→
[B,16×8×8]
→
[B,1024]
```

训练：

> **保留 B，把后面维度全部合并。**

---

## 母题 5：Linear 输入检查

```python
Linear(1024,10)
```

输入最后一维必须是：

```text
1024
```

训练：

> **看到 Linear，先检查最后一维和 in_features。**

---

# 9. 错题本

## 错题 1：把 ReLU 当成主要特征提取器

**错误理解：** ReLU 自己在识别边缘或物体。

**正确理解：** Conv 做加权组合，ReLU 用 `max(0,x)` 引入非线性。

**记忆：**

> **Conv 组合，ReLU 截断。**

---

## 错题 2：认为 Pooling 会把不同 channels 混起来

**正确理解：**

> 标准 MaxPool2d 对每个 channel 独立做 H/W 池化。

---

## 错题 3：把 `MaxPool2d(2,2)` 的两个 2 当成重复参数

```text
第一个 2 = kernel_size
第二个 2 = stride
```

---

## 错题 4：认为 Flatten 会打乱元素

**正确理解：**

> Flatten 主要是改变 Shape 的组织方式。

---

## 错题 5：把 `view()` 当成轴交换

**正确理解：**

```text
view → reshape
permute → reorder axes
```

---

## 错题 6：认为 Linear 只能接受二维 Tensor

**正确理解：**

```text
Linear(20,10)
[4,7,20] → [4,7,10]
```

最后一维匹配即可。

---

## 错题 7：把 out_features 永远当类别数

**正确理解：**

> out_features 本质是输出特征数；只有分类器最后一层时通常等于类别数。

---

# 10. 重点掌握清单

## 必须会解释

- [ ] ReLU 与非线性
- [ ] 激活与激活模式
- [ ] MaxPool 的 kernel/stride
- [ ] 为什么 Pooling 通常不改 C
- [ ] Flatten 的作用
- [ ] Linear 的 in_features/out_features
- [ ] logits

## 必须会手算

- [ ] ReLU 小向量
- [ ] 4×4 MaxPool2d(2,2)
- [ ] `[B,16,8,8] → [B,1024]`
- [ ] `Linear(1024,10)` 输入输出 Shape

## 必须会看代码判断

- [ ] `nn.ReLU()` → Shape 通常不变
- [ ] `nn.MaxPool2d(2,2)` → C 通常不变，H/W 减小
- [ ] `flatten(start_dim=1)` → 保留 B
- [ ] `view(x.size(0),-1)` → 保留 B，其他压平
- [ ] `Linear(a,b)` → 最后一维 a→b

---

# 11. 触发式记忆

看到 `nn.ReLU()`：

> `max(0,x)`，引入非线性。

看到 `nn.MaxPool2d(2,2)`：

> 2×2 窗口、步长 2、各 channel 独立取最大。

看到 `torch.flatten(x,start_dim=1)`：

> 保留 Batch，把其余维度合并。

看到 `x.view(x.size(0),-1)`：

> 保留 B，其余自动压平。

看到 `nn.Linear(1024,10)`：

> 输入最后一维 1024，输出最后一维 10。

---

# Day 3-1 最终必须牢牢记住

`Conv → ReLU → Pooling → Flatten → Linear`

最关键的一句话：

> **这四个组件分别解决“非线性、空间压缩、特征整理、输出映射”四个不同问题。**
