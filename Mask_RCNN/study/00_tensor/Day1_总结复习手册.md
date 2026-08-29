# Day 1 总结：Tensor、图片与 Shape

## 1. 核心概念表

| 概念 | 是什么 | 看到什么时想到它 |
|---|---|---|
| Tensor | PyTorch 中保存多维数据的基本结构 | `torch.randn(...)`、`x.shape` |
| B | Batch，一次处理多少张图片 | `[B,C,H,W]` 第 0 维 |
| C | Channel，通道/特征维 | `[B,C,H,W]` 第 1 维 |
| H | Height，高度/行 | 第 2 维 |
| W | Width，宽度/列 | 第 3 维 |
| integer indexing | 取某一维中的一个具体元素 | `x[:,0]` |
| slicing | 取某一维上的一个范围 | `x[:,0:1]` |
| RGB channel | 图片原始颜色通道 | `C=3` |
| feature channel | CNN 学到的特征通道 | `C=64/256/...` |
| shape | 每一维分别有多大 | `x.shape` |
| ndim | 一共有几个维度 | `x.ndim` |
| numel | Tensor 中一共有多少个元素 | `x.numel()` |
| dtype | 元素数据类型 | `x.dtype` |
| device | Tensor 所在设备 | `x.device` |

---

## 2. 核心数据流

普通图片常见：

```text
HWC
```

进入 PyTorch 卷积前通常整理成：

```text
BCHW
```

例如：

```text
[1,3,640,640]
```

经过：

```python
Conv2d(3,64,3,padding=1)
```

得到：

```text
[1,64,640,640]
```

空间位置 `(H,W)` 上的描述也发生变化：

```text
输入：
[R,G,B]

→ Conv2d →

输出：
[f1,f2,...,f64]
```

---

## 3. Tensor 索引速查

假设：

```text
x.shape = [1,3,640,640]
```

| 表达式 | 含义 | 结果 shape |
|---|---|---|
| `x` | 完整 Tensor | `[1,3,640,640]` |
| `x[0]` | 第 0 张图 | `[3,640,640]` |
| `x[0:1]` | 第 0 张图，但保留 B | `[1,3,640,640]` |
| `x[:,0]` | 所有图片的第 0 通道 | `[1,640,640]` |
| `x[:,0:1]` | 第 0 通道，但保留 C | `[1,1,640,640]` |
| `x[0,0]` | 第 0 图、第 0 通道 | `[640,640]` |
| `x[0,:,100,200]` | 一个位置的全部 RGB | `[3]` |
| `x[0,0,100,200]` | 一个具体值 | `[]` |
| `x[:,:,0:3,0:3]` | 左上角 3×3 | `[1,3,3,3]` |

最重要规则：

```text
整数索引
→ 取一个具体元素
→ 对应维度通常消失

切片
→ 取一个范围
→ 对应维度通常保留
```

---

## 4. RGB channel 与 feature channel

| | RGB channel | CNN feature channel |
|---|---|---|
| 来源 | 图片本身 | 卷积层学习 |
| 典型数量 | 3 | 64、256、512… |
| 表示 | R/G/B 颜色信息 | 网络学到的特征响应 |
| 一个位置上的向量 | `[R,G,B]` | `[f1,...,f64]` |

最关键区别：

> **CNN 的 64 个 channel 不是 64 种颜色，而是 64 维特征描述。**

---

## 5. Conv2d 与 weight.shape

```python
conv = nn.Conv2d(
    in_channels=3,
    out_channels=64,
    kernel_size=3
)
```

可以直接读成：

| 参数 | 含义 |
|---|---|
| `3` | 输入通道数 |
| `64` | 输出通道数 |
| `3` | 3×3 kernel |

权重：

```text
conv.weight.shape
=
[64,3,3,3]
```

排列：

```text
[C_out,C_in,K_h,K_w]
```

因此单个 filter 的主要权重 shape：

```text
[3,3,3]
```

具体卷积计算放在 Day 2。

---

## 6. Tensor 基本属性速查

| 写法 | 含义 |
|---|---|
| `x.shape` | 每一维大小 |
| `x.ndim` | 维度数量 |
| `x.numel()` | 元素总数 |
| `x.dtype` | 数据类型 |
| `x.device` | CPU/GPU |
| `x[...]` | 具体 values / 切片 |

---

## 7. 核心完整代码

```python
import torch
import torch.nn as nn


x = torch.randn(1, 3, 640, 640)

print("x.shape:", x.shape)
print("x.ndim:", x.ndim)
print("x.numel():", x.numel())
print("x.dtype:", x.dtype)
print("x.device:", x.device)

print("\n===== 索引 =====")
print("x[0].shape:", x[0].shape)
print("x[:,0].shape:", x[:, 0].shape)
print("x[:,0:1].shape:", x[:, 0:1].shape)
print("x[0,0,0:3,0:3].shape:", x[0, 0, 0:3, 0:3].shape)
print("x[0,:,100,200].shape:", x[0, :, 100, 200].shape)
print("x[0,0,100,200].shape:", x[0, 0, 100, 200].shape)

conv = nn.Conv2d(
    in_channels=3,
    out_channels=64,
    kernel_size=3,
    stride=1,
    padding=1
)

y = conv(x)

print("\n===== Conv 前后 =====")
print("x.shape:", x.shape)
print("y.shape:", y.shape)
print("conv.weight.shape:", conv.weight.shape)
print("x 某位置:", x[0, :, 100, 200].shape)
print("y 同一位置:", y[0, :, 100, 200].shape)
```

---

# 8. 母题

## 母题 1：整数索引还是切片？

已知：

```python
x = torch.randn(8, 3, 224, 224)
```

求：

```python
x[:,1].shape
x[:,1:2].shape
```

推导：

```text
x[:,1]
→ C 用整数索引
→ C 维消失
→ [8,224,224]

x[:,1:2]
→ C 用切片
→ C 维保留，长度变成 1
→ [8,1,224,224]
```

这道母题训练：

> **看到整数索引先检查“哪一维会消失”；看到切片先检查“这一维还保不保留”。**

---

## 母题 2：固定空间位置后还剩什么？

已知：

```python
x = torch.randn(8, 3, 224, 224)
```

求：

```python
x[0,:,100,100].shape
```

推导：

- B 用整数索引 → 消失
- C 用 `:` → 保留
- H 用整数索引 → 消失
- W 用整数索引 → 消失

所以：

```text
shape = [3]
```

这道母题训练：

> **固定 B/H/W 后，只剩 C，因此一个空间位置对应一个 C 维向量。**

---

## 母题 3：四个维度全部用整数索引

```python
x[0,2,100,100].shape
```

B、C、H、W 都被取成具体元素，因此最后没有任何维度：

```text
torch.Size([])
```

这道母题训练：

> **标量 Tensor 的 shape 是 `[]`，不是 `[1]`。**

---

## 母题 4：从 RGB 到 64 个 feature channels

```python
conv = nn.Conv2d(3, 64, 3, padding=1)
x = torch.randn(1, 3, 640, 640)
y = conv(x)
```

Shape：

```text
x: [1,3,640,640]
y: [1,64,640,640]
```

同一空间位置：

```text
x[0,:,100,200] → [3]
y[0,:,100,200] → [64]
```

这道母题训练：

> **看到 `out_channels=64`，想到输出每个空间位置被描述成 64 维特征。**

---

## 母题 5：看懂 `conv.weight.shape`

```python
Conv2d(3,64,3)
```

对应：

```text
weight.shape = [64,3,3,3]
```

含义：

```text
64 个 filter
每个 filter 覆盖 3 个输入通道
空间大小 3×3
```

这道母题训练：

> **看到 weight 四维，立刻按 `[C_out,C_in,K_h,K_w]` 解释。**

---

# 9. 错题本

## 错题 1：把 `x[:,0]` 和 `x[:,0:1]` 当成一样的 Shape

**错误理解：**

> 都只取第 0 个通道，所以 Shape 应该一样。

**正确理解：**

`0` 是整数索引，会移除 C 维；`0:1` 是切片，会保留 C 维。

```text
x[:,0]   → [B,H,W]
x[:,0:1] → [B,1,H,W]
```

**记忆：**

> **整数“拿出一个”，切片“保留一个范围”。**

---

## 错题 2：把 `[1,3,640,640]` 理解成“有 1×3×640×640 个维度”

**错误理解：**

> Shape 里的数字很多，所以维度数也是这些数字相乘。

**正确理解：**

```text
[1,3,640,640]
```

只有 4 个维度，分别是 B、C、H、W；这些数字表示每一维有多大。

**记忆：**

> **shape 看“每一维多大”，ndim 看“总共有几维”。**

---

## 错题 3：把 H/W 的方向记反

**正确理解：**

```text
H = 行 = 上下
W = 列 = 左右
```

**记忆：**

> **先行后列，也就是先 H 后 W。**

---

## 错题 4：把 CNN 的 64 channel 理解成 64 种颜色

**错误理解：**

> RGB 有 3 个 channel，所以 64 channel 可能是 64 种颜色。

**正确理解：**

RGB channel 是输入颜色；CNN feature channel 是卷积学习出来的特征响应。

**记忆：**

> **输入 channel 看颜色，深层 channel 看特征。**

---

## 错题 5：认为 `x[0,0,100,200]` 的 shape 是 `[1]`

**正确理解：**

B/C/H/W 四维全部被整数索引移除，结果是 0 维 Tensor：

```text
torch.Size([])
```

**记忆：**

> **所有维度都被“点名取走”后，只剩一个数，不剩维度。**

---

## 错题 6：把 `x.ndim` 写成 `x.ndim()`

**正确写法：**

```python
x.ndim
```

`ndim` 是属性，不是函数。

---

# 10. 重点掌握清单

## 必须会解释

- [ ] B、C、H、W 分别是什么
- [ ] RGB channel 与 feature channel 的区别
- [ ] 整数索引与切片的区别
- [ ] 一个空间位置为什么对应一个 C 维向量
- [ ] `shape / ndim / numel / dtype / device` 的区别

## 必须会推导

- [ ] `x[0].shape`
- [ ] `x[:,0].shape`
- [ ] `x[:,0:1].shape`
- [ ] `x[0,:,100,200].shape`
- [ ] `x[0,0,100,200].shape`
- [ ] `Conv2d(3,64,3,padding=1)` 前后的 C 变化

## 必须会看代码判断

- [ ] 看到 `:` 知道该维全部保留
- [ ] 看到整数索引知道对应维可能消失
- [ ] 看到 `0:1` 知道这是切片
- [ ] 看到 `out_channels=64` 知道输出 C=64
- [ ] 看到 `conv.weight.shape` 能按 `[C_out,C_in,K_h,K_w]` 解读

---

# 11. 触发式记忆

看到：

```python
x[:,0]
```

想到：

> C 用整数索引，C 维消失。

看到：

```python
x[:,0:1]
```

想到：

> C 用切片，C 维保留。

看到：

```python
x[b,:,h,w]
```

想到：

> 固定一张图和一个空间位置，留下一个 C 维向量。

看到：

```python
Conv2d(3,64,3)
```

想到：

> 输入 3 通道，输出 64 个 feature channels。

看到：

```text
[64,3,3,3]
```

想到：

> `[C_out,C_in,K_h,K_w]`。

---

# Day 1 最终必须牢牢记住

```text
HWC
→ PyTorch 常整理成 BCHW
```

```text
x[B,C,H,W]
```

```text
整数索引
→ 维度通常消失

切片
→ 维度通常保留
```

```text
RGB：
C=3，表示颜色

CNN：
C=64/256/...
表示特征
```

最关键的一句话：

> **Day 1 真正要练成的不是记住几个索引结果，而是看到一行 Tensor 操作时，先判断“它动了哪一维、哪一维被保留、哪一维会消失”，再预测最终 Shape。**
