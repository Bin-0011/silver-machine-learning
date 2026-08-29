# Day 1：图片、Tensor 与 Shape

Day 1 是整条 Mask R-CNN 学习路线的起点。后面的 Conv2d、ResNet、FPN、RPN、RoIAlign 都会不断处理 Tensor，所以今天先把“图片进入 PyTorch 后到底长什么样”弄清楚。

这一课不追求复杂模型，只建立四个基础能力：

- 看懂 `[B, C, H, W]`
- 看懂 Tensor 索引
- 能在运行前预测索引后的 shape
- 分清 RGB channel 与 CNN feature channel

---

## 1. 从普通图片到 PyTorch Tensor

普通图片常见的数据排列是 `H × W × C`。例如一张 640×640 的 RGB 图片可以写成 `640 × 640 × 3`。

- `H`：Height，高度，对应“行”
- `W`：Width，宽度，对应“列”
- `C`：Channel，通道

RGB 图片中 `C=3`，三个通道分别是 `R、G、B`。因此一个像素位置不是一个数字，而是三个数字，例如 `[R,G,B]`。

PyTorch 的 `Conv2d` 常使用 `B × C × H × W`，多出来的 `B` 是 Batch，表示一次送进网络多少张图片。

例如：

```python
x = torch.randn(1, 3, 640, 640)
```

得到：

```text
x.shape = [1, 3, 640, 640]
```

含义可以直接读成：

| 维度 | 含义 | 当前值 |
|---|---|---:|
| B | 一次处理几张图片 | 1 |
| C | 每张图片有几个通道 | 3 |
| H | 每张图片有多少行 | 640 |
| W | 每张图片有多少列 | 640 |

因此 `x[B,C,H,W]` 可以理解成：

`x[第几张图片, 第几个通道, 第几行, 第几列]`

例如 `x[0,1,100,200]` 表示：第 0 张图片、第 1 个通道、第 100 行、第 200 列上的一个具体数值。

---

## 2. `torch.randn()`：先造一张“假图片”练结构

`torch.randn(1, 3, 640, 640)` 不会读取真实图片，而是创建一个 shape 为 `[1,3,640,640]` 的随机 Tensor。

`torch.randn()` 生成的是标准正态分布随机数，所以里面可能有正数，也可能有负数。今天使用它的目的只是模拟图片结构，让我们专心练 shape 和索引。

容易混淆的是：

| 函数 | 数值特点 |
|---|---|
| `torch.randn()` | 标准正态分布，可以出现负数 |
| `torch.rand()` | `[0,1)` 均匀分布，都是非负数 |

---

## 3. Tensor 索引：先看“位置”，再看“规则”

假设：

```python
x = torch.randn(1, 3, 640, 640)
```

它的四个索引位置固定对应：

```text
x[B, C, H, W]
```

例如：

```python
x[:, 0]
```

完整理解是：

```python
x[:, 0, :, :]
```

其中：

- 第 1 个位置 `:` 对应 B，表示所有图片都保留
- 第 2 个位置 `0` 对应 C，表示只取第 0 个通道
- 后面的 H、W 没写，默认全部保留

所以 `x[:,0]` 表示所有图片的第 0 个通道。对于 RGB 输入，可以把 `0、1、2` 分别看成 `R、G、B`。

---

## 4. 整数索引和切片：为什么一个会“掉维”，一个不会

这是 Day 1 最重要的索引规则。

### 整数索引

```python
x[:, 0]
```

这里在 C 维使用了整数 `0`，意思是“取这个维度中的一个具体元素”。这个维度会被移除：

```text
[1,3,640,640]
→ x[:,0]
[1,640,640]
```

### 切片索引

```python
x[:, 0:1]
```

`0:1` 是切片，表示在 C 维取一个范围 `[0,1)`。虽然里面只有一个通道，但 C 这一维仍然保留：

```text
[1,3,640,640]
→ x[:,0:1]
[1,1,640,640]
```

可以先用这句记：

> **整数索引：取一个具体元素，对应维度通常消失；切片：取一个范围，对应维度通常保留。**

---

## 5. 从 Tensor 中取一个 3×3 小区域

看这一行：

```python
x[0, 0, 0:3, 0:3]
```

逐维翻译：

- `B=0`：第 0 张图片，整数索引，B 维消失
- `C=0`：第 0 个通道，整数索引，C 维消失
- `H=0:3`：第 0～2 行，切片，H 维保留，长度变成 3
- `W=0:3`：第 0～2 列，切片，W 维保留，长度变成 3

因此输出 shape 是：

```text
[3,3]
```

它就是第 0 张图、第 0 个通道左上角的 3×3 小块。

---

## 6. 一个空间位置为什么会得到一个 Channel 向量

再看：

```python
x[0, :, 100, 200]
```

这里固定了：

- 第 0 张图片
- 第 100 行
- 第 200 列

只把 C 维全部保留下来，因此结果 shape 是：

```text
[3]
```

对于 RGB 图片，这 3 个数字就是该位置的 `[R,G,B]`。

所以要开始建立这个认识：

> **固定一个空间位置 `(H,W)` 后，这个位置上会有一个 C 维向量。**

RGB 输入时 `C=3`，所以是 3 维；经过 CNN 后，C 可能变成 64、256、512，此时同一个空间位置就会变成更高维的 feature vector。

---

## 7. RGB 的 3 个 channel 与 CNN 的 64 个 channel

RGB 的 3 个 channel 来自图片本身：

```text
R
G
B
```

但 CNN 中的 64 个 channel 是网络学习得到的 64 个特征通道，不是 64 种颜色。

例如：

```python
conv = torch.nn.Conv2d(
    in_channels=3,
    out_channels=64,
    kernel_size=3,
    stride=1,
    padding=1
)

y = conv(x)
```

输入：

```text
[B,3,H,W]
```

输出：

```text
[B,64,H,W]
```

于是同一个空间位置的描述从：

```text
[R,G,B]
```

变成：

```text
[f1,f2,...,f64]
```

这里的 `f1～f64` 可以先理解为网络学到的不同特征响应。

---

## 8. `Conv2d(3,64,3)` 与 `conv.weight.shape`

Day 1 只需要先认识这个关系，具体卷积怎么算放到 Day 2。

```python
conv = torch.nn.Conv2d(
    in_channels=3,
    out_channels=64,
    kernel_size=3
)
```

参数含义：

- `in_channels=3`：输入有 3 个通道
- `out_channels=64`：输出 64 个 feature channels
- `kernel_size=3`：空间窗口是 3×3

打印：

```python
print(conv.weight.shape)
```

会得到：

```text
[64,3,3,3]
```

PyTorch 的排列是：

```text
[C_out, C_in, K_h, K_w]
```

因此：

- 整层有 64 个 filter
- 每个 filter 的主要权重 shape 是 `[3,3,3]`
- 一个 filter 会同时覆盖全部 3 个输入通道

---

## 9. Tensor 调试时常看的几个属性

以后调试深度学习代码，经常会检查这些：

| 属性/函数 | 回答什么问题 |
|---|---|
| `x.shape` | 每一维分别多大 |
| `x.ndim` | 一共有几个维度 |
| `x.numel()` | 一共有多少个元素 |
| `x.dtype` | 每个元素是什么数据类型 |
| `x.device` | Tensor 在 CPU 还是 GPU |

例如：

```python
print(x.shape)
print(x.ndim)
print(x.numel())
print(x.dtype)
print(x.device)
```

对于 `[1,3,640,640]`，`ndim=4`。这里的“4”表示一共有 B、C、H、W 四个维度，不是元素总数。

---

## 10. 真实图片中的 HWC → CHW

许多图片工具得到的数据常见排列是：

```text
HWC
```

而 PyTorch 单张图片常用：

```text
CHW
```

所以项目中经常看到：

```python
image = image.permute(2, 0, 1)
```

表示：

```text
HWC → CHW
```

再增加 Batch 维后才是：

```text
BCHW
```

Day 1 只需要知道这种转换存在，`permute()` 的细节以后再学。

---

## 11. 今日完整练习代码

```python
import torch
import torch.nn as nn


# 1. 模拟 1 张 640×640 RGB 图片
x = torch.randn(1, 3, 640, 640)

print("完整 Tensor shape:", x.shape)
print("ndim:", x.ndim)
print("numel:", x.numel())
print("dtype:", x.dtype)
print("device:", x.device)


# 2. 取 RGB 三个通道
r_channel = x[:, 0]
g_channel = x[:, 1]
b_channel = x[:, 2]

print("\nR 通道 shape:", r_channel.shape)
print("G 通道 shape:", g_channel.shape)
print("B 通道 shape:", b_channel.shape)


# 3. 取 R 通道，但保留 C 维
r_channel_keep_dim = x[:, 0:1]
print("\n保留 C 维的 R 通道 shape:", r_channel_keep_dim.shape)


# 4. 第 0 张图、第 0 通道左上角 3×3
patch = x[0, 0, 0:3, 0:3]
print("\n左上角 3×3:")
print(patch)
print("patch.shape:", patch.shape)


# 5. 同一个空间位置的全部 RGB 值
pixel_features = x[0, :, 100, 200]
print("\n(H=100, W=200) 的 RGB 向量:")
print(pixel_features)
print("shape:", pixel_features.shape)


# 6. 验证常见索引后的 shape
print("\n===== Shape 验证 =====")
print("x.shape:", x.shape)
print("x[0].shape:", x[0].shape)
print("x[:,0].shape:", x[:, 0].shape)
print("x[:,0:1].shape:", x[:, 0:1].shape)
print("x[0,:,100,200].shape:", x[0, :, 100, 200].shape)
print("x[0,0,100,200].shape:", x[0, 0, 100, 200].shape)


# 7. 模拟经过一层卷积
conv = nn.Conv2d(
    in_channels=3,
    out_channels=64,
    kernel_size=3,
    stride=1,
    padding=1
)

y = conv(x)

print("\n===== Conv2d 前后 =====")
print("x.shape:", x.shape)
print("y.shape:", y.shape)
print("conv.weight.shape:", conv.weight.shape)

print("x 某个位置:", x[0, :, 100, 200].shape)
print("y 同一个位置:", y[0, :, 100, 200].shape)
```

---

# Day 1 完成标准

- [ ] `B、C、H、W` 分别表示什么？
- [ ] 为什么 PyTorch 中图片常写成 `[B,C,H,W]`？
- [ ] `x[:,0]` 表示什么？为什么 C 维会消失？
- [ ] `x[:,0:1]` 表示什么？为什么 C 维会保留？
- [ ] `x[0,0,0:3,0:3]` 为什么得到 `[3,3]`？
- [ ] `x[0,:,100,200]` 为什么得到 `[3]`？
- [ ] `x[0,0,100,200]` 为什么得到 `torch.Size([])`？
- [ ] RGB 的 3 个 channel 与 CNN 的 64 个 feature channels 有什么区别？
- [ ] `Conv2d(3,64,3)` 中的 3、64、3 分别表示什么？
- [ ] `conv.weight.shape=[64,3,3,3]` 四维分别是什么？
- [ ] `shape`、`ndim`、`numel()`、`dtype`、`device` 分别表示什么？
- [ ] 看到一个简单 Tensor 索引时，能否先不运行代码，自己判断 shape？

# Day 1 最终必须牢牢记住

```text
普通图片常见：
HWC

PyTorch Conv2d 常见：
BCHW
```

```text
x[B,C,H,W]
=
x[第几张图, 第几个通道, 第几行, 第几列]
```

```text
整数索引
→ 取一个具体元素
→ 对应维度通常消失

切片索引
→ 取一个范围
→ 对应维度通常保留
```

```text
RGB 输入：
[B,3,H,W]

经过 Conv2d：

[B,64,H,W]
```

最关键的一句话：

> **RGB 的 3 个 channel 表示输入颜色信息；CNN 的 feature channels 表示网络学习得到的特征描述。先学会看懂 Tensor 的 Shape 和索引，后面才能真正追踪 CNN、ResNet 和 Mask R-CNN 的数据流。**
