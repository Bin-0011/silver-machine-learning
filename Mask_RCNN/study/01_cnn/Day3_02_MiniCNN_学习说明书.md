# Day 3-2：把组件组织成完整 MiniCNN

Day 3-1 已经分别学过：

`Conv → ReLU → Pooling → Flatten → Linear`

现在不再增加新的网络运算，而是学习另一件同样重要的事：

> **怎样把已经学过的层组织成一个真正的 PyTorch 模型。**

这一部分的重点从“单个层怎么算”转到：

`模型有哪些层 → 数据怎样依次流过 → 每层 Shape 怎样衔接 → 代码为什么这样组织 → 出错时在哪里查`

完整目标：

`[B,3,32,32] → Conv1 → ReLU1 → Pool1 → Conv2 → ReLU2 → Pool2 → Flatten → Linear → [B,10] logits`

---

## 1. MiniCNN 的整体结构

先只看数据流：

```text
输入
[B,3,32,32]

→ Conv1
[B,8,32,32]

→ ReLU1
[B,8,32,32]

→ Pool1
[B,8,16,16]

→ Conv2
[B,16,16,16]

→ ReLU2
[B,16,16,16]

→ Pool2
[B,16,8,8]

→ Flatten
[B,1024]

→ Linear
[B,10]

→ logits
```

这次的核心不是再学习 ReLU 或 Pooling 的数学，而是学会：

> **上一层的输出必须成为下一层能够接受的输入。**

---

## 2. `class MiniCNN(nn.Module)`：把模型写成一个模块

完整模型从：

```python
class MiniCNN(nn.Module):
```

开始。

这里第一次真正使用 PyTorch 的“模型类”写法。

### `nn.Module`

`nn.Module` 是 PyTorch 神经网络模块的基础类。

Conv、Linear 等层本身都是 Module；我们写的 `MiniCNN` 也继承它，就可以把多个子层组织成一个整体模型。

### `__init__()`

```python
def __init__(self):
```

负责：

> **定义模型里“有哪些层”。**

例如：

```python
self.conv1
self.relu1
self.pooling1
self.conv2
self.relu2
self.pooling2
self.fc
```

### `super().__init__()`

```python
super().__init__()
```

负责初始化父类 `nn.Module` 提供的模型能力。

当前阶段只需要知道：

> 自定义 PyTorch 模型时，这行是标准结构的一部分。

---

## 3. 第一组：Conv1 → ReLU1 → Pool1

定义：

```python
self.conv1 = nn.Conv2d(
    in_channels=3,
    out_channels=8,
    kernel_size=3,
    stride=1,
    padding=1,
    bias=False
)

self.relu1 = nn.ReLU()

self.pooling1 = nn.MaxPool2d(
    kernel_size=2,
    stride=2
)
```

Shape：

```text
[B,3,32,32]
→ Conv1
[B,8,32,32]
→ ReLU1
[B,8,32,32]
→ Pool1
[B,8,16,16]
```

这里 `conv1.weight.shape` 是：

```text
[8,3,3,3]
```

因为：

```text
[C_out,C_in,K_h,K_w]
```

---

## 4. 第二组为什么从 `in_channels=8` 开始

第二层：

```python
self.conv2 = nn.Conv2d(
    in_channels=8,
    out_channels=16,
    kernel_size=3,
    stride=1,
    padding=1,
    bias=False
)
```

为什么不是 `in_channels=3`？

因为上一组已经输出：

```text
[B,8,16,16]
```

所以第二层的输入已经不是 RGB，而是上一层得到的 8 个 feature channels。

于是：

```text
[B,8,16,16]
→ Conv2
[B,16,16,16]
→ ReLU2
[B,16,16,16]
→ Pool2
[B,16,8,8]
```

`conv2.weight.shape`：

```text
[16,8,3,3]
```

这个地方就是“层与层必须接得上”的典型例子。

---

## 5. 分类头为什么是 `Linear(16*8*8,10)`

第二组 Pooling 后：

```text
[B,16,8,8]
```

Flatten：

```text
[B,16×8×8]
=
[B,1024]
```

因此 Linear 必须写成：

```python
self.fc = nn.Linear(
    in_features=16 * 8 * 8,
    out_features=10
)
```

否则 Flatten 后的特征数与 `in_features` 对不上。

这里：

```text
fc.weight.shape = [10,1024]
fc.bias.shape   = [10]
```

---

## 6. `forward()`：真正规定数据流顺序

`__init__()` 只是准备好了所有层。

真正规定 Tensor 怎么走的是：

```python
def forward(self, x):
```

当前数据流：

```python
x = self.conv1(x)
x = self.relu1(x)
x = self.pooling1(x)

x = self.conv2(x)
x = self.relu2(x)
x = self.pooling2(x)

x = torch.flatten(x, start_dim=1)

logits = self.fc(x)

return logits
```

可以把两部分区分成：

```text
__init__
→ 有什么工具

forward
→ 数据怎样依次使用这些工具
```

---

## 7. 为什么写 `self.conv1(x)`

定义时：

```python
self.conv1 = nn.Conv2d(...)
```

`conv1` 是当前 `MiniCNN` 对象的一个属性。

所以调用时：

```python
x = self.conv1(x)
```

不能随便写成：

```python
x = conv1(x)
```

因为当前作用域不一定有一个独立变量 `conv1`。

也不能写：

```python
x.conv1()
```

因为 `x` 是 Tensor，它没有 `conv1()` 方法。

---

## 8. `self.fc` 和 `logits` 不一样

```python
self.fc = nn.Linear(...)
```

这里：

```text
self.fc
= 一个 Linear 模块
```

执行：

```python
logits = self.fc(x)
```

得到：

```text
logits
= Linear 计算后的 Tensor
```

因此：

> **fc 是计算模块，logits 是计算结果。**

---

## 9. `model = MiniCNN()` 与 `model(x)`

```python
model = MiniCNN()
```

会实例化模型，运行 `__init__()`，把各层创建出来。

这时候：

> 模型结构已经存在，但数据还没有经过网络。

真正前向传播：

```python
output = model(dummy_input)
```

PyTorch 的 `nn.Module` 会让：

```text
model(dummy_input)
→ 调用模型
→ 进入 forward(dummy_input)
→ 按 forward 顺序计算
→ return logits
→ 赋给 output
```

正常使用模型时写：

```python
model(x)
```

而不是主动写：

```python
model.forward(x)
```

---

## 10. `if __name__ == "__main__":`

完整脚本底部：

```python
if __name__ == "__main__":
```

可以先理解成：

> **只有直接运行当前 `.py` 文件时，下面的测试代码才执行。**

这样这个文件未来被其他文件 import 时，测试部分不会自动跑起来。

---

## 11. dummy input：先验证数据流

```python
dummy_input = torch.randn(1,3,32,32)
```

它不是实际图片，只是：

> 用随机数字模拟一张 32×32 RGB 图片，用来验证模型能否跑通以及 Shape 是否正确。

含义：

```text
1 → B
3 → C
32 → H
32 → W
```

---

## 12. 完整可运行 MiniCNN

```python
import torch
import torch.nn as nn


class MiniCNN(nn.Module):
    def __init__(self):
        super().__init__()

        # 第一组
        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=8,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
        )

        self.relu1 = nn.ReLU()

        self.pooling1 = nn.MaxPool2d(
            kernel_size=2,
            stride=2
        )

        # 第二组
        self.conv2 = nn.Conv2d(
            in_channels=8,
            out_channels=16,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
        )

        self.relu2 = nn.ReLU()

        self.pooling2 = nn.MaxPool2d(
            kernel_size=2,
            stride=2
        )

        # 分类头
        self.fc = nn.Linear(
            in_features=16 * 8 * 8,
            out_features=10
        )

    def forward(self, x):
        print("输入:", x.shape)

        x = self.conv1(x)
        print("conv1:", x.shape)

        x = self.relu1(x)
        print("relu1:", x.shape)

        x = self.pooling1(x)
        print("pool1:", x.shape)

        x = self.conv2(x)
        print("conv2:", x.shape)

        x = self.relu2(x)
        print("relu2:", x.shape)

        x = self.pooling2(x)
        print("pool2:", x.shape)

        x = torch.flatten(x, start_dim=1)
        print("flatten:", x.shape)

        logits = self.fc(x)
        print("logits:", logits.shape)

        return logits


if __name__ == "__main__":
    dummy_input = torch.randn(1,3,32,32)

    model = MiniCNN()

    output = model(dummy_input)

    print("最终输出:", output.shape)
```

---

## 13. 纸上执行：不运行先推 Shape

| 执行位置 | Shape |
|---|---|
| `dummy_input` | `[1,3,32,32]` |
| `conv1` | `[1,8,32,32]` |
| `relu1` | `[1,8,32,32]` |
| `pooling1` | `[1,8,16,16]` |
| `conv2` | `[1,16,16,16]` |
| `relu2` | `[1,16,16,16]` |
| `pooling2` | `[1,16,8,8]` |
| `flatten` | `[1,1024]` |
| `fc` | `[1,10]` |
| `return` | `[1,10]` |

以后看模型代码，先纸上推，再运行核对。

---

## 14. 逐行代码审查：固定问 5 件事

每一行重要代码都可以检查：

1. 这行是什么对象 / 函数 / 方法？
2. 输入 Tensor Shape 是什么？
3. 内部做了什么？
4. 输出 Shape / 含义是什么？
5. 下一步为什么需要这个输出？

例如：

```python
x = self.conv1(x)
```

应该能解释：

```text
对象：Conv2d 模块
输入：[B,3,32,32]
计算：8 个 [3,3,3] filter 做局部乘加
输出：[B,8,32,32]
下一步：ReLU 对卷积响应引入非线性
```

这套方法以后可以直接迁移到 ResNet、FPN、RPN 和 Mask R-CNN。

---

## 15. 输入尺寸从 32×32 改成 64×64 会怎样

当前：

```python
self.fc = nn.Linear(
    16 * 8 * 8,
    10
)
```

它绑定了前面最终 feature map 是：

```text
[B,16,8,8]
```

如果输入改成：

```text
[B,3,64,64]
```

两次 Pool 后：

```text
64×64
→ 32×32
→ 16×16
```

所以：

```text
[B,16,16,16]
```

Flatten：

```text
[B,4096]
```

但当前 fc 仍然要求：

```text
in_features=1024
```

因此会在：

```python
self.fc(x)
```

这里报 Shape mismatch。

调试时要形成这条链：

> **输入尺寸改变 → 继续推每层 H/W → 看最终 Flatten → 检查 Linear.in_features。**

---

# Day 3-2 完成标准

- [ ] `class MiniCNN(nn.Module)` 表示什么？
- [ ] `nn.Module` 在这里是什么？
- [ ] `__init__()` 负责什么？
- [ ] `forward()` 负责什么？
- [ ] `super().__init__()` 当前阶段需要怎样理解？
- [ ] 为什么 `conv2.in_channels=8`？
- [ ] `conv1.weight.shape` 为什么是 `[8,3,3,3]`？
- [ ] `conv2.weight.shape` 为什么是 `[16,8,3,3]`？
- [ ] 为什么分类头写 `Linear(16*8*8,10)`？
- [ ] 为什么调用层时写 `self.conv1(x)`？
- [ ] `self.fc` 和 `logits` 有什么区别？
- [ ] `model = MiniCNN()` 做了什么？
- [ ] `output = model(dummy_input)` 会触发什么？
- [ ] `if __name__ == "__main__":` 有什么作用？
- [ ] dummy input 的四维分别表示什么？
- [ ] 能否不运行代码推导整条 MiniCNN Shape？
- [ ] 能否对 `conv1/relu1/pool1/conv2/relu2/pool2/flatten/fc` 按 5 个问题逐行审查？
- [ ] 如果输入从 32×32 改成 64×64，能否定位错误发生在 Linear？
- [ ] 为什么 `Linear.in_features` 必须与 Flatten 后最后一维一致？

# Day 3-2 最终必须牢牢记住

```text
__init__
= 定义模型有哪些层
```

```text
forward
= 定义数据按什么顺序流动
```

```text
上一层输出 Shape
= 下一层输入条件
```

```text
MiniCNN：
[B,3,32,32]
→ [B,8,32,32]
→ [B,8,16,16]
→ [B,16,16,16]
→ [B,16,8,8]
→ [B,1024]
→ [B,10]
```

最关键的一句话：

> **写模型不是把层名堆在一起，而是建立一条能够从输入一路接到输出的数据流；每一层都必须同时检查“作用、输入、输出、Shape 和下一步依赖”。**
