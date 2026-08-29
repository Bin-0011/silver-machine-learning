# Day 3-2 总结：MiniCNN

## 1. 核心结构表

| 结构 | 作用 | 典型代码 |
|---|---|---|
| `nn.Module` | PyTorch 模型基础类 | `class MiniCNN(nn.Module)` |
| `__init__()` | 定义模型有哪些层 | `self.conv1=...` |
| `forward()` | 定义数据流顺序 | `x=self.conv1(x)` |
| `self.xxx` | 当前模型实例拥有的模块 | `self.fc` |
| `model(x)` | 执行模型前向传播 | 触发 `forward(x)` |
| dummy input | 模拟输入验证 Shape | `torch.randn(1,3,32,32)` |
| `return logits` | 返回最终模型输出 | `[B,10]` |

---

## 2. MiniCNN 完整数据流

```text
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

---

## 3. 完整代码

```python
import torch
import torch.nn as nn


class MiniCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(
            3, 8,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
        )

        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2,2)

        self.conv2 = nn.Conv2d(
            8, 16,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
        )

        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2,2)

        self.fc = nn.Linear(
            16 * 8 * 8,
            10
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)

        x = torch.flatten(x,start_dim=1)

        logits = self.fc(x)

        return logits


if __name__ == "__main__":
    x = torch.randn(1,3,32,32)

    model = MiniCNN()
    logits = model(x)

    print("输入:", x.shape)
    print("输出:", logits.shape)
```

---

## 4. Shape 与层职责表

| 位置 | Shape | 当前职责 |
|---|---|---|
| Input | `[B,3,32,32]` | RGB Tensor |
| Conv1 | `[B,8,32,32]` | 提取 8 个 feature channels |
| ReLU1 | `[B,8,32,32]` | 引入非线性 |
| Pool1 | `[B,8,16,16]` | H/W 减半 |
| Conv2 | `[B,16,16,16]` | 组合成 16 个 feature channels |
| ReLU2 | `[B,16,16,16]` | 引入非线性 |
| Pool2 | `[B,16,8,8]` | H/W 再减半 |
| Flatten | `[B,1024]` | 整理成 feature vector |
| Linear | `[B,10]` | 映射成 10 个输出 |
| logits | `[B,10]` | 原始类别分数 |

---

## 5. 关键 PyTorch 写法对照

| 写法 | 含义 |
|---|---|
| `class MiniCNN(nn.Module)` | 定义 PyTorch 模型 |
| `super().__init__()` | 初始化父类 Module |
| `self.conv1=...` | 把卷积层注册为模型子模块 |
| `def forward(self,x)` | 定义前向数据流 |
| `self.conv1(x)` | 让 x 通过 conv1 |
| `model=MiniCNN()` | 创建模型实例 |
| `model(x)` | 执行前向传播 |
| `return logits` | 返回最终输出 |
| `if __name__=="__main__"` | 直接运行文件时执行测试 |

---

## 6. 逐行审查模板

看到任意关键行，固定检查：

```text
1. 这是什么？
2. 输入 Shape？
3. 内部做什么？
4. 输出 Shape？
5. 下一步为什么需要它？
```

例如：

```python
x = self.pool1(x)
```

```text
是什么：MaxPool2d
输入：[B,8,32,32]
计算：每个 channel 独立对 2×2 区域取最大值
输出：[B,8,16,16]
下一步：Conv2 接收 8 个 feature channels
```

---

# 7. 母题

## 母题 1：为什么 Conv2 的 `in_channels=8`

因为上一层：

```text
Pool1 输出
[B,8,16,16]
```

所以 Conv2 必须接收 8 个输入 channels。

训练：

> **上一层 C_out = 下一层 C_in。**

---

## 母题 2：为什么 Linear 是 1024

Pool2 输出：

```text
[B,16,8,8]
```

Flatten：

```text
16×8×8=1024
```

所以：

```python
Linear(1024,10)
```

训练：

> **Linear.in_features 必须和 Flatten 后最后一维对齐。**

---

## 母题 3：`model(x)` 做了什么

```text
model(x)
→ nn.Module 调用机制
→ forward(x)
→ 按 forward 顺序执行
→ return logits
```

训练：

> **模型实例是可以像函数一样接收 Tensor 的。**

---

## 母题 4：32×32 改成 64×64

```text
64×64
→ Pool1 32×32
→ Pool2 16×16
→ [B,16,16,16]
→ Flatten [B,4096]
```

当前：

```python
Linear(1024,10)
```

不匹配。

训练：

> **输入尺寸变化要一路追踪到分类头。**

---

## 母题 5：找代码连接错误

假设：

```python
self.conv1 = nn.Conv2d(3,8,3,padding=1)
self.conv2 = nn.Conv2d(16,32,3,padding=1)
```

如果中间没有把 C 从 8 改成 16，那么 Conv2 会接不上。

训练：

> **每一层都检查上一层输出 C 是否等于下一层输入 C。**

---

# 8. 错题本

## 错题 1：认为 `__init__()` 就是在执行前向传播

**正确理解：**

```text
__init__ → 定义层
forward → 使用层处理数据
```

---

## 错题 2：写成 `x.conv1()`

**正确理解：**

```text
x = Tensor
self.conv1 = 模型里的 Conv2d 模块
```

所以：

```python
self.conv1(x)
```

---

## 错题 3：把 `self.fc` 和 logits 当成同一个东西

```text
self.fc = 模块
logits = 模块计算结果
```

---

## 错题 4：只看每一层能不能单独运行，不看层与层是否接上

需要检查：

```text
上一层输出 C
下一层输入 C
最终 Flatten
Linear.in_features
```

---

## 错题 5：认为输入图片尺寸变化只影响前面的 Conv

**正确理解：**

输入 H/W 会一路影响：

```text
Pool
→ 最终 feature map
→ Flatten
→ Linear.in_features
```

---

## 错题 6：直接调用 `model.forward(x)` 当成常规用法

正常使用：

```python
model(x)
```

让 `nn.Module` 的调用机制触发 forward。

---

# 9. 重点掌握清单

## 必须会解释

- [ ] `nn.Module`
- [ ] `__init__`
- [ ] `forward`
- [ ] `self`
- [ ] `model(x)`
- [ ] dummy input
- [ ] `return logits`

## 必须会推导

- [ ] MiniCNN 全部 Shape
- [ ] `conv1.weight.shape`
- [ ] `conv2.weight.shape`
- [ ] Flatten 后 1024
- [ ] Linear 输出 `[B,10]`
- [ ] 64×64 输入导致的 4096

## 必须会查 Bug

- [ ] C_in/C_out 是否接上
- [ ] H/W 是否推导正确
- [ ] Flatten 后是否等于 Linear.in_features
- [ ] forward 是否漏层
- [ ] 输入尺寸改变后哪一层最先报错

---

# 10. 触发式记忆

看到 `self.conv1=...`：

> 模型拥有一层 conv1。

看到 `x=self.conv1(x)`：

> 数据真正通过 conv1。

看到 `def forward(self,x)`：

> 数据流路线图。

看到 `model(x)`：

> 进入 forward。

看到 `Linear(16*8*8,10)`：

> 前面最终 feature map 必须是 `[B,16,8,8]`。

看到 `mat1 and mat2 shapes cannot be multiplied`：

> 优先检查 Flatten 与 Linear.in_features。

---

# Day 3-2 最终必须牢牢记住

```text
模型结构：
__init__

数据流：
forward
```

```text
上一层输出
必须能成为下一层输入
```

```text
MiniCNN：
Conv → ReLU → Pool
→ Conv → ReLU → Pool
→ Flatten
→ Linear
→ logits
```

最关键的一句话：

> **MiniCNN 是第一次把“单层知识”变成“完整模型数据流”：会写模型不仅要认识每个层，还要确保每一层在 Shape 和语义上都能顺利接到下一层。**
