可以把“MiniCNN 逐行代码审查”做成一种固定训练法，不是只看“有没有报错”，而是每一行都回答 **5 个问题**：

> **这行是什么？输入是什么？输出是什么？Shape 怎么变？为什么下一步需要它？**

你现在这份 `MiniCNN` 很适合这样练。

## 先建立统一审查模板

以后看到任何一行代码，都按这 5 项检查：

1. **语法层**：这是什么对象/函数/方法？
2. **数据层**：输入 Tensor 是什么 shape？
3. **计算层**：这行内部做了什么？
4. **输出层**：输出 shape / 含义是什么？
5. **依赖层**：为什么下一行需要这个输出？

例如：

```python
x = self.conv1(x)
```

你要能回答：

```text
是什么：
调用 conv1 卷积层

输入：
[B, 3, 32, 32]

做什么：
8 个 [3,3,3] filter 在输入上滑动
每个位置做乘加
产生 8 张 feature maps

输出：
[B, 8, 32, 32]

为什么下一步需要：
ReLU 要对 Conv 得到的响应值做非线性激活
```

---

# 1. 从 import 开始审

```python
import torch
import torch.nn as nn
```

你要检查：

```text
torch
→ Tensor、随机输入、flatten 等基础操作

torch.nn
→ 神经网络层和 Module 基类

nn
→ torch.nn 的别名
```

然后问自己：

> 为什么不用 `torch.Conv2d`？

因为：

```python
Conv2d
ReLU
MaxPool2d
Linear
Module
```

都在：

```python
torch.nn
```

下面。

---

# 2. 审类定义

```python
class MiniCNN(nn.Module):
```

你要知道：

> `MiniCNN` 是一个自定义神经网络类，继承 `nn.Module`。

这里不是普通“数据类”，而是 PyTorch 模型的标准写法。

然后：

```python
def __init__(self):
    super().__init__()
```

要回答：

```text
__init__
→ 定义模型有哪些层

super().__init__()
→ 初始化父类 nn.Module
```

这一步很重要，因为 PyTorch 需要 `nn.Module` 去登记：

```text
参数
子模块
训练/推理状态
state_dict
.to(device)
.parameters()
```

---

# 3. 审第一层 Conv

```python
self.conv1 = nn.Conv2d(
    in_channels=3,
    out_channels=8,
    kernel_size=3,
    stride=1,
    padding=1,
    bias=False
)
```

你应该逐项说：

```text
in_channels=3
→ 输入是 RGB 三通道

out_channels=8
→ 有 8 个 filter
→ 输出 8 个 feature maps

kernel_size=3
→ 每个 filter 空间大小 3×3

stride=1
→ 每次移动 1 格

padding=1
→ 外围补 1 圈
→ 在 kernel=3, stride=1 时保持 H/W

bias=False
→ 不额外给每个输出 channel 加 bias
```

然后推 shape：

```text
输入：
[B,3,32,32]

输出：
[B,8,32,32]
```

再继续追问：

> `conv1.weight.shape` 是多少？

答案：

```text
[8,3,3,3]
```

因为：

```text
[C_out,C_in,kH,kW]
```

---

# 4. 审 ReLU

```python
self.relu1 = nn.ReLU()
```

你要回答：

```text
作用：
max(0,x)

输入：
[B,8,32,32]

输出：
[B,8,32,32]

shape：
不变

为什么需要：
给 Conv 的线性计算加入非线性
```

然后再说清：

```text
Conv
→ 产生响应

ReLU
→ 负值截成 0
→ 形成激活/不激活模式
```

---

# 5. 审 Pooling

```python
self.pooling1 = nn.MaxPool2d(
    kernel_size=2,
    stride=2
)
```

你要回答：

```text
kernel_size=2
→ 每次看 2×2

stride=2
→ 每次移动 2 格

操作：
每个 channel 独立取局部最大值

输入：
[B,8,32,32]

输出：
[B,8,16,16]
```

关键问：

> 为什么 C 还是 8？

因为：

```text
MaxPool2d
不跨 channel 混合
只压缩每个 channel 自己的 H/W
```

---

# 6. 第二层 Conv 要重点审“输入通道为什么是 8”

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

这里要特别问：

> 为什么 `in_channels=8`，不是 3？

因为上一层已经输出：

```text
[B,8,16,16]
```

所以第二层看到的不是 RGB，而是：

```text
8 个 feature channels
```

这就是 CNN“层层组合特征”的关键。

然后：

```text
输入：
[B,8,16,16]

单个 filter：
[8,3,3]

filter 数：
16

weight：
[16,8,3,3]

输出：
[B,16,16,16]
```

---

# 7. 第二组 ReLU + Pool

```python
x = self.relu2(x)
x = self.pooling2(x)
```

你应该直接推：

```text
[B,16,16,16]
↓ ReLU
[B,16,16,16]
↓ Pool
[B,16,8,8]
```

如果这一步还需要计算很久，说明 shape 推导还没形成条件反射。

---

# 8. 审 Flatten

```python
x = torch.flatten(x, start_dim=1)
```

这里要回答：

```text
输入：
[B,16,8,8]

start_dim=1：
从 C 维开始一直展平到最后

B 保留

16×8×8=1024

输出：
[B,1024]
```

然后一定要问：

> 为什么不是 `[B,16,64]`？

因为 `flatten(start_dim=1)` 是把：

```text
C
H
W
```

全部合并成一个维度，不是只合并 H/W。

---

# 9. 审 Linear

```python
self.fc = nn.Linear(
    in_features=16 * 8 * 8,
    out_features=10
)
```

要回答：

```text
in_features:
1024

out_features:
10
```

然后：

```text
fc.weight.shape:
[10,1024]

fc.bias.shape:
[10]
```

数据流：

```text
[B,1024]
↓ Linear
[B,10]
```

这里也要说清：

> `10` 本质是 `out_features`，因为这是分类器最后一层，所以这里被设计成 `num_classes=10`。

---

# 10. 审 forward 的职责

```python
def forward(self, x):
```

这里你要知道：

```text
__init__
→ 定义“有什么层”

forward
→ 定义“数据按什么顺序流过这些层”
```

这是非常重要的区别。

可以把：

```python
self.conv1 = ...
self.relu1 = ...
```

理解成：

> 准备工具。

而：

```python
x = self.conv1(x)
x = self.relu1(x)
```

理解成：

> 真正让数据流过这些工具。

---

# 11. 为什么是 `self.conv1(x)`？

你之前的注释是对的。

```python
x = self.conv1(x)
```

因为：

```text
self.conv1
→ MiniCNN 实例里的卷积模块
```

不能写：

```python
conv1(x)
```

除非当前作用域里真的有一个叫 `conv1` 的局部变量。

也不能写：

```python
x.conv1()
```

因为：

```text
x 是 Tensor
```

Tensor 没有 `conv1()` 方法。

---

# 12. 审 `logits = self.fc(x)`

这一行特别适合区分：

```text
模块
vs
数据
```

```python
self.fc
```

是：

```text
Linear 模块
```

而：

```python
logits
```

是：

```text
这个模块计算后的 Tensor
```

所以：

```text
self.fc
↓ 输入 x
↓
得到 logits
```

不是：

```text
fc = logits
```

---

# 13. 审 `return logits`

```python
return logits
```

你要问：

> 为什么 return logits，不 return x？

因为当前模型定义的最终目标就是输出：

```text
[B,10]
```

也就是分类原始分数。

其实如果你写：

```python
return x
```

而前一行已经：

```python
x = self.fc(x)
```

也可以。

但：

```python
logits = self.fc(x)
return logits
```

语义更清晰。

---

# 14. 审测试代码

```python
if __name__ == "__main__":
```

你应该知道：

> 只有直接运行当前 `.py` 文件时，下面的测试代码才执行。

如果未来别人：

```python
from 02_mini_CNN import MiniCNN
```

测试部分不会自动运行。

---

# 15. 审 dummy input

```python
dummy_input = torch.randn(1, 3, 32, 32)
```

解释：

```text
1
→ B，一张图片

3
→ RGB

32,32
→ H,W
```

这里不是实际图片，只是：

> 用随机数模拟一批输入，用来验证模型的数据流和 shape。

---

# 16. 审实例化

```python
model = MiniCNN()
```

这一步：

```text
调用 __init__
↓
创建 conv1
relu1
pooling1
conv2
relu2
pooling2
fc
```

模型结构此时已经存在。

但：

> 数据还没通过网络。

---

# 17. 审真正的 forward

```python
output = model(dummy_input)
```

这行非常关键。

你表面上没有写：

```python
model.forward(dummy_input)
```

但 PyTorch 的 `nn.Module` 会让：

```python
model(dummy_input)
```

最终触发模型的：

```python
forward(dummy_input)
```

学习阶段你可以理解成：

```text
model(dummy_input)
↓
进入 forward
↓
数据按你定义的流程流动
↓
return logits
↓
赋值给 output
```

以后不要养成直接手动调用：

```python
model.forward(x)
```

的习惯，正常写：

```python
model(x)
```

即可。

---

# 18. 最终你要做“纸上执行”

逐行审查的最后一步，不看运行结果，自己写一张表：

| 执行位置          | Shape          |
| ------------- | -------------- |
| `dummy_input` | `[1,3,32,32]`  |
| `conv1`       | `[1,8,32,32]`  |
| `relu1`       | `[1,8,32,32]`  |
| `pooling1`    | `[1,8,16,16]`  |
| `conv2`       | `[1,16,16,16]` |
| `relu2`       | `[1,16,16,16]` |
| `pooling2`    | `[1,16,8,8]`   |
| `flatten`     | `[1,1024]`     |
| `fc`          | `[1,10]`       |
| `return`      | `[1,10]`       |

然后再运行程序，核对预测和真实结果。

---

# 我建议你这次不要让我直接审完

最有学习价值的方式是你自己做第一轮。

拿你现在的 `02_mini_CNN.py`，给每一个关键行加这种注释：

```python
x = self.conv1(x)

# 1. 输入：
# [B,3,32,32]

# 2. 当前层：
# Conv2d(3,8,3,stride=1,padding=1)

# 3. 内部计算：
# 8 个 [3,3,3] filter 扫描输入
# 每个位置逐元素乘法 + 求和

# 4. 输出：
# [B,8,32,32]

# 5. 下一步为什么需要：
# ReLU 对卷积响应加入非线性
```

然后所有关键行都这样写：

```text
conv1
relu1
pool1
conv2
relu2
pool2
flatten
fc
```

这就是你现阶段非常适合的**逐行代码审查训练**。

## 最后再加一层“找 Bug”审查

完成上面的理解审查后，再检查：

```text
① 代码能不能运行？
② 每层输入输出 channel 能否接上？
③ H/W 推导是否正确？
④ Flatten 后维度是否等于 Linear.in_features？
⑤ 最终 out_features 是否符合任务类别数？
⑥ forward 有没有漏掉定义过的层？
⑦ 有没有变量名/对象类型理解错误？
⑧ 如果输入从 32×32 改成 64×64，会在哪里报错？
```

最后这个问题尤其重要。

你的模型目前：

```python
self.fc = nn.Linear(16 * 8 * 8, 10)
```

是**绑定 32×32 输入尺寸的**。

如果输入突然改成：

```text
[B,3,64,64]
```

经过两次 Pool 后：

```text
[B,16,16,16]
```

Flatten 就会得到：

```text
[B,4096]
```

但 `fc` 还期待：

```text
1024
```

于是会在 Linear 这里报 shape mismatch。

如果你能自己定位到这里，就说明你已经开始具备真正的 **“看数据流找 Bug”** 能力了。
