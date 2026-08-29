import torch

# .默认是 float32 浮点数类型
x = torch.tensor(
    [[[
        [1., 2., 3., 4., 5.],
        [6., 7., 8., 9., 10.],
        [11., 12., 13., 14., 15.],
        [16., 17., 18., 19., 20.],
        [21., 22., 23., 24., 25.]
    ]]]
)

print("x.shape =", x.shape)
print("\n")
print("x=", x)
print("\n")

# 单通道卷积 in_channels=1
# 定义一个 3*3 卷积核
conv = torch.nn.Conv2d(
    in_channels=1,
    out_channels=1,
    kernel_size=3,
    stride=1,
    padding=0,
    bias=False
)

print("conv.weight.shape =", conv.weight.shape)
print("\n")

# 初始化卷积核权重为全 1
with torch.no_grad():
    conv.weight[:] = torch.tensor(
        [[[
            [1., 1., 1.],
            [1., 1., 1.],
            [1., 1., 1.]
        ]]]
    )

# 进行卷积操作
y = conv(x)

print("y=", y)
print("\n")
print("y.shape =", y.shape)

print("\n===== 自动求导信息 =====")

print("x.requires_grad =", x.requires_grad)
print("conv.weight.requires_grad =", conv.weight.requires_grad)
print("y.requires_grad =", y.requires_grad)

print("x.grad_fn =", x.grad_fn)
print("conv.weight.grad_fn =", conv.weight.grad_fn)
print("y.grad_fn =", y.grad_fn)

print("\n===== 实验1：正常 forward =====")

y1 = conv(x)

print("conv.weight.requires_grad =", conv.weight.requires_grad)
print("y1.requires_grad =", y1.requires_grad)
print("y1.grad_fn =", y1.grad_fn)


print("\n===== 实验2：no_grad forward =====")

with torch.no_grad():
    y2 = conv(x)

print("conv.weight.requires_grad =", conv.weight.requires_grad)
print("y2.requires_grad =", y2.requires_grad)
print("y2.grad_fn =", y2.grad_fn)

print("\n===== 实验3：手动冻结权重 =====")

conv.weight.requires_grad_(False)

print("conv.weight.requires_grad =", conv.weight.requires_grad)

print("\n===== 实验4：改变 Conv2d 的参数 =====")

conv = torch.nn.Conv2d(
    1,
    1,
    kernel_size=3,
    stride=1,
    padding=0
)

z1 = conv(x)

print("padding=0 stride=1")
print(z1)
print("z1.shape =", z1.shape)

conv = torch.nn.Conv2d(
    1,
    1,
    kernel_size=3,
    stride=1,
    padding=1
)

z2 = conv(x)

print("\npadding=1 stride=1")
print(z2)
print("z2.shape =", z2.shape)

conv = torch.nn.Conv2d(
    1,
    1,
    kernel_size=3,
    stride=2,
    padding=1
)

z3 = conv(x)

print("\npadding=1 stride=2")
# print("conv.weight.shape =", conv.weight.shape)
print(z3)
print("z3.shape =", z3.shape)

print("\n===== 实验4：改变输入通道 in_channels =====")
# 3 输入通道
x = torch.randn(1, 3, 5, 5)

conv = torch.nn.Conv2d(
    in_channels=3,
    out_channels=1,
    kernel_size=3,
    bias=False
)

print("conv.weight.shape =", conv.weight.shape)
# print("conv.bias =", conv.bias)

conv = torch.nn.Conv2d(
    in_channels=3,
    out_channels=64,
    kernel_size=3
)

print("conv.weight.shape =", conv.weight.shape)
# print("conv.bias =", conv.bias)
print("conv.bias.shape =", conv.bias.shape)

conv = torch.nn.Conv2d(
    3,
    64,
    kernel_size=3,
    bias=True
)

print("conv.weight.shape =", conv.weight.shape)
# print("conv.bias =", conv.bias)
print("conv.bias.shape =", conv.bias.shape)
