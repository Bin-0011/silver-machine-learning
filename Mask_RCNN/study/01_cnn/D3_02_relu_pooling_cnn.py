import torch
import torch.nn as nn

# shape [1, 5]
x = torch.tensor([
    [-2., -1., 0., 1., 2.]
])

relu = nn.ReLU()

# shape [1, 5]
y = relu(x)

print("x:", x)
print("y:", y)

print("x.shape:", x.shape)
print("y.shape:", y.shape)


print("\n===== 2D Max Pooling =====")
# nn.MaxPool2d

x = torch.tensor([
    [[[
        1., 2., 3., 4.,
        ],
       [
        5., 6., 7., 8.,
       ],
       [
        9.,10.,11.,12.,
       ],
       [
        13.,14.,15.,16.
       ]]]
])

pool = nn.MaxPool2d(
    kernel_size=2,
    stride=2
)

y = pool(x)

print("x:", x)
print("y:", y)

print("x.shape:", x.shape)
print("y.shape:", y.shape)


print("\n===== 2D Convolution + ReLU + Max Pooling =====")
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

x = conv(x)
print("conv:", x.shape)

x = relu(x)
print("relu:", x.shape)

x = pool(x)
print("pool:", x.shape)

x = torch.flatten(x, start_dim=1)  # [B, C, H, W] -> [B, C*H*W]
print("flattened:", x.shape)

fc = nn.Linear(
    8 * 16 * 16,
    10
)

x = fc(x)
print("fc:", x.shape)
