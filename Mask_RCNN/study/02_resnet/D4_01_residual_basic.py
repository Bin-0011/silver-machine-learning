import torch

x = torch.tensor([
    [1., 2., 3.]
])

fx = torch.tensor([
    [0.1, -0.2, 0.5]
])

y = x + fx

print(y)