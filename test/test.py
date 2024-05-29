import torch

a = torch.Tensor([[1, 2], [2, 3]]).unsqueeze(2)
b = a.repeat(1, 1, 3).reshape(-1, 2 * 3)
d = a[:, (0, -1)]
print(d[0])
c = torch.nn.SiLU()(b)
print(c)