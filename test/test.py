import torch

a = torch.Tensor([[[1, 2, 5], [3, 4, 1]], [[4, 5, 7], [6, 7, 0]]])
b = a.reshape(-1, 2 * 3)
print(b)
b = a.reshape(-1, 2, 3)
print(b)
print(torch.sum(b, dim = 1))