import torch

def RMSE(yhat, y):
    return torch.sqrt(torch.mean(yhat - y) ** 2)