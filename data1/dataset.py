# tao dataset
import torch
from .gen_data import *

from torch.utils.data import Dataset, DataLoader, random_split;

class Data(DataLoader):
    def __init__(self, x_data, y_data):
        self.x = x_data
        self.y = y_data
    def __len__(self) -> int:
        return self.y.shape[0]
    def __getitem__(self, index):
        x = self.x[index]
        y = self.y[index]
        return x, y
    
data = Data(data_raw[0], data_raw[1])
train_data, test_data = random_split(data, [0.5, 0.5])
test_dataloader = DataLoader(test_data, batch_size = 64, shuffle = True);
train_dataloader = DataLoader(train_data, batch_size = 64, shuffle = True);