# tao data tu function
import torch 

def test_func(x, y):
    return torch.exp(torch.sin(torch.pi * x) + y ** 2)

def from_function_to_dataset(function, data_range = [-1, 1], data_dim = 1, num_of_data = 2000):
    """
        Remember: function always calculated with batch !!!
    """    
    x = torch.rand(num_of_data, data_dim) * (data_range[1] - data_range[0]) + data_range[0]
    y = test_func(*x.T)
    return x.to(torch.float), y.to(torch.float)

data_raw = from_function_to_dataset(function = test_func, data_dim = 2)
print(data_raw)
    