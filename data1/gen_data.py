import torch 

def test_func(x):
    return torch.exp(torch.sin(torch.pi * x[:, 0]) + x[:, 1] ** 2)

def from_function_to_dataset(function, data_range = [-1, 1], data_dim = 1, num_of_data = 4000):
    """
        Remember: function always calculated with batch !!!
        Args:
            function: function we want to gen data
            data_dim: number of variables in the input
            num_of_data: amount of data we want to gen
    """    
    x = torch.rand(num_of_data, data_dim) * (data_range[1] - data_range[0]) + data_range[0]
    y = test_func(*x.T) + (torch.rand(1)[0] - 0.5) / 1e7
    return x, y

data_raw = from_function_to_dataset(function = test_func, data_dim = 2)
    