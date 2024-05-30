# tao data tu function
import numpy as np

def test_func(x, y):
    return np.exp(np.sin(np.pi * x) + y * y)

def from_function_to_dataset(function, data_range = [-2, 2], data_dim = 1, num_of_data = 10000):
    """
        Remember: function always calculated with batch !!!
    """    
    x = np.random.rand(num_of_data, data_dim) * (data_range[1] - data_range[0]) + data_range[0]
    y = test_func(*x.T)
    return x, y

data_raw = from_function_to_dataset(function = test_func, data_dim = 2)
print(data_raw)
    