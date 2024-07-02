import torch

def generate_data(func, num_samples, ranges, noise_std):
    """
    Generate data samples with noise.

    Args:
        func (function): The function used to generate the data samples.
        num_samples (int): The number of data samples to generate.
        ranges (list of tuple): A list of ranges for each dimension.
        noise_std (float): The standard deviation of the noise.
        
    Returns:
        x (Tensor): The input data samples.
        data_with_noise (Tensor): The output data samples with noise.
    """
    # Generate data samples using the provided function
    x = []
    for range in ranges:
        a, b = range
        x.append(torch.rand(num_samples) * (b - a) + a)
    x = torch.stack(x, dim=1)
    data = func(x)
    # Add noise to the data samples
    noise = torch.normal(0, noise_std, size=data.shape)
    data_with_noise = data + noise

    return x, data_with_noise