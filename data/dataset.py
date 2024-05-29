# tao dataset
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from .gen_data import generate_data

class FunctionDataset(Dataset):
    def __init__(self, func, num_samples, range, noise_std):
        """
        Args:
            func (function): The function used to generate the data samples.
            num_samples (int): The number of data samples to generate.
            range (tuple): The range of the data samples.
            noise_std (float): The standard deviation of the noise.
        """
        self.x, self.y = generate_data(func, num_samples, range, noise_std)

    def plot(self):
        """
        Plot the data samples (1D only)
        """
        dimension = len(self.x[0])
        if dimension == 1:
            plt.figure()
            plt.scatter(self.x.squeeze(), self.y)
            plt.xlabel('x')
            plt.ylabel('y')
            plt.title('Data samples')
            plt.show()
        else:
            print('Cannot plot data samples with dimension greater than 2')
    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]
