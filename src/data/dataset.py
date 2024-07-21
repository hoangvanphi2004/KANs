# tao dataset
import matplotlib.pyplot as plt
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
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
            print('Cannot plot data samples with dimension greater than 1')
    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]
    
    def get_dataloader(self, batch_size, shuffle=True):
        return DataLoader(self, batch_size=batch_size, shuffle=shuffle)

class FunctionDatamodule(pl.LightningDataModule):
    def __init__(self, func, train_samples, val_samples, test_samples, range, noise_std, batch_size):
        super(FunctionDatamodule, self).__init__()
        self.func = func
        self.num_samples = {'train': train_samples, 'val': val_samples, 'test': test_samples}
        self.range = range
        self.noise_std = noise_std
        self.batch_size = batch_size

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            self.train = FunctionDataset(self.func, self.num_samples['train'], self.range, self.noise_std)
            self.val = FunctionDataset(self.func, self.num_samples['val'], self.range, self.noise_std)
        if stage == 'test' or stage is None:
            self.test = FunctionDataset(self.func, self.num_samples['test'], self.range, self.noise_std)

    def train_dataloader(self):
        return self.train.get_dataloader(self.batch_size)

    def val_dataloader(self):
        return self.val.get_dataloader(self.batch_size)
    
    def test_dataloader(self):
        return self.test.get_dataloader(self.batch_size)