import torch
import pytorch_lightning as pl
import torch.optim as optim
import rootutils

ROOTPATH = rootutils.setup_root(__file__, indicator=".project_root", pythonpath=True)

from src.model.kan.layer import KANLayer
from src.model.kan.utils import *
from torch import nn
from src.model.kan.activations import *
import matplotlib.pyplot as plt
from src.model.kan.sparsification_reg import LossWithSparsificationRegularization
        
class KAN(pl.LightningModule):
    def __init__(self, G = 5, k = 3, width = [2, 5, 1], b = nn.SiLU(), default_grid_range = [-1, 1], threshold = 5e-2) -> None:
        """
        Args:
            G: number of grids of an activation 
            k: the degree of an activation
            width: number of nodes
            b: basis function
            default_grid_range: range that activations function work
            threshold: threshold to prune
        """
        super(KAN, self).__init__()
        self.G = G
        self.k = k
        self.width = width
        self.layer = []
        self.bias = []
        self.threshold = threshold

        if b == 'SiLU':
            b = nn.SiLU()
        elif b == 'Sigmoid':
            b = nn.Sigmoid()
        elif b == 'Tanh':
            b = nn.Tanh()
        else:
            raise ValueError(f"Activation function {b} not recognized.")

        for i in range(len(width) - 1):
            one_layer = KANLayer(width[i], width[i + 1], k = k, G = G, b = b, default_grid_range = default_grid_range)
            self.layer.append(one_layer)
            bias = nn.Linear(width[i + 1], 1, bias=False).requires_grad_(True)
            self.bias.append(bias)
        self.layer = nn.ModuleList(self.layer)
        self.bias = nn.ModuleList(self.bias)

    def forward(self, x):
        self.acts_value = []
        self.acts_value.append(x)
        self.acts_scale = []
        for i in range(len(self.layer)):
            output = self.layer[i](x) 
            x = output[0] + self.bias[i].weight
            #x += self.bias[i].weight;
            self.acts_value.append(x)
            output_of_splines = torch.mean(torch.abs(output[3]), dim = 0);
            input_range = self.layer[i].knots.data[:, -1] - self.layer[i].knots.data[:, 0] + 1e-4
            self.acts_scale.append((output_of_splines / input_range).reshape(self.layer[i].num_in_node, self.layer[i].num_out_node))
        return x
    
    def initial_grid_from_other_model(self, model, x):
        """
        This function transfer the old model to the new one (try to 
        predict the same result when the same input is fed into model)
        The transmission is conducted base on a sample data
        Args:
            model: other model
            x: sample
            
        Example:
            new_model: KAN
            old_model: KAN
            x: sample-dataset
            ->
            new_model.initial_grid_from_other_model(old_model, x = x)
        """
        model(x)
        # self.forward(x)
        for i in range(len(self.layer)):
            coarser_grid = model.layer[i].knots.data
            #x_range = coarser_grid[:, [0, -1]]
            #x = torch.cat([(x_range[:, [0]] + i * (x_range[:, [-1]] - x_range[:, [0]]) / (self.G * 10)) for i in range(self.G * 10)], dim = 1).to(self.device)
            self.layer[i].extend_grid(model.layer[i], model.acts_value[i])
        for i in range(len(self.layer)): 
            self.layer[i].b = model.layer[i].b
            self.layer[i].scale_b = model.layer[i].scale_b
            self.layer[i].scale_spline = model.layer[i].scale_spline
            self.layer[i].mask = model.layer[i].mask
        self.bias = model.bias
        
    def update_grid_from_sample(self, x):
        """
        We would like to change the the position of grids to adapt
        to the distribution of sample, this function handle it
        
        Args:
            x: sample
        """
        for i in range(len(self.layer)):
            self.forward(x)
            self.layer[i].update_grid_range(self.acts_value[i])
    
    def plot(self):
        """
        This function plot all activation function of the model
        Example:
            new_model: KAN
            ->
            new_model.plot()
        """
        w = len(self.layer)
        h = 0
        for i in range(len(self.width) - 1):
            h = max(self.width[i] * self.width[i + 1], h)
        
        for i in range(w):
            x = torch.linspace(-3, 3, 100).unsqueeze(1).repeat(1, self.layer[i].num_in_node)
            y = self.layer[i](x)[2]
            for j in range(h):
                if j < y.shape[1]:
                    x_j = torch.linspace(self.layer[i].knots[j][0] - 1, self.layer[i].knots[j][-1] + 1, 100).detach().numpy()
                    y_j = y[:, j].flatten().detach().numpy()
                    plt.subplot(w, h, i * h + j + 1)
                    plt.plot(x_j, y_j)
                    plt.xlim(self.layer[i].knots[j][0] - 1, self.layer[i].knots[j][-1] + 1)
                    plt.ylim(-3, 3)

        plt.show()        
    
    def pruning(self):
        """
        This function in charge of pruning useless activation functions
        """
        threshold = self.threshold
        mask = [torch.ones((self.width[0], ))]
        for l in range(len(self.acts_scale) - 1):
            input_mask = torch.max(self.acts_scale[l], dim = 0)[0] > threshold
            output_mask = torch.max(self.acts_scale[l + 1], dim = 1)[0] > threshold
            overall_mask = input_mask * output_mask
            mask.append(overall_mask.float())
        mask.append(torch.ones((self.width[-1], )))
        
        for l in range(len(self.layer)):
            recent_mask = (mask[l].unsqueeze(1).repeat(1, mask[l + 1].shape[0]) * mask[l + 1].unsqueeze(0).repeat(mask[l].shape[0], 1)).reshape(-1)
            self.layer[l].mask = self.layer[l].mask * recent_mask

    def training_step(self, batch, batch_idx, stop_grid = 2):
        x, y = batch
        if self.current_epoch < stop_grid:
            self.update_grid_from_sample(x)

        y_pred = self(x).reshape(-1)
        loss = self.loss_func(y_pred, y)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss
    
    def on_train_end(self):
        self.pruning()

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x).reshape(-1)
        loss = self.loss_func(y_pred, y)
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x).reshape(-1)
        loss = self.loss_func(y_pred, y)
        self.log('test_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def configure_loss_func(self, loss_func='MSE', lamb=0., lamb_l1=1, lamb_entropy=2, lamb_l1_coef=0., lamb_l1_coef_diff=0.):
        if loss_func == 'MSE':
            loss_func = nn.MSELoss()
        else:
            raise ValueError(f"Loss function {loss_func} not recognized.")
        self.loss_func = LossWithSparsificationRegularization(model=self, loss_func=loss_func, lamb=lamb, lamb_l1=lamb_l1, lamb_entropy=lamb_entropy, lamb_l1_coef=lamb_l1_coef, lamb_l1_coef_diff=lamb_l1_coef_diff)

    def configure_optimizers(self, is_lbfgs=False, lr=1e-3):
        if is_lbfgs:
            self.is_lbfgs = True
            optimizer = optim.LBFGS(self.parameters(), lr=lr, history_size=10, line_search_fn="strong_wolfe", tolerance_grad=1e-32, tolerance_change=1e-32)
        else:
            self.is_lbfgs = False
            optimizer = optim.Adam(self.parameters(), lr=lr)
        return optimizer

    def optimizer_step(
        self, epoch, batch_idx, optimizer, optimizer_idx):
        if self.is_lbfgs:
            def closure():
                optimizer.zero_grad()
                loss = self.training_step(self.current_batch, batch_idx)
                loss.backward()
                return loss
            optimizer.step(closure)
        else:
            super().optimizer_step(epoch, batch_idx, optimizer, optimizer_idx)