import torch
import activations
from torch import nn
from activations import *

class KANLayer(nn.Module):
    def __init__(self, num_in_node, num_out_node, k = 3, G = 5, b = nn.SiLU(), default_grid_range = [-1, 1], device = "cpu") -> None:
        super().__init__()
        self.num_in_node = num_in_node
        self.num_out_node = num_out_node
        self.size = num_in_node * num_out_node
        
        self.k = k
        self.G = G
        self.b = b
        self.default_grid_range = default_grid_range
        # self.knots: (size, G + 1)
        self.knots = torch.linspace(default_grid_range[0], default_grid_range[1], G + 1).repeat(self.size, 1);

        self.noise = torch.rand(self.size, G + 1)
        # self.coef: (size, G + k)
        self.coef = nn.Parameter(curve_to_coef(self.knots, self.noise, self.knots, k, device))
        
        self.device = device
    def forward(self, x):
        
        # x : (batch, num_in_node)
        pre_acts = x.unsqueeze(2).repeat(1, 1, self.num_out_node).reshape(-1, self.size).T
        # after here all is (, batch)
        y = coef_to_curve(pre_acts, self.knots, self.coef, self.k, self.device)
        post_splines = y
        y = self.b(pre_acts) + y
        post_acts = y
        # y : (num_out_node, batch)
        y = torch.sum(y.reshape(self.num_out_node, -1, self.num_in_node), dim = 2)
        return y.T, pre_acts.T, post_splines.T, post_acts.T
        
    def extend_grid(self, coarser_layer, x):
        """
            This function is used in case we want to change recent gird to new grid
        """
        
        x_val = x.unsqueeze(2).repeat(1, 1, self.num_out_node).reshape(-1, self.size).T
        new_grids_generator = KANLayer(1, self.size, k = self.k, G = coarser_layer.G);
        # new_grids_generator.knots : (size, batch) (size, coarser_layer.G + 1)
        new_grids_generator.coef.data = curve_to_coef(new_grids_generator.knots, coarser_layer.knots, new_grids_generator.knots, self.k, self.device)
        y_val = coef_to_curve(x_val, coarser_layer.knots, coarser_layer.coef, coarser_layer.k, self.device);
        input_grid = torch.linspace(-1, 1, self.G + 1).to(self.device)
        self.knots = new_grids_generator(input_grid.unsqueeze(dim = 1))[0].T
        self.coef.data = curve_to_coef(x_val, y_val, self.knots, self.k, self.device)
        
    def update_grid_range(self, x):
        """
            This function is used in case we want to update the grid range according to sample
        """
        # x: (batch, num_in_node)
        x_eval = x.unsqueeze(2).repeat(1, 1, self.num_out_node).reshape(-1, self.size).T
        y_eval = coef_to_curve(x_eval, self.knots, self.coef, self.k, self.device)
        # x: (size, batch)
        x_pos = torch.sort(x_eval, dim = 1)[0]
        grid_range = x_pos[:, [0, -1]]
        self.knots = torch.cat([grid_range[:, [0]] - 0.01 + i * (grid_range[:, [-1]] - grid_range[:, [0]] + 0.01) / self.G for i in range(self.G + 1)], dim = 1)
        self.coef.data = curve_to_coef(x_eval, y_eval, self.knots, self.k, self.device)
        
#----------------------Test space---------------------#
# a = KANLayer(3, 5)
# b = KANLayer(3, 5, G = 10)
# x = torch.linspace(-1, 1, 3).repeat(1, 64)
# y = a(x)
# print(y[0].shape, y[1].shape, y[2].shape, y[3].shape)
# b.extend_grid(a, x)
# print(b.knots.shape, b.coef.data.shape)
# a.update_grid_range(x)
# print(a.knots)
#------------------------------------------------------#

# init
# forward
# extend_grid (mo rong grid, fit grid cu vao grid moi)
# update_grid_range (mo rong grid size theo sample)
# remain_node (giu lai cac node quan trong)
# unlock(tam thoi bo qua)
# lock(tam thoi bo qua)