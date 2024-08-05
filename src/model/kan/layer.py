import torch
from torch import nn
from .activations import *

class KANLayer(nn.Module):
    def __init__(self, num_in_node, num_out_node, k = 3, G = 5, b = nn.SiLU(), default_grid_range = [-1, 1]) -> None:
        super(KANLayer, self).__init__()
        self.num_in_node = num_in_node
        self.num_out_node = num_out_node
        self.size = num_in_node * num_out_node
        
        self.k = k
        self.G = G
        self.b = b
        self.default_grid_range = default_grid_range
    
        self.scale_b = nn.Parameter(torch.ones(self.size, 1, ), requires_grad = True)
        
        self.scale_spline = nn.Parameter(torch.ones(self.size, 1), requires_grad = True)
        
        # self.knots: (size, G + 1)
        self.knots = nn.Parameter(torch.linspace(default_grid_range[0], default_grid_range[1], G + 1).unsqueeze(0).repeat(self.size, 1), requires_grad=False)

        self.noise = torch.rand(self.size, G + 1)
        # self.coef: (size, G + k)
        self.coef = nn.Parameter(curve_to_coef(self.knots, self.noise, self.knots, k), requires_grad = True)
        
        self.mask = torch.ones(size=(self.size, ))

        
    def forward(self, x):
        
        # x : (batch, num_in_node)
        # pre_acts : (size, batch)
        pre_acts = x.unsqueeze(1).repeat(1, self.num_out_node, 1).reshape(-1, self.size).T
        # after here all is (, batch)
        y = coef_to_curve(x_eval=pre_acts, grid=self.knots, coef=self.coef, k=self.k)
        y = y * self.mask.unsqueeze(1).repeat(1, y.shape[1])
        post_splines = y
        #print(self.scale_b)
        y = self.scale_b.repeat(1, x.shape[0]) * self.b(pre_acts) + self.scale_spline.repeat(1, x.shape[0]) * y
        
        post_acts = y
        y = y.reshape(self.num_in_node, self.num_out_node, -1).permute(2, 0, 1)
        # y : (batch, num_out_node)
        y = torch.sum(y, dim = 1)
        return y, pre_acts.T, post_splines.T, post_acts.T
        
    def extend_grid(self, coarser_layer, x):
        """
            This function is used in case we want to change recent gird to new grid
        """
        
        x_eval = x.unsqueeze(2).repeat(1, self.num_out_node, 1).reshape(-1, self.size).T
        #new_grids_generator = KANLayer(num_in_node=1, num_out_node=self.size, k = 1, G = coarser_layer.G);
        # new_grids_generator.knots : (size, batch) (size, coarser_layer.G + 1)
        
        # new_grids_generator.coef.data = curve_to_coef(x_eval=new_grids_generator.knots, y_eval=coarser_layer.knots, grid=new_grids_generator.knots, k = 1, device=self.device)
        y_eval = coef_to_curve(x_eval=x_eval, grid=coarser_layer.knots, coef=coarser_layer.coef, k=coarser_layer.k)
        # input_grid = torch.linspace(-1, 1, self.G + 1).to(self.device)
        
        #grid_range = coarser_layer.knots.data[:, [0, -1]]
        
        #self.knots.data = torch.cat([grid_range[:, [0]] + i * (grid_range[:, [-1]] - grid_range[:, [0]]) / self.G for i in range(self.G + 1)], dim = 1).to(self.device)

        #self.update_grid_range(x_eval)
        # print(grid_range)
        # print(coarser_layer.knots.data)
        # print(x_eval.min(), x_eval.max(), "->\n" , y_eval.min(), y_eval.max(),"from\n", self.knots.data.min(), self.knots.data.max())
        # print("->end of grid")
        
        #self.knots.data = torch.sort(new_grids_generator(input_grid.unsqueeze(dim = 1))[0].T, dim = 1)[0]

        x_pos = torch.sort(x_eval, dim = 1)[0]
        grid_range = torch.cat([x_pos[:, [0]], x_pos[:, [-1]]], dim = 1)
        ids = [int(x_eval.shape[1] / self.G * i) for i in range(self.G)] + [-1]
        grid_adapt = x_pos[:, ids]
        grid_uniform = torch.cat([grid_range[:, [0]] - 0.01 + i * (grid_range[:, [-1]] - grid_range[:, [0]] + 0.02) / self.G for i in range(self.G + 1)], dim = 1)
        self.knots.data = grid_adapt * 0.98 + grid_uniform * 0.02

        self.coef.data = curve_to_coef(x_eval=x_eval, y_eval=y_eval, grid=self.knots, k=self.k)
        # print("->end of coef")
        
    def update_grid_range(self, x):
        """
            This function is used in case we want to update the grid range according to sample
        """
        # x: (batch, num_in_node)
        x_eval = x.unsqueeze(2).repeat(1, self.num_out_node, 1).reshape(-1, self.size).T
        y_eval = coef_to_curve(x_eval=x_eval, grid=self.knots, coef=self.coef, k=self.k)
        # x_pos: (size, batch)
        x_pos = torch.sort(x_eval, dim = 1)[0]
        grid_range = torch.cat([x_pos[:, [0]], x_pos[:, [-1]]], dim = 1)
        ids = [int(x_eval.shape[1] / self.G * i) for i in range(self.G)] + [-1]
        grid_adapt = x_pos[:, ids]
        grid_uniform = torch.cat([grid_range[:, [0]] - 0.01 + i * (grid_range[:, [-1]] - grid_range[:, [0]] + 0.02) / self.G for i in range(self.G + 1)], dim = 1)
        self.knots.data = grid_adapt * 0.98 + grid_uniform * 0.02
        self.coef.data = curve_to_coef(x_eval=x_eval, y_eval=y_eval, grid=self.knots, k=self.k)

    
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
