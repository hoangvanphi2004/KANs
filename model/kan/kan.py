import torch
from .layer import KANLayer
from .utils import *
from torch import nn
from .activations import *
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

class KAN(nn.Module):
    def __init__(self, G = 5, k = 3, width = [2, 5, 1], b = nn.SiLU(), default_gird_range = [-1, 1], device = "cpu") -> None:
        super(KAN, self).__init__()
        self.G = G
        self.k = k
        self.width = width
        self.layer = []
        self.bias = []
        self.device = device
        for i in range(len(width) - 1):
            one_layer = KANLayer(width[i], width[i + 1], k = k, G = G, b = b, default_grid_range = default_gird_range, device = device)
            self.layer.append(one_layer)
            
            bias = nn.Linear(width[i + 1], 1, bias=False, device=device).requires_grad_(True)
            self.bias.append(bias)
        self.layer = nn.ModuleList(self.layer)
        self.bias = nn.ModuleList(self.bias)
        
    def forward(self, x):
        self.acts_value = []
        self.acts_value.append(x)
        for i in range(len(self.layer)):
            output = self.layer[i](x) 
            x = output[0] + self.bias[i].weight
            #x += self.bias[i].weight;
            self.acts_value.append(x)
        return x
    
    def initial_grid_from_other_model(self, model):
        # model(x)
        # self.forward(x)
        for i in range(len(self.layer)):
            coarser_grid = model.layer[i].knots.data
            x_range = coarser_grid[:, [0, -1]]
            x = torch.cat([(x_range[:, [0]] + i * (x_range[:, [-1]] - x_range[:, [0]]) / (self.G * 10)) for i in range(self.G * 10)], dim = 1).to(self.device)
            self.layer[i].extend_grid(model.layer[i], x)
        # for i in range(len(self.layer)): 
            # print(model.pre_acts[i].T, model.splines[i].T, model.layer[i].knots, self.layer[i].knots)
            # print(model.pre_acts[i].T.shape, model.splines[i].T.shape, self.layer[i].knots.shape)
            # print(self.layer[i].coef.data)
            # self.layer[i].coef.data = curve_to_coef(x_eval = model.pre_acts[i].T, y_eval = model.splines[i].T, grid=self.layer[i].knots, k=self.k, device=self.device)
            self.layer[i].b = model.layer[i].b
            self.layer[i].scale_b = model.layer[i].scale_b
            self.layer[i].scale_spline = model.layer[i].scale_spline
        self.bias = model.bias
            # print(self.layer[i].coef.data)
            # print(" stop please T^T ")
    def update_grid_from_sample(self, x):
        self.forward(x)
        for i in range(len(self.layer)):
            self.layer[i].update_grid_range(self.acts_value[i])
    
    def plot(self):
        w = len(self.layer)
        h = 0
        for i in range(len(self.width) - 1):
            h = max(self.width[i] * self.width[i + 1], h)
        
        for i in range(w):
            x = torch.linspace(-3, 3, 100).unsqueeze(1).repeat(1, self.layer[i].num_in_node).to(torch.double)
            y = self.layer[i](x)[2]
            for j in range(h):
                x_j = torch.linspace(-3, 3, 100).detach().cpu().numpy()
                if j < y.shape[1]:
                    y_j = y[:, j].flatten().detach().cpu().numpy()
                    plt.subplot(w, h, i * h + j + 1)
                    plt.plot(x_j, y_j)
                    plt.xlim(-1, 1)
                    plt.ylim(-2, 2)

        plt.show()        
            
    def train_model(self, data_loader, test_dataloader, optimizer, loss_func, epochs = 20, stop_grid = 2, is_LBFGS = False):
        size = len(data_loader.dataset)
        self.train()
        train_loss_list = []
        test_loss_list = []
        train_loss = 0
        cnt = 0;
        for t in range(epochs):
            print(f"Epoch {t + 1}\n -------------------------------");
            first = True
            pbar = tqdm(data_loader, desc='description', ncols=100)
            for batch, (X, y) in enumerate(pbar):
                if first:
                    first = False
                    
                if t < stop_grid:
                    self.update_grid_from_sample(X)
                    
                predict = self.forward(X).reshape(-1)
                #print(predict, y, "------")
                loss = loss_func(predict, y)
                
                def closure():
                    loss.backward()
                    optimizer.zero_grad()
                    return loss
                
                if is_LBFGS:
                    optimizer.step(closure)
                else:
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()
                
                train_loss = loss.item();
                cnt += 1;
                if batch % 30 == 0:
                    #print([par for par in self.parameters()])
                    loss, current = loss.item(), (batch + 1) * len(X)
                    pbar.set_description(f"loss: {loss} [{current}/{size}]"); 
                    
            test_loss = self.test_model(test_dataloader, loss_func = loss_func);    
            train_loss = train_loss
            
            train_loss_list.append(train_loss)
            test_loss_list.append(test_loss)
        
        return train_loss_list, test_loss_list
    def test_model(self, data_loader, loss_func):
        self.eval()
        epochs_loss = 0
        cnt = 0;
        for batch, (X, y) in enumerate(data_loader):
            predict = self.forward(X).reshape(-1)
            loss = loss_func(predict, y)
            
            epochs_loss += loss.item();
            cnt += 1;
        
        print(f"test_loss: {epochs_loss / cnt}")
        
        return epochs_loss / cnt
        
    
#----------------------Test space------------------------#
# a = KAN(width = [2, 5, 1])
# b = KAN(width = [2, 5, 1], G = 10)
# x = torch.linspace(-1, 1, 5).repeat(1, 64)
# b.initial_grid_from_other_model(a, x)
# a.update_grid_from_sample(x)
#--------------------------------------------------------#

# init
# forward
# train
# test