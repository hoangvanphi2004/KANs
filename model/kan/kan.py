import torch
from .layer import KANLayer
from .utils import *
from torch import nn
from .spline import *
import numpy as np
import matplotlib.pyplot as plt

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
        self.pre_acts = []
        self.post_acts = []
        self.splines = []
        for i in range(len(self.layer)):
            output = self.layer[i](x) 
            x = output[0]
            #x += self.bias[i].weight;
            self.acts_value.append(x)
            self.pre_acts.append(output[1])
            self.post_acts.append(output[3])
            self.splines.append(output[2])
        return x
    
    def initial_grid_from_other_model(self, model, x):
        model(x)
        self.forward(x)
        for i in range(len(self.layer)):
            self.layer[i].extend_grid(model.layer[i], self.acts_value[i])
        for i in range(len(self.layer)): 
            print(model.pre_acts[i].T, model.splines[i].T, model.layer[i].knots, self.layer[i].knots)
            print(model.pre_acts[i].T.shape, model.splines[i].T.shape, self.layer[i].knots.shape)
            print(self.layer[i].coef.data)
            self.layer[i].coef.data = curve_to_coef(x_eval = model.pre_acts[i].T, y_eval = model.splines[i].T, grid=self.layer[i].knots, k=self.k, device=self.device)
            self.layer[i].b = model.layer[i].b
            self.layer[i].scale_b = model.layer[i].scale_b
            self.layer[i].scale_spline = model.layer[i].scale_spline
            print(self.layer[i].coef.data)
            print(" stop please T^T ")
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

        plt.show()        
            
    def train_model(self, data_loader, optimizer, loss_func, is_LBFGS = False):
        def closure():
            loss.backward()
            optimizer.zero_grad()
            return loss
        size = len(data_loader.dataset)
        self.train()
        first_x = 0;
        first = True
        for batch, (X, y) in enumerate(data_loader):
            
            if first:
                first_x = X
                first = False
                
            predict = self.forward(X).reshape(-1)
            #print(predict, y, "------")
            loss = loss_func(predict, y)
            
            if is_LBFGS:
                optimizer.step(closure)
            else:
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
            
            epochs_loss = loss.item();
            if batch % 30 == 0:
                #print([par for par in self.parameters()])
                loss, current = loss.item(), (batch + 1) * len(X)
                print(f"loss: {loss} [{current}/{size}]"); 
                
        return first_x
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