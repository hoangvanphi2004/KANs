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
        for i in range(len(self.layer)): 
            self.layer[i].b = model.layer[i].b
            self.layer[i].scale_b = model.layer[i].scale_b
            self.layer[i].scale_spline = model.layer[i].scale_spline
        self.bias = model.bias
    def update_grid_from_sample(self, x):
        for i in range(len(self.layer)):
            self.forward(x)
            self.layer[i].update_grid_range(self.acts_value[i])
    
    def plot(self):
        w = len(self.layer)
        h = 0
        for i in range(len(self.width) - 1):
            h = max(self.width[i] * self.width[i + 1], h)
        
        for i in range(w):
            x = torch.linspace(-3, 3, 100).unsqueeze(1).repeat(1, self.layer[i].num_in_node)
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
            
    def train_model(self, train_data, test_data, optimizer, loss_func, epochs = 20, stop_grid = 2, is_LBFGS = False):
        self.train()
        train_loss_list = []
        test_loss_list = []
        train_loss = 0
        cnt = 0;
        pbar = tqdm(range(epochs), desc='description', ncols=100)
        for t in pbar:
            X = train_data[0]
            y = train_data[1]
            if t < stop_grid:
                self.update_grid_from_sample(X)
                
            global loss
            def closure():
                global loss
                optimizer.zero_grad()
                predict = self.forward(X).reshape(-1)
                #print(predict, y, "------")
                loss = loss_func(predict, y)
                loss.backward()
                
                return loss
            
            if is_LBFGS:
                optimizer.step(closure)
            else:
                predict = self.forward(X).reshape(-1)
                #print(predict, y, "------")
                loss = loss_func(predict, y)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
            
            train_loss = loss.item();
            cnt += 1;
            pbar.set_description(f"loss: {loss}"); 
                    
            test_loss = self.test_model(test_data, loss_func = loss_func);    
            train_loss = train_loss
            
            train_loss_list.append(train_loss)
            test_loss_list.append(test_loss)
        
        return train_loss_list, test_loss_list
    def test_model(self, test_data, loss_func):
        self.eval()
        epochs_loss = 0
        cnt = 0;
        X = test_data[0]
        y = test_data[1]
        
        predict = self.forward(X).reshape(-1)
        loss = loss_func(predict, y)
        
        epochs_loss += loss.item();
        cnt += 1;
        
        #print(f"test_loss: {epochs_loss / cnt}")
        
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