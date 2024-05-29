import torch
from layer import KANLayer
from torch import nn

class KAN(nn.Module):
    def __init__(self, G = 5, k = 3, width = [2, 5, 1], b = nn.SiLU(), default_gird_range = [-1, 1], device = "cpu") -> None:
        super().__init__()
        self.width = width
        self.layer = []
        for i in range(len(width) - 1):
            one_layer = KANLayer(width[i], width[i + 1], k = k, G = G, b = b, default_grid_range = default_gird_range, device = device)
            self.layer.append(one_layer)
        
        self.layer = nn.ModuleList(self.layer)
        
    def forward(self, x):
        for i in range(len(self.layer)):
            x = self.layer[i](x)[0]
        return x
    
    def initial_grid_from_other_model(self, model, x):
        for i in range(len(self.layer)):
            self.layer[i].extend_grid(model.layer[i], x)
            x = model.layer[i](x)[0]
            
    def update_grid_from_sample(self, x):
        for i in range(len(self.layer)):
            temp = self.layer[i](x)[0]
            self.layer[i].update_grid_range(x)
            x = temp
            
    def train(self, data_loader, optimizer, loss_func, epochs):
        size = len(data_loader.dataset)
        self.train()
        for t in range(epochs):
            print(f"Epoch {t + 1}\n -------------------------------");
            for batch, (X, y) in enumerate(data_loader):
                self.update_grid_from_sample(X)
                predict = self.forward(X)
                loss = loss_func(predict, y)
                
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                
                epochs_loss = loss.item();
                if batch % 30 == 0:
                    loss, current = loss.item(), (batch + 1) * len(X)
                    print(f"loss: {loss} [{current}/{size}]"); 
    def test(self):
        pass
    
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