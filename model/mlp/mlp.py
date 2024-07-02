import torch
import torch.nn as nn
from tqdm import tqdm

class MLP(nn.Module):
    def __init__(self, width, activation_class=nn.ReLU, device='cpu'):
        super(MLP, self).__init__()
        self.width = width
        self.device = device
        self.layers = nn.ModuleList()
        for i in range(len(width) - 1):
            linear_layer = nn.Linear(width[i], width[i + 1]).to(device)
            nn.init.xavier_uniform_(linear_layer.weight)  # Xavier initialization
            self.layers.append(linear_layer)
            if i != len(width) - 2:
                self.layers.append(activation_class().to(device))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x.squeeze()

    def train_model(self, train_loader, val_loader, optimizer, loss_func, epochs=20, is_lbfgs=False):
        train_loss_list = []
        val_loss_list = []
        pbar = tqdm(range(epochs))

        for epoch in pbar:
            self.train()
            train_loss = 0
            for x, y in train_loader:
                x, y = x.to(self.device), y.to(self.device)
                    
                def closure():
                    optimizer.zero_grad()
                    y_pred = self(x)
                    loss = loss_func(y_pred, y)
                    loss.backward()
                    return loss
                
                if is_lbfgs:
                    loss = optimizer.step(closure)
                else:
                    optimizer.zero_grad()
                    y_pred = self(x)
                    loss = loss_func(y_pred, y)
                    loss.backward()
                    optimizer.step()

                train_loss += loss.item()
                
            train_loss /= len(train_loader)
            train_loss_list.append(train_loss)

            self.eval()
            val_loss = 0
            with torch.no_grad():
                for x, y in val_loader:
                    x, y = x.to(self.device), y.to(self.device)
                    y_pred = self(x)
                    loss = loss_func(y_pred, y)
                    val_loss += loss.item()
            val_loss /= len(val_loader)
            val_loss_list.append(val_loss)
            pbar.set_description(f"train_loss: {train_loss}, val_loss: {val_loss}")

        return train_loss_list, val_loss_list

    def test_model(self, test_loader, loss_func):
        self.eval()
        test_loss = 0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(self.device), y.to(self.device)
                y_pred = self(x)
                loss = loss_func(y_pred, y)
                test_loss += loss.item()
        test_loss /= len(test_loader)
        return test_loss