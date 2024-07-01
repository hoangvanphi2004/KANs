import torch 
import torch.nn as nn 
from tqdm import tqdm

class MLP(nn.Module):
    def __init__(self, width, activation=nn.ReLU(), device='cpu'):
        super(MLP, self).__init__()
        self.width = width
        self.device = device
        self.layer = []
        for i in range(len(width) - 1):
            one_layer = nn.Linear(width[i], width[i + 1], bias=True).to(device)
            self.layer.append(one_layer)
            if i != len(width) - 2:
                self.layer.append(activation)
        self.layer = nn.ModuleList(self.layer)

    def forward(self, x):
        for i in range(len(self.layer)):
            x = self.layer[i](x)
        return x
    
    def train_model(self, train_loader, val_loader, optimizer, loss_func, epochs=20):
        train_loss_list = []
        val_loss_list = []
        pbar = tqdm(range(epochs))

        for epoch in pbar:
            self.train()
            train_loss = 0
            val_loss = 0
            for x, y in train_loader:
                x, y = x.to(self.device), y.to(self.device)
                optimizer.zero_grad()
                y_pred = self(x)
                loss = loss_func(y_pred, y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            train_loss /= len(train_loader)
            train_loss_list.append(train_loss)

            self.eval()
            with torch.no_grad():  
                for x, y in val_loader:
                    x, y = x.to(self.device), y.to(self.device)
                    y_pred = self(x)
                    loss = loss_func(y_pred, y)
                    val_loss += loss.item()
            val_loss /= len(val_loader)
            val_loss_list.append(val_loss)
            pbar.set_description(f"train_loss: {train_loss}, val_loss: {val_loss}")

            self.train()

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