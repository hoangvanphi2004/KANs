import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim

class MLP(pl.LightningModule):
    def __init__(self, width, activation_class='ReLU'):
        super(MLP, self).__init__()
        self.width = width
        self.layers = nn.ModuleList()
        
        if activation_class == 'ReLU':
            activation_class = nn.ReLU
        elif activation_class == 'Sigmoid':
            activation_class = nn.Sigmoid
        elif activation_class == 'Tanh':
            activation_class = nn.Tanh
        else:
            raise ValueError(f"Activation function {activation_class} not recognized.")

        for i in range(len(width) - 1):
            linear_layer = nn.Linear(width[i], width[i + 1])
            nn.init.xavier_uniform_(linear_layer.weight)
            self.layers.append(linear_layer)
            if i != len(width) - 2: 
                self.layers.append(activation_class())

        self.loss_func = nn.MSELoss()

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x.squeeze()

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        loss = self.loss_func(y_pred, y)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        loss = self.loss_func(y_pred, y)
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        loss = self.loss_func(y_pred, y)
        self.log('test_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

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