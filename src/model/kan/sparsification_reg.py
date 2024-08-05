import torch
import torch.nn as nn

class LossWithSparsificationRegularization(nn.Module):
    def __init__(self, model, loss_func, lamb=0., lamb_l1=1, lamb_entropy=10, lamb_l1_coef=0., lamb_l1_coef_diff=0.):
        super().__init__()
        self.model = model
        self.loss_func = loss_func
        self.lamb = lamb
        self.lamb_l1 = lamb_l1
        self.lamb_entropy = lamb_entropy
        self.lamb_l1_coef = lamb_l1_coef
        self.lamb_l1_coef_diff = lamb_l1_coef_diff

    def forward(self, y_pred, y_true):
        reg = 0
        l_pred = self.loss_func(y_pred, y_true)
        for i in range(len(self.model.layer)):
            # L1 regularization
            l1 = torch.sum(self.model.acts_scale[i])
            # Entropy regularization
            p = self.model.acts_scale[i].reshape(-1, ) / l1 # (size, 1)
            entropy = -torch.sum(p * torch.log(p + 1e-4))
            # Coefficient L1 regularization
            coef_l1 = torch.sum(torch.mean(torch.abs(self.model.layer[i].coef), dim=1))
            # Coefficient L1 difference regularization
            coef_diff_l1 = torch.sum(torch.mean(torch.abs(torch.diff(self.model.layer[i].coef, dim=1)), dim=1))

            reg += self.lamb_l1 * l1 + self.lamb_entropy * entropy + self.lamb_l1_coef * coef_l1 + self.lamb_l1_coef_diff * coef_diff_l1
        reg = self.lamb * reg
        return l_pred, reg

