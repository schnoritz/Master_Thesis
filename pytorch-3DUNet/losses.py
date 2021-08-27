import torch
import torch.nn as nn


class RMSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, y_hat, y):
        return torch.sqrt(self.mse(y_hat, y))


class HeteroscedasticAleatoricLoss(nn.Module):
    def __init__(self):
        super(HeteroscedasticAleatoricLoss, self).__init__()

    def forward(self, y_hat, y, log_sigma):

        return torch.div(torch.sum(0.5*torch.exp(-log_sigma)*torch.pow(torch.abs(y-y_hat), 2) + 0.5*log_sigma), torch.numel(y))


if __name__ == "__main__":

    y_hat = torch.randn(2, 5, 32, 32, 32)
    y = torch.randn(2, 5, 32, 32, 32)

    criterion = RMSELoss()
    loss = criterion(y_hat, y)

    print(loss)
