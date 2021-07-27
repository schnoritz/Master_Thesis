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
        return torch.sum(0.5 * (torch.exp(log_sigma) *
                                torch.square(torch.norm(y_hat-y)) + torch.square(log_sigma)))


if __name__ == "__main__":

    y_hat = torch.randn(32, 10, 32, 32, 32)
    y = torch.randn(32, 10, 32, 32, 32)
    log_sigma = torch.randn(32, 10, 32, 32, 32)

    criterion = HeteroscedasticAleatoricLoss()
    loss = criterion(y_hat, y, log_sigma)
