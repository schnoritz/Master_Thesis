import torch
import torch.nn as nn
import torch.nn.functional as nnf

# implementation of this architecture: https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/u-net-architecture.png


def add_skip_connection(x, skip):
    connection = skip.pop()
    if x.shape[2:] != connection.shape[2:]:
        print(f"Warning! Interpolation x: {x.shape}, skip connection: {connection.shape}")
        x = nnf.interpolate(input=x, size=connection.shape[2:], mode="trilinear", align_corners=False)

    return torch.cat((connection, x), dim=1)


def block(in_channels=None, out_channels=None, kernel_size=None, stride=None, padding=None, bias=None):

    block_list = [
        nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
        nn.BatchNorm3d(out_channels),
        nn.ReLU(inplace=True)
    ]

    return block_list


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, UQ=False, sample=False, p=0.1):

        super(DoubleConv, self).__init__()
        self.sample = sample
        self.conv = block(in_channels, out_channels, 3, 1, 1, False)

        if UQ:
            self.conv.append(nn.Dropout(p=p))

        self.conv.extend(block(out_channels, out_channels, 3, 1, 1, False))

        if UQ:
            self.conv.append(nn.Dropout(p=p))

        self.conv = nn.Sequential(*self.conv)

        if sample == "upsample":
            self.upsample = nn.ConvTranspose3d(
                in_channels=int((2/3)*in_channels), out_channels=int((2/3)*in_channels), kernel_size=2, stride=2)

        if sample == "downsample":
            self.downsample = nn.MaxPool3d(kernel_size=2, stride=2)

    def forward(self, x, skip_connection=None):

        if self.sample == "downsample":
            skip = self.conv(x)
            x = self.downsample(skip)

            return x, skip

        elif self.sample == "upsample":
            x = self.upsample(x)
            x = add_skip_connection(x, skip_connection)
            return self.conv(x)

        else:
            return self.conv(x)


class Dose3DUNET(nn.Module):
    def __init__(
        self, in_channels=5, out_channels=1, UQ=False, features=[64, 128], p=0.1
    ):

        super(Dose3DUNET, self).__init__()
        self.DOWN = nn.ModuleList()
        self.UP = nn.ModuleList()
        self.UQ = UQ

        if UQ:
            self.SIGMA = [
                *block(3*features[0], features[0], kernel_size=3, stride=1, padding=1, bias=False),
                *block(features[0], features[0], kernel_size=3, stride=1, padding=1, bias=False),
                *block(features[0], out_channels, kernel_size=1, stride=1, padding=0, bias=False)]

            self.SIGMA = nn.Sequential(*self.SIGMA)

        self.LAST_CONV = block(3*features[0], features[0], kernel_size=3, stride=1, padding=1, bias=False)
        if UQ:
            self.LAST_CONV.append(nn.Dropout(p=p))
        self.LAST_CONV.extend(block(features[0], features[0], kernel_size=3, stride=1, padding=1, bias=False))
        if UQ:
            self.LAST_CONV.append(nn.Dropout(p=p))
        self.LAST_CONV.extend(block(features[0], out_channels, kernel_size=1, stride=1, padding=0, bias=False))
        if UQ:
            self.LAST_CONV.append(nn.Dropout(p=p))

        self.LAST_CONV = nn.Sequential(*self.LAST_CONV)

        self.FIRST_CONV = block(in_channels, features[0]//2, kernel_size=3, stride=1, padding=1, bias=False)
        if UQ:
            self.FIRST_CONV.append(nn.Dropout(p=p))
        self.FIRST_CONV.extend(block(features[0]//2, features[0], kernel_size=3, stride=1, padding=1, bias=False))
        if UQ:
            self.FIRST_CONV.append(nn.Dropout(p=p))
        self.FIRST_CONV = nn.Sequential(*self.FIRST_CONV)

        self.LAST_DECONV = nn.ConvTranspose3d(
            in_channels=2*features[0],
            out_channels=2*features[0],
            kernel_size=2, stride=2)

        self.FIRST_POOL = nn.MaxPool3d(kernel_size=2, stride=2)

        # DOWNSAMPLING
        for size in features:
            self.DOWN.append(DoubleConv(size, 2*size, UQ=UQ, sample="downsample", p=p))

        # UPSAMPLING
        features = features[::-1]
        for size in features:
            self.UP.append(DoubleConv(6*size, 2*size, UQ=UQ, sample="upsample", p=p))

        # BOTTLENECK
        self.BOTTLENECK = DoubleConv(2*features[0], 4*features[0], UQ=UQ, p=p)

    def forward(self, x):

        skip_connections = []

        x = self.FIRST_CONV(x)
        skip_connections.append(x)
        x = self.FIRST_POOL(x)

        for down in self.DOWN:
            x, skip = down(x)
            skip_connections.append(skip)

        x = self.BOTTLENECK(x)

        for up in self.UP:
            x = up(x, skip_connections)

        x = self.LAST_DECONV(x)
        x = add_skip_connection(x, skip_connections)
        pred = self.LAST_CONV(x)

        if hasattr(self, 'SIGMA'):
            sigma = self.SIGMA(x)
            return pred, sigma

        else:
            return pred

    def get_uncertainty(self, x, T):

        # implementation after arXiv:1703.04977 formula (8) and (9)
        # with E[X^2] + E[X]^2 + A[sigma^2]
        # with E = prediction and A aleatoric uncertainty
        if self.UQ:
            self.UQ_mode()
            y_hat_sq = []
            y_hat = []
            sigma_sq = []
            for num in range(T):
                print(num)
                pred, sigma = self.forward(x)
                y_hat_sq.append(torch.square(pred))
                y_hat.append(pred)
                sigma_sq.append(torch.exp(sigma))

            y_hat_sq = torch.div(torch.sum(torch.stack(y_hat_sq), dim=0), T)
            y_hat = torch.square(torch.div(torch.sum(torch.stack(y_hat), dim=0), T))
            sigma_sq = torch.div(torch.sum(torch.stack(sigma_sq), dim=0), T)

            variance = y_hat_sq - y_hat + sigma_sq

            print(variance.shape)

            self.non_UQ_mode()
            return variance
        else:
            print("Model is not initialized for uncertainty quantification!")

    def UQ_mode(self):

        if self.UQ:
            for layer in self.LAST_CONV:
                if isinstance(layer, nn.Dropout):
                    layer.train()
                else:
                    layer.eval()

            for layer in self.FIRST_CONV:
                if isinstance(layer, nn.Dropout):
                    layer.train()
                else:
                    layer.eval()

            for part in self.DOWN:
                for layer in part.conv:
                    if isinstance(layer, nn.Dropout):
                        layer.train()
                    else:
                        layer.eval()

            for part in self.UP:
                for layer in part.conv:
                    if isinstance(layer, nn.Dropout):
                        layer.train()
                    else:
                        layer.eval()

            for layer in self.BOTTLENECK.conv:
                if isinstance(layer, nn.Dropout):
                    layer.train()
                else:
                    layer.eval()

            else:
                pass

    def non_UQ_mode(self):

        if self.UQ:
            self.train()
        else:
            pass


def test():
    # mit x = torch.randn((batch_size, in_channels, W, H, D))

    import numpy as np
    from losses import HeteroscedasticAleatoricLoss, RMSELoss
    from torch import optim

    UQ = False

    # input = torch.load("/Users/simongutwein/Studium/Masterarbeit/test_data/p0_1/training_data.pt")
    # target = torch.load("/Users/simongutwein/Studium/Masterarbeit/test_data/p0_1/target_data.pt")
    input = torch.randn((5, 512, 512, 110))
    target = torch.randn((1, 512, 512, 110))

    inputs = list()
    targets = list()
    for i in np.linspace(128, 384-64, 7, endpoint=True):
        for ii in np.linspace(128, 384-64, 7, endpoint=True):
            inputs.append(input[:, int(i-16):int(i+16), int(ii-16):int(ii+16), input.shape[-1]//2-16:input.shape[-1]//2+16])
            targets.append(target[:, int(i-16):int(i+16), int(ii-16):int(ii+16), input.shape[-1]//2-16:input.shape[-1]//2+16])

    print(len(inputs), len(targets))

    if UQ:
        criterion = HeteroscedasticAleatoricLoss()
    else:
        criterion = RMSELoss()

    model = Dose3DUNET(in_channels=5, out_channels=1, UQ=UQ, p=0.1)
    optimizer = optim.Adam(model.parameters(),
                           0.0001, (0.9, 0.99), 10E-8)

    bs = 2

    for i in range(10):
        epoch_loss = 0
        for ii in range(0, len(inputs), bs):

            input = torch.stack(inputs[ii:ii+bs])
            target = torch.stack(targets[ii:ii+bs])

            print(f"Inputsize is: {input.shape}")

            if UQ:
                pred, sigma = model(input)
                print(f"Outputsize is: {pred.shape}, {sigma.shape}")
                loss = criterion(target, pred, sigma)

            else:
                pred = model(input)
                print(f"Outputsize is: {pred.shape}")
                loss = criterion(target, pred)

            assert pred.shape[2:] == input.shape[2:]

            epoch_loss += loss.item()
            print(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


if __name__ == "__main__":
    test()
