
import torch
import torch.nn as nn
import torch.nn.functional as nnf
import torchvision.transforms.functional as tf

# implementation of this architecture: https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/u-net-architecture.png


def add_skip_connection(x, skip):
    connection = skip.pop()
    if x.shape[2:] != connection.shape[2:]:
        print(f"Warning! Interpolation x: {x.shape}, skip connection: {connection.shape}")
        x = nnf.interpolate(input=x, size=connection.shape[2:], mode="trilinear", align_corners=False)

    return torch.cat((connection, x), dim=1)


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, sample=False, p=0.2):

        super(DoubleConv, self).__init__()
        self.sample = sample
        self.conv = nn.Sequential()

        if sample == "upsample":
            self.conv.add_module("upsample", nn.ConvTranspose3d(
                in_channels=in_channels, out_channels=in_channels, kernel_size=2, stride=2))

        self.conv.add_module("main_block", nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(p=p),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(p=p)))

        if sample == "downsample":
            self.conv.add_module("downsample", nn.MaxPool3d(kernel_size=2, stride=2))

    def forward(self, x):

        return self.conv(x)


class Dose3DUNET(nn.Module):
    def __init__(
        self, in_channels=5, out_channels=1, features=[64, 128], p=0.2
    ):

        super(Dose3DUNET, self).__init__()
        self.DOWN = nn.ModuleList()
        self.UP = nn.ModuleList()
        self.SIGMA = nn.Sequential(
            nn.Conv3d(
                3*features[0],
                features[0],
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False
            ),
            nn.BatchNorm3d(features[0]),
            nn.ReLU(inplace=True),
            nn.Conv3d(
                features[0],
                features[0],
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False
            ),
            nn.BatchNorm3d(features[0]),
            nn.ReLU(inplace=True),
            nn.Conv3d(
                features[0],
                out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False
            ),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.LAST_CONV = nn.Sequential(
            nn.Conv3d(
                3*features[0],
                features[0],
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False
            ),
            nn.BatchNorm3d(features[0]),
            nn.ReLU(inplace=True),
            nn.Conv3d(
                features[0],
                features[0],
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False
            ),
            nn.BatchNorm3d(features[0]),
            nn.ReLU(inplace=True),
            nn.Conv3d(
                features[0],
                out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False
            ),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.FIRST_CONV = nn.Sequential(
            nn.Conv3d(
                in_channels,
                features[0]//2,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False
            ),
            nn.BatchNorm3d(features[0]//2),
            nn.ReLU(inplace=True),
            nn.Dropout(p=p),
            nn.Conv3d(
                features[0]//2,
                features[0],
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False
            ),
            nn.BatchNorm3d(features[0]),
            nn.ReLU(inplace=True),
            nn.Dropout(p=p)
        )
        self.LAST_DECONV = nn.ConvTranspose3d(
            in_channels=2*features[0],
            out_channels=2*features[0],
            kernel_size=2, stride=2)

        self.FIRST_POOL = nn.MaxPool3d(kernel_size=2, stride=2)

        # DOWNSAMPLING
        for size in features:
            self.DOWN.append(DoubleConv(size, 2*size, sample="downsample", p=p))

        # UPSAMPLING
        features = features[::-1]
        for size in features:
            self.UP.append(DoubleConv(6*size, 2*size, sample="upsample", p=p))

        # BOTTLENECK
        self.BOTTLENECK = DoubleConv(2*features[0], 4*features[0], p=p)

    def forward(self, x):

        skip = []
        x = self.FIRST_CONV(x)
        skip.append(x)
        x = self.FIRST_POOL(x)

        for down in self.DOWN:
            x = down(x)
            skip.append(x)

        x = self.BOTTLENECK(x)

        for up in self.UP:
            x = add_skip_connection(x, skip)
            x = up(x)

        x = self.LAST_DECONV(x)
        x = add_skip_connection(x, skip)

        out = self.LAST_CONV(x)
        sigma = self.SIGMA(x)

        return out, sigma

    def get_uncertainty(self, x, T):

        state = (0, 0, 0)
        self.UQ()
        for num in range(T):
            print(num)
            pred, _ = self.forward(x)
            state = self.update_state(state, pred)

        (count, mean, m2) = state
        (mean, variance, sampleVariance) = (mean, m2 / count, m2 / (count - 1))

        self.non_UQ()
        return mean, variance, sampleVariance

    def update_state(self, state, new_val):

        (count, mean, m2) = state
        count += 1
        delta = mean - new_val
        mean += delta / count
        delta2 = new_val - mean
        m2 += delta * delta2
        return (count, mean, m2)

    def UQ(self):

        print("Model IS set for uncertrainty quantification!")

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

    def non_UQ(self):

        print("Model IS NOT set for uncertrainty quantification!")
        self.train()


def test():
    # mit x = torch.randn((batch_size, in_channels, W, H, D))

    import numpy as np
    from losses import HeteroscedasticAleatoricLoss
    from torch import optim

    input = torch.load("/Users/simongutwein/Studium/Masterarbeit/test_data/p0_1/training_data.pt")
    target = torch.load("/Users/simongutwein/Studium/Masterarbeit/test_data/p0_1/target_data.pt")

    inputs = list()
    targets = list()
    for i in np.linspace(128, 384-64, 7, endpoint=True):
        for ii in np.linspace(128, 384-64, 7, endpoint=True):
            inputs.append(input[:, int(i-16):int(i+16), int(ii-16):int(ii+16), input.shape[-1]-16:input.shape[-1]+16])
            targets.append(target[:, int(i-16):int(i+16), int(ii-16):int(ii+16), input.shape[-1]-16:input.shape[-1]+16])

    print(len(inputs), len(targets))

    criterion = HeteroscedasticAleatoricLoss()
    model = Dose3DUNET(in_channels=5, out_channels=1, p=0.1)
    optimizer = optim.Adam(model.parameters(),
                           0.0001, (0.9, 0.99), 10E-8)

    bs = 2

    for i in range(10):
        epoch_loss = 0
        for ii in range(0, len(inputs), bs):

            input = torch.stack(inputs[ii:ii+bs])
            target = torch.stack(targets[ii:ii+bs])

            print(input.shape, target.shape)

            pred, sigma = model(input)
            print(f"Inputsize is: {input.shape}")
            print(f"Outputsize is: {pred.shape}, {sigma.shape}")
            assert pred.shape[2:] == input.shape[2:]

            loss = criterion(target, pred, sigma)
            epoch_loss += loss.item()
            print(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


if __name__ == "__main__":
    test()
