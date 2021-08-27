from typing import final
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
    def __init__(self, in_channels, out_channels, sample=False):

        super(DoubleConv, self).__init__()
        self.sample = sample
        self.first_conv = nn.Sequential(*block(in_channels, out_channels, 3, 1, 1, False))
        self.second_conv = nn.Sequential(*block(out_channels, out_channels, 3, 1, 1, False), *block(out_channels, out_channels, 3, 1, 1, False))

        if sample == "upsample":
            self.upsample = nn.ConvTranspose3d(
                in_channels=int((2/3)*in_channels), out_channels=int((2/3)*in_channels), kernel_size=2, stride=2)

        if sample == "downsample":
            self.downsample = nn.MaxPool3d(kernel_size=2, stride=2)

    def forward(self, x, skip_connection=None):

        if self.sample == "downsample":

            res_tensor = self.first_conv(x)
            out = self.second_conv(res_tensor)
            out = torch.add(out, res_tensor)
            skip = out
            x = self.downsample(skip)

            return x, skip

        elif self.sample == "upsample":

            x = self.upsample(x)
            x = add_skip_connection(x, skip_connection)

            res_tensor = self.first_conv(x)
            out = self.second_conv(res_tensor)
            out = torch.add(out, res_tensor)

            return out

        else:

            res_tensor = self.first_conv(x)
            out = self.second_conv(res_tensor)
            out = torch.add(out, res_tensor)

            return out


class ResDose3DUNET(nn.Module):
    def __init__(
        self, in_channels=5, out_channels=1, features=[64, 128]
    ):

        super(ResDose3DUNET, self).__init__()
        self.DOWN = nn.ModuleList()
        self.UP = nn.ModuleList()

        self.first_conv = nn.Sequential(*block(3*features[0], features[0], kernel_size=3, stride=1, padding=1, bias=False))
        self.second_conv = nn.Sequential(*block(features[0], features[0], kernel_size=3, stride=1, padding=1, bias=False))
        self.final_conv = nn.Sequential(*block(features[0], out_channels, kernel_size=1, stride=1, padding=0, bias=False))

        self.FIRST_CONV = nn.Sequential(
            *block(in_channels, features[0]//2, kernel_size=3, stride=1, padding=1, bias=False),
            *block(features[0]//2, features[0], kernel_size=3, stride=1, padding=1, bias=False))

        self.LAST_DECONV = nn.ConvTranspose3d(in_channels=2*features[0], out_channels=2*features[0], kernel_size=2, stride=2)

        self.FIRST_POOL = nn.MaxPool3d(kernel_size=2, stride=2)

        # DOWNSAMPLING
        for size in features:
            self.DOWN.append(DoubleConv(size, 2*size, sample="downsample"))

        # UPSAMPLING
        features = features[::-1]
        for size in features:
            self.UP.append(DoubleConv(6*size, 2*size, sample="upsample"))

        # BOTTLENECK
        self.BOTTLENECK = DoubleConv(2*features[0], 4*features[0])

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
        res_tensor = self.first_conv(x)
        x = self.second_conv(res_tensor)
        x = self.second_conv(x)
        x = torch.add(res_tensor, x)
        x = self.final_conv(x)

        return x


if __name__ == "__main__":

    print("start")
    input = torch.randn(2, 5, 32, 32, 32)
    target = torch.randn(2, 1, 32, 32, 32)

    device = torch.device("cpu")
    input = input.to(device)
    model = ResDose3DUNET()
    loss = torch.nn.MSELoss()
    optim = torch.optim.SGD(params=model.parameters(), lr=0.01, momentum=0.8)
    model.to(device)
    pred = model(input)
    loss_val = loss(pred, target)
    optim.zero_grad()
    loss_val.backward()
    optim.step()
    #print("pred start")
    #pred = model(input)
    print(pred.shape)
