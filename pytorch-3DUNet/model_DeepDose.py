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
        self.conv = nn.Sequential(*block(in_channels, in_channels, 3, 1, 1, False), *block(in_channels, out_channels, 3, 1, 1, False))

        if sample == "upsample":
            self.upsample = nn.ConvTranspose3d(in_channels=int((2/3)*in_channels), out_channels=int((2/3)*in_channels), kernel_size=2, stride=2)

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
            x = self.conv(x)
            return x

        else:
            x = self.conv(x)
            return x


class Dose3DUNET(nn.Module):
    def __init__(
        self, in_channels=5, out_channels=1, features=[64, 128]
    ):

        super(Dose3DUNET, self).__init__()
        self.DOWN = nn.ModuleList()
        self.UP = nn.ModuleList()

        self.FIRST_CONV = nn.Sequential(
            *block(in_channels, features[0]//2, kernel_size=3, stride=1, padding=1, bias=False),
            *block(features[0]//2, features[0], kernel_size=3, stride=1, padding=1, bias=False))
        self.FIRST_POOL = nn.MaxPool3d(kernel_size=2, stride=2)

        # DOWNSAMPLING
        for size in features:
            self.DOWN.append(DoubleConv(size, 2*size, sample="downsample"))

        # BOTTLENECK
        self.BOTTLENECK = DoubleConv(2*features[-1], 4*features[-1])

        # UPSAMPLING
        for size in reversed(features):
            self.UP.append(DoubleConv(6*size, 2*size, sample="upsample"))

        self.LAST_DECONV = nn.ConvTranspose3d(2*features[0], 2*features[0], kernel_size=2, stride=2)

        self.LAST_CONV = nn.Sequential(
            *block(3*features[0], features[0], kernel_size=3, stride=1, padding=1, bias=False),
            *block(features[0], features[0], kernel_size=3, stride=1, padding=1, bias=False),
            *block(features[0], out_channels, kernel_size=1, stride=1, padding=0, bias=False))

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
        x = self.LAST_CONV(x)

        return x


if __name__ == "__main__":

    from time import time

    for ps, bs in zip([32, 64], [64, 4]):

        input = torch.randn(bs, 5, ps, ps, ps)
        device = torch.device("cuda")
        input = input.to(device)
        model = Dose3DUNET(features=[64, 128])
        model.to(device)
        pred = model(input)

        time_needed = 0
        for i in range(10):

            input = torch.randn(bs, 5, ps, ps, ps)
            device = torch.device("cuda")
            input = input.to(device)
            model = Dose3DUNET(features=[64, 128])
            model.to(device)
            start = time()
            pred = model(input)
            time_needed += time()-start
            torch.cuda.empty_cache()
            #print("pred start")
            #pred = model(input)
        print(ps, time_needed/10)
