import torch
import torch.nn as nn
import torch.nn.functional as nnf
import torchvision.transforms.functional as tf

# implementation of this architecture: https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/u-net-architecture.png


def upsample(x):
    return nnf.interpolate(input=x, scale_factor=2, mode="trilinear", align_corners=False)


def downsample(x):
    down = nn.MaxPool3d(kernel_size=2, stride=2)
    return down(x)


def add_skip_connection(x, skip):
    connection = skip.pop()
    if x.shape[2:] != connection.shape[2:]:
        print(
            f"Warning! Interpolation x: {x.shape}, skip connection: {connection.shape}")
        x = nnf.interpolate(
            input=x, size=connection.shape[2:], mode="trilinear", align_corners=False)

    return torch.cat((connection, x), dim=1)


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, sample=False):

        super(DoubleConv, self).__init__()
        self.sample = sample
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3,
                      stride=1, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3,
                      stride=1, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):

        if self.sample == "upsample":
            x = self.conv(x)
            return upsample(x)

        if self.sample == "downsample":
            x = downsample(x)
            return self.conv(x)

        return self.conv(x)


class Dose3DUNET(nn.Module):
    def __init__(
        self, in_channels=5, out_channels=1, features=[64, 128]
    ):

        super(Dose3DUNET, self).__init__()
        self.DOWN = nn.ModuleList()
        self.UP = nn.ModuleList()
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
            nn.Conv3d(
                features[0]//2,
                features[0],
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False
            ),
            nn.BatchNorm3d(features[0]),
            nn.ReLU(inplace=True)
        )

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

        skip = []
        x = self.FIRST_CONV(x)
        skip.append(x)

        for down in self.DOWN:
            x = down(x)
            skip.append(x)

        x = downsample(x)
        x = self.BOTTLENECK(x)
        x = upsample(x)

        for up in self.UP:

            x = add_skip_connection(x, skip)
            x = up(x)

        x = add_skip_connection(x, skip)
        x = self.LAST_CONV(x)
        return x


def test():
    # mit x = torch.randn((batch_size, in_channels, W, H, D))
    x = torch.randn((2, 5, 32, 32, 32))
    # x.to("cuda")

    model = Dose3DUNET()
    # model.to("cuda")
    # print(model)
    preds = model(x)
    print(f"Inputsize is: {x.shape}")
    print(f"Outputsize is: {preds.shape}")
    assert preds.shape[2:] == x.shape[2:]


if __name__ == "__main__":
    test()
