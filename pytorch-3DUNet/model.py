import torch
import torch.nn as nn
import torch.nn.functional as nnf
import torchvision.transforms.functional as tf

# implementation of this architecture: https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/u-net-architecture.png


def upsample(x):
    return nnf.interpolate(input=x, scale_factor=2, mode="trilinear", align_corners=False)


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):

        super(DoubleConv, self).__init__()
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
        return self.conv(x)


class Dose3DUNET(nn.Module):
    def __init__(
        self, in_channels=5, out_channels=1, features=[64, 128, 256]
    ):

        super(Dose3DUNET, self).__init__()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)

        # DOWN
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature

        # UP
        for num in range(len(features)-1, 0, -1):
            self.ups.append(DoubleConv(3*features[num], features[num]))
        self.ups.append(nn.Conv3d(3*features[0], out_channels, kernel_size=1))

        self.bottleneck = nn.Sequential(nn.MaxPool3d(kernel_size=2, stride=2), DoubleConv(
            features[-1], features[-1]*2))
        #self.final_conv = nn.Conv3d(3*features[0], out_channels, kernel_size=1)

    def forward(self, x):

        print(x.shape)
        skip_connections = []
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        x = upsample(x)

        skip_connections = skip_connections[::-1]

        for num, up in enumerate(self.ups):

            skip_connection = skip_connections[num]
            x = upsample(x)

            # check if shapes match
            if x.shape[2:] != skip_connection.shape[2:]:
                x = nnf.interpolate(
                    input=x, size=skip_connection.shape[2:], mode="trilinear", align_corners=False)
                print(
                    "Warning! Interpolation due to not matching shape.")

            # add skip connection to upsampled tensor
            x = torch.cat((skip_connection, x), dim=1)
            x = up(x)

        return x


def test():
    # mit x = torch.randn((batch_size, in_channels, W, H, D))
    x = torch.randn((2, 5, 64, 64, 64))

    model = Dose3DUNET(in_channels=5, out_channels=1,
                       features=[64, 128, 256])
    # print(model)
    preds = model(x)
    print(f"Inputsize is: {x.shape}")
    print(f"Outputsize is: {preds.shape}")
    assert preds.shape[2:] == x.shape[2:]


if __name__ == "__main__":
    test()
