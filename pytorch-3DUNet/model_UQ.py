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
    def __init__(self, in_channels, out_channels, sample=False, p=0.1):

        super(DoubleConv, self).__init__()
        self.sample = sample
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3,
                      stride=1, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(p=p),
            nn.Conv3d(out_channels, out_channels, kernel_size=3,
                      stride=1, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(p=p)
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
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)
        self.last_conv = nn.Sequential(
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
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False
            ),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.sigma = nn.Sequential(
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
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False
            ),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.first_conv = nn.Sequential(
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
        self.DOWN = []
        for size in features:
            self.DOWN.append(DoubleConv(size, 2*size, sample="downsample"))

        # UPSAMPLING
        self.UP = []
        features = features[::-1]
        for size in features:
            self.UP.append(DoubleConv(6*size, 2*size, sample="upsample"))

        # BOTTLENECK
        self.BOTTLENECK = DoubleConv(2*features[0], 4*features[0])

    def forward(self, x):

        skip = []
        x = self.first_conv(x)
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

        sigma = self.sigma(x)
        prediction = self.last_conv(x)

        return prediction, sigma

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
    x = torch.randn((2, 5, 32, 32, 32))

    model = Dose3DUNET(in_channels=5, out_channels=1)
    uncertainty = model.get_uncertainty(x, T=3)
    print(uncertainty[0].shape, uncertainty[1].shape, uncertainty[2].shape)
    preds, sigma = model(x)
    print(f"Inputsize is: {x.shape}")
    print(f"Outputsize is: {preds.shape}, {sigma.shape}")
    assert preds.shape[2:] == x.shape[2:]


if __name__ == "__main__":
    test()
