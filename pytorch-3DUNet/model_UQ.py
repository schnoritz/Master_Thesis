import torch
import torch.nn as nn
import torch.nn.functional as nnf
import torchvision.transforms.functional as tf

# implementation of this architecture: https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/u-net-architecture.png


def upsample(x):
    return nnf.interpolate(input=x, scale_factor=2, mode="trilinear", align_corners=False)


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, p=0.2):

        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3,
                      stride=1, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout3d(p=p),
            nn.Conv3d(out_channels, out_channels, kernel_size=3,
                      stride=1, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout3d(p=p)
        )

        for layer in self.conv:
            if isinstance(layer, nn.Dropout3d):
                layer.eval()

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
                print(
                    f"Warning! Interpolation due to not matching shape. x: {x.shape}, skip connection: {skip_connection.shape}")
                x = nnf.interpolate(
                    input=x, size=skip_connection.shape[2:], mode="trilinear", align_corners=False)

            # add skip connection to upsampled tensor
            x = torch.cat((skip_connection, x), dim=1)
            x = up(x)

        return x

    def get_uncertainty(self, x, T):

        state = (0, 0, 0)
        self.UQ()
        for num in range(T):
            print(num)
            pred = self.forward(x)
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

        for part in self.downs:
            for layer in part.conv:
                if isinstance(layer, nn.Dropout3d):
                    layer.train()
                else:
                    layer.eval()

        for part in self.ups:
            if isinstance(part, DoubleConv):
                for layer in part.conv:
                    if isinstance(layer, nn.Dropout3d):
                        layer.train()
                    else:
                        layer.eval()

        for part in self.bottleneck:
            if isinstance(part, DoubleConv):
                for layer in part.conv:
                    if isinstance(layer, nn.Dropout3d):
                        layer.train()
                    else:
                        layer.eval()

    def non_UQ(self):

        print("Model IS NOT set for uncertrainty quantification!")

        for part in self.downs:
            for layer in part.conv:
                if isinstance(layer, nn.Dropout3d):
                    layer.eval()
                else:
                    layer.train()

        for part in self.ups:
            if isinstance(part, DoubleConv):
                for layer in part.conv:
                    if isinstance(layer, nn.Dropout3d):
                        layer.eval()
                    else:
                        layer.train()

        for part in self.bottleneck:
            if isinstance(part, DoubleConv):
                for layer in part.conv:
                    if isinstance(layer, nn.Dropout3d):
                        layer.eval()
                    else:
                        layer.train()


def test():
    # mit x = torch.randn((batch_size, in_channels, W, H, D))
    x = torch.randn((2, 5, 32, 32, 32))

    model = Dose3DUNET(in_channels=5, out_channels=1)
    uncertainty = model.get_uncertainty(x, T=10)
    print(uncertainty[0].shape, uncertainty[1].shape, uncertainty[2].shape)
    preds = model(x)
    print(f"Inputsize is: {x.shape}")
    print(f"Outputsize is: {preds.shape}")
    assert preds.shape[2:] == x.shape[2:]


if __name__ == "__main__":
    test()
