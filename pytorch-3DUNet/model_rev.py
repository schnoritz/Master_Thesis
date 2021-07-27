import random
import torch.nn.functional as F
import torch.optim as optim
import torch
import torch.nn as nn
import torch.nn.functional as nnf
import torchvision.transforms.functional as tf
import revtorch.revtorch as rv

# implementation of this architecture: https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/u-net-architecture.png


class ResidualInner(nn.Module):
    def __init__(self, channels, groups):
        super(ResidualInner, self).__init__()
        self.gn = nn.GroupNorm(groups, channels)
        self.conv = nn.Conv3d(channels, channels, 3, padding=1, bias=False)

    def forward(self, x):
        x = self.gn(x)
        x = F.leaky_relu(x)
        x = self.conv(x)
        return x


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


def makeReversibleSequence(channels):
    innerChannels = channels // 2
    groups = channels // 4
    fBlock = ResidualInner(innerChannels, groups)
    gBlock = ResidualInner(innerChannels, groups)
    return rv.ReversibleBlock(fBlock, gBlock)


def makeReversibleComponent(channels, blockCount):
    modules = []
    for _ in range(blockCount):
        modules.append(makeReversibleSequence(channels))
    return rv.ReversibleSequence(nn.ModuleList(modules))


class EncoderModule(nn.Module):
    def __init__(self, inChannels, outChannels, depth, downsample=True):
        super(EncoderModule, self).__init__()
        self.downsample = downsample
        self.conv = nn.Conv3d(inChannels, outChannels, 1)
        self.reversibleBlocks = makeReversibleComponent(outChannels, depth)

    def forward(self, x):
        if self.downsample:
            x = F.max_pool3d(x, 2)
        x = self.conv(x)  # increase number of channels
        x = self.reversibleBlocks(x)
        return x


class DecoderModule(nn.Module):
    def __init__(self, inChannels, outChannels, depth, upsample=True):
        super(DecoderModule, self).__init__()
        self.reversibleBlocks = makeReversibleComponent(inChannels, depth)
        self.upsample = upsample
        self.conv = nn.Conv3d(inChannels, outChannels, 1)

    def forward(self, x):
        x = self.reversibleBlocks(x)
        x = self.conv(x)
        if self.upsample:
            x = F.interpolate(x, scale_factor=2,
                              mode="trilinear", align_corners=False)
        return x


class RevDose3DUNET(nn.Module):
    def __init__(self, in_channels=5, out_channels=1, features=[64, 128], depth=1):
        super(RevDose3DUNET, self).__init__()

        self.depth = depth

        # hier schauen ob ich einen reversible block ohne upsampling/downsampling nutzen kann
        self.first_conv = EncoderModule(
            in_channels, features[0], self.depth, downsample=False)
        # self.last_conv = nn.Sequential(DecoderModule(
        #     3*features[0], features[0], self.depth+1, upsample=False), nn.Conv3d(features[0], out_channels, kernel_size=1, padding=0, bias=False))
        self.last_conv = nn.Sequential(DecoderModule(
            3*features[0], 1, self.depth+1, upsample=False))

        DOWN = []
        for size in features:
            DOWN.append(EncoderModule(
                size, 2*size, self.depth, downsample=True))
        self.encoders = nn.ModuleList(DOWN)

        self.bottleneck = EncoderModule(
            2*features[-1], 4*features[-1], depth, downsample=False)

        features = features[::-1]
        UP = []
        for size in features:
            UP.append(DecoderModule(
                6*size, 2*size, self.depth, upsample=True))
        self.decoders = nn.ModuleList(UP)

    def forward(self, x):

        skips = []
        x = self.first_conv(x)
        skips.append(x)
        for down in self.encoders:
            x = down(x)
            skips.append(x)

        x = downsample(x)
        x = self.bottleneck(x)
        x = upsample(x)

        for up in self.decoders:
            x = add_skip_connection(x, skips)
            x = up(x)

        x = add_skip_connection(x, skips)
        x = self.last_conv(x)

        return x


def test():
    # mit x = torch.randn((batch_size, in_channels, W, H, D))
    x = torch.randn((2, 5, 512, 512, 32))
    model = RevDose3DUNET()
    preds = model(x)
    print(preds.shape)


if __name__ == "__main__":
    test()
