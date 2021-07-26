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
        x = self.conv(F.leaky_relu(self.gn(x)))
        return x


def makeReversibleSequence(channels):
    innerChannels = channels // 2
    groups = 64 // 2
    fBlock = ResidualInner(innerChannels, groups)
    gBlock = ResidualInner(innerChannels, groups)
    #gBlock = nn.Sequential()
    return rv.ReversibleBlock(fBlock, gBlock)


def makeReversibleComponent(channels, blockCount):
    modules = []
    for i in range(blockCount):
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
        if self.upsample:
            self.conv = nn.Conv3d(inChannels, outChannels, 1)

    def forward(self, x):
        x = self.reversibleBlocks(x)
        if self.upsample:
            x = self.conv(x)
            x = F.interpolate(x, scale_factor=2,
                              mode="trilinear", align_corners=False)
        return x


class NoNewReversible(nn.Module):
    def __init__(self, in_channels=5, out_channels=1, features=[64, 128, 256]):
        super(NoNewReversible, self).__init__()
        depth = 1
        self.features = features

        self.firstConv = nn.Conv3d(
            in_channels, self.features[0], 3, padding=1, bias=False)
        self.lastConv = DecoderModule(features[0], out_channels, depth)

        # create encoder levels
        encoderModules = []
        in_channels = features[0]
        for feature in features:
            encoderModules.append(EncoderModule(in_channels, feature, depth))
            in_channels = feature

        self.encoders = nn.ModuleList(encoderModules)

        self.bottleneck = EncoderModule(
            self.features[-1], 2*self.features[-1], depth)

        # create decoder levels
        decoderModules = []
        for feature in reversed(features):
            decoderModules.append(DecoderModule(2*feature, feature, depth))

        self.decoders = nn.ModuleList(decoderModules)

    def forward(self, x):
        x = self.firstConv(x)

        inputStack = []
        for i in range(len(self.features)):
            x = self.encoders[i](x)
            inputStack.append(x)

        x = self.bottleneck(x)

        for i in range(len(self.features)):
            x = self.decoders[i](x)
            x = x + inputStack.pop()

        x = self.lastConv(x)
        x = torch.sigmoid(x)
        return x


def test():
    # mit x = torch.randn((batch_size, in_channels, W, H, D))
    x = torch.randn((10, 5, 32, 32, 32))
    y = torch.randn((10, 1, 32, 32, 32))
    criterion = torch.nn.MSELoss()
    model = NoNewReversible()
    optimizer = torch.optim.Adam(model.parameters(), lr=1E-4)
    # print(model)
    for epoch in range(100):
        print(epoch)
        preds = model(x)
        optimizer.zero_grad()
        loss = criterion(preds, y)
        loss.backward()
        optimizer.step()
        print(loss.item())


if __name__ == "__main__":
    test()
