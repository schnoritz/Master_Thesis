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


def upsample(x):
    return nnf.interpolate(input=x, scale_factor=2, mode="trilinear", align_corners=False)


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
            in_channels, self.features[0], kernel_size=3, padding=1, bias=False)
        self.lastConv = nn.Conv3d(
            3*self.features[0], out_channels, kernel_size=1)

        # create encoder levels
        encoderModules = []
        in_channels = features[0]
        for feature in features[1:]:
            encoderModules.append(EncoderModule(in_channels, feature, depth))
            in_channels = feature

        self.encoders = nn.ModuleList(encoderModules)

        self.bottleneck = EncoderModule(
            self.features[-1], 2*self.features[-1], depth)

        # create decoder levels
        decoderModules = []
        features = features[1:]
        for feature in reversed(features):
            decoderModules.append(DecoderModule(3*feature, feature, depth))
        decoderModules.append(self.lastConv)

        self.decoders = nn.ModuleList(decoderModules)

    def forward(self, x):
        print("-"*50)
        print("DOWN:")

        inputStack = []
        print(x.shape)
        x = self.firstConv(x)
        inputStack.append(x)
        print(x.shape)
        for encoder in self.encoders:
            x = encoder(x)
            print(x.shape)
            inputStack.append(x)
        print("-"*50)
        print("BOTTLENECK:")
        x = self.bottleneck(x)
        print(x.shape)
        print("-"*50)
        print("UP:")

        x = upsample(x)
        print(x.shape)

        for decoder in self.decoders:
            x = torch.cat((inputStack.pop(), x), dim=1)
            print(x.shape)
            x = decoder(x)
            print(x.shape)

        print("-"*50)
        print("OUT:")

        print(x.shape)
        print("-"*50)
        return x


def test():
    # mit x = torch.randn((batch_size, in_channels, W, H, D))
    x = torch.randn((3, 5, 32, 32, 32))
    y = torch.randn((3, 1, 32, 32, 32))
    criterion = torch.nn.MSELoss()
    model = NoNewReversible()
    optimizer = torch.optim.Adam(model.parameters(), lr=1E-4)
    # print(model)
    for epoch in range(1):
        # print(epoch)
        preds = model(x)
        # optimizer.zero_grad()
        # loss = criterion(preds, y)
        # loss.backward()
        # optimizer.step()
        # print(loss.item())


if __name__ == "__main__":
    test()
