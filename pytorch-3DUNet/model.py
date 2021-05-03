import torch
import torch.nn as nn
import torchvision.transforms.functional as tf

#implementation of this architecture: https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/u-net-architecture.png


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

class UNET(nn.Module):
    def __init__(
        self, in_channels=5, out_channels=1, features=[64, 128, 256, 512]
    ):
        super(UNET, self).__init__()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)

        #DOWN
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature

        #UP
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose3d(feature*2, feature,
                        kernel_size=2, stride=2)
            )
            self.ups.append(DoubleConv(feature*2, feature))

        self.bottleneck = DoubleConv(features[-1], features[-1]*2)
        self.final_conv = nn.Conv3d(features[0], out_channels, kernel_size=1)


    def forward(self, x):
        skip_connections = []
        for down in self.downs:
            print(x.shape)
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx//2]
            
            # not needed due to the fact that input dimension is always (32, 32, 32)
            # if x.shape != skip_connection.shape:
            #     x = tf.resize(x, size=skip_connection.shape[2:])

            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx+1](concat_skip)
            print(x.shape)

        return self.final_conv(x)

def test():
    # mit x = torch.randn((batch_size, in_channels, W, H, D))
    x = torch.randn((10, 5, 32, 32, 32))
    
    model = UNET(in_channels=5, out_channels=1) 
    #print(model)
    preds = model(x)  
    print(f"Inputsize is: {x.shape}")    
    print(f"Outputsize is: {preds.shape}")
    assert preds.shape[2:] == x.shape[2:]

if __name__ == "__main__":
    test()
