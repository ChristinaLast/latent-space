import torch
import torch.nn as nn
import numpy as np


class Decoder(nn.Module):
    def __init__(self, device) -> None:
        super().__init__()
        # Upsample the image
        self.decoder = []
        self.fact = 2
        self.conv_depth = 8  # int(log(input_size)/log(2)).ceilfiles = [f for f in os.listdir('.') if re.match(r'[0-9]+.*\.jpg', f)]

        self.Nc_init = 2048
        # fully connected layer
        for n in torch.arange(0, self.conv_depth):
            self.decoder.append(
                Up(
                    self.Nc_init // np.power(self.fact, n),
                    self.Nc_init // np.power(self.fact, n + 1),
                    bilinear=True,
                )
            )

        self.decoder.append(
            DoubleConv(self.Nc_init // np.power(self.fact, self.conv_depth), 3)
        )
        self.decoder = nn.ModuleList(self.decoder)

    def forward(self, x):
        for layer in self.decoder:
            x = layer(x)
        return x
    


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(
        self, in_channels, out_channels, mid_channels=None, bilinear=True
    ):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            if not mid_channels:
                mid_channels = out_channels
            self.up = nn.Upsample(
                scale_factor=2, mode="bilinear", align_corners=True
            )

            # add a conv layer to divide by two the number of channels
            self.conv = DoubleConv(in_channels, out_channels, mid_channels)
        else:  # might not work
            self.up = nn.ConvTranspose2d(
                in_channels, in_channels // 2, kernel_size=2, stride=2
            )
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1):
        x1 = self.up(x1)
        return self.conv(x1)


class DoubleConv(nn.Module):
    """(convolution => [BN] => PReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.PReLU(),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.PReLU(),
        )

    def forward(self, x):
        return self.double_conv(x)