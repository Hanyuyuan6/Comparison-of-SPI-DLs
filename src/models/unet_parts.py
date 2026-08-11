import torch
import torch.nn as nn

# Standard U-Net encoder/decoder blocks (independent implementation).


class DoubleConv(nn.Module):
    """Two (3x3 Conv -> BN -> ReLU) stages; spatial size unchanged."""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        mid_channels = mid_channels or out_channels
        layers = []
        for cin, cout in ((in_channels, mid_channels), (mid_channels, out_channels)):
            layers += [
                nn.Conv2d(cin, cout, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(cout),
                nn.ReLU(inplace=True),
            ]
        self.double_conv = nn.Sequential(*layers)

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Halve the spatial size (maxpool), then DoubleConv."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels),
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Double the spatial size, concatenate the encoder skip, then DoubleConv."""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x_low, x_skip):
        x_low = self.up(x_low)
        # pad if sizes mismatch (no-op for power-of-two inputs)
        if x_low.shape[-2:] != x_skip.shape[-2:]:
            dh = x_skip.size(-2) - x_low.size(-2)
            dw = x_skip.size(-1) - x_low.size(-1)
            x_low = nn.functional.pad(x_low, (dw // 2, dw - dw // 2, dh // 2, dh - dh // 2))
        return self.conv(torch.cat([x_skip, x_low], dim=1))


class OutConv(nn.Module):
    """1x1 conv mapping to the desired number of output channels."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)
