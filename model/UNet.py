import torch
import torch.nn as nn
import torch.nn.functional as F


class UNetDown(nn.Module):
    def __init__(self, in_size, out_size, bn=True):
        super(UNetDown, self).__init__()
        layers = [nn.Conv2d(in_size, out_size, 4, 2, 1, bias=False)]
        if bn:
            layers.append(nn.BatchNorm2d(out_size))
        layers.append(nn.ReLU(inplace=True))  # 用 ReLU 替代 LeakyReLU
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class UNetUp(nn.Module):
    def __init__(self, in_size, out_size):
        super(UNetUp, self).__init__()
        layers = [
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(in_size, out_size, 3, stride=1, padding=1),
            nn.BatchNorm2d(out_size),
            nn.ReLU(inplace=True)
        ]
        self.model = nn.Sequential(*layers)
        self.fuse = nn.Sequential(
            nn.Conv2d(out_size * 2, out_size, 3, padding=1),
            nn.BatchNorm2d(out_size),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, skip_input):
        x = self.model(x)
        x = torch.cat((x, skip_input), dim=1)
        x = self.fuse(x)
        return x


class RGB_UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super(RGB_UNet, self).__init__()
        self.down1 = UNetDown(in_channels, 64, bn=False)
        self.down2 = UNetDown(64, 128)
        self.down3 = UNetDown(128, 256)
        self.down4 = UNetDown(256, 256)
        self.down5 = UNetDown(256, 256, bn=False)

        self.up1 = UNetUp(256, 256)
        self.up2 = UNetUp(256, 256)
        self.up3 = UNetUp(256, 128)
        self.up4 = UNetUp(128, 64)

        self.final = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(64, out_channels, kernel_size=3,  padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)

        u1 = self.up1(d5, d4)
        u2 = self.up2(u1, d3)
        u3 = self.up3(u2, d2)
        u4 = self.up4(u3, d1)

        out = self.final(u4)
        return out