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


class Concat(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Concat, self).__init__()
        self.fuse = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.gate = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1),
            nn.Sigmoid()
        )

    def forward(self, rgb, edge):
        x = torch.cat((rgb, edge), dim=1)
        fused = self.fuse(x)
        gated = self.gate(x)
        out = fused * gated + rgb * (1 - gated)
        return out


class RGB_UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super(RGB_UNet, self).__init__()
        self.down1 = UNetDown(in_channels, 64, bn=False)
        self.concat1 = Concat(128, 64)
        self.down2 = UNetDown(64, 128)
        self.concat2 = Concat(256, 128)
        self.down3 = UNetDown(128, 256)
        self.concat3 = Concat(512, 256)
        self.down4 = UNetDown(256, 256)
        self.concat4 = Concat(512, 256)
        self.down5 = UNetDown(256, 256, bn=False)
        self.concat5 = Concat(512, 256)

        self.up1 = UNetUp(256, 256)
        self.concat6 = Concat(512, 256)
        self.up2 = UNetUp(256, 256)
        self.concat7 = Concat(512, 256)
        self.up3 = UNetUp(256, 128)
        self.concat8 = Concat(256, 128)
        self.up4 = UNetUp(128, 64)
        self.concat9 = Concat(128, 64)


        self.final = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(64, out_channels, kernel_size=3, padding=1),
            nn.Tanh()
        )


    def forward(self, x, edge):

        d1 = self.down1(x)
        d1 = self.concat1(d1, edge['down1'])
        d2 = self.down2(d1)
        d2 = self.concat2(d2, edge['down2'])
        d3 = self.down3(d2)
        d3 = self.concat3(d3, edge['down3'])
        d4 = self.down4(d3)
        d4 = self.concat4(d4, edge['down4'])
        d5 = self.down5(d4)
        d5 = self.concat5(d5, edge['down5'])

        u1 = self.up1(d5, d4)
        u1 = self.concat6(u1, edge['u1'])
        u2 = self.up2(u1, d3)
        u2 = self.concat7(u2, edge['u2'])
        u3 = self.up3(u2, d2)
        u3 = self.concat8(u3, edge['u3'])
        u4 = self.up4(u3, d1)
        u4 = self.concat9(u4, edge['u4'])

        out = self.final(u4)
        return out


class Edge_UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=3):
        super(Edge_UNet, self).__init__()
        self.edge_pre = nn.Sequential(
            nn.Conv2d(in_channels, 3, kernel_size=1),  # 将 edge 单通道映射到 RGB 维度
            nn.BatchNorm2d(3),
            nn.ReLU(inplace=True)
        )
        self.down1 = UNetDown(out_channels, 64, bn=False)
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
            nn.Conv2d(64, out_channels, kernel_size=3, padding=1),
            # nn.Sigmoid()
        )

        self.edge_final = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(64, 1, kernel_size=3, padding=1),
            # nn.Sigmoid()
        )

    def forward(self, x):
        x=self.edge_pre(x)
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)

        u1 = self.up1(d5, d4)
        u2 = self.up2(u1, d3)
        u3 = self.up3(u2, d2)
        u4 = self.up4(u3, d1)

        edge_out=self.edge_final(u4)
        out = self.final(u4)
        edge = {'down1': d1, 'down2': d2, 'down3': d3, 'down4': d4, 'down5': d5, 'u1': u1, 'u2': u2, 'u3': u3, 'u4': u4,
                'out': out, }
        return edge_out, edge
