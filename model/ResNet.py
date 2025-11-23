import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

# ---- Basic Conv Block ----
class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size,
                              stride=stride, padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

# ---- Multi-Scale RFB ----
class RFB(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(RFB, self).__init__()
        self.relu = nn.ReLU(True)
        self.branch0 = nn.Sequential(BasicConv2d(in_channel, out_channel, 1))
        self.branch1 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, (1, 3), padding=(0, 1)),
            BasicConv2d(out_channel, out_channel, (3, 1), padding=(1, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=3, dilation=3)
        )
        self.branch2 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, (1, 5), padding=(0, 2)),
            BasicConv2d(out_channel, out_channel, (5, 1), padding=(2, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=5, dilation=5)
        )
        self.branch3 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, (1, 7), padding=(0, 3)),
            BasicConv2d(out_channel, out_channel, (7, 1), padding=(3, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=7, dilation=7)
        )
        self.conv_cat = BasicConv2d(4 * out_channel, out_channel, 3, padding=1)
        self.conv_res = BasicConv2d(in_channel, out_channel, 1)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        x_cat = self.conv_cat(torch.cat((x0, x1, x2, x3), 1))
        x = self.relu(x_cat + self.conv_res(x))
        return x

# ---- Feature Fusion + Reconstruction ----
class aggregation(nn.Module):
    # dense aggregation, it can be replaced by other aggregation model, such as DSS, amulet, and so on.
    # used after MSF
    def __init__(self, channel):
        super(aggregation, self).__init__()
        self.relu = nn.ReLU(True)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.upsample1 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)
        self.conv_upsample1 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample2 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample3 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample4 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample5 = BasicConv2d(2*channel, 2*channel, 3, padding=1)

        self.conv_concat2 = BasicConv2d(2*channel, 2*channel, 3, padding=1)
        self.conv_concat3 = BasicConv2d(3*channel, 3*channel, 3, padding=1)
        self.conv4 = BasicConv2d(3*channel, 3*channel, 3, padding=1)
        self.reconstruct = nn.Sequential(
            BasicConv2d(3*channel, 32, 3, padding=1),
            nn.Conv2d(32, 3, 1),
            nn.Tanh()  # output in [-1, 1]
        )

    def forward(self, x1, x2, x3):
        x1_1 = x1
        x2_1 = self.conv_upsample1(self.upsample(x1)) + x2
        x3_1 = self.conv_upsample2(self.upsample(self.upsample(x1))) \
               + self.conv_upsample3(self.upsample(x2)) + x3

        x2_2 = torch.cat((x2_1, self.conv_upsample4(self.upsample(x1_1))), 1)
        x2_2 = self.conv_concat2(x2_2)

        x3_2 = torch.cat((x3_1, self.conv_upsample5(self.upsample(x2_2))), 1)
        x3_2 = self.conv_concat3(x3_2)

        x = self.conv4(x3_2)
        x = self.reconstruct(x)
        x=self.upsample1(x)

        return x

# ---- Final Enhancement Network ----
class Net(nn.Module):
    def __init__(self, channel=32):
        super(Net, self).__init__()
        resnet = models.resnet50(pretrained=True)
        self.conv1 = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool
        )
        self.layer1 = resnet.layer1  # 256
        self.layer2 = resnet.layer2  # 512
        self.layer3 = resnet.layer3  # 1024
        self.layer4 = resnet.layer4  # 2048

        self.rfb2 = RFB(512, channel)
        self.rfb3 = RFB(1024, channel)
        self.rfb4 = RFB(2048, channel)

        self.decoder = aggregation(channel)

    def forward(self, x):
        x = self.conv1(x)
        x1 = self.layer1(x)   # not used
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        f2 = self.rfb2(x2)
        f3 = self.rfb3(x3)
        f4 = self.rfb4(x4)
        out = self.decoder(f4, f3, f2)
        return out
