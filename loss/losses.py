# loss.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg16
from pytorch_msssim import ssim

class L1Loss(nn.Module):
    def __init__(self):
        super(L1Loss, self).__init__()
        self.loss = nn.L1Loss()

    def forward(self, pred, target):
        return self.loss(pred, target)

class VGGPerceptualLoss(nn.Module):
    def __init__(self, resize=True):
        super(VGGPerceptualLoss, self).__init__()
        self.vgg = vgg16(pretrained=True).features[:16].eval()
        for param in self.vgg.parameters():
            param.requires_grad = False
        self.resize = resize

    def forward(self, pred, target):
        if self.resize:
            pred = F.interpolate(pred, size=(224, 224), mode='bilinear', align_corners=False)
            target = F.interpolate(target, size=(224, 224), mode='bilinear', align_corners=False)
        return F.l1_loss(self.vgg(pred), self.vgg(target))

class SSIMLoss(nn.Module):
    def __init__(self):
        super(SSIMLoss, self).__init__()

    def forward(self, pred, target):
        return 1 - ssim(pred, target, data_range=2.0, size_average=True)

class BCEWithLogitsLoss(nn.Module):
    def __init__(self):
        super(BCEWithLogitsLoss, self).__init__()
        self.loss = nn.BCEWithLogitsLoss()

    def forward(self, pred, target):
        return self.loss(pred, target)




class DiceLoss(nn.Module):
    def __init__(self,  smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        # logits: 模型原始输出，shape=[B,1,H,W]，未经过sigmoid
        # targets: 0-1标签，float tensor，shape=[B,1,H,W]


        probs = torch.sigmoid(logits)  # 转成概率
        probs_flat = probs.view(probs.size(0), -1)
        targets_flat = targets.view(targets.size(0), -1)

        intersection = (probs_flat * targets_flat).sum(dim=1)
        dice_score = (2 * intersection + self.smooth) / (probs_flat.sum(dim=1) + targets_flat.sum(dim=1) + self.smooth)
        dice_loss = 1 - dice_score.mean()

        return dice_loss


class EdgeLoss(nn.Module):
    def __init__(self,device,bce_weight=0.5, dice_weight=0.5):
        super(EdgeLoss, self).__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.BCEWithLogitsLoss=BCEWithLogitsLoss().to(device)
        self.DiceLoss=DiceLoss().to(device)

    def forward(self, pred, target):
        bce_loss = self.BCEWithLogitsLoss(pred, target)
        dice_loss = self.DiceLoss(pred, target)
        loss=self.bce_weight * bce_loss + self.dice_weight * dice_loss
        return loss,{'bce':bce_loss.item(),'dice':dice_loss.item()}

class RGBLoss(nn.Module):
    def __init__(self, device,lambda_l1=1.0, lambda_percep=0.1, lambda_ssim=0.2):
        super(RGBLoss, self).__init__()
        self.l1 = L1Loss().to(device)
        self.percep = VGGPerceptualLoss().to(device)
        self.ssim = SSIMLoss().to(device)
        self.lambda_l1 = lambda_l1
        self.lambda_percep = lambda_percep
        self.lambda_ssim = lambda_ssim

    def forward(self, pred, target):
        l1_loss = self.l1(pred, target)
        perceptual_loss = self.percep(pred, target)
        ssim_loss = self.ssim(pred, target)

        total = self.lambda_l1 * l1_loss + self.lambda_percep * perceptual_loss + self.lambda_ssim * ssim_loss
        return total, {'L1': l1_loss.item(), 'Perceptual': perceptual_loss.item(), 'SSIM': ssim_loss.item()}
