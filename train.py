import os

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

from options.train_option import TrainOptions
import random
from torchvision.transforms import transforms
from data.dataset import GetTrainingPairs
from model.UNet import RGB_UNet
from loss.losses import RGBLoss, EdgeLoss
import time

os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'


def set_seed(seed):
    random.seed(seed)  # Python 内置随机
    os.environ['PYTHONHASHSEED'] = str(seed)  # 固定 Python 哈希种子（很重要）
    np.random.seed(seed)  # NumPy 随机
    torch.manual_seed(seed)  # PyTorch CPU 随机
    torch.cuda.manual_seed(seed)  # PyTorch GPU 随机
    torch.cuda.manual_seed_all(seed)  # 多 GPU 随机

    # cuDNN 固定配置（会稍微牺牲性能，但换来可复现性）
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def worker_init_fn(worker_id):
    np.random.seed(42 + worker_id)
    torch.manual_seed(42 + worker_id)


if __name__ == '__main__':
    opt = TrainOptions().initialize().parse_args()
    seed = opt.seed
    start_epoch = opt.start_epoch
    num_epoch = opt.num_epoch
    batch_size = opt.batch_size
    num_workers = opt.num_workers
    device = opt.device
    lr = opt.lr
    lr_b1 = opt.b1
    lr_b2 = opt.b2
    raw_path = opt.raw_path
    reference_path = opt.reference_path
    raw_edge = opt.raw_edge
    reference_edge = opt.reference_edge
    checkpoint = opt.checkpoint
    height = opt.image_height
    width = opt.image_width

    set_seed(seed)
    torch.use_deterministic_algorithms(True)

    os.makedirs(checkpoint, exist_ok=True)

    ## Data pipeline
    transforms_ = [
        transforms.Resize((height, width)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]

    edge_transforms_ = [
        # transforms.Resize((height, width),interpolation=InterpolationMode.NEAREST),
        transforms.ToTensor(),
    ]

    g = torch.Generator()
    g.manual_seed(seed)
    dataloader = DataLoader(
        GetTrainingPairs(raw_path, reference_path, raw_edge, reference_edge, transforms_=transforms_,
                         edge_transforms_=edge_transforms_),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        worker_init_fn=worker_init_fn,
        generator=g
    )

    # model
    model_rgb = RGB_UNet()
    model_rgb = model_rgb.to(device)
    model_rgb.train()



    # losses
    rgb_losses = RGBLoss(device)


    # optimizer
    optimizer_rgb = torch.optim.Adam(model_rgb.parameters(), lr=lr, betas=(lr_b1, lr_b2))
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=300, eta_min=1e-6)
    # train pipeline
    for epoch in range(start_epoch, num_epoch + 1):
        epoch_loss = 0
        for iteration, batch in enumerate(dataloader):
            # load image tensor
            raw_images, reference_images, raw_edge, reference_edge = batch['A'], batch['B'], batch['C'], batch['D']
            raw_images, reference_images, raw_edge, reference_edge = raw_images.to(device), reference_images.to(
                device), raw_edge.to(device), reference_edge.to(device)

            # start time
            t0 = time.time()

            # clear the gradients
            optimizer_rgb.zero_grad()


            # model forward
            out_rgb = model_rgb(raw_images)
            # compute loss
            rgb_loss, dict_rgb = rgb_losses(out_rgb, reference_images)
            rgb_loss.backward()

            # rgb_loss.backward()
            epoch_loss += rgb_loss.item()
            optimizer_rgb.step()
            # scheduler.step()
            t1 = time.time()
            print(
                "===> Epoch[{}]({}/{}): L1Loss: {:.4f} || PerceptualLoss: {:.4f} || RGB_SSIMLoss: {:.4f} || Loss: {:.4f} || Learning rate: lr={} || Timer: {:.4f} sec."
                .format(epoch,
                        iteration, len(dataloader), dict_rgb['L1'], dict_rgb['Perceptual'], dict_rgb['SSIM']
                        , rgb_loss.item() ,
                        optimizer_rgb.param_groups[0]['lr'], (t1 - t0)))

        print("===> Epoch {} Complete: Avg. Loss: {:.4f}".format(epoch, epoch_loss / len(dataloader)))

        # save checkpoint
        if epoch % 5 == 0:
            torch.save(model_rgb.state_dict(), checkpoint + 'epoch_' + str(epoch) + '.pth')
