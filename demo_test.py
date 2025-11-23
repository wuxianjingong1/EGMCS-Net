import os

import numpy as np
import torch
from torch.utils.data import DataLoader

from options.test_option import TestOptions
from torchvision.transforms import transforms
from data.dataset import GetTestImage
from model.UNet_edge import RGB_UNet,Edge_UNet
# from model.UNet import RGB_UNet,Edge_UNet
from torchvision.utils import save_image
from evaluation.evaluation import evaluate_metrics




if __name__ == '__main__':
    opt = TestOptions().initialize().parse_args()
    batch_size = opt.batch_size
    num_workers = opt.num_workers
    device = opt.device
    raw_path = opt.raw_path
    raw_edge = opt.raw_edge
    reference_path = opt.reference_path
    checkpoint = opt.checkpoint
    output_dir = opt.output_dir
    height = opt.image_height
    width = opt.image_width

    os.makedirs(output_dir, exist_ok=True)


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

    dataloader = DataLoader(
        GetTestImage(raw_path,raw_edge, transforms_=transforms_,edge_transforms_=edge_transforms_),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    # model
    model_rgb = RGB_UNet()
    model_rgb = model_rgb.to(device)
    model_rgb.eval()

    model_edge = Edge_UNet()
    model_edge = model_edge.to(device)
    model_edge.eval()

    mses = []
    psnrs = []
    ssims = []
    lpipss = []
    min_mse = np.inf
    min_psnr = np.inf
    min_ssim = np.inf
    min_lpips = np.inf

    for index in range(5,301,5):
        print(index)
        # model_rgb.load_state_dict(torch.load(checkpoint + 'epoch_'+str(index)+'.pth'))
        model_rgb.load_state_dict(torch.load(checkpoint + 'rgb_epoch_'+str(index)+'.pth'))
        model_edge.load_state_dict(torch.load(checkpoint + 'edge_epoch_'+str(index)+'.pth'))
        # test pipeline
        for i, batch in enumerate(dataloader):
            # load image tensor
            raw_image,raw_edge,name = batch['test'],batch['edge'],batch['name'][0]
            raw_image,raw_edge= raw_image.to(device),raw_edge.to(device)
            out_edge, edge = model_edge(raw_edge)
            pre_image = model_rgb(raw_image,edge)
            # pre_image = model_rgb(raw_image)
            save_image(pre_image, os.path.join(output_dir, name),normalize=True)

        mse, psnr, ssim, lpips = evaluate_metrics(output_dir, reference_path)
        if mse < min_mse:
            min_mse = mse
            min_psnr = psnr
            min_ssim = ssim
            min_lpips = lpips
        mses.append(mse)
        psnrs.append(psnr)
        ssims.append(ssim)
        lpipss.append(lpips)
        if index == 300:
            print('mses:' + str(mses))
            print('psnrs:' + str(psnrs))
            print('ssims:' + str(ssims))
            print('lpipss:' + str(lpipss))
            print(f"Min MSE: {min_mse:.4f}")
            print(f"Min PSNR: {min_psnr:.4f}")
            print(f"Min SSIM: {min_ssim:.4f}")
            print(f"Min LPIPS: {min_lpips:.4f}")

