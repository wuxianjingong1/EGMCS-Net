import os

import torch
from torch.utils.data import DataLoader

from options.test_option import TestOptions
from torchvision.transforms import transforms
from data.dataset import GetTestImage
from model.UNet import RGB_UNet
from torchvision.utils import save_image




if __name__ == '__main__':
    opt = TestOptions().initialize().parse_args()
    batch_size = opt.batch_size
    num_workers = opt.num_workers
    device = opt.device
    raw_path = opt.raw_path
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

    dataloader = DataLoader(
        GetTestImage(raw_path, transforms_=transforms_),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    # model
    model = RGB_UNet()
    model = model.to(device)
    model.load_state_dict(torch.load(checkpoint+'epoch_40.pth'))
    model.eval()

    # test pipeline
    for i, batch in enumerate(dataloader):
        print(i)
        # load image tensor
        raw_image,name = batch['test'],batch['name'][0]
        raw_image= raw_image.to(device)
        pre_image = model(raw_image)
        save_image(pre_image, os.path.join(output_dir, name),normalize=True)
