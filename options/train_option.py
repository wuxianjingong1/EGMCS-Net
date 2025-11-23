import argparse

class TrainOptions():
    """This class includes train options.

    """

    def initialize(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--seed', type=int, default=42, help='# of test examples.')
        parser.add_argument('--start_epoch', type=int, default=1, help='the epoch to start training.')
        parser.add_argument('--num_epoch', type=int, default=300, help='the epoch to finish training')
        parser.add_argument('--batch_size', type=int, default=16, help='batch_size')
        parser.add_argument('--num_workers', type=int, default=8, help='num_workers')
        parser.add_argument('--device', type=str, default='cuda:0', help='train on which device')
        parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
        parser.add_argument('--b1', type=float, default=0.9,help='decay of 1st order momentum')
        parser.add_argument('--b2', type=float, default=0.999, help='decay of 2nd order momentum')
        parser.add_argument('--raw_path', type=str, default='UIEB/train/raw', help='raw image path')
        parser.add_argument('--reference_path', type=str, default='UIEB/train/reference', help='reference image path')
        parser.add_argument('--raw_edge', type=str, default='UIEB/train/raw_edge', help='raw edge path')
        parser.add_argument('--reference_edge', type=str, default='UIEB/train/reference_edge', help='reference edge path')
        parser.add_argument('--checkpoint', type=str, default='checkpoint/', help='checkpoint path')
        # image resize
        parser.add_argument('--image_height', type=int, default=256, help='resize images to this size')
        parser.add_argument('--image_width', type=int, default=256, help='resize images to this size')
        return parser
