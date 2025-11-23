import argparse

class TestOptions():
    """This class includes test options.

    """

    def initialize(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--batch_size', type=int, default=1, help='batch_size')
        parser.add_argument('--num_workers', type=int, default=8, help='num_workers')
        parser.add_argument('--device', type=str, default='cuda:0', help='train on which device')
        parser.add_argument('--raw_path', type=str, default='UIEB/test/raw', help='raw image path')
        parser.add_argument('--reference_path', type=str, default='UIEB/test/reference', help='reference image path')
        parser.add_argument('--raw_edge', type=str,default='UIEB/test/raw_edge',help='raw edge path')
        parser.add_argument('--checkpoint', type=str, default='checkpoint/edge_two_loss/', help='checkpoint path')
        parser.add_argument('--output_dir', type=str, default='output/', help='output path')
        # image resize
        parser.add_argument('--image_height', type=int, default=256, help='resize images to this size')
        parser.add_argument('--image_width', type=int, default=256, help='resize images to this size')
        return parser
