import cv2
import numpy as np
from PIL import Image
import os

def get_canny_edge(input_path, save_path=None):
    # 1. 读取图像并转为灰度图
    img = cv2.imread(input_path)
    img = cv2.resize(img, (256,256), interpolation=cv2.INTER_LINEAR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 2. 应用 Canny 边缘检测
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    edge = cv2.Canny(gray, threshold1=50, threshold2=100)
    if save_path:
        cv2.imwrite(save_path, edge)

    return edge



def batch_canny_edge(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for fname in os.listdir(input_dir):
        if fname.lower().endswith(('.jpg', '.png', '.jpeg')):
            in_path = os.path.join(input_dir, fname)
            out_path = os.path.join(output_dir, fname)
            get_canny_edge(in_path, save_path=out_path)


batch_canny_edge('../UIEB/test/raw', '../UIEB/test/raw_edge')
