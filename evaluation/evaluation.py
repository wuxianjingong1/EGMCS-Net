import os
import torch
import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as compare_ssim
import lpips
import math

# -------- 指标函数 --------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



def calculate_mse(img1, img2):
    # 输入均为numpy数组，值域[0,255]
    img1=img1.astype(np.float32)/ 255.0
    img2=img2.astype(np.float32)/ 255.0
    mse = np.mean(( img1-img2 ) ** 2)
    return mse

def calculate_psnr(mse, max_pixel=1.0):
    if mse == 0:
        return float('inf')
    psnr = 20 * math.log10(max_pixel / math.sqrt(mse))
    return psnr

def calculate_ssim(img1, img2):
    # 单通道或多通道（RGB）均可
    # skimage的SSIM，multichannel=True 表示彩色图
    img1 = img1.astype(np.float32) / 255.0
    img2 = img2.astype(np.float32) / 255.0
    ssim = compare_ssim(img1, img2, channel_axis=-1, data_range=1.0)
    return ssim

def calculate_lpips(img1, img2, loss_fn):
    # 输入是PIL图转成torch.tensor，值归一化[-1,1]
    # img shape: (H,W,C), 需要转成 (1,C,H,W)
    def preprocess(pil_img):
        img = np.array(pil_img).astype(np.float32) / 255.0
        img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)  # 1,C,H,W
        img = img * 2 - 1  # [0,1] -> [-1,1]
        return img

    img1_t = preprocess(img1).to(device)
    img2_t = preprocess(img2).to(device)
    with torch.no_grad():
        dist = loss_fn(img1_t, img2_t)
    return dist.item()

# -------- 主函数 --------

def evaluate_metrics(enhanced_dir, gt_dir):
    enhanced_list = sorted([f for f in os.listdir(enhanced_dir) if f.endswith(('.png','.jpg','.jpeg'))])
    gt_list = sorted([f for f in os.listdir(gt_dir) if f.endswith(('.png','.jpg','.jpeg'))])

    assert len(enhanced_list) == len(gt_list), "增强图和GT数量不匹配"

    loss_fn = lpips.LPIPS(net='alex').to(device)

    mse_all, psnr_all, ssim_all, lpips_all = [], [], [], []

    for en_name, gt_name in zip(enhanced_list, gt_list):
        en_path = os.path.join(enhanced_dir, en_name)
        gt_path = os.path.join(gt_dir, gt_name)

        en_img = Image.open(en_path).convert('RGB').resize((256, 256), Image.BICUBIC)
        gt_img = Image.open(gt_path).convert('RGB').resize((256, 256), Image.BICUBIC)

        en_np = np.array(en_img)
        gt_np = np.array(gt_img)

        mse = calculate_mse(en_np, gt_np)
        psnr = calculate_psnr(mse)
        ssim = calculate_ssim(en_np, gt_np)
        lpips_score = calculate_lpips(en_img, gt_img, loss_fn)

        mse_all.append(mse)
        psnr_all.append(psnr)
        ssim_all.append(ssim)
        lpips_all.append(lpips_score)

    #     print(f"{en_name} | MSE: {mse:.4f}, PSNR: {psnr:.4f}, SSIM: {ssim:.4f}, LPIPS: {lpips_score:.4f}")
    #
    # print("\n=== Average metrics ===")
    # print(f"Avg MSE: {np.mean(mse_all):.4f}")
    # print(f"Avg PSNR: {np.mean(psnr_all):.4f}")
    # print(f"Avg SSIM: {np.mean(ssim_all):.4f}")
    # print(f"Avg LPIPS: {np.mean(lpips_all):.4f}")
    return np.mean(mse_all),np.mean(psnr_all),np.mean(ssim_all),np.mean(lpips_all)

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    enhanced_images_dir = './enhanced'  # 你的增强图文件夹路径
    gt_images_dir = './gt'              # 你的GT图文件夹路径

    evaluate_metrics(enhanced_images_dir, gt_images_dir)
