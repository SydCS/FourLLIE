import os

import cv2
import torch

import data.util as data_util
import utils.util as util

# 读取文件夹中的图片文件名
low_folder = 'Eval/Huawei/low'
high_folder = 'Eval/Huawei/high'

low_images = os.listdir(low_folder)
high_images = os.listdir(high_folder)

# 创建输出目录
output_folder_amp_l_phase_h = 'tsne/Huawei/amp_l_phase_h'
output_folder_amp_h_phase_l = 'tsne/Huawei/amp_h_phase_l'
output_folder_amp_lk_phase_l = 'tsne/Huawei/amp_lk_phase_l'
os.makedirs(output_folder_amp_l_phase_h, exist_ok=True)
os.makedirs(output_folder_amp_h_phase_l, exist_ok=True)
os.makedirs(output_folder_amp_lk_phase_l, exist_ok=True)

K = 10

for image_name in low_images:
    # 读取低光照图片
    low_path = os.path.join(low_folder, image_name)
    low_tensor = data_util.read_img_seq([low_path])

    # 读取高光照图片
    high_path = os.path.join(high_folder, image_name)
    high_tensor = data_util.read_img_seq([high_path])

    # 进行傅里叶变换
    _, _, H, W = low_tensor.shape
    low_fft = torch.fft.fft2(low_tensor, norm="backward")
    high_fft = torch.fft.fft2(high_tensor, norm="backward")

    # 获取幅度谱和相位谱
    low_amp = torch.abs(low_fft)
    low_pha = torch.angle(low_fft)

    high_amp = torch.abs(high_fft)
    high_pha = torch.angle(high_fft)

    # 交换 amplitude
    amp_h_phase_l = high_amp * torch.exp(1j * low_pha)
    amp_l_phase_h = low_amp * torch.exp(1j * high_pha)

    # amplitude * K
    amp_lk_phase_l = low_amp * K * torch.exp(1j * low_pha)

    # 逆变换
    tensor_amp_h_phase_l = torch.fft.ifft2(amp_h_phase_l, s=(H, W), norm="backward").real
    tensor_amp_l_phase_h = torch.fft.ifft2(amp_l_phase_h, s=(H, W), norm="backward").real
    tensor_amp_lk_phase_l = torch.fft.ifft2(amp_lk_phase_l, s=(H, W), norm="backward").real

    # 保存结果图片
    amp_h_phase_l_path = os.path.join(output_folder_amp_h_phase_l, image_name)
    amp_l_phase_h_path = os.path.join(output_folder_amp_l_phase_h, image_name)
    amp_lk_phase_l_path = os.path.join(output_folder_amp_lk_phase_l, image_name)

    img_amp_h_phase_l = util.tensor2img(tensor_amp_h_phase_l)
    cv2.imwrite(amp_h_phase_l_path, img_amp_h_phase_l)

    tensor_amp_l_phase_h = util.tensor2img(tensor_amp_l_phase_h)
    cv2.imwrite(amp_l_phase_h_path, tensor_amp_l_phase_h)

    tensor_amp_lk_phase_l = util.tensor2img(tensor_amp_lk_phase_l)
    cv2.imwrite(amp_lk_phase_l_path, tensor_amp_lk_phase_l)
