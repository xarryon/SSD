import cv2
import numpy as np
import torch
import torch.nn.functional as F

def normal_quantize(image, bits=4, denoise=False):
    """
    带噪声图像量化函数
    
    参数:
        image: 输入图像(灰度)
        bits: 量化位数
        denoise: 是否进行去噪预处理
        
    返回:
        量化后的图像
    """
    # 预处理去噪
    if denoise:
        image = cv2.medianBlur(image, 3)
        image = cv2.GaussianBlur(image, (3, 3), 0)
    
    # 计算量化级别
    levels = 2 ** bits

    quantized = torch.floor(image / (256 / levels)) * (255 / (levels - 1))
    
    return quantized


def local_stats(image, window_size=5):
    """计算图像的局部均值和标准差"""
    pad = window_size // 2
    padded = F.pad(image, (pad, pad), 'constant', 1)
    
    local_mean = torch.zeros_like(image, dtype=torch.float32)
    local_std = torch.zeros_like(image, dtype=torch.float32)
    
    for i in range(pad, padded.shape[0]-pad):
        for j in range(pad, padded.shape[1]-pad):
            window = padded[i-pad:i+pad+1, j-pad:j+pad+1]
            local_mean[i-pad, j-pad] = torch.mean(window)
            local_std[i-pad, j-pad] = torch.std(window)
    
    return local_mean, local_std


def adaptive_quantize(image, bits=4, window_size=3, alpha=0.5):
    """
    基于局部统计的自适应量化
    
    参数:
        image: 输入图像(灰度,0-255)
        bits: 量化位数
        window_size: 局部统计窗口大小(奇数)
        alpha: 调整参数(0-1)，控制局部标准差的影响程度
        
    返回:
        量化后的图像
    """
    # 计算全局量化步长
    levels = 2 ** bits
    global_step = 256 / levels
    
    # 计算局部统计量
    local_mean, local_std = local_stats(image, window_size)
    
    # 归一化局部标准差
    norm_std = (local_std - torch.min(local_std)) / (torch.max(local_std) - torch.min(local_std) + 1e-6)
    
    # 自适应调整量化步长
    adaptive_step = global_step * (1 + alpha * norm_std)
    
    # 应用量化
    quantized = torch.floor((image - local_mean + global_step/2) / adaptive_step) * adaptive_step + local_mean
    
    # 确保值在有效范围内
    quantized = torch.clip(quantized, 0, 255)
    
    return quantized