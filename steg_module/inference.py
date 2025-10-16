#!/usr/bin/env python
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import yaml
from easydict import EasyDict
from model import Model
import modules.Unet_common as common
import warnings
import random
import os

import torchvision.transforms as T
from datasets import SSD_Dataset
from torch.utils.data import DataLoader
from utils.common import to, get_random_images, concatenate_images, save_images, o2t
from tqdm import tqdm
from kornia.losses import psnr_loss, ssim_loss
from torchvision.utils import save_image as save_image_torch

from modules.id.id_loss import IDLoss
from collections import OrderedDict

from network.noise_layers import *
from network.noise_layers.noise_layer import Random_Noise_Selected
from steg_module.datasets import SSD_Dataset
from modules.denoiser.denoise_module import DnCNN, Enhancement

warnings.filterwarnings("ignore")

def seed_torch(seed=2025):
    seed = int(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def gauss_noise(shape):
    noise = torch.zeros(shape).cuda()
    for i in range(noise.shape[0]):
        noise[i] = torch.randn(noise[i].shape).cuda()

    return noise


def image_to_patches(x, patch_size):
    if isinstance(patch_size, int):
        patch_h = patch_w = patch_size
    else:
        patch_h, patch_w = patch_size

    assert x.shape[2] % patch_h == 0 and x.shape[3] % patch_w == 0, "Image size must be divisible by patch size."

    patches = x.unfold(2, patch_h, patch_h).unfold(3, patch_w, patch_w)  # (B, C, num_h, num_w, patch_h, patch_w)
    patches = patches.permute(0, 2, 3, 1, 4, 5)  # (B, num_h, num_w, C, patch_h, patch_w)
    patches = patches.reshape(x.shape[0], -1, x.shape[1], patch_h, patch_w)  # (B, num_patches, C, patch_h, patch_w)
    
    return patches


def shuffle_patches(patches):
    """
    Args:
        patches: (B, num_patches, C, P, P)
    Returns:
        shuffled_patches: (B, num_patches, C, P, P)
        indices: (B, num_patches), 打乱顺序的索引
    """
    B, num_patches = patches.shape[0], patches.shape[1]
    
    indices = torch.stack([torch.randperm(num_patches) for _ in range(B)], dim=0).to(patches.device)
    
    shuffled_patches = torch.gather(patches, 1, indices.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(-1, -1, patches.shape[2], patches.shape[3], patches.shape[4]))
    
    return shuffled_patches, indices


def shuffled_patches_to_image(shuffled_patches, original_size, patch_size):
    if isinstance(patch_size, int):
        patch_h = patch_w = patch_size
    else:
        patch_h, patch_w = patch_size

    B, num_patches, C, P1, P2 = shuffled_patches.shape
    H, W = original_size
    num_h, num_w = H // patch_h, W // patch_w

    shuffled_patches = shuffled_patches.reshape(B, num_h, num_w, C, patch_h, patch_w)
    shuffled_patches = shuffled_patches.permute(0, 3, 1, 4, 2, 5)  # (B, C, num_h, patch_h, num_w, patch_w)
    shuffled_image = shuffled_patches.reshape(B, C, H, W)  # (B, C, H, W)

    return shuffled_image


def restore_patches(shuffled_patches, indices):
    B, num_patches = indices.shape[0], indices.shape[1]
    inv_indices = torch.argsort(indices, dim=1)  # (B, num_patches)
    
    restored_patches = torch.gather(shuffled_patches, 1, inv_indices.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(-1, -1, shuffled_patches.shape[2], shuffled_patches.shape[3], shuffled_patches.shape[4]))
    
    return restored_patches


def patches_to_image(patches, original_size, patch_size):
    """
    Args:
        patches: (B, num_patches, C, P, P)
        original_size: (H, W)
        patch_size: int or (int, int)
    Returns:
        x: (B, C, H, W)
    """
    if isinstance(patch_size, int):
        patch_h = patch_w = patch_size
    else:
        patch_h, patch_w = patch_size

    B, num_patches, C, P1, P2 = patches.shape
    H, W = original_size
    num_h, num_w = H // patch_h, W // patch_w

    patches = patches.reshape(B, num_h, num_w, C, patch_h, patch_w)
    patches = patches.permute(0, 3, 1, 4, 2, 5)  # (B, C, num_h, patch_h, num_w, patch_w)
    patches = patches.reshape(B, C, num_h * patch_h, num_w * patch_w)  # (B, C, H, W)
    
    return patches


def get_parameter_number(net):
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}


def load(name, net, optim = None):
    state_dicts = torch.load(name)
    new_state_dict = OrderedDict()
    
    for k, v in state_dicts['net'].items():
        name = 'module.' + k
        new_state_dict[name] = v
        
    net.load_state_dict(new_state_dict)
    
    if optim is not None:
        try:
            optim.load_state_dict(state_dicts['opt'])
        except:
            print('Cannot load optimizer for some reason or other')


def load_dataset(c):
    transform_val = T.Compose([
        # T.CenterCrop(c.cropsize_val),
        T.ToTensor(),
    ])

    # Test data loader
    testloader = DataLoader(
        SSD_Dataset(c, transforms_=transform_val, mode="test"),
        batch_size=c.batchsize_test,
        shuffle=False,
        pin_memory=True,
        num_workers=12,
        drop_last=True
    )
    
    return testloader


def init_model(mod, c):
    for key, param in mod.named_parameters():
        split = key.split('.')
        if param.requires_grad:
            param.data = c.init_scale * torch.randn(param.data.shape).cuda()
            if split[-2] == 'conv5':
                param.data.fill_(0.)


def del_files(folder_path):
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        os.remove(file_path)


def load_noise(c, name=None):
    if name == None:
        noise_layer = Random_Noise_Selected(c.noise_layers).to()
        noise_layer.eval()
    else:
        noise_layer = eval(name)
        
    return noise_layer
    
                
def main(path) -> None:
    seed_torch()
    
    with open(path, 'r') as f:
        c = EasyDict(yaml.load(f, Loader=yaml.SafeLoader))
    
    test_loader = load_dataset(c)
    
    dncnn_weight = torch.load('first.pth')
    dncnn = DnCNN()
    dncnn.load_state_dict(dncnn_weight, strict=True)
    for param in dncnn.parameters():
        param.requires_grad = False
    dncnn = dncnn.cuda().eval()
    denoiser_weight = torch.load('second.pth')
    new_state_dict = {}
    for key, value in denoiser_weight.items():
        if key.startswith('module.'):  
            new_key = key[7:]  
            new_state_dict[new_key] = value
        else:
            new_state_dict[key] = value
    denoiser = Enhancement()
    denoiser.load_state_dict(new_state_dict, strict=True)
    for param in denoiser.parameters():
        param.requires_grad = False
    denoiser = denoiser.cuda().eval()
    
    net = Model(c.size, c.patch_size, c.clamp, 'cuda:0').to()
    weight = torch.load('weights/model_checkpoint.pt')
    net.load_state_dict(weight['net'], strict=True)
    net.eval()
    for param in net.parameters():
        param.requires_grad = False
    net.cuda()
    
    dwt = common.DWT()
    iwt = common.IWT()

    id = IDLoss().cuda().eval()
    
    # load noise_layer
    noise_name = 'Identity()' 
    noise_layer = load_noise(c, noise_name) 
    
    #################
    #     val:      #
    #################
    cnt = 0
    saved_iterations = [1, 2, 3, 4]
    saved_all = None
    id_list = []
    
    with torch.no_grad():
        psnr_s_list = []
        psnr_c_list = []
        ssim_s_list = []
        ssim_c_list = []
        
        net.eval()
        test_pbar = tqdm(
            iterable=None,
            unit="batch",
            total=len(test_loader),
            leave=False,
            desc="Test",)
        
        for x in test_loader:
            # to(x, device)
            x = x.cuda()
            
            if c.batchsize_test == 1:
                cover = x
                secret = x.clone()

            elif c.batchsize_test == 2:
                cover = x.narrow(0, 0, 1)
                secret = x.narrow(0, 1, 1)
                
            # cover_input = sp_spatial(cover, c.cropsize_val // 2)
            cover_input = dwt(cover)
            
            secret_input = dwt(secret) * 0.5
            secret_input = secret_input.narrow(1, 0, c.channels_in)
            
            sec_id = id.extract_feats(2 * secret_input - 1)
            sec_inp, sec_bac = id.extract_hidden(2 * secret_input - 1)
            

            output_steg, output_z_sets, _ = net(cover_input, secret_input, condition = [secret_input, [sec_inp, sec_bac], 0, 1]) # secret_shuffled / secret_input
            steg_img = iwt(output_steg)
            
            noise_steg_img = noise_layer([o2t(steg_img), cover, None]) # other
            
            de_steg_img = 0.5 * (denoiser(dncnn(noise_steg_img)) + 1) # other
            
            output_steg = dwt(de_steg_img)
            
            output_z_gauss = torch.zeros_like(output_z_sets[-1])
            
            for _ in range(1):
                if _ == 0:
                    steg_id = id.extract_feats(output_steg.narrow(1, 0, c.channels_in) - 1)
                    steg_inp, steg_bac = id.extract_hidden(output_steg.narrow(1, 0, c.channels_in) - 1)
                    cover_rev_sets, secret_rev_sets = net(output_steg, output_z_gauss, rev = True, condition = [0.5 * output_steg.narrow(1, 0, c.channels_in), [steg_inp, steg_bac], 0, 1]) # cover_input condition = True
                    secret_rev = secret_rev_sets[-1]
                else:
                    sec_rev_id = id.extract_feats(secret_rev * 2 - 1)
                    sec_rev_inp, sec_rev_bac = id.extract_hidden(secret_rev * 2 - 1)
                    secret_rev_sets = net(output_steg, output_z_gauss, rev = True, condition = [secret_rev, [sec_rev_inp, sec_rev_bac], 0, 1])[1]
                    secret_rev = secret_rev_sets[-1]

            cover_rev = iwt(cover_rev_sets[-1])
            id_sim = id(secret_rev, 0.5 * (output_steg.narrow(1,0,3))).cpu().item()
            
            psnr_s = -psnr_loss(secret_rev.detach(), secret_input, 2).item()
            psnr_c = -psnr_loss(cover.detach(), steg_img, 2).item()
            
            ssim_s = 1 - 2 * ssim_loss(secret_rev.detach(), secret_input, window_size=11, reduction="mean").item()
            ssim_c = 1 - 2 * ssim_loss(cover.detach(), steg_img, window_size=11, reduction="mean").item()
            
            psnr_s_list.append(psnr_s)
            psnr_c_list.append(psnr_c)
            
            ssim_s_list.append(ssim_s)
            ssim_c_list.append(ssim_c)
            
            id_list.append(id_sim)
            
            test_pbar.update(1)
            
            if cnt in saved_iterations:
                if saved_all is None:
                    saved_all = get_random_images(cover, secret, steg_img, F.interpolate(secret_rev, scale_factor=2), de_steg_img)
                else:
                    saved_all = concatenate_images(saved_all, cover, secret, steg_img, F.interpolate(secret_rev, scale_factor=2), de_steg_img)
               
            cnt += 1
            
            
            save_image_torch(torch.cat((de_steg_img, 0.5 * F.interpolate(output_steg.narrow(1,0,3), scale_factor=2), F.interpolate(secret_rev, scale_factor=2)), dim = 3), 'exp.png')
            
        # if accelerator.is_local_main_process:
        psnr_s_m, psnr_c_m, ssim_s_m, ssim_c_m = np.mean(psnr_s_list), np.mean(psnr_c_list), np.mean(ssim_s_list), np.mean(ssim_c_list)
        print(" NoiseType:{} \n PSNR_S:{:.2f}, SSIM_S:{:.4f} | PSNR_C:{:.2f}, SSIM_C:{:.4f}, ID:{:.4f} \n".format(noise_name, psnr_s_m, ssim_s_m, psnr_c_m, ssim_c_m, np.mean(id_list)))