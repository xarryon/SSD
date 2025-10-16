import torch
import torch.nn as nn
import numpy as np
from . import *

class Random_Noise(nn.Module):
    # batch中每个样本执行不用的noise变换 #
    def __init__(self):
        super(Random_Noise, self).__init__()
        
        GB = GaussianBlur().eval()
        MB = MedianBlur().eval() 
        
        GN = GaussianNoise().eval() 
        SP = SaltPepper().eval() 
        
        B = Brightness().eval() 
        C = Contrast().eval() 
        S = Saturation().eval() 
        H = Hue().eval()
        
        Rz = Resize().eval()
        jpeg = JpegTest().eval()
        
        self.noise =  [GB, GN, MB, SP, B, C, S, H, Rz, jpeg]
        
        self.blur = [GB, MB]
        self.noise = [GN, SP]
        self.color = [B, C, S, H]
        self.others = [Rz, jpeg]
        
        self.noise_sets = [self.blur, self.noise, self.color, self.others]
        
    def forward(self, img, mode = 0):
        
        if torch.min(img) > 0:
            img = img * 2 - 1
            
        forward_image = img.clone().detach()
        noised_image = torch.zeros_like(forward_image)
        
        if mode == 0:
            for index in range(forward_image.shape[0]):
                random_noise_layer = np.random.choice(self.noise, 1)[0] # 随机抽取任意变换
                noised_image[index] += random_noise_layer(forward_image[index].clone().unsqueeze(0))[0]
        
        elif mode == 1:
            for index in range(forward_image.shape[0]):
                random_noise_set = np.random.choice(self.noise_sets, 1)[0] # 随机抽取任意变换
                random_noise_layer = np.random.choice(random_noise_set, 1)[0] 
                noised_image[index] += random_noise_layer(forward_image[index].clone().unsqueeze(0))[0]
                    
        noised_image_gap = noised_image.clamp(-1, 1) - forward_image
    
        if torch.min(img) > 0:
            return 0.5 * (1 + (img + noised_image_gap))
        
        else:
            return img + noised_image_gap


class ALL_Noise(nn.Module):
    # batch中每个样本执行全部noise变换 #
    def __init__(self):
        super(ALL_Noise, self).__init__()
        
        GB = GaussianBlur().eval()
        GN = GaussianNoise().eval() 
        MB = MedianBlur().eval() 
        SP = SaltPepper().eval() 
        B = Brightness().eval() 
        C = Contrast().eval() 
        S = Saturation().eval() 
        H = Hue().eval()
        Rz = Resize().eval()
        jpeg = JpegTest().eval()
        
        self.noise =  [GB, GN, MB, SP, B, C, S, H, Rz, jpeg]


    def forward(self, img):
        forward_image = img.clone().detach()
        
        noised_image = torch.zeros_like(forward_image)
        noised_image = noised_image.repeat(len(self.noise),1,1,1)
        
        for index in range(len(self.noise)):
            noised_image[index] = self.noise[index]([forward_image.clone(), 0])[0]

        noised_image_gap = noised_image.clamp(-1, 1) - forward_image

        return img + noised_image_gap


class Random_SubNoise(nn.Module):
    # batch中每个样本执行不用的noise变换 #
    def __init__(self):
        super(Random_SubNoise, self).__init__()
        
        GB = GaussianBlur().eval()
        MB = MedianBlur().eval() 
        B = Brightness().eval() 
        C = Contrast().eval() 
        S = Saturation().eval() 
        H = Hue().eval()
            
        self.blur = [GB, MB]
        self.color = [B, C, S, H]
        self.noises = [GB, MB, B, C, S, H]
        
    def forward(self, img, mode = 0):
        forward_image = img.clone().detach()

        if mode == 0:        
            noised_image = torch.zeros_like(forward_image).repeat(2, 1, 1, 1)
            
            for index in range(forward_image.shape[0]):
                random_noise_blur_layer = np.random.choice(self.blur, 1)[0] # 随机抽取任意变换
                noised_image[index] += random_noise_blur_layer(forward_image[index].clone().unsqueeze(0))[0]

                random_noise_color_layer = np.random.choice(self.color, 1)[0] # 随机抽取任意变换
                noised_image[index * 2 + 1] += random_noise_color_layer(forward_image[index].clone().unsqueeze(0))[0]

            noised_image_gap = noised_image.clamp(-1, 1) - torch.cat((forward_image, forward_image), dim = 0)
            img = torch.cat((img, img), dim = 0)
            
            return img + noised_image_gap

        elif mode == 1:
            noised_image = torch.zeros_like(forward_image)
            
            for index in range(forward_image.shape[0]):
                random_noise_layer = np.random.choice(self.noises, 1)[0] # 随机抽取任意变换
                noised_image[index] += random_noise_layer(forward_image[index].clone().unsqueeze(0))[0]

            noised_image_gap = noised_image.clamp(-1, 1) - forward_image
        
            return img + noised_image_gap


class Random_Noise_Selected(nn.Module):
    # batch中每个样本执行不用的noise变换 #
    def __init__(self, layers):
        super(Random_Noise_Selected, self).__init__()
        for i in range(len(layers)):
            layers[i] = eval(layers[i])
        self.noise = nn.Sequential(*layers)
        
    def forward(self, image_cover_mask):
        image, cover_image, mask = image_cover_mask[0], image_cover_mask[1], image_cover_mask[2]
        
        forward_image = image.clone().detach() * 2 - 1 # [-1, 1]
        forward_cover_image = cover_image.clone().detach() * 2 - 1 # [-1, 1]
        noised_image_C = torch.zeros_like(forward_image)

        for index in range(forward_image.shape[0]):
            random_noise_layer_C = np.random.choice(self.noise, 1)[0] # 随机抽取任意变换
            noised_image_C[index] = random_noise_layer_C([forward_image[index].clone().unsqueeze(0), forward_cover_image[index].clone().unsqueeze(0)])

        noised_image_gap_C = noised_image_C.clamp(-1, 1) - forward_image
        noised_image_gap_C.to(torch.float32)
        
        noise_img = (2 * image - 1) + noised_image_gap_C

        return torch.clamp(noise_img, -1, 1)