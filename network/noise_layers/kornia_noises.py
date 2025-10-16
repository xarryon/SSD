import torch.nn as nn
import kornia
import numpy as np
import random
# Kornia based noises


# intensity
class GaussianBlur(nn.Module):

    def __init__(self, kernel_size=(3,3), sigma=(2,2), p = 1, train = False):
        super(GaussianBlur, self).__init__()
        self.kernel_size = kernel_size
        self.train_label = train
        self.sigma = sigma
        self.p = p
        
    def forward(self, image_cover_mask):
        if self.train_label:
            kz = int(random.uniform(2, 15))
            kz = kz - 1 if kz % 2 == 0 else kz
            self.kernel_size = (kz, kz)

        self.transform = kornia.augmentation.RandomGaussianBlur(kernel_size=self.kernel_size, sigma=self.sigma, p=1)
        image = image_cover_mask[0]
        
        return self.transform(image).type(image.dtype)


class BoxBlur(nn.Module):

    def __init__(self, kernel_size=(3,3), p=1, train=False):
        super(BoxBlur, self).__init__()
        self.train_label = train
        self.kernel_size = kernel_size
        
    def forward(self, image_cover_mask):
        if self.train_label:
            kz = int(random.uniform(2, 15))
            kz = kz - 1 if kz % 2 == 0 else kz
            self.kernel_size = (kz, kz)
        self.transform = kornia.augmentation.RandomBoxBlur(kernel_size=self.kernel_size, p=1)
        
        image = image_cover_mask[0]
        return self.transform(image).type(image.dtype)


class MotionBlur(nn.Module):

    def __init__(self, kernel_size=(3,3), angle=35, direction=0.5, p=1, train=False):
        super(MotionBlur, self).__init__()
        self.train_label = train
        self.kernel_size = kernel_size
        self.angle = angle
        self.direction = direction
        
    def forward(self, image_cover_mask):
        if self.train_label:
            kz = int(random.uniform(2, 15))
            kz = kz + 1 if kz % 2 == 0 else kz
            self.kernel_size = (kz, kz)
            
        self.transform = kornia.augmentation.RandomMotionBlur(kernel_size=self.kernel_size, angle=self.angle, direction=self.direction, p=1)
        
        image = image_cover_mask[0]
        return self.transform(image).type(image.dtype)
    

class GaussianNoise(nn.Module):

    def __init__(self, mean=0, std=0.1, p=1):
        super(GaussianNoise, self).__init__()
        self.transform = kornia.augmentation.RandomGaussianNoise(mean=mean, std=std, p=p)

    def forward(self, image_cover_mask):
        image = image_cover_mask[0]
        #mask = mask[:, 0: 3, :, :]
        return self.transform(image).type(image.dtype) #image * mask + self.transform(image) * (1 - mask)


class MedianBlur(nn.Module):

    def __init__(self, kernel_size=(3,3), train=False):
        super(MedianBlur, self).__init__()
        self.train_label = train
        self.kernel_size = kernel_size

    def forward(self, image_cover_mask):
        if self.train_label:
            kz = int(random.uniform(2, 15))
            kz = kz - 1 if kz % 2 == 0 else kz
            self.kernel_size = (kz, kz)
        
        self.transform = kornia.augmentation.RandomMedianBlur(kernel_size=self.kernel_size, p=1)
        
        image = image_cover_mask[0]
        
        return self.transform(image).type(image.dtype)


class Brightness(nn.Module):

    def __init__(self, brightness=0.5, p=1):
        super(Brightness, self).__init__()
        self.transform = kornia.augmentation.ColorJitter(brightness=brightness, p=p)

    def forward(self, image_cover_mask):
        image = image_cover_mask[0]
        out = (image + 1 ) / 2
        colorjitter = self.transform(out).type(image.dtype)
        colorjitter = (colorjitter * 2) - 1
        return colorjitter


class Contrast(nn.Module):

    def __init__(self, contrast=0.5, p=1):
        super(Contrast, self).__init__()
        self.transform = kornia.augmentation.ColorJitter(contrast=contrast, p=p)

    def forward(self, image_cover_mask):
        image = image_cover_mask[0]
        out = (image + 1) / 2
        colorjitter = self.transform(out).type(image.dtype)
        colorjitter = (colorjitter * 2) - 1
        return colorjitter


class Saturation(nn.Module):

    def __init__(self, saturation=0.5, p=1):
        super(Saturation, self).__init__()
        self.transform = kornia.augmentation.ColorJitter(saturation=saturation, p=p)

    def forward(self, image_cover_mask):
        image = image_cover_mask[0]
        out = (image + 1) / 2
        colorjitter = self.transform(out).type(image.dtype)
        colorjitter = (colorjitter * 2) - 1
        return colorjitter


class Hue(nn.Module):

    def __init__(self, hue=0.1, p=1):
        super(Hue, self).__init__()
        self.transform = kornia.augmentation.ColorJitter(hue=hue, p=p)

    def forward(self, image_cover_mask):
        image = image_cover_mask[0]
        out = (image + 1) / 2
        colorjitter = self.transform(out).type(image.dtype)
        colorjitter = (colorjitter * 2) - 1
        return colorjitter


# geometric
class Flipping(nn.Module):

    def __init__(self, p=1):
        super(Flipping, self).__init__()
        self.transform = kornia.augmentation.RandomHorizontalFlip(p=p)

    def forward(self, image_cover_mask):
        image = image_cover_mask[0]
        return self.transform(image).type(image.dtype)
    
    
class Rotation(nn.Module):

    def __init__(self, degrees=2.5, p=1):
        super(Rotation, self).__init__()
        self.transform = kornia.augmentation.RandomRotation(degrees=degrees, p=p)

    def forward(self, image_cover_mask):
        image = image_cover_mask[0]
        return self.transform(image).type(image.dtype)


class Affine(nn.Module):

    def __init__(self, degrees=0, translate=0.1, scale=[0.7,0.7], shear=30, p=1):
        super(Affine, self).__init__()
        self.transform = kornia.augmentation.RandomAffine(degrees=degrees, translate=translate, scale=scale, shear=shear, p=p)

    def forward(self, image_cover_mask):
        image = image_cover_mask[0]
        return self.transform(image).type(image.dtype)


class Elastic(nn.Module):

    def __init__(self, kernel = (63, 63), sigma = (32, 32), p=1):
        super(Elastic, self).__init__()
        self.transform = kornia.augmentation.RandomElasticTransform(kernel_size=kernel, sigma=sigma, p=p)

    def forward(self, image_cover_mask):
        image = image_cover_mask[0]
        return self.transform(image).type(image.dtype)


class Grayscale(nn.Module):
    def __init__(self):
        super(Grayscale, self).__init__()
    
    def forward(self, image_cover_mask):
        image = kornia.color.rgb_to_grayscale(image_cover_mask[0]) 
        
        return image.type(image.dtype).repeat(1, 3, 1, 1)
    
