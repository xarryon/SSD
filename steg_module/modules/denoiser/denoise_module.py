import torch
from torch import nn
import torch.nn.functional as F
import random
import kornia

import numpy as np

from torchvision.transforms import Resize
from torchvision.transforms.functional import InterpolationMode
from torchvision.utils import save_image
from kornia.augmentation import RandomMedianBlur, RandomGaussianBlur
import torch.nn as nn
import steg_module.modules.denoiser.basicblock as B
import torch.nn.utils.spectral_norm as SpectralNorm


class ConvINRelu(nn.Module):
	"""
	A sequence of Convolution, Instance Normalization, and ReLU activation
	"""

	def __init__(self, channels_in, channels_out, stride):
		super(ConvINRelu, self).__init__()

		self.layers = nn.Sequential(
			nn.Conv2d(channels_in, channels_out, 3, stride, padding=1),
			nn.InstanceNorm2d(channels_out),
			nn.ReLU(inplace=True)
		)

	def forward(self, x):
		return self.layers(x)


class ConvBlock(nn.Module):
	'''
	Network that composed by layers of ConvINRelu
	'''

	def __init__(self, in_channels, out_channels, blocks=1, stride=1):
		super(ConvBlock, self).__init__()

		layers = [ConvINRelu(in_channels, out_channels, stride)] if blocks != 0 else []
		for _ in range(blocks - 1):
			layer = ConvINRelu(out_channels, out_channels, 1)
			layers.append(layer)

		self.layers = nn.Sequential(*layers)

	def forward(self, x):
		return self.layers(x)


class Down(nn.Module):
    def __init__(self, in_channels, out_channels, blocks):
        super(Down, self).__init__()
        self.layer = torch.nn.Sequential(
            ConvBlock(in_channels, in_channels, stride=2),
            ConvBlock(in_channels, out_channels, blocks=blocks)
        )

    def forward(self, x):
        return self.layer(x)
    
    
class UP(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UP, self).__init__()
        self.conv = ConvBlock(in_channels, out_channels)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        return self.conv(x)



def l2_norm(input, axis=1):
    norm = torch.norm(input, 2, axis, True)
    output = torch.div(input, norm+1e-10)
    return output


class SeparableConv2d(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size=1,stride=1,padding=0,dilation=1,bias=False):
        super(SeparableConv2d,self).__init__()

        self.conv1 = nn.Conv2d(in_channels,out_channels,kernel_size,stride,padding,dilation,bias=bias)

    def forward(self,x):
        x = self.conv1(x)

        return x
 
   
class Block(nn.Module):
    def __init__(self,in_filters,out_filters,reps,strides=1,start_with_relu=True,grow_first=True):
        super(Block, self).__init__()

        if out_filters != in_filters or strides!=1:
            self.skip = nn.Conv2d(in_filters,out_filters,1,stride=strides, bias=False)
            self.skipbn = nn.BatchNorm2d(out_filters)
        else:
            self.skip=None

        self.relu = nn.ReLU(inplace=True)
        rep=[]

        filters=in_filters
        if grow_first:
            rep.append(self.relu)
            rep.append(SeparableConv2d(in_filters,out_filters,3,stride=1,padding=1,bias=False))
            rep.append(nn.BatchNorm2d(out_filters))
            filters = out_filters

        for i in range(reps-1):
            rep.append(self.relu)
            rep.append(SeparableConv2d(filters,filters,3,stride=1,padding=1,bias=False))
            rep.append(nn.BatchNorm2d(filters))

        if not grow_first:
            rep.append(self.relu)
            rep.append(SeparableConv2d(in_filters,out_filters,3,stride=1,padding=1,bias=False))
            rep.append(nn.BatchNorm2d(out_filters))

        if not start_with_relu:
            rep = rep[1:]
        else:
            rep[0] = nn.ReLU(inplace=False)

        if strides != 1:
            rep.append(nn.MaxPool2d(3,strides,1))
        self.rep = nn.Sequential(*rep)

    def forward(self,inp):
        x = self.rep(inp)

        if self.skip is not None:
            skip = self.skip(inp)
            skip = self.skipbn(skip)
        else:
            skip = inp

        x+=skip
        return x


class Denoise_module_1(nn.Module):
    def __init__(self):
        super(Denoise_module_1, self).__init__()
        self.fcn = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 3, 3, padding=1))
        
    def random_noise(self, image):
        forward_image = image.clone().detach()
        noised_image = torch.zeros_like(image)
        
        for index in range(image.shape[0]):
            random_noise_layer = np.random.choice(self.noise, 1)[0] # 随机抽取任意变换
            noised_image[index] += random_noise_layer(forward_image[index].clone().unsqueeze(0))[0]

        noise = image - noised_image
        
        return noised_image, noise

    
    def forward(self, image, train = False):
        if train:
            noised_img, noise = self.random_noise(image)
            pred_noise = self.fcn(noised_img)
            pred_img = pred_noise + noised_img
            pred_img = pred_img.clamp(-1, 1)

            return pred_img, noise, pred_noise
        
        else:
            pred_noise = self.fcn(image)
            pred_img = pred_noise + image
            pred_img = pred_img.clamp(-1, 1)
            
            return pred_img


# class Denoise_module_1(nn.Module):
#     def __init__(self):
#         super(Denoise_module_1, self).__init__()
#         self.fcn = nn.Sequential(
#             nn.Conv2d(3, 32, 3, padding=1),
#             nn.BatchNorm2d(32),
#             nn.LeakyReLU(inplace=True),
#             nn.Conv2d(32, 64, 3, padding=1),
#             nn.BatchNorm2d(64),
#             nn.LeakyReLU(inplace=True),
#             nn.Conv2d(64, 128, 3, padding=1),
#             nn.BatchNorm2d(128),
#             nn.LeakyReLU(inplace=True),
#             nn.Conv2d(128, 256, 3, padding=1),
#             nn.BatchNorm2d(256),
#             nn.LeakyReLU(inplace=True),
#             nn.Conv2d(256, 128, 3, padding=1),
#             nn.BatchNorm2d(128),
#             nn.LeakyReLU(inplace=True),
#             nn.Conv2d(128, 64, 3, padding=1),
#             nn.BatchNorm2d(64),
#             nn.LeakyReLU(inplace=True),
#             nn.Conv2d(64, 32, 3, padding=1),
#             nn.BatchNorm2d(32),
#             nn.LeakyReLU(inplace=True),
#             nn.Conv2d(32, 3, 3, padding=1))
        
#     def random_noise(self, image):
#         forward_image = image.clone().detach()
#         noised_image = torch.zeros_like(image)
        
#         for index in range(image.shape[0]):
#             random_noise_layer = np.random.choice(self.noise, 1)[0] # 随机抽取任意变换
#             noised_image[index] += random_noise_layer(forward_image[index].clone().unsqueeze(0))[0]

#         noise = image - noised_image
        
#         return noised_image, noise

    
#     def forward(self, image, train = False):
#         if train:
#             noised_img, noise = self.random_noise(image)
#             pred_noise = self.fcn(noised_img)
#             pred_img = pred_noise + noised_img
#             pred_img = pred_img.clamp(-1, 1)

#             return pred_img, noise, pred_noise
        
#         else:
#             pred_noise = self.fcn(image)
            
#             return pred_noise


class Denoise_module_2(nn.Module):
    def __init__(self):
        super(Denoise_module_2, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, 2, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(32, 64, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(64) # 128

        self.block1 = Block(64, 128, 2, 2, start_with_relu=False, grow_first=True) # 64
        self.block2 = Block(128, 256, 2, 2, start_with_relu=True, grow_first=True) # 32
        self.block3 = Block(256, 512, 2, 2, start_with_relu=True, grow_first=True) # 16
        
        self.hidden = Block(512, 512, 3, 1) # 16
        
        self.decoder4_1 = nn.Sequential(
            nn.UpsamplingNearest2d(scale_factor=2),
            SeparableConv2d(512, 256, 3, 1, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True))
        self.decoder4_2 = Block(256, 256, 3, 1) # 32
        
        self.decoder3_1 = nn.Sequential(
            nn.UpsamplingNearest2d(scale_factor=2),
            SeparableConv2d(256, 128, 3, 1, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True))
        self.decoder3_2 = Block(128, 128, 3, 1) # 64
        
        self.decoder2_1 = nn.Sequential(
            nn.UpsamplingNearest2d(scale_factor=2),
            SeparableConv2d(128, 64, 3, 1, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True))
        self.decoder2_2 = Block(64, 64, 3, 1) # 128
        
        self.decoder1_1 = nn.Sequential(
            nn.UpsamplingNearest2d(scale_factor=2),
            SeparableConv2d(64, 32, 3, 1, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True))
        self.decoder1_2 = nn.Sequential(
            nn.Conv2d(32, 3, 1, 1, bias=False),) # 256
        
        self.gap = nn.AdaptiveAvgPool2d(1)
    
    
    def forward(self, input, train = False):
        
        x = self.conv1(input)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        enc_0 = self.relu(x)

        enc_1 = self.block1(enc_0) #　128
        enc_2 = self.block2(enc_1) #　256
        enc_3 = self.block3(enc_2) #　512
        
        hidden = self.hidden(enc_3)
        
        rec = self.decoder4_1(hidden)
        rec0 = self.decoder4_2(rec + enc_2)

        rec = self.decoder3_1(rec0) 
        rec1 = self.decoder3_2(rec + enc_1)
        
        rec = self.decoder2_1(rec1)
        rec2 = self.decoder2_2(rec + enc_0)
        
        rec = self.decoder1_1(rec2)
        pred = self.decoder1_2(rec)
        
        rec_image = pred.clone().detach()
        gap = rec_image.clamp(-1, 1) - rec_image
        
        return pred + gap


class Denoise(nn.Module):
    def __init__(self, module1_pth, module2_pth, device):
        super(Denoise, self).__init__()
        self.module_1 = Denoise_module_1()
        self.module_2 = Denoise_module_2()

        weight1 = torch.load(module1_pth, map_location = device)
        weight1_state_dict = {k[7:]:v for k,v in weight1.items()}
        self.module_1.load_state_dict(weight1_state_dict)
        self.module_1.eval()
        
        weight2 = torch.load(module2_pth, map_location = device)
        weight2_state_dict = {k[7:]:v for k,v in weight2.items()}
        self.module_2.load_state_dict(weight2_state_dict)
        self.module_2.eval()
        
        self.MB = RandomMedianBlur(p=1).eval()
        
    def forward(self, input):
        input = self.MB(input.clone())
        input = self.module_1(input, False)
        input = self.module_2(input, False)
        
        return input


class FeatureExDilatedResNet(nn.Module):
    def __init__(self, ngf = 64):
        super(FeatureExDilatedResNet, self).__init__()
        self.conv1 = nn.Sequential(
            SpectralNorm(nn.Conv2d(3, ngf, 3, 1, 1)),
            nn.LeakyReLU(0.2),
            )
        self.stem1_1 = resnet_block(ngf, dilation=[7,5])
        self.stem1_2 = resnet_block(ngf, dilation=[7,5])

        self.conv2 = nn.Sequential(
            SpectralNorm(nn.Conv2d(ngf, ngf*2, 3, 2, 1)),
            nn.LeakyReLU(0.2),
            )
        self.stem2_1 = resnet_block(ngf*2, dilation=[5,3])
        self.stem2_2 = resnet_block(ngf*2, dilation=[5,3])

        self.conv3 = nn.Sequential(
            SpectralNorm(nn.Conv2d(ngf*2, ngf*4, 3, 2, 1)),
            nn.LeakyReLU(0.2),
            )

        self.stem3_1 = resnet_block(ngf*4, dilation=[3,1])
        self.stem3_2 = resnet_block(ngf*4, dilation=[3,1])

        self.conv4 = nn.Sequential(
            SpectralNorm(nn.Conv2d(ngf*4, ngf*2, 3, 1, 1)),
            nn.LeakyReLU(0.2),
            )

    def forward(self, img): #
        fea1 = self.stem1_2(self.stem1_1(self.conv1(img)))
        fea2 = self.stem2_2(self.stem2_1(self.conv2(fea1)))
        fea3 = self.stem3_2(self.stem3_1(self.conv3(fea2)))
        fea4 = self.conv4(fea3)
        return fea4


def resnet_block(in_channels, conv_layer = nn.Conv2d, norm_layer = nn.BatchNorm2d, kernel_size = 3, dilation = [1,1], bias=True):
    return ResnetBlock(in_channels,conv_layer, norm_layer, kernel_size, dilation, bias=bias)


class ResnetBlock(nn.Module):
    def __init__(self, in_channels,conv_layer = nn.Conv2d, norm_layer = nn.BatchNorm2d, kernel_size = 3, dilation = [1,1], bias=True):
        super(ResnetBlock, self).__init__()
        self.stem = nn.Sequential(
            SpectralNorm(conv_layer(in_channels, in_channels, kernel_size=kernel_size, stride=1, dilation=dilation[0], padding=((kernel_size-1)//2)*dilation[0], bias=bias)),
            nn.LeakyReLU(0.2),
            SpectralNorm(conv_layer(in_channels, in_channels, kernel_size=kernel_size, stride=1, dilation=dilation[1], padding = ((kernel_size-1)//2)*dilation[1], bias=bias)),
        )
    def forward(self, x):
        out = self.stem(x) + x
        return out


class UpDilateResBlock(nn.Module):
    def __init__(self, dim, dilation=[2,1] ):
        super(UpDilateResBlock, self).__init__()
        self.Model0 = nn.Sequential(
            SpectralNorm(nn.Conv2d(dim, dim, 3, 1, ((3-1)//2)*dilation[0], dilation[0])),
            nn.LeakyReLU(0.2),
            SpectralNorm(nn.Conv2d(dim, dim, 3, 1, ((3-1)//2)*dilation[0], dilation[0])),
            nn.LeakyReLU(0.2),
        )
        self.Model1 = nn.Sequential(
            SpectralNorm(nn.Conv2d(dim, dim, 3, 1, ((3-1)//2)*dilation[1], dilation[1])),
            nn.LeakyReLU(0.2),
            SpectralNorm(nn.Conv2d(dim, dim, 3, 1, ((3-1)//2)*dilation[1], dilation[1])),
            nn.LeakyReLU(0.2),
        )
    def forward(self, x):
        out = x + self.Model0(x)
        out2 = out + self.Model1(out)
        return out2

             
class Enhancement(nn.Module):
    def __init__(self):
        super(Enhancement, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, 2, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(32, 64, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(64) # 128

        self.block1 = Block(64, 128, 2, 2, start_with_relu=False, grow_first=True) # 64
        self.block2 = Block(128, 256, 2, 2, start_with_relu=True, grow_first=True) # 32
        self.block3 = Block(256, 512, 2, 2, start_with_relu=True, grow_first=True) # 16
        self.block4 = Block(512, 1024, 2, 2, start_with_relu=True, grow_first=True) # 8
        
        self.decoder5_1 = nn.Sequential(
            nn.UpsamplingNearest2d(scale_factor=2),
            SeparableConv2d(1024, 512, 3, 1, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True))
        self.decoder5_2 = Block(512, 512, 3, 1) # 32

        self.decoder4_1 = nn.Sequential(
            nn.UpsamplingNearest2d(scale_factor=2),
            SeparableConv2d(512, 256, 3, 1, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True))
        self.decoder4_2 = Block(256, 256, 3, 1) # 32
        
        self.decoder3_1 = nn.Sequential(
            nn.UpsamplingNearest2d(scale_factor=2),
            SeparableConv2d(256, 128, 3, 1, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True))
        self.decoder3_2 = Block(128, 128, 3, 1) # 64
        
        self.decoder2_1 = nn.Sequential(
            nn.UpsamplingNearest2d(scale_factor=2),
            SeparableConv2d(128, 64, 3, 1, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True))
        self.decoder2_2 = Block(64, 64, 3, 1) # 128
        
        self.decoder1_1 = nn.Sequential(
            nn.UpsamplingNearest2d(scale_factor=2),
            SeparableConv2d(64, 32, 3, 1, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True))
        self.decoder1_2 = nn.Sequential(
            nn.Conv2d(32, 3, 1, 1, bias=False),) # 256
        
        self.gap = nn.AdaptiveAvgPool2d(1)

        self.fuse4 = nn.Sequential(nn.Conv2d(512 * 2, 512, kernel_size=3, stride=1, padding=1),
                                   nn.ReLU(True))
        self.fuse3 = nn.Sequential(nn.Conv2d(256 * 2, 256, kernel_size=3, stride=1, padding=1),
                                   nn.ReLU(True))
        self.fuse2 = nn.Sequential(nn.Conv2d(128 * 2, 128, kernel_size=3, stride=1, padding=1),
                                   nn.ReLU(True))
        self.fuse1 = nn.Sequential(nn.Conv2d(64 * 2, 64, kernel_size=3, stride=1, padding=1),
                                   nn.ReLU(True))
        
        self.MB = RandomMedianBlur(p=1).eval()
        
    def forward(self, input):
        input = self.MB(input)
        
        x = self.conv1(input)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        enc_0 = self.relu(x)

        enc_1 = self.block1(enc_0) #　128
        enc_2 = self.block2(enc_1) #　256
        enc_3 = self.block3(enc_2) #　512
        enc_4 = self.block4(enc_3) #  1024
        
        rec = self.decoder5_1(enc_4)
        rec_ = self.decoder5_2(self.fuse4(torch.cat((rec, enc_3), dim = 1)))

        rec = self.decoder4_1(rec_)
        rec0 = self.decoder4_2(self.fuse3(torch.cat((rec, enc_2), dim = 1)))

        rec = self.decoder3_1(rec0) 
        rec1 = self.decoder3_2(self.fuse2(torch.cat((rec, enc_1), dim = 1)))
        
        rec = self.decoder2_1(rec1)
        rec2 = self.decoder2_2(self.fuse1(torch.cat((rec, enc_0), dim = 1)))
        
        rec = self.decoder1_1(rec2)
        pred = self.decoder1_2(rec)
        
        rec_image = pred.clone().detach()
        gap = rec_image.clamp(-1, 1) - rec_image
        
        return pred + gap
    
    
class DnCNN(nn.Module):
    def __init__(self, in_nc=3, out_nc=3, nc=64, nb=20, act_mode='R'):

        super(DnCNN, self).__init__()
        assert 'R' in act_mode or 'L' in act_mode, 'Examples of activation function: R, L, BR, BL, IR, IL'
        bias = True

        m_head = B.conv(in_nc, nc, mode='C'+act_mode[-1], bias=bias)
        m_body = [B.conv(nc, nc, mode='C'+act_mode, bias=bias) for _ in range(nb-2)]
        m_tail = B.conv(nc, out_nc, mode='C', bias=bias)

        self.model = B.sequential(m_head, *m_body, m_tail)

    def forward(self, x):
        x = 0.5 * (x + 1)
        n = self.model(x)
        oup = x - n
        return 2 * oup - 1
    