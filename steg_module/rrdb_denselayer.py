import torch
import torch.nn as nn
import torch.nn.functional as F1
import steg_module.modules.module_util as mutil
from steg_module.modules.block import ConvBlock, Down, UP, Down_1x1, UP_1x1, AdaIN
import torchvision.transforms.functional as F


# Dense connection
class ResidualDenseBlock_out(nn.Module):
    def __init__(self, input, output, hidden=32, bias=True, final=False):
        super(ResidualDenseBlock_out, self).__init__()
        self.conv1 = nn.Conv2d(input, hidden, 3, 1, 1, bias=bias)
        self.conv2 = nn.Conv2d(input + hidden, hidden, 3, 1, 1, bias=bias)
        self.conv3 = nn.Conv2d(input + 2 * hidden, hidden, 3, 1, 1, bias=bias)
        self.conv4 = nn.Conv2d(input + 3 * hidden, hidden, 3, 1, 1, bias=bias)
        self.conv5 = nn.Conv2d(input + 4 * hidden, output, 3, 1, 1, bias=bias)
        self.lrelu = nn.LeakyReLU(inplace=True)
        
        self.sigmoid = nn.Sigmoid()
        self.final = final
        # initialization
        mutil.initialize_weights([self.conv5], 0.)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        
        if self.final:
            x5 = 2 * self.sigmoid(x5)
            
        return x5


class ResidualUnetBlock_out(nn.Module):
    def __init__(self, input, output, hidden = 16, bias = True, size = 'identity'):
        super(ResidualUnetBlock_out, self).__init__()
        self.conv1 = ConvBlock(input, hidden)
        
        self.down1 = Down(hidden, hidden * 2, blocks=1)
        self.down2 = Down(hidden * 2, hidden * 4, blocks=1)
        self.down3 = Down(hidden * 4, hidden * 8, blocks=1)
        
        self.up3 = UP(hidden * 8, hidden * 4)
        self.up2 = UP(hidden * 4, hidden * 2)
        self.up1 = UP(hidden * 2, hidden)
        
        self.fuse2 = nn.Sequential(nn.Conv2d(hidden * 4 * 2, hidden * 4, kernel_size=1, stride=1, padding=0),
                                   nn.LeakyReLU(True))
        self.fuse1 = nn.Sequential(nn.Conv2d(hidden * 2 * 2, hidden * 2, kernel_size=1, stride=1, padding=0),
                                   nn.LeakyReLU(True))
        
        self.size = size
        
        if self.size == 'up':
            self.layer = UP_1x1(hidden, output)
        elif self.size == 'down':
            self.layer = Down_1x1(hidden, output)
        else:
            self.layer = nn.Sequential(nn.Conv2d(hidden, output, kernel_size=1, stride=1, padding=0),
                                       nn.LeakyReLU(True))

        mutil.initialize_weights([self.layer], 0.) 
                
    def forward(self, x):
        d0 = self.conv1(x)
        d1 = self.down1(d0)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        
        up3 = self.up3(d3)
        up2 = self.up2(self.fuse2(torch.cat((up3, d2), dim = 1)))
        up1 = self.up1(self.fuse1(torch.cat((up2, d1), dim = 1)))
        
        out = self.layer(up1)
            
        return out


class ResidualUnetBlock_out_blur(nn.Module):
    def __init__(self, input, output, hidden = 16, bias = True, size = 'identity'):
        super(ResidualUnetBlock_out_blur, self).__init__()
        self.conv1 = ConvBlock(input, hidden)
        
        self.down1 = Down(hidden, hidden * 2, blocks=1)
        self.down2 = Down(hidden * 2, hidden * 4, blocks=1)
        self.down3 = Down(hidden * 4, hidden * 8, blocks=1)
        
        self.up3 = UP(hidden * 8, hidden * 4)
        self.up2 = UP(hidden * 4, hidden * 2)
        self.up1 = UP(hidden * 2, hidden)
        
        self.fuse2 = nn.Sequential(nn.Conv2d(hidden * 4 * 2, hidden * 4, kernel_size=1, stride=1, padding=0),
                                   nn.LeakyReLU(True))
        self.fuse1 = nn.Sequential(nn.Conv2d(hidden * 2 * 2, hidden * 2, kernel_size=1, stride=1, padding=0),
                                   nn.LeakyReLU(True))
        
        self.size = size
        
        if self.size == 'up':
            self.layer = UP_1x1(hidden, output)
        elif self.size == 'down':
            self.layer = Down_1x1(hidden, output)
        else:
            self.layer = nn.Sequential(nn.Conv2d(hidden, output, kernel_size=1, stride=1, padding=0),
                                       nn.LeakyReLU(True))

        mutil.initialize_weights([self.layer], 0.) 
                
    def forward(self, x):
        d0 = self.conv1(x)
        
        d1 = self.down1(d0)
        d1 = F.gaussian_blur(d1, kernel_size=[3, 3], sigma=[0.25, 0.25])
        d2 = self.down2(d1)
        d2 = F.gaussian_blur(d2, kernel_size=[3, 3], sigma=[0.25, 0.25])
        d3 = self.down3(d2)
        d3 = F.gaussian_blur(d3, kernel_size=[3, 3], sigma=[0.25, 0.25])
        
        up3 = self.up3(d3)
        up2 = self.up2(self.fuse2(torch.cat((up3, d2), dim = 1)))
        up1 = self.up1(self.fuse1(torch.cat((up2, d1), dim = 1)))
        
        out = self.layer(up1)
            
        return out


class SEAttention(nn.Module):
    def __init__(self, in_channels, out_channels, reduction=8):
        super(SEAttention, self).__init__()
        self.se = nn.Sequential(
            nn.Linear(in_channels, out_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(out_channels // reduction, out_channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.se(x)
        return x.unsqueeze(-1).unsqueeze(-1)
    

class ResidualTwoStreams(nn.Module):
    def __init__(self, input, output, hidden = 16, id = False, size = 'identity', blur=False):
        super(ResidualTwoStreams, self).__init__()
        self.conv1 = ConvBlock(input, hidden)
        
        self.down1 = Down(hidden, hidden * 2, blocks=1)
        self.down2 = Down(hidden * 2, hidden * 4, blocks=1)
        self.down3 = Down(hidden * 4, hidden * 8, blocks=1)
        
        self.up3_b = UP(hidden * 8, hidden * 4)
        self.up2_b = UP(hidden * 4, hidden * 2)
        self.up1_b = UP(hidden * 2, hidden)

        self.fuse2_b = nn.Sequential(nn.Conv2d(hidden * 4 * 2, hidden * 4, kernel_size=1, stride=1, padding=0),
                                   nn.LeakyReLU(True))
        self.fuse1_b = nn.Sequential(nn.Conv2d(hidden * 2 * 2, hidden * 2, kernel_size=1, stride=1, padding=0),
                                   nn.LeakyReLU(True))
        
        self.up3_a = UP(hidden * 8, hidden * 4)
        self.up2_a = UP(hidden * 4, hidden * 2)
        self.up1_a = UP(hidden * 2, hidden)

        self.fuse2_a = nn.Sequential(nn.Conv2d(hidden * 4 * 2, hidden * 4, kernel_size=1, stride=1, padding=0),
                                   nn.LeakyReLU(True))
        self.fuse1_a = nn.Sequential(nn.Conv2d(hidden * 2 * 2, hidden * 2, kernel_size=1, stride=1, padding=0),
                                   nn.LeakyReLU(True))
        
        self.size = size
        
        if self.size == 'up':
            self.layer_a = UP_1x1(hidden, output)
            self.layer_b = UP_1x1(hidden, output)
        elif self.size == 'down':
            self.layer_a = Down_1x1(hidden, output)
            self.layer_b = Down_1x1(hidden, output)
        else:
            self.layer_b = nn.Sequential(nn.Conv2d(hidden, output, kernel_size=1, stride=1, padding=0),
                                       nn.LeakyReLU(True))
            self.layer_a = nn.Sequential(nn.Conv2d(hidden, output, kernel_size=1, stride=1, padding=0),
                                       nn.LeakyReLU(True))

        mutil.initialize_weights([self.layer_a], 0.) 
        mutil.initialize_weights([self.layer_b], 0.) 
        
        self.blur = blur
                
    def forward(self, x, condition = None):
        d0 = self.conv1(x)

        d1 = self.down1(d0)
        if self.blur:
            d1 = F.gaussian_blur(d1, kernel_size=[3, 3], sigma=[0.5, 0.5])
        d2 = self.down2(d1)
        if self.blur:
            d2 = F.gaussian_blur(d2, kernel_size=[3, 3], sigma=[0.5, 0.5])
        d3 = self.down3(d2)
        if self.blur:
            d3 = F.gaussian_blur(d3, kernel_size=[3, 3], sigma=[0.5, 0.5])

        up3_a = self.up3_a(d3)
        up2_a = self.up2_a(self.fuse2_a(torch.cat((up3_a, d2), dim = 1)))
        up1_a = self.up1_a(self.fuse1_a(torch.cat((up2_a, d1), dim = 1)))
        out_a = self.layer_a(up1_a)
        
        up3_b = self.up3_b(d3)
        up2_b = self.up2_b(self.fuse2_b(torch.cat((up3_b, d2), dim = 1)))
        up1_b = self.up1_b(self.fuse1_b(torch.cat((up2_b, d1), dim = 1)))
        out_b = self.layer_b(up1_b)
        
        return out_a, out_b