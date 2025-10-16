import torch
import torch.nn as nn
import torch.nn.functional as F

class TConvINRelu(nn.Module):
    """
    A sequence of Convolution, Instance Normalization, and ReLU activation
    """

    def __init__(self, channels_in, channels_out, stride):
        super(TConvINRelu, self).__init__()

        self.layers = nn.Sequential(
            nn.ConvTranspose2d(channels_in, channels_out, 2, stride, padding=0),
            nn.InstanceNorm2d(channels_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.layers(x)

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


class TConvBlock(nn.Module):
    '''
    Network that composed by layers of TConvINRelu
    '''

    def __init__(self, in_channels, out_channels, blocks=1, stride=2):
        super(TConvBlock, self).__init__()

        layers = [TConvINRelu(in_channels, out_channels, stride)] if blocks != 0 else []
        for _ in range(blocks - 1):
            layer = TConvINRelu(out_channels, out_channels, 2)
            layers.append(layer)

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class Down(nn.Module):
    def __init__(self, in_channels, out_channels, blocks):
        super(Down, self).__init__() # 先下采样
        self.layer = torch.nn.Sequential(
            ConvBlock(in_channels, in_channels, stride=2),
            ConvBlock(in_channels, out_channels, blocks=blocks)
        )

    def forward(self, x):
        return self.layer(x)


class Down_1x1(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Down_1x1, self).__init__() # 先下采样
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)

class UP(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UP, self).__init__()
        self.conv = ConvBlock(in_channels, out_channels)

    def forward(self, x): # 先上采样
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        return self.conv(x)


class UP_1x1(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UP_1x1, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x): # 先上采样
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        return self.conv(x)
    

class TwoConvBlocks(nn.Module):
    '''
    Network that composed by layers of ConvINRelu
    '''
    def __init__(self, in_channels, out_channels, blocks=1, stride=1, harr=True, in_1=3, in_2=3):
        super(TwoConvBlocks, self).__init__()
        if harr:
            self.split_len1 = in_1 * 4
            self.split_len2 = in_2 * 4
        else:
            self.split_len1 = in_1
            self.split_len2 = in_2
            
        self.layers_cover = ConvBlock(in_channels, out_channels, blocks, stride)
        self.layers_secret = ConvBlock(in_channels, out_channels, blocks, stride)


    def forward(self, x):
        x1, x2 = (x.narrow(1, 0, self.split_len1),
                  x.narrow(1, self.split_len1, self.split_len2))
        
        x1 = self.layers_cover(x1)
        x2 = self.layers_cover(x2)
        
        return torch.cat((x1, x2), 1)


class TwoDown(nn.Module):
    def __init__(self, in_channels, out_channels, blocks, splits):
        super(TwoDown, self).__init__() # 先下采样
        self.layer1 = Down(in_channels, out_channels, blocks)
        self.layer2 = Down(in_channels, out_channels, blocks)
        self.splits = splits
        
    def forward(self, x):
        x1, x2 = (x.narrow(1, 0, self.splits),
                  x.narrow(1, self.splits, self.splits))
        x1, x2= self.layer1(x1), self.layer2(x1)
        
        return torch.cat((x1, x2), dim = 1)


class TwoUp(nn.Module):
    def __init__(self, in_channels, out_channels, splits):
        super(TwoUp, self).__init__() # 先下采样
        self.layer1 = UP(in_channels, out_channels)
        self.layer2 = UP(in_channels, out_channels)
        self.splits = splits
        
    def forward(self, x):
        x1, x2 = (x.narrow(1, 0, self.splits),
                  x.narrow(1, self.splits, self.splits))
        x1, x2= self.layer1(x1), self.layer2(x1)
        
        return torch.cat((x1, x2), dim = 1)


class SEAttention(nn.Module):
    def __init__(self, in_channels, out_channels, reduction=8):
        super(SEAttention, self).__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels // reduction, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=out_channels // reduction, out_channels=out_channels, kernel_size=1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.se(x) * x
        return x


class ChannelAttention(nn.Module):
    def __init__(self, in_channels, out_channels, reduction=8):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.max_pool = nn.AdaptiveMaxPool2d((1, 1))

        self.fc = nn.Sequential(nn.Conv2d(in_channels=in_channels, out_channels=out_channels // reduction, kernel_size=1, bias=False),
                                nn.ReLU(inplace=True),
                                nn.Conv2d(in_channels=out_channels // reduction, out_channels=out_channels, kernel_size=1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class CBAMAttention(nn.Module):
    def __init__(self, in_channels, out_channels, reduction=8):
        super(CBAMAttention, self).__init__()
        self.ca = ChannelAttention(in_channels=in_channels, out_channels=out_channels, reduction=reduction)
        self.sa = SpatialAttention()

    def forward(self, x):
        x = self.ca(x) * x
        x = self.sa(x) * x
        return x


class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)


class CoordAttention(nn.Module):
    def __init__(self, in_channels, out_channels, reduction=8):
        super(CoordAttention, self).__init__()
        self.pool_w, self.pool_h = nn.AdaptiveAvgPool2d((1, None)), nn.AdaptiveAvgPool2d((None, 1))
        temp_c = max(8, in_channels // reduction)
        self.conv1 = nn.Conv2d(in_channels, temp_c, kernel_size=1, stride=1, padding=0)

        self.bn1 = nn.InstanceNorm2d(temp_c)
        self.act1 = h_swish() # nn.SiLU() # nn.Hardswish() # nn.SiLU()

        self.conv2 = nn.Conv2d(temp_c, out_channels, kernel_size=1, stride=1, padding=0)
        self.conv3 = nn.Conv2d(temp_c, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        short = x
        n, c, H, W = x.shape
        x_h, x_w = self.pool_h(x), self.pool_w(x).permute(0, 1, 3, 2)
        x_cat = torch.cat([x_h, x_w], dim=2)
        out = self.act1(self.bn1(self.conv1(x_cat)))
        x_h, x_w = torch.split(out, [H, W], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)
        out_h = torch.sigmoid(self.conv2(x_h))
        out_w = torch.sigmoid(self.conv3(x_w))
        return short * out_w * out_h


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, reduction, stride, attention=None):
        super(BasicBlock, self).__init__()

        self.change = None
        if (in_channels != out_channels or stride != 1):
            self.change = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, padding=0,
                          stride=stride, bias=False),
                nn.InstanceNorm2d(out_channels)
            )

        self.left = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1,
                      stride=stride, bias=False),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(out_channels)
        )

        if attention == 'se':
            print('SEAttention')
            self.attention = SEAttention(in_channels=out_channels, out_channels=out_channels, reduction=reduction)
        elif attention == 'cbam':
            print('CBAMAttention')
            self.attention = CBAMAttention(in_channels=out_channels, out_channels=out_channels, reduction=reduction)
        elif attention == 'coord':
            print('CoordAttention')
            self.attention = CoordAttention(in_channels=out_channels, out_channels=out_channels, reduction=reduction)
        else:
            print('None Attention')
            self.attention = nn.Identity()

    def forward(self, x):
        identity = x
        x = self.left(x)
        x = self.attention(x)

        if self.change is not None:
            identity = self.change(identity)

        x += identity
        x = F.relu(x)
        return x


class BottleneckBlock(nn.Module):
    def __init__(self, in_channels, out_channels, reduction, stride, attention=None):
        super(BottleneckBlock, self).__init__()

        self.change = None
        if (in_channels != out_channels or stride != 1):
            self.change = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, padding=0,
                          stride=stride, bias=False),
                nn.InstanceNorm2d(out_channels)
            )

        self.left = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1,
                      stride=stride, padding=0, bias=False),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=1, padding=0, bias=False),
            nn.InstanceNorm2d(out_channels)
        )

        if attention == 'se':
            # print('SEAttention')
            self.attention = SEAttention(in_channels=out_channels, out_channels=out_channels, reduction=reduction)
        elif attention == 'cbam':
            # print('CBAMAttention')
            self.attention = CBAMAttention(in_channels=out_channels, out_channels=out_channels, reduction=reduction)
        elif attention == 'coord':
            # print('CoordAttention')
            self.attention = CoordAttention(in_channels=out_channels, out_channels=out_channels, reduction=reduction)
        else:
            # print('None Attention')
            self.attention = nn.Identity()

    def forward(self, x):
        identity = x
        x = self.left(x)
        x = self.attention(x)

        if self.change is not None:
            identity = self.change(identity)

        x += identity
        x = F.relu(x)
        return x


class ResBlock(nn.Module):

    def __init__(self, in_channels, out_channels, blocks=1, block_type="BottleneckBlock", reduction=8, stride=1, attention=None):
        super(ResBlock, self).__init__()

        layers = [eval(block_type)(in_channels, out_channels, reduction, stride, attention=attention)] if blocks != 0 else []
        for _ in range(blocks - 1):
            layer = eval(block_type)(out_channels, out_channels, reduction, 1, attention=attention)
            layers.append(layer)

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)
    

class AdaIN(nn.Module):
    """
    Adaptive Instance Normalization (AdaIN) layer
    """
    def __init__(self):
        super(AdaIN, self).__init__()
    
    def forward(self, content, style, epsilon=1e-5):
        """
        Args:
            content: content features (N, C, H, W)
            style: style features (N, C, H', W')
            epsilon: small constant to avoid division by zero
        Returns:
            normalized content features with style statistics
        """
        content_mean, content_std = self.calc_mean_std(content, epsilon)
        style_mean, style_std = self.calc_mean_std(style, epsilon)
        normalized_content = (content - content_mean) / content_std
        
        output = normalized_content * style_std + style_mean
        
        return output
    
    @staticmethod
    def calc_mean_std(features, epsilon=1e-5):
        """
        Calculate mean and standard deviation across spatial dimensions
        Args:
            features: tensor of shape (N, C, H, W)
            epsilon: small constant to avoid division by zero
        Returns:
            mean and std of shape (N, C, 1, 1)
        """
        batch_size, channels = features.size()[:2]
        
        features_reshaped = features.view(batch_size, channels, -1)
        mean = features_reshaped.mean(dim=2).view(batch_size, channels, 1, 1)
        std = features_reshaped.std(dim=2).view(batch_size, channels, 1, 1)
        
        std = std + epsilon
        
        return mean, std