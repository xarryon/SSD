from . import *

class DW_Decoder(nn.Module):

    def __init__(self, message_length, blocks=2, channels=64, attention=None, scale = 1):
        super(DW_Decoder, self).__init__()

        self.conv1 = ConvBlock(3, 16 * scale, blocks=blocks)
        self.down1 = Down(16 * scale, 32 * scale, blocks=blocks)
        self.down2 = Down(32 * scale, 64 * scale, blocks=blocks)
        self.down3 = Down(64 * scale, 128 * scale, blocks=blocks)
        self.down4 = Down(128 * scale, 256 * scale, blocks=blocks)

        self.up3 = UP(256 * scale, 128 * scale)
        self.att3 = ResBlock(128 * 2 * scale, 128 * scale, blocks=blocks, attention=attention)

        self.up2 = UP(128 * scale, 64 * scale)
        self.att2 = ResBlock(64 * 2 * scale, 64 * scale, blocks=blocks, attention=attention)

        self.up1 = UP(64 * scale, 32 * scale)
        self.att1 = ResBlock(32 * 2 * scale, 32 * scale, blocks=blocks, attention=attention)

        self.up0 = UP(32 * scale, 16 * scale)
        self.att0 = ResBlock(16 * 2 * scale, 16 * scale, blocks=blocks, attention=attention)

        self.Conv_1x1 = nn.Conv2d(16 * scale, 1, kernel_size=1, stride=1, padding=0, bias=False)

        self.message_layer = nn.Linear(message_length * message_length, message_length)
        self.message_length = message_length
        
        
    def forward(self, x, hidden=False):
        d0 = self.conv1(x)
        d1 = self.down1(d0)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        
        u3 = self.up3(d4)
        u3 = torch.cat((d3, u3), dim=1)
        u3 = self.att3(u3)

        u2 = self.up2(u3)
        u2 = torch.cat((d2, u2), dim=1)
        u2 = self.att2(u2)

        u1 = self.up1(u2)
        u1 = torch.cat((d1, u1), dim=1)
        u1 = self.att1(u1)

        u0 = self.up0(u1)
        u0 = torch.cat((d0, u0), dim=1)
        u0 = self.att0(u0)
        
        residual = self.Conv_1x1(u0)
        
        tmp = residual.clone()
        
        message = F.interpolate(residual, 
                                size = (self.message_length, self.message_length),
                                mode = 'nearest')
        
        message = message.view(message.shape[0], -1)
        message = self.message_layer(message)

        if hidden:
            return message, tmp
        else:
            return message


class DW_Decoder_Multi(nn.Module):

    def __init__(self, message_length, blocks=2, channels=64, attention=None):
        super(DW_Decoder_Multi, self).__init__()

        self.conv1 = ConvBlock(3, 16, blocks=blocks)
        self.down1 = Down(16, 32, blocks=blocks)
        self.down2 = Down(32, 64, blocks=blocks)
        self.down3 = Down(64, 128, blocks=blocks)
        self.down4 = Down(128, 256, blocks=blocks)

        self.up3 = UP(256, 128)
        self.att3 = ResBlock(128 * 2, 128, blocks=blocks, attention=attention)

        self.up2 = UP(128, 64)
        self.att2 = ResBlock(64 * 2, 64, blocks=blocks, attention=attention)

        self.up1 = UP(64, 32)
        self.att1 = ResBlock(32 * 2, 32, blocks=blocks, attention=attention)

        self.up0 = UP(32, 16)
        self.att0 = ResBlock(16 * 2, 16, blocks=blocks, attention=attention)

        self.Conv_1x1 = nn.Conv2d(16, 1, kernel_size=1, stride=1, padding=0, bias=False)
        
        self.message_layer_high = nn.Linear(128, message_length)
        self.message_layer_shallow = nn.Linear(message_length * message_length, message_length)
        self.message_length = message_length
        
        self.ada = nn.AdaptiveAvgPool2d(1)
        
    def forward(self, x):
        d0 = self.conv1(x)
        d1 = self.down1(d0)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        
        u3 = self.up3(d4)
        u3 = torch.cat((d3, u3), dim=1)
        u3 = self.att3(u3)

        u2 = self.up2(u3)
        u2 = torch.cat((d2, u2), dim=1)
        u2 = self.att2(u2)

        u1 = self.up1(u2)
        u1 = torch.cat((d1, u1), dim=1)
        u1 = self.att1(u1)

        u0 = self.up0(u1)
        u0 = torch.cat((d0, u0), dim=1)
        u0 = self.att0(u0)
        
        u3_avg = self.ada(u3)
        message_1 = self.message_layer_high(u3_avg[:,:,0,0]) # 空间维度降为1
        
        residual_shallow = self.Conv_1x1(u0) # 通道维度降为1
        message_2 = F.interpolate(residual_shallow, 
                                size = (self.message_length, self.message_length),
                                mode = 'nearest')
        message_2 = message_2.view(message_2.shape[0], -1)
        message_2 = self.message_layer_shallow(message_2)


        return message_1, message_2
    

class DW_Decoder_v2(nn.Module):

    def __init__(self, message_length, blocks=2, channels=64, attention=None):
        super(DW_Decoder_v2, self).__init__()

        self.conv1 = ConvBlock(3, 16, blocks=blocks)
        self.down1 = Down(16, 32, blocks=blocks)
        self.down2 = Down(32, 64, blocks=blocks)
        self.down3 = Down(64, 128, blocks=blocks)
        self.down4 = Down(128, 256, blocks=blocks)

        self.up3 = UP(256, 128)
        self.att3 = ResBlock(128 * 2, 128, blocks=blocks, attention=attention)

        self.up2 = UP(128, 64)
        self.att2 = ResBlock(64 * 2, 64, blocks=blocks, attention=attention)

        self.up1 = UP(64, 32)
        self.att1 = ResBlock(32 * 2, 32, blocks=blocks, attention=attention)

        self.up0 = UP(32, 16)
        self.att0 = ResBlock(16 * 2, 16, blocks=blocks, attention=attention)

        self.Conv_1x1 = nn.Conv2d(16, 16, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, x):
        d0 = self.conv1(x)
        d1 = self.down1(d0)
        d2 = self.down2(d1)
        d3 = self.down3(d2)

        d4 = self.down4(d3)

        u3 = self.up3(d4)
        u3 = torch.cat((d3, u3), dim=1)
        u3 = self.att3(u3)

        u2 = self.up2(u3)
        u2 = torch.cat((d2, u2), dim=1)
        u2 = self.att2(u2)

        u1 = self.up1(u2)
        u1 = torch.cat((d1, u1), dim=1)
        u1 = self.att1(u1)

        u0 = self.up0(u1)
        u0 = torch.cat((d0, u0), dim=1)
        u0 = self.att0(u0)

        residual = self.Conv_1x1(u0)

        message = F.interpolate(residual, size=(8, 8), mode='nearest')

        return message
    

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

if __name__ == '__main__':
    dec = DW_Decoder_v2(128)
    x = torch.randn(2,3,256,256)
    y = dec(x)
    print(y.shape)
