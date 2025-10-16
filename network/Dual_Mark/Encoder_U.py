from . import *
from math import sqrt
from torch.cuda.amp import autocast as autocast

class DW_Encoder(nn.Module):

    def __init__(self, message_length, blocks=2, channels=64, attention=None, Adain = False):
        super(DW_Encoder, self).__init__()
        
        # 编码压缩过程
        self.conv1 = ConvBlock(3, 16, blocks=blocks)
        self.down1 = Down(16, 32, blocks=blocks)
        self.down2 = Down(32, 64, blocks=blocks)
        self.down3 = Down(64, 128, blocks=blocks)
        self.down4 = Down(128, 256, blocks=blocks)

        # 解码扩增过程
        self.up3 = UP(256, 128)
        self.linear3 = nn.Linear(message_length, message_length * message_length)
        # self.Conv_message3 = nn.Sequential(ConvBlock(1, channels, blocks = blocks//2),
        #                                    ResBlock(channels, channels * 2, blocks = blocks//2, attention=attention))
        self.Conv_message3 = ConvBlock(1, channels * 2, blocks = blocks)
        self.att3 = ResBlock(128 * 2 + channels * 2, 128, blocks=blocks, attention=attention)

        self.up2 = UP(128, 64)
        self.linear2 = nn.Linear(message_length, message_length * message_length)
        # self.Conv_message2 = nn.Sequential(ConvBlock(1, channels, blocks=blocks//2),
        #                                    ResBlock(channels, channels, blocks=blocks//2, attention=attention))
        self.Conv_message2 = ConvBlock(1, channels, blocks = blocks)
        self.att2 = ResBlock(64 * 2 + channels, 64, blocks=blocks, attention=attention)

        self.up1 = UP(64, 32)
        self.linear1 = nn.Linear(message_length, message_length * message_length)
        # self.Conv_message1 = nn.Sequential(ConvBlock(1, channels, blocks=blocks//2),
        #                                    ResBlock(channels, channels // 2, blocks=blocks//2, attention=attention))
        self.Conv_message1 = ConvBlock(1, channels // 2, blocks = blocks)
        self.att1 = ResBlock(32 * 2 + channels // 2, 32, blocks=blocks, attention=attention)

        self.up0 = UP(32, 16)
        self.linear0 = nn.Linear(message_length, message_length * message_length)
        # self.Conv_message0 = nn.Sequential(ConvBlock(1, channels, blocks=blocks//2),
        #                                    ResBlock(channels, channels // 4, blocks=blocks//2, attention=attention))
        self.Conv_message0 = ConvBlock(1, channels // 4, blocks = blocks)
        self.att0 = ResBlock(16 * 2 + channels // 4, 16, blocks=blocks, attention=attention)

        self.Conv_1x1 = nn.Conv2d(16 + 3, 3, kernel_size=1, stride=1, padding=0)

        self.message_length = message_length
        self.adain = Adain
        
    # @autocast(True)
    def forward(self, x, watermark, mask = None, hidden = False):
        d0 = self.conv1(x)
        d1 = self.down1(d0)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)

        u3 = self.up3(d4)
        expanded_message = self.linear3(watermark)
        expanded_message = expanded_message.view(-1, 1, self.message_length, self.message_length) # 拓展到二维 
        expanded_message = F.interpolate(expanded_message,
                                        size=(d3.shape[2], d3.shape[3]),
                                        mode='nearest')
        expanded_message = self.Conv_message3(expanded_message)
        
        if mask is not None:
            mask_rez = F.interpolate(mask,
                                    size=(d3.shape[2], d3.shape[3]),
                                    mode='nearest')
            if self.adain:
                img_feat_mean = torch.mean(u3, dim = [2, 3], keepdim = True)
                img_feat_std = torch.std(u3, dim = [2, 3], unbiased = False, keepdim = True)
                expanded_message = F.instance_norm(expanded_message) * img_feat_std + img_feat_mean
            
            expanded_message = expanded_message * mask_rez[:, 0].unsqueeze(1)
        
        
        u3 = torch.cat((d3, u3, expanded_message), dim=1)
        u3 = self.att3(u3)
        
        tmp = u3.clone()
        
        u2 = self.up2(u3)
        expanded_message = self.linear2(watermark)
        expanded_message = expanded_message.view(-1, 1, self.message_length, self.message_length)
        expanded_message = F.interpolate(expanded_message, size=(d2.shape[2], d2.shape[3]),
                                                           mode='nearest')
        expanded_message = self.Conv_message2(expanded_message)#通道维度拓展为64
        
        
        if mask is not None:
            mask_rez = F.interpolate(mask,
                                    size=(d2.shape[2], d2.shape[3]),
                                    mode='nearest')
            if self.adain:
                img_feat_mean = torch.mean(u2, dim = [2, 3], keepdim = True)
                img_feat_std = torch.std(u2, dim = [2, 3], unbiased = False, keepdim = True)
                expanded_message = F.instance_norm(expanded_message) * img_feat_std + img_feat_mean
            
            expanded_message = expanded_message * mask_rez[:, 0].unsqueeze(1)
            
        u2 = torch.cat((d2, u2, expanded_message), dim=1) # encoder特征 decoder特征 message
        u2 = self.att2(u2) # 融合时候采用注意力

        u1 = self.up1(u2) # 融合后特征进行上采样
        expanded_message = self.linear1(watermark)
        expanded_message = expanded_message.view(-1, 1, self.message_length , self.message_length)
        expanded_message = F.interpolate(expanded_message, size=(d1.shape[2], d1.shape[3]),
                                                           mode='nearest')
        expanded_message = self.Conv_message1(expanded_message)
        
        if mask is not None:
            mask_rez = F.interpolate(mask,
                                    size=(d1.shape[2], d1.shape[3]),
                                    mode='nearest')
            if self.adain:
                img_feat_mean = torch.mean(u1, dim = [2, 3], keepdim = True)
                img_feat_std = torch.std(u1, dim = [2, 3], unbiased = False, keepdim = True)
                expanded_message = F.instance_norm(expanded_message) * img_feat_std + img_feat_mean
                
            expanded_message = expanded_message * mask_rez[:, 0].unsqueeze(1)
                
        u1 = torch.cat((d1, u1, expanded_message), dim=1)
        u1 = self.att1(u1)

        u0 = self.up0(u1)
        expanded_message = self.linear0(watermark)
        expanded_message = expanded_message.view(-1, 1, self.message_length, self.message_length)
        expanded_message = F.interpolate(expanded_message, size=(d0.shape[2], d0.shape[3]),
                                                           mode='nearest')
        expanded_message = self.Conv_message0(expanded_message)
        
        if mask is not None:
            mask_rez = F.interpolate(mask,
                                    size=(d0.shape[2], d0.shape[3]),
                                    mode='nearest')
            if self.adain:
                img_feat_mean = torch.mean(u0, dim = [2, 3], keepdim = True)
                img_feat_var = torch.std(u0, dim = [2, 3], unbiased = False, keepdim = True)
                expanded_message = F.instance_norm(expanded_message) * img_feat_var + img_feat_mean
            
            expanded_message = expanded_message * mask_rez[:, 0].unsqueeze(1)
            
        u0 = torch.cat((d0, u0, expanded_message), dim=1)
        u0 = self.att0(u0)

        image = self.Conv_1x1(torch.cat((x, u0), dim=1)) # 每次上采样都会把message融入其中

        forward_image = image.clone().detach()
        gap = forward_image.clamp(-1, 1) - forward_image

        if hidden:
            return image + gap, expanded_message, tmp
        else:
            return image + gap, expanded_message


class DW_Encoder(nn.Module):

    def __init__(self, message_length, blocks=2, channels=64, attention=None, Adain = False, scale = 1):
        super(DW_Encoder, self).__init__()
        
        # 编码压缩过程
        self.conv1 = ConvBlock(3, 16 * scale, blocks=blocks)
        self.down1 = Down(16 * scale, 32 * scale, blocks=blocks)
        self.down2 = Down(32 * scale, 64 * scale, blocks=blocks)
        self.down3 = Down(64 * scale, 128 * scale, blocks=blocks)
        self.down4 = Down(128 * scale, 256 * scale, blocks=blocks)

        self.channels = channels * scale
        
        # 解码扩增过程
        self.up3 = UP(256 * scale, 128 * scale)
        self.linear3 = nn.Linear(message_length, message_length * message_length)
        # self.Conv_message3 = nn.Sequential(ConvBlock(1, channels, blocks = blocks//2),
        #                                    ResBlock(channels, channels * 2, blocks = blocks//2, attention=attention))
        self.Conv_message3 = ConvBlock(1, self.channels * 2, blocks = blocks)
        self.att3 = ResBlock(128 * 2 * scale + self.channels * 2, 128 * scale, blocks=blocks, attention=attention)

        self.up2 = UP(128 * scale, 64 * scale)
        self.linear2 = nn.Linear(message_length, message_length * message_length)
        # self.Conv_message2 = nn.Sequential(ConvBlock(1, channels, blocks=blocks//2),
        #                                    ResBlock(channels, channels, blocks=blocks//2, attention=attention))
        self.Conv_message2 = ConvBlock(1, self.channels, blocks = blocks)
        self.att2 = ResBlock(64 * 2  * scale + self.channels, 64 * scale, blocks=blocks, attention=attention)

        self.up1 = UP(64 * scale, 32 * scale)
        self.linear1 = nn.Linear(message_length, message_length * message_length)
        # self.Conv_message1 = nn.Sequential(ConvBlock(1, channels, blocks=blocks//2),
        #                                    ResBlock(channels, channels // 2, blocks=blocks//2, attention=attention))
        self.Conv_message1 = ConvBlock(1, self.channels // 2, blocks = blocks)
        self.att1 = ResBlock(32 * 2 * scale + self.channels // 2, 32 * scale, blocks=blocks, attention=attention)

        self.up0 = UP(32 * scale, 16 * scale)
        self.linear0 = nn.Linear(message_length, message_length * message_length)
        # self.Conv_message0 = nn.Sequential(ConvBlock(1, channels, blocks=blocks//2),
        #                                    ResBlock(channels, channels // 4, blocks=blocks//2, attention=attention))
        self.Conv_message0 = ConvBlock(1, self.channels // 4, blocks = blocks)
        self.att0 = ResBlock(16 * 2 * scale + self.channels // 4, 16 * scale, blocks=blocks, attention=attention)

        self.Conv_1x1 = nn.Conv2d(16 * scale + 3, 3, kernel_size=1, stride=1, padding=0)

        self.message_length = message_length
        self.adain = Adain
        
    # @autocast(True)
    def forward(self, x, watermark, mask = None, hidden = False):
        d0 = self.conv1(x)
        d1 = self.down1(d0)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)

        u3 = self.up3(d4)
        expanded_message = self.linear3(watermark)
        expanded_message = expanded_message.view(-1, 1, self.message_length, self.message_length) # 拓展到二维 
        expanded_message = F.interpolate(expanded_message,
                                        size=(d3.shape[2], d3.shape[3]),
                                        mode='nearest')
        expanded_message = self.Conv_message3(expanded_message)
        
        if mask is not None:
            mask_rez = F.interpolate(mask,
                                    size=(d3.shape[2], d3.shape[3]),
                                    mode='nearest')
            if self.adain:
                img_feat_mean = torch.mean(u3, dim = [2, 3], keepdim = True)
                img_feat_std = torch.std(u3, dim = [2, 3], unbiased = False, keepdim = True)
                expanded_message = F.instance_norm(expanded_message) * img_feat_std + img_feat_mean
            
            expanded_message = expanded_message * mask_rez[:, 0].unsqueeze(1)
        
        
        u3 = torch.cat((d3, u3, expanded_message), dim=1)
        u3 = self.att3(u3)
        
        tmp = u3.clone()
        
        u2 = self.up2(u3)
        expanded_message = self.linear2(watermark)
        expanded_message = expanded_message.view(-1, 1, self.message_length, self.message_length)
        expanded_message = F.interpolate(expanded_message, size=(d2.shape[2], d2.shape[3]),
                                                           mode='nearest')
        expanded_message = self.Conv_message2(expanded_message)#通道维度拓展为64
        
        
        if mask is not None:
            mask_rez = F.interpolate(mask,
                                    size=(d2.shape[2], d2.shape[3]),
                                    mode='nearest')
            if self.adain:
                img_feat_mean = torch.mean(u2, dim = [2, 3], keepdim = True)
                img_feat_std = torch.std(u2, dim = [2, 3], unbiased = False, keepdim = True)
                expanded_message = F.instance_norm(expanded_message) * img_feat_std + img_feat_mean
            
            expanded_message = expanded_message * mask_rez[:, 0].unsqueeze(1)
            
        u2 = torch.cat((d2, u2, expanded_message), dim=1) # encoder特征 decoder特征 message
        u2 = self.att2(u2) # 融合时候采用注意力

        u1 = self.up1(u2) # 融合后特征进行上采样
        expanded_message = self.linear1(watermark)
        expanded_message = expanded_message.view(-1, 1, self.message_length , self.message_length)
        expanded_message = F.interpolate(expanded_message, size=(d1.shape[2], d1.shape[3]),
                                                           mode='nearest')
        expanded_message = self.Conv_message1(expanded_message)
        
        if mask is not None:
            mask_rez = F.interpolate(mask,
                                    size=(d1.shape[2], d1.shape[3]),
                                    mode='nearest')
            if self.adain:
                img_feat_mean = torch.mean(u1, dim = [2, 3], keepdim = True)
                img_feat_std = torch.std(u1, dim = [2, 3], unbiased = False, keepdim = True)
                expanded_message = F.instance_norm(expanded_message) * img_feat_std + img_feat_mean
                
            expanded_message = expanded_message * mask_rez[:, 0].unsqueeze(1)
                
        u1 = torch.cat((d1, u1, expanded_message), dim=1)
        u1 = self.att1(u1)

        u0 = self.up0(u1)
        expanded_message = self.linear0(watermark)
        expanded_message = expanded_message.view(-1, 1, self.message_length, self.message_length)
        expanded_message = F.interpolate(expanded_message, size=(d0.shape[2], d0.shape[3]),
                                                           mode='nearest')
        expanded_message = self.Conv_message0(expanded_message)
        
        if mask is not None:
            mask_rez = F.interpolate(mask,
                                    size=(d0.shape[2], d0.shape[3]),
                                    mode='nearest')
            if self.adain:
                img_feat_mean = torch.mean(u0, dim = [2, 3], keepdim = True)
                img_feat_var = torch.std(u0, dim = [2, 3], unbiased = False, keepdim = True)
                expanded_message = F.instance_norm(expanded_message) * img_feat_var + img_feat_mean
            
            expanded_message = expanded_message * mask_rez[:, 0].unsqueeze(1)
            
        u0 = torch.cat((d0, u0, expanded_message), dim=1)
        u0 = self.att0(u0)

        image = self.Conv_1x1(torch.cat((x, u0), dim=1)) # 每次上采样都会把message融入其中

        forward_image = image.clone().detach()
        gap = forward_image.clamp(-1, 1) - forward_image

        if hidden:
            return image + gap, expanded_message, tmp
        else:
            return image + gap, expanded_message
        

class DW_EncoderV3(nn.Module):

    def __init__(self, message_length, blocks=2, channels=64, attention=None):
        super(DW_EncoderV3, self).__init__()

        self.conv1 = ConvBlock(3, 16, blocks=blocks)
        self.down1 = Down(16, 32, blocks=blocks)
        self.down2 = Down(32, 64, blocks=blocks)
        self.down3 = Down(64, 128, blocks=blocks)

        self.down4 = Down(128, 256, blocks=blocks)

        self.up3 = UP(256, 128)
        self.linear3_exp = nn.Linear(message_length, message_length * message_length)
        self.linear3_ide = nn.Linear(message_length, message_length * message_length)
        self.Conv_message3_exp = ConvBlock(1, channels, blocks=blocks)
        self.Conv_message3_ide = ConvBlock(1, channels, blocks=blocks)
        self.att3 = ResBlock(128 * 2 + channels, 128, blocks=blocks, attention=attention)

        self.up2 = UP(128, 64)
        self.linear2_exp = nn.Linear(message_length, message_length * message_length)
        self.linear2_ide = nn.Linear(message_length, message_length * message_length)
        self.Conv_message2_exp = ConvBlock(1, channels, blocks=blocks)
        self.Conv_message2_ide = ConvBlock(1, channels, blocks=blocks)
        self.att2 = ResBlock(64 * 2 + channels, 64, blocks=blocks, attention=attention)

        self.up1 = UP(64, 32)
        self.linear1_exp = nn.Linear(message_length, message_length * message_length)
        self.linear1_ide = nn.Linear(message_length, message_length * message_length)
        self.Conv_message1_exp = ConvBlock(1, channels, blocks=blocks)
        self.Conv_message1_ide = ConvBlock(1, channels, blocks=blocks)
        self.att1 = ResBlock(32 * 2 + channels, 32, blocks=blocks, attention=attention)

        self.up0 = UP(32, 16)
        self.linear0_exp = nn.Linear(message_length, message_length * message_length)
        self.linear0_ide = nn.Linear(message_length, message_length * message_length)
        self.Conv_message0_exp = ConvBlock(1, channels, blocks=blocks)
        self.Conv_message0_ide = ConvBlock(1, channels, blocks=blocks)
        self.att0 = ResBlock(16 * 2 + channels, 16, blocks=blocks, attention=attention)

        self.Conv_1x1 = nn.Conv2d(16 + 3, 3, kernel_size=1, stride=1, padding=0)

        self.message_length = message_length

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

    # @autocast(True)
    def forward(self, x, watermark_exp, watermark_ide, mask = None):
        d0 = self.conv1(x)
        d1 = self.down1(d0)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)

        u3 = self.up3(d4)
        expanded_message_exp = self.linear3_exp(watermark_exp)
        expanded_message_exp = expanded_message_exp.view(-1, 1, self.message_length, self.message_length) #拓展到二维 
        expanded_message_ide = self.linear3_ide(watermark_ide)
        expanded_message_ide = expanded_message_ide.view(-1, 1, self.message_length, self.message_length) #拓展到二维 

        expanded_message_exp = F.interpolate(expanded_message_exp,
                                        size=(d3.shape[2], d3.shape[3]),
                                        mode='nearest')
        expanded_message_ide = F.interpolate(expanded_message_ide,
                                        size=(d3.shape[2], d3.shape[3]),
                                        mode='nearest')
        
        expanded_message_exp = self.Conv_message3_exp(expanded_message_exp)
        expanded_message_ide = self.Conv_message3_ide(expanded_message_ide)
        
        if mask is not None:
            mask_rez = F.interpolate(mask,
                                    size=(d3.shape[2], d3.shape[3]),
                                    mode='nearest')
            expanded_message_exp = expanded_message_exp * mask_rez[:, 0].unsqueeze(1)
            expanded_message_ide = expanded_message_ide * (1 - mask_rez[:, 0].unsqueeze(1))
            
        expanded_message = expanded_message_exp + expanded_message_ide
            
        u3 = torch.cat((d3, u3, expanded_message), dim=1)
        u3 = self.att3(u3)

        u2 = self.up2(u3)
        expanded_message_exp = self.linear2_exp(watermark_exp)
        expanded_message_exp = expanded_message_exp.view(-1, 1, self.message_length, self.message_length)
        expanded_message_ide = self.linear2_ide(watermark_ide)
        expanded_message_ide = expanded_message_ide.view(-1, 1, self.message_length, self.message_length)
        
        expanded_message_exp = F.interpolate(expanded_message_exp,
                                        size=(d2.shape[2], d2.shape[3]),
                                        mode='nearest')
        expanded_message_ide = F.interpolate(expanded_message_ide,
                                        size=(d2.shape[2], d2.shape[3]),
                                        mode='nearest')
        
        expanded_message_exp = self.Conv_message2_exp(expanded_message_exp)
        expanded_message_ide = self.Conv_message2_ide(expanded_message_ide)
        
        if mask is not None:
            mask_rez = F.interpolate(mask,
                                    size=(d2.shape[2], d2.shape[3]),
                                    mode='nearest')
            expanded_message_exp = expanded_message_exp * mask_rez[:, 0].unsqueeze(1)
            expanded_message_ide = expanded_message_ide * (1 - mask_rez[:, 0].unsqueeze(1))
            
        expanded_message = expanded_message_exp + expanded_message_ide
        
        u2 = torch.cat((d2, u2, expanded_message), dim=1) # encoder特征 decoder特征 message
        u2 = self.att2(u2) # 融合时候采用注意力

        u1 = self.up1(u2) # 融合后特征进行上采样
        expanded_message_exp = self.linear1_exp(watermark_exp)
        expanded_message_exp = expanded_message_exp.view(-1, 1, self.message_length, self.message_length)
        expanded_message_ide = self.linear1_ide(watermark_ide)
        expanded_message_ide = expanded_message_ide.view(-1, 1, self.message_length, self.message_length)
        
        expanded_message_exp = F.interpolate(expanded_message_exp, size=(d1.shape[2], d1.shape[3]),
                                                           mode='nearest')
        expanded_message_ide = F.interpolate(expanded_message_ide, size=(d1.shape[2], d1.shape[3]),
                                                           mode='nearest')
        
        expanded_message_exp = self.Conv_message1_exp(expanded_message_exp)
        expanded_message_ide = self.Conv_message1_ide(expanded_message_ide)
        
        if mask is not None:
            mask_rez = F.interpolate(mask,
                                    size=(d1.shape[2], d1.shape[3]),
                                    mode='nearest')
            expanded_message_exp = expanded_message_exp * mask_rez[:, 0].unsqueeze(1)
            expanded_message_ide = expanded_message_ide * (1 - mask_rez[:, 0].unsqueeze(1))
        
        expanded_message = expanded_message_exp + expanded_message_ide
        
        u1 = torch.cat((d1, u1, expanded_message), dim=1)
        u1 = self.att1(u1)

        u0 = self.up0(u1)
        expanded_message_exp = self.linear0_exp(watermark_exp)
        expanded_message_exp = expanded_message_exp.view(-1, 1, self.message_length, self.message_length)
        expanded_message_ide = self.linear0_ide(watermark_ide)
        expanded_message_ide = expanded_message_ide.view(-1, 1, self.message_length, self.message_length)
        
        expanded_message_exp = F.interpolate(expanded_message_exp, size=(d0.shape[2], d0.shape[3]),
                                                           mode='nearest')
        expanded_message_ide = F.interpolate(expanded_message_ide, size=(d0.shape[2], d0.shape[3]),
                                                           mode='nearest')
        
        expanded_message_exp = self.Conv_message0_exp(expanded_message_exp)
        expanded_message_ide = self.Conv_message0_ide(expanded_message_ide)
        
        if mask is not None:
            mask_rez = F.interpolate(mask,
                                    size=(d0.shape[2], d0.shape[3]),
                                    mode='nearest')
            expanded_message_exp = expanded_message_exp * mask_rez[:, 0].unsqueeze(1)
            expanded_message_ide = expanded_message_ide * (1 - mask_rez[:, 0].unsqueeze(1))
        
        expanded_message = expanded_message_exp + expanded_message_ide
        
        u0 = torch.cat((d0, u0, expanded_message), dim=1)
        u0 = self.att0(u0)

        image = self.Conv_1x1(torch.cat((x, u0), dim=1)) # 每次上采样都会把message融入其中

        forward_image = image.clone().detach()
        
        '''read_image = torch.zeros_like(forward_image)

        for index in range(forward_image.shape[0]):
            single_image = ((forward_image[index].clamp(-1, 1).permute(1, 2, 0) + 1) / 2 * 255).add(0.5).clamp(0, 255).to('cpu', torch.uint8).numpy()
            im = Image.fromarray(single_image)
            read = np.array(im, dtype=np.uint8)
            read_image[index] = self.transform(read).unsqueeze(0).to(image.device)

        gap = read_image - forward_image'''
        
        gap = forward_image.clamp(-1, 1) - forward_image

        return image + gap


class DW_Encoder_v2(nn.Module):

    def __init__(self, message_length, blocks=2, channels=64, attention=None):
        super(DW_Encoder_v2, self).__init__()

        self.conv1 = ConvBlock(3, 16, blocks=blocks)
        self.down1 = Down(16, 32, blocks=blocks)
        self.down2 = Down(32, 64, blocks=blocks)
        self.down3 = Down(64, 128, blocks=blocks)

        self.down4 = Down(128, 256, blocks=blocks)

        self.up3 = UP(256, 128)
        self.Conv_message3 = ConvBlock(16, channels, blocks=blocks)
        self.att3 = ResBlock(128 * 2 + channels, 128, blocks=blocks, attention=attention)

        self.up2 = UP(128, 64)
        self.Conv_message2 = ConvBlock(16, channels, blocks=blocks)
        self.att2 = ResBlock(64 * 2 + channels, 64, blocks=blocks, attention=attention)

        self.up1 = UP(64, 32)
        self.Conv_message1 = ConvBlock(16, channels, blocks=blocks)
        self.att1 = ResBlock(32 * 2 + channels, 32, blocks=blocks, attention=attention)

        self.up0 = UP(32, 16)
        self.Conv_message0 = ConvBlock(16, channels, blocks=blocks)
        self.att0 = ResBlock(16 * 2 + channels, 16, blocks=blocks, attention=attention)

        self.Conv_1x1 = nn.Conv2d(16 + 3, 3, kernel_size=1, stride=1, padding=0)

        self.message_length = message_length


    def forward(self, x, watermark):
        d0 = self.conv1(x)
        d1 = self.down1(d0)
        d2 = self.down2(d1)
        d3 = self.down3(d2)

        d4 = self.down4(d3)

        u3 = self.up3(d4)
        expanded_message = F.interpolate(watermark, size = (d3.shape[2], d3.shape[3]),
                                                           mode='nearest')
        expanded_message = self.Conv_message3(expanded_message)
        u3 = torch.cat((d3, u3, expanded_message), dim=1)
        u3 = self.att3(u3)

        u2 = self.up2(u3)
        
        expanded_message = F.interpolate(watermark, size = (d2.shape[2], d2.shape[3]),
                                                           mode='nearest')
        expanded_message = self.Conv_message2(expanded_message)#通道维度拓展为64
        u2 = torch.cat((d2, u2, expanded_message), dim=1) # encoder特征 decoder特征 message
        u2 = self.att2(u2) # 融合时候采用注意力

        u1 = self.up1(u2) # 融合后特征进行上采样
        expanded_message = F.interpolate(watermark, size = (d1.shape[2], d1.shape[3]),
                                                           mode='nearest')
        expanded_message = self.Conv_message1(expanded_message)
        u1 = torch.cat((d1, u1, expanded_message), dim=1)
        u1 = self.att1(u1)

        u0 = self.up0(u1)
        expanded_message = F.interpolate(watermark, size=(d0.shape[2], d0.shape[3]),
                                                           mode='nearest')
        expanded_message = self.Conv_message0(expanded_message)
        u0 = torch.cat((d0, u0, expanded_message), dim=1)
        u0 = self.att0(u0)

        image = self.Conv_1x1(torch.cat((x, u0), dim=1)) # 每次上采样都会把message融入其中

        forward_image = image.clone().detach()

        gap = forward_image.clamp(-1, 1) - forward_image

        return image + gap


class DW_Encoder_ref(nn.Module):

    def __init__(self, message_length, blocks=2, channels=64, attention=None):
        super(DW_Encoder_ref, self).__init__()

        self.conv1 = ConvBlock(3, 16, blocks=blocks)
        self.down1 = Down(16, 32, blocks=blocks)
        self.down2 = Down(32, 64, blocks=blocks)
        self.down3 = Down(64, 128, blocks=blocks)

        self.down4 = Down(128, 256, blocks=blocks)

        self.up3 = UP(256, 128)
        self.linear3 = nn.Linear(message_length, message_length * message_length)
        self.Conv_message3 = ConvBlock(1, channels, blocks=blocks)
        self.att3 = ResBlock(128 * 2 + channels, 128, blocks=blocks, attention=attention)

        self.up2 = UP(128, 64)
        self.linear2 = nn.Linear(message_length, message_length * message_length)
        self.Conv_message2 = ConvBlock(1, channels, blocks=blocks)
        self.att2 = ResBlock(64 * 2 + channels, 64, blocks=blocks, attention=attention)

        self.up1 = UP(64, 32)
        self.linear1 = nn.Linear(message_length, message_length * message_length)
        self.Conv_message1 = ConvBlock(1, channels, blocks=blocks)
        self.att1 = ResBlock(32 * 2 + channels, 32, blocks=blocks, attention=attention)

        self.up0 = UP(32, 16)
        self.linear0 = nn.Linear(message_length, message_length * message_length)
        self.Conv_message0 = ConvBlock(1, channels, blocks=blocks)
        self.att0 = ResBlock(16 * 2 + channels, 16, blocks=blocks, attention=attention)

        self.Conv_1x1 = nn.Conv2d(16 + 3, 3, kernel_size=1, stride=1, padding=0)

        self.message_length = message_length

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])


    def forward(self, x, watermark):
        d0 = self.conv1(x)
        d1 = self.down1(d0)
        d2 = self.down2(d1)
        d3 = self.down3(d2)

        d4 = self.down4(d3)

        u3 = self.up3(d4)
        expanded_message = self.linear3(watermark)
        expanded_message = expanded_message.view(-1, 1, self.message_length, self.message_length) #拓展到二维 
        expanded_message = F.interpolate(expanded_message, size=(d3.shape[2], d3.shape[3]),
                                                           mode='nearest')
        expanded_message = self.Conv_message3(expanded_message)
        u3 = torch.cat((d3, u3, expanded_message), dim=1)
        u3 = self.att3(u3)

        u2 = self.up2(u3)
        expanded_message = self.linear2(watermark)
        expanded_message = expanded_message.view(-1, 1, self.message_length, self.message_length)
        expanded_message = F.interpolate(expanded_message, size=(d2.shape[2], d2.shape[3]),
                                                           mode='nearest')
        expanded_message = self.Conv_message2(expanded_message)#通道维度拓展为64
        u2 = torch.cat((d2, u2, expanded_message), dim=1) # encoder特征 decoder特征 message
        u2 = self.att2(u2) # 融合时候采用注意力

        u1 = self.up1(u2) # 融合后特征进行上采样
        expanded_message = self.linear1(watermark)
        expanded_message = expanded_message.view(-1, 1, self.message_length, self.message_length)
        expanded_message = F.interpolate(expanded_message, size=(d1.shape[2], d1.shape[3]),
                                                           mode='nearest')
        expanded_message = self.Conv_message1(expanded_message)
        u1 = torch.cat((d1, u1, expanded_message), dim=1)
        u1 = self.att1(u1)

        u0 = self.up0(u1)
        expanded_message = self.linear0(watermark)
        expanded_message = expanded_message.view(-1, 1, self.message_length, self.message_length)
        expanded_message = F.interpolate(expanded_message, size=(d0.shape[2], d0.shape[3]),
                                                           mode='nearest')
        expanded_message = self.Conv_message0(expanded_message)
        u0 = torch.cat((d0, u0, expanded_message), dim=1)
        u0 = self.att0(u0)

        image = self.Conv_1x1(torch.cat((x, u0), dim=1)) # 每次上采样都会把message融入其中

        forward_image = image.clone().detach()
        '''read_image = torch.zeros_like(forward_image)

        for index in range(forward_image.shape[0]):
            single_image = ((forward_image[index].clamp(-1, 1).permute(1, 2, 0) + 1) / 2 * 255).add(0.5).clamp(0, 255).to('cpu', torch.uint8).numpy()
            im = Image.fromarray(single_image)
            read = np.array(im, dtype=np.uint8)
            read_image[index] = self.transform(read).unsqueeze(0).to(image.device)

        gap = read_image - forward_image'''
        gap = forward_image.clamp(-1, 1) - forward_image

        return image + gap
    
    
class Down(nn.Module):
    def __init__(self, in_channels, out_channels, blocks):
        super(Down, self).__init__() # 先下采样
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

    def forward(self, x): # 先上采样
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        return self.conv(x)


class DW_Encoder_de(nn.Module):

    def __init__(self, message_length, blocks=2, channels=64, attention=None, Adain = False):
        super(DW_Encoder_de, self).__init__()
        
        # 编码压缩过程
        self.conv1 = ConvBlock(3, 16, blocks=blocks)
        self.down1 = Down(16, 32, blocks=blocks)
        self.down2 = Down(32, 64, blocks=blocks)
        self.down3 = Down(64, 128, blocks=blocks)
        self.down4 = Down(128, 256, blocks=blocks)

        # 解码扩增过程
        self.up3 = UP(256, 128)
        self.linear3 = nn.Linear(message_length, message_length * message_length)

        self.Conv_message3 = ConvBlock(1, channels * 2, blocks = blocks)
        self.att3 = ResBlock(128 * 2 + channels * 2, 128, blocks=blocks, attention=attention)
        self.att3_n = ResBlock(128 * 2, 128, blocks=blocks, attention=attention)

        self.up2 = UP(128, 64)
        self.up2_n = UP(128, 64)
        self.linear2 = nn.Linear(message_length, message_length * message_length)

        self.Conv_message2 = ConvBlock(1, channels, blocks = blocks)
        self.att2 = ResBlock(64 * 2 + channels, 64, blocks=blocks, attention=attention)
        self.att2_n = ResBlock(64 * 2, 64, blocks=blocks, attention=attention)
        

        self.up1 = UP(64, 32)
        self.up1_n = UP(64, 32)
        self.linear1 = nn.Linear(message_length, message_length * message_length)

        self.Conv_message1 = ConvBlock(1, channels // 2, blocks = blocks)
        self.att1 = ResBlock(32 * 2 + channels // 2, 32, blocks=blocks, attention=attention)
        self.att1_n = ResBlock(32 * 2, 32, blocks=blocks, attention=attention)
        

        self.up0 = UP(32, 16)
        self.up0_n = UP(32, 16)
        self.linear0 = nn.Linear(message_length, message_length * message_length)

        self.Conv_message0 = ConvBlock(1, channels // 4, blocks = blocks)
        self.att0 = ResBlock(16 * 2 + channels // 4, 16, blocks=blocks, attention=attention)
        self.att0_n = ResBlock(16 * 2, 16, blocks=blocks, attention=attention)
        
        self.Conv_1x1 = nn.Conv2d(16 + 3, 3, kernel_size=1, stride=1, padding=0)
        self.Conv_1x1_n = nn.Conv2d(16 + 3, 3, kernel_size=1, stride=1, padding=0)
        
        self.message_length = message_length
        self.adain = Adain
        
    # @autocast(True)
    def forward(self, x, watermark, mask = None, hidden = False):
        d0 = self.conv1(x)
        d1 = self.down1(d0)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)

        u3 = self.up3(d4)
        expanded_message = self.linear3(watermark)
        expanded_message = expanded_message.view(-1, 1, self.message_length, self.message_length) # 拓展到二维 
        expanded_message = F.interpolate(expanded_message,
                                        size=(d3.shape[2], d3.shape[3]),
                                        mode='nearest')
        expanded_message = self.Conv_message3(expanded_message)
        
        if mask is not None:
            mask_rez = F.interpolate(mask,
                                    size=(d3.shape[2], d3.shape[3]),
                                    mode='nearest')
            if self.adain:
                img_feat_mean = torch.mean(u3, dim = [2, 3], keepdim = True)
                img_feat_std = torch.std(u3, dim = [2, 3], unbiased = False, keepdim = True)
                expanded_message = F.instance_norm(expanded_message) * img_feat_std + img_feat_mean
            
            expanded_message = expanded_message * mask_rez[:, 0].unsqueeze(1)
        
        u3_m = torch.cat((d3, u3, expanded_message), dim=1)
        u3_m = self.att3(u3_m)
        u3_n = torch.cat((d3, u3), dim=1)
        u3_n = self.att3_n(u3_n)
        
        u2_m = self.up2(u3_m)
        u2_n = self.up2_n(u3_n)
        expanded_message = self.linear2(watermark)
        expanded_message = expanded_message.view(-1, 1, self.message_length, self.message_length)
        expanded_message = F.interpolate(expanded_message, size=(d2.shape[2], d2.shape[3]),
                                                           mode='nearest')
        expanded_message = self.Conv_message2(expanded_message) # 通道维度拓展为64
        
        if mask is not None:
            mask_rez = F.interpolate(mask,
                                    size=(d2.shape[2], d2.shape[3]),
                                    mode='nearest')
            if self.adain:
                img_feat_mean = torch.mean(u2, dim = [2, 3], keepdim = True)
                img_feat_std = torch.std(u2, dim = [2, 3], unbiased = False, keepdim = True)
                expanded_message = F.instance_norm(expanded_message) * img_feat_std + img_feat_mean
            
            expanded_message = expanded_message * mask_rez[:, 0].unsqueeze(1)
            
        u2_m = torch.cat((d2, u2_m, expanded_message), dim=1)
        u2_m = self.att2(u2_m)
        u2_n = torch.cat((d2, u2_n), dim=1)
        u2_n = self.att2_n(u2_n)
        
        u1_m = self.up1(u2_m) # 融合后特征进行上采样
        u1_n = self.up1_n(u2_n) # 融合后特征进行上采样
        expanded_message = self.linear1(watermark)
        expanded_message = expanded_message.view(-1, 1, self.message_length , self.message_length)
        expanded_message = F.interpolate(expanded_message, size=(d1.shape[2], d1.shape[3]),
                                                           mode='nearest')
        expanded_message = self.Conv_message1(expanded_message)
        
        if mask is not None:
            mask_rez = F.interpolate(mask,
                                    size=(d1.shape[2], d1.shape[3]),
                                    mode='nearest')
            if self.adain:
                img_feat_mean = torch.mean(u1, dim = [2, 3], keepdim = True)
                img_feat_std = torch.std(u1, dim = [2, 3], unbiased = False, keepdim = True)
                expanded_message = F.instance_norm(expanded_message) * img_feat_std + img_feat_mean
                
            expanded_message = expanded_message * mask_rez[:, 0].unsqueeze(1)
                
        u1_m = torch.cat((d1, u1_m, expanded_message), dim=1)
        u1_m = self.att1(u1_m)
        u1_n = torch.cat((d1, u1_n), dim=1)
        u1_n = self.att1_n(u1_n)
        
        u0_m = self.up0(u1_m)
        u0_n = self.up0_n(u1_n)
        expanded_message = self.linear0(watermark)
        expanded_message = expanded_message.view(-1, 1, self.message_length, self.message_length)
        expanded_message = F.interpolate(expanded_message, size=(d0.shape[2], d0.shape[3]),
                                                           mode='nearest')
        expanded_message = self.Conv_message0(expanded_message)
        
        if mask is not None:
            mask_rez = F.interpolate(mask,
                                    size=(d0.shape[2], d0.shape[3]),
                                    mode='nearest')
            if self.adain:
                img_feat_mean = torch.mean(u0, dim = [2, 3], keepdim = True)
                img_feat_var = torch.std(u0, dim = [2, 3], unbiased = False, keepdim = True)
                expanded_message = F.instance_norm(expanded_message) * img_feat_var + img_feat_mean
            
            expanded_message = expanded_message * mask_rez[:, 0].unsqueeze(1)
            
        u0_m = torch.cat((d0, u0_m, expanded_message), dim=1)
        u0_m = self.att0(u0_m)
        u0_n = torch.cat((d0, u0_n), dim=1)
        u0_n = self.att0_n(u0_n)
        
        image_m = self.Conv_1x1(torch.cat((x, u0_m), dim=1))
        image_n = self.Conv_1x1_n(torch.cat((x, u0_n), dim=1))

        forward_image_m = image_m.clone().detach()
        gap_m = forward_image_m.clamp(-1, 1) - forward_image_m
        
        forward_image_n = image_n.clone().detach()
        gap_n = forward_image_n.clamp(-1, 1) - forward_image_n
        

        return image_m + gap_m, image_n + gap_n
    
