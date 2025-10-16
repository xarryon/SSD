import torch
from torch import nn
import torch.nn.functional as F
import random
from .ConvBlock import ConvBlock
from torchvision.transforms import Resize
from torchvision.transforms.functional import InterpolationMode


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


# class SeparableConv2d(nn.Module):
#     def __init__(self,in_channels,out_channels,kernel_size=1,stride=1,padding=0,dilation=1,bias=False):
#         super(SeparableConv2d,self).__init__()

#         self.conv1 = nn.Conv2d(in_channels,in_channels,kernel_size,stride,padding,dilation,groups=in_channels,bias=bias)
#         self.pointwise = nn.Conv2d(in_channels,out_channels,1,1,0,1,1,bias=bias)

#     def forward(self,x):
#         x = self.conv1(x)
#         x = self.pointwise(x)
#         return x


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


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)
    

class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class AttentionPool2d(nn.Module):
    def __init__(self, spacial_dim: int, embed_dim: int, num_heads: int, output_dim: int = None):
        super().__init__()
        self.positional_embedding = nn.Parameter(torch.randn(spacial_dim ** 2, embed_dim) / embed_dim ** 0.5)

        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads
        self.emd_dim = embed_dim
        
        # self.ln_1 = LayerNorm(512)
        # self.feedforward = nn.Sequential(nn.Linear(512, 2048, False),
        #                                  QuickGELU(),
        #                                  nn.Linear(2048, 512, False))
        # self.ln_2 = LayerNorm(512)
        # self.proj = nn.Conv2d(512, 1, 1)
        
    def forward(self, q, k ,v):
        b, c, w, h = k.shape
        q = q.flatten(start_dim=2).permute(2, 0, 1)  # NCHW -> (HW)NC
        k = k.flatten(start_dim=2).permute(2, 0, 1)  # NCHW -> (HW)NC
        v = v.flatten(start_dim=2).permute(2, 0, 1)  # NCHW -> (HW)NC
        
        q = q + self.positional_embedding[:, None, :].to(q.dtype)  # (HW+1)NC
        k = k + self.positional_embedding[:, None, :].to(q.dtype)  # (HW+1)NC
        
        x_all, _ = F.multi_head_attention_forward(
                        query=q, key=k, value=v,
                        embed_dim_to_check=q.shape[-1],
                        num_heads=self.num_heads,
                        q_proj_weight=self.q_proj.weight,
                        k_proj_weight=self.k_proj.weight,
                        v_proj_weight=self.v_proj.weight,
                        in_proj_weight=None,
                        in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
                        bias_k=None,
                        bias_v=None,
                        add_zero_attn=False,
                        dropout_p=0,
                        out_proj_weight=self.c_proj.weight,
                        out_proj_bias=self.c_proj.bias,
                        use_separate_proj_weight=True,
                        training=self.training,
                        need_weights=False)
    
        
        spa_feat = x_all.permute(1,2,0).view(b, c, h, w)
        
        return  spa_feat

        
        

class Reconstructor4(nn.Module):
    def __init__(self):
        super(Reconstructor4, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, 2, 0, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(32, 64, 3, bias=False)
        self.bn2 = nn.BatchNorm2d(64)

        self.block1 = Block(64, 128, 2, 2, start_with_relu=False, grow_first=True)
        self.block2 = Block(128, 256, 2, 2, start_with_relu=True, grow_first=True)
        self.block3 = Block(256, 512, 2, 2, start_with_relu=True, grow_first=True)
        
        self.decoder4_1 = nn.Sequential(
            nn.UpsamplingNearest2d(scale_factor=2),
            SeparableConv2d(512, 512, 3, 1, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True))
        self.decoder4_2 = Block(512, 512, 3, 1)
        
        self.decoder3_1 = nn.Sequential(
            nn.UpsamplingNearest2d(scale_factor=2),
            SeparableConv2d(512, 256, 3, 1, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True))
        self.decoder3_2 = Block(256, 256, 3, 1)
        
        self.decoder2_1 = nn.Sequential(
            nn.UpsamplingNearest2d(scale_factor=2),
            SeparableConv2d(256, 128, 3, 1, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True))
        self.decoder2_2 = Block(128, 128, 3, 1)
        
        self.decoder1_1 = nn.Sequential(
            nn.UpsamplingNearest2d(scale_factor=2),
            SeparableConv2d(128, 32, 3, 1, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True))
        self.decoder1_2 = nn.Sequential(
            nn.Conv2d(32, 3, 1, 1, bias=False),)
    
    
    def forward(self, input):
        
        x = self.conv1(input)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        
        rec = self.decoder4_1(x)
        rec0 = self.decoder4_2(rec)

        rec = self.decoder3_1(rec0) 
        rec1 = self.decoder3_2(rec)
        
        rec = self.decoder2_1(rec1)
        rec2 = self.decoder2_2(rec)
        
        rec = self.decoder1_1(rec2)
        pred = self.decoder1_2(rec)

        forward_image = pred.clone().detach()
        gap = forward_image.clamp(-1, 1) - forward_image
        
        return pred + gap
        
        

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.sqz = nn.Sequential(nn.BatchNorm2d(512), 
                                 SeparableConv2d(512, 32, 3, 1, 1, bias=False),
                                 nn.BatchNorm2d(32),
                                 nn.ReLU(inplace=True))
        
        self.decoder5 = nn.Sequential(
            nn.UpsamplingNearest2d(scale_factor=2),
            SeparableConv2d(32, 512, 3, 1, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True))
        self.decoder6 = Block(512, 512, 3, 1)
        
        self.decoder7 = nn.Sequential(
            nn.UpsamplingNearest2d(scale_factor=2),
            SeparableConv2d(512, 512, 3, 1, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True))
        self.decoder8 = Block(512, 512, 3, 1)
        
        self.decoder9 = nn.Sequential(
            nn.UpsamplingNearest2d(scale_factor=2),
            SeparableConv2d(512, 256, 3, 1, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True))
        self.decoder10 = Block(256, 256, 3, 1)
        
        self.decoder11 = nn.Sequential(
            nn.UpsamplingNearest2d(scale_factor=2),
            SeparableConv2d(256, 64, 3, 1, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True))
        
        self.decoder12 = nn.Sequential(
            nn.Conv2d(64, 3, 1, 1, bias=False),)
        
        self.gap = nn.AdaptiveAvgPool2d(1)
        
        self.mapping = nn.Sequential(nn.Flatten(),
                                    nn.Linear(512 * 14 * 14, 512),
                                    nn.BatchNorm1d(512, affine=True))
        
        
    def forward(self, input, hidden = False):
            
        sqz = self.sqz(input)
        rec = self.decoder5(sqz)
        rec0 = self.decoder6(rec)

        feat = self.mapping(rec0.clone())
            
        rec = self.decoder7(rec0) 
        rec1 = self.decoder8(rec)
        
        rec = self.decoder9(rec1)
        rec2 = self.decoder10(rec)
        
        rec = self.decoder11(rec2)
        pred = self.decoder12(rec)

        forward_image = pred.clone().detach()
        gap = forward_image.clamp(-1, 1) - forward_image
        
        if not hidden: 
            return pred + gap
        else:
            return pred + gap, l2_norm(feat)
        

class Reconstructor4(nn.Module):
    def __init__(self):
        super(Reconstructor4, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, 2, 0, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(32, 64, 3, bias=False)
        self.bn2 = nn.BatchNorm2d(64)

        self.block1 = Block(64, 128, 2, 2, start_with_relu=False, grow_first=True)
        self.block2 = Block(128, 256, 2, 2, start_with_relu=True, grow_first=True)
        self.block3 = Block(256, 512, 2, 2, start_with_relu=True, grow_first=True)
        
        self.decoder4_1 = nn.Sequential(
            nn.UpsamplingNearest2d(scale_factor=2),
            SeparableConv2d(512, 512, 3, 1, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True))
        self.decoder4_2 = Block(512, 512, 3, 1)
        
        self.decoder3_1 = nn.Sequential(
            nn.UpsamplingNearest2d(scale_factor=2),
            SeparableConv2d(512, 256, 3, 1, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True))
        self.decoder3_2 = Block(256, 256, 3, 1)
        
        self.decoder2_1 = nn.Sequential(
            nn.UpsamplingNearest2d(scale_factor=2),
            SeparableConv2d(256, 128, 3, 1, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True))
        self.decoder2_2 = Block(128, 128, 3, 1)
        
        self.decoder1_1 = nn.Sequential(
            nn.UpsamplingNearest2d(scale_factor=2),
            SeparableConv2d(128, 32, 3, 1, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True))
        self.decoder1_2 = nn.Sequential(
            nn.Conv2d(32, 3, 1, 1, bias=False),)
        
        self.hidden = []
        self.gap = nn.AdaptiveAvgPool2d(1)
        
    def forward(self, input, hidden = False):
        
        x = self.conv1(input)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        
        rec = self.decoder4_1(x)
        rec0 = self.decoder4_2(rec)

        self.hidden_0 = l2_norm(self.gap(rec0.clone()).squeeze(-1).squeeze(-1))
        
        rec = self.decoder3_1(rec0) 
        rec1 = self.decoder3_2(rec)
        
        self.hidden_1 = l2_norm(self.gap(rec1.clone()).squeeze(-1).squeeze(-1))
        
        rec = self.decoder2_1(rec1)
        rec2 = self.decoder2_2(rec)
        
        self.hidden_2 = l2_norm(self.gap(rec2.clone()).squeeze(-1).squeeze(-1))
        
        rec = self.decoder1_1(rec2)
        pred = self.decoder1_2(rec)

        forward_image = pred.clone().detach()
        gap = forward_image.clamp(-1, 1) - forward_image
        
        if not hidden:
            return pred + gap
        else:
            return pred + gap, [self.hidden_0, self.hidden_1, self.hidden_2]



class Reconstructor4_R(nn.Module):
    def __init__(self):
        super(Reconstructor4_R, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, 2, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(32, 64, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(64)

        self.block1 = Block(64, 128, 2, 2, start_with_relu=False, grow_first=True)
        self.block2 = Block(128, 256, 2, 2, start_with_relu=True, grow_first=True)
        self.block3 = Block(256, 512, 2, 2, start_with_relu=True, grow_first=True)
        
        self.decoder4_1 = nn.Sequential(
            SeparableConv2d(512, 512, 3, 1, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True))
        self.decoder4_2 = Block(512, 512, 3, 1)
        
        self.decoder3_1 = nn.Sequential(
            nn.UpsamplingNearest2d(scale_factor=2),
            SeparableConv2d(512, 256, 3, 1, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)) # 64
        self.decoder3_2 = Block(256, 256, 3, 1)
        
        self.decoder2_1 = nn.Sequential(
            nn.UpsamplingNearest2d(scale_factor=2),
            SeparableConv2d(256, 128, 3, 1, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)) # 128
        self.decoder2_2 = Block(128, 128, 3, 1)
        
        self.decoder1_1 = nn.Sequential(
            nn.UpsamplingNearest2d(scale_factor=2),
            SeparableConv2d(128, 64, 3, 1, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True))  # 256
        self.decoder1_2 = Block(64, 64, 3, 1)
        
        self.decoder0_1 = nn.Sequential(
            nn.UpsamplingNearest2d(scale_factor=2),
            SeparableConv2d(64, 32, 3, 1, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True))  # 256
        self.decoder0_2 = nn.Sequential(
            nn.Conv2d(32, 3, 1, 1, bias=False),)
        
        self.gap = nn.AdaptiveAvgPool2d(1)
        
    def forward(self, input, hidden = False):
        
        x = self.conv1(input)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        enc_0 = self.relu(x)

        enc_1 = self.block1(enc_0) #　128
        enc_2 = self.block2(enc_1) #　256
        enc_3 = self.block3(enc_2) #　512
        
        rec = self.decoder4_1(enc_3)
        rec0 = self.decoder4_2(rec)

        hidden_feat0 = l2_norm(self.gap(rec0.clone()).squeeze(-1).squeeze(-1))
        
        rec = self.decoder3_1(rec0) # 256
        rec1 = self.decoder3_2(rec + enc_2)
        
        hidden_feat1 = l2_norm(self.gap(rec1.clone()).squeeze(-1).squeeze(-1))
        
        rec = self.decoder2_1(rec1)
        rec2 = self.decoder2_2(rec + enc_1)
        
        hidden_feat2 = l2_norm(self.gap(rec2.clone()).squeeze(-1).squeeze(-1))
        
        rec = self.decoder1_1(rec2)
        rec3 = self.decoder1_2(rec + enc_0)

        rec = self.decoder0_1(rec3)
        pred = self.decoder0_2(rec)
        
        forward_image = pred.clone().detach()
        gap = forward_image.clamp(-1, 1) - forward_image
        
        if not hidden:
            return pred + gap
        else:
            return pred + gap, [hidden_feat0, hidden_feat1, hidden_feat2]
        
        
class Reconstructor5(nn.Module):
    def __init__(self):
        super(Reconstructor5, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, 2, 0, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(32, 64, 3, bias=False)
        self.bn2 = nn.BatchNorm2d(64)

        self.block1 = Block(64, 128, 2, 2, start_with_relu=False, grow_first=True)
        self.block2 = Block(128, 256, 2, 2, start_with_relu=True, grow_first=True)
        self.block3 = Block(256, 512, 2, 2, start_with_relu=True, grow_first=True)
        
        self.decoder4_1 = nn.Sequential(
            nn.UpsamplingNearest2d(scale_factor=2),
            SeparableConv2d(512, 512, 3, 1, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True))
        self.decoder4_2 = Block(512, 512, 3, 1)
        
        self.decoder3_1 = nn.Sequential(
            nn.UpsamplingNearest2d(scale_factor=2),
            SeparableConv2d(512, 256, 3, 1, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True))
        self.decoder3_2 = Block(256, 256, 3, 1)
        
        self.decoder2_1 = nn.Sequential(
            nn.UpsamplingNearest2d(scale_factor=2),
            SeparableConv2d(256, 128, 3, 1, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True))
        self.decoder2_2 = Block(128, 128, 3, 1)
        
        self.decoder1_1 = nn.Sequential(
            nn.UpsamplingNearest2d(scale_factor=2),
            SeparableConv2d(128, 32, 3, 1, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True))
        self.decoder1_2 = nn.Sequential(
            nn.Conv2d(32, 3, 1, 1, bias=False))
    
    
    def forward(self, input):
        
        x = self.conv1(input)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        
        rec = self.decoder4_1(x)
        rec0 = self.decoder4_2(rec)

        rec = self.decoder3_1(rec0) 
        rec1 = self.decoder3_2(rec)
        
        rec = self.decoder2_1(rec1)
        rec2 = self.decoder2_2(rec)
        
        rec = self.decoder1_1(rec2)
        pred = self.decoder1_2(rec)
        
        return pred
    

class Reconstructor6(nn.Module):
    def __init__(self, eps):
        super(Reconstructor6, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, 2, 0, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(32, 64, 3, bias=False)
        self.bn2 = nn.BatchNorm2d(64)

        self.block1 = Block(64, 128, 2, 2, start_with_relu=False, grow_first=True)
        self.block2 = Block(128, 256, 2, 2, start_with_relu=True, grow_first=True)
        self.block3 = Block(256, 512, 2, 2, start_with_relu=True, grow_first=True)
        
        self.decoder4_1 = nn.Sequential(
            nn.UpsamplingNearest2d(scale_factor=2),
            SeparableConv2d(512, 512, 3, 1, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True))
        self.decoder4_2 = Block(512, 512, 3, 1)
        
        self.decoder3_1 = nn.Sequential(
            nn.UpsamplingNearest2d(scale_factor=2),
            SeparableConv2d(512, 256, 3, 1, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True))
        self.decoder3_2 = Block(256, 256, 3, 1)
        
        self.decoder2_1 = nn.Sequential(
            nn.UpsamplingNearest2d(scale_factor=2),
            SeparableConv2d(256, 128, 3, 1, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True))
        self.decoder2_2 = Block(128, 128, 3, 1)
        
        self.decoder1_1 = nn.Sequential(
            nn.UpsamplingNearest2d(scale_factor=2),
            SeparableConv2d(128, 32, 3, 1, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True))
        self.decoder1_2 = nn.Sequential(
            nn.Conv2d(32, 3, 1, 1, bias=False),
            nn.Tanh())
    

        self.eps = eps
        
    def forward(self, input):
        
        x = self.conv1(input)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        
        rec = self.decoder4_1(x)
        rec0 = self.decoder4_2(rec)

        rec = self.decoder3_1(rec0) 
        rec1 = self.decoder3_2(rec)
        
        rec = self.decoder2_1(rec1)
        rec2 = self.decoder2_2(rec)
        
        rec = self.decoder1_1(rec2)
        pred = self.decoder1_2(rec) * self.eps
        
        return pred
    