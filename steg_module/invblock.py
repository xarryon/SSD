import torch
import torch.nn as nn
from ssd.steg_module.rrdb_denselayer import ResidualUnetBlock_out, ResidualTwoStreams

class INV_block(nn.Module):
    def __init__(self, clamp, subnet_constructor_cover=ResidualTwoStreams, subnet_constructor_secret=ResidualUnetBlock_out, harr=True, in_1=3, in_2=3, id=False):
        super().__init__()
        if harr:
            self.split_len1 = in_1 * 4
            self.split_len2 = in_2
        else:
            self.split_len1 = in_1 
            self.split_len2 = in_2
            
        self.clamp = clamp
        
        self.r = subnet_constructor_cover(self.split_len1, self.split_len2, hidden = 64, id = id)
        self.f = subnet_constructor_secret(self.split_len2, self.split_len1)
        
    def e(self, s):
        return torch.exp(self.clamp * 2 * (torch.sigmoid(s) - 0.5))

    def forward(self, cs, sg, rev=False, condition=None):
        x1, x2 = cs, sg

        if not rev:
            t2 = self.f(x2)
            y1 = x1 + t2 
            s1, t1 = self.r(y1, condition) 
            y2 = self.e(s1) * x2 + t1

        else:
            s1, t1 = self.r(x1, condition)
            y2 = (x2 - t1) / self.e(s1)
            t2 = self.f(y2)
            y1 = (x1 - t2)

        return y1, y2
