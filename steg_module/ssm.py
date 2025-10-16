from ssd.steg_module.model import *
from ssd.steg_module.invblock import INV_block
import torch
from ssd.steg_module.rrdb_denselayer import ResidualTwoStreams
import ssd.steg_module.modules.Unet_common as common



class SSM(nn.Module):
    def __init__(self, clamp, length):
        super(SSM, self).__init__()
        
        if type(clamp) is not list:
            clamp = [clamp for _ in range(length)]
            
        self.inv1 = INV_block(subnet_constructor_cover=ResidualTwoStreams,clamp=clamp[0])
        self.inv2 = INV_block(subnet_constructor_cover=ResidualTwoStreams,clamp=clamp[1])
        self.inv3 = INV_block(subnet_constructor_cover=ResidualTwoStreams,clamp=clamp[2])
        self.inv4 = INV_block(subnet_constructor_cover=ResidualTwoStreams,clamp=clamp[3])

        self.inv13 = INV_block(subnet_constructor_cover=ResidualTwoStreams, clamp=clamp[4])
        self.inv14 = INV_block(subnet_constructor_cover=ResidualTwoStreams, clamp=clamp[5])
        self.inv15 = INV_block(subnet_constructor_cover=ResidualTwoStreams, clamp=clamp[6])
        self.inv16 = INV_block(subnet_constructor_cover=ResidualTwoStreams, clamp=clamp[7])
        self.iwt = common.IWT()
        
            
    def forward(self, cs, sg, rev=False, hidden = False, condition = None):
        
        if not rev:
            cs_show = self.iwt(cs)
            steg1, z1 = self.inv1(cs, sg, rev=False, condition = [condition[1], condition[3]])
            steg1_show = self.iwt(steg1)
            steg2, z2 = self.inv2(steg1, z1, rev=False, condition = [condition[1], condition[3]])
            steg2_show = self.iwt(steg2)
            steg3, z3 = self.inv3(steg2, z2, rev=False, condition = [condition[1], condition[3]])
            steg3_show = self.iwt(steg3)
            steg4, z4 = self.inv4(steg3, z3, rev=False, condition = [condition[1], condition[3]])
            steg4_show = self.iwt(steg4)

            steg7, z5 = self.inv13(steg4, z4, rev=False, condition = [condition[1], condition[3]])
            steg5_show = self.iwt(steg7)
            steg8, z6 = self.inv14(steg7, z5, rev=False, condition = [condition[1], condition[3]])
            steg6_show = self.iwt(steg8)
            steg9, z7 = self.inv15(steg8, z6, rev=False, condition = [condition[1], condition[3]])
            steg7_show = self.iwt(steg9)
            steg10, z8 = self.inv16(steg9, z7, rev=False, condition = [condition[1], condition[3]])
            
            return steg10, torch.stack((z1, z2, z3, z4, z5, z6, z7, z8)), torch.stack((steg7_show, steg6_show, steg5_show, steg4_show,
                                                                                        steg3_show, steg2_show, steg1_show, cs_show))

        
        else:
            cov1, sec1 = self.inv16(cs, sg, rev=True, condition = [condition[1], condition[3]])
            cov1_show = self.iwt(cov1)
            cov2, sec2 = self.inv15(cov1, sec1, rev=True, condition = [condition[1], condition[3]])
            cov2_show = self.iwt(cov2)
            cov3, sec3 = self.inv14(cov2, sec2, rev=True, condition = [condition[1], condition[3]])
            cov3_show = self.iwt(cov3)
            cov4, sec4 = self.inv13(cov3, sec3, rev=True, condition = [condition[1], condition[3]])
            cov4_show = self.iwt(cov4)
            cov7, sec7 = self.inv4(cov4, sec4, rev=True, condition = [condition[1], condition[3]])
            cov7_show = self.iwt(cov7)
            cov8, sec8 = self.inv3(cov7, sec7, rev=True, condition = [condition[1], condition[3]])
            cov8_show = self.iwt(cov8)
            cov9, sec9 = self.inv2(cov8, sec8, rev=True, condition = [condition[1], condition[3]])
            cov9_show = self.iwt(cov9)
            cov10, sec10 = self.inv1(cov9, sec9, rev=True, condition = [condition[1], condition[3]])
            cov10_show = self.iwt(cov10)
            
            return torch.stack((cov1_show, cov2_show, cov3_show, cov4_show, cov7_show, cov8_show, cov9_show, cov10_show)), \
                torch.stack((sec1, sec2, sec3, sec4, sec7, sec8, sec9, sec10))
