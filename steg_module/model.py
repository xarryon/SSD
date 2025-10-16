import torch
import torch.nn as nn
from ssd.steg_module.ssm import SSM
from ssd.steg_module.utils.common import exp_spatial, sp_spatial
from torchvision.utils import save_image
import steg_module.modules.Unet_common as common
from steg_module.rrdb_denselayer import ResidualUnetBlock_out
from steg_module.modules.id.facenet import FaceNet_random 
from easydict import EasyDict

import yaml
import os

class Model(nn.Module):
    def __init__(self, length, patch_size, clamp, device): # denoiser
        super(Model, self).__init__()
        self.length = length
        self.patch_size = patch_size
        self.clamp = clamp
        
        self.model = SSM(self.clamp, self.length)
        
        self.dwt = common.DWT()
        
    def forward(self, x1, x2, rev=False, hidden = False, condition = None):
        if not rev:
            if hidden:
                out1, out2, out3 = self.model(x1, x2, condition = condition)
                
                return out1, out2, out3
            
            else:

                out1, out2, out3 = self.model(x1, x2, condition = condition)
                
                return out1, out2, out3

        else:

            out1, out2 = self.model(x1, x2, rev=True, condition=condition)
            
            return out1, out2

def load_INN(model_cfg, pth, num):
    with open(model_cfg, 'r') as f:
        c = EasyDict(yaml.load(f, Loader=yaml.SafeLoader))
        
    stegnet = Model(c.size, c.patch_size, c.clamp, 'cuda:0').to()
    pretrained_weight = torch.load(os.path.join(pth, num))
    stegnet.load_state_dict(pretrained_weight['net'], strict=True)
    stegnet.eval()
    
    for param in stegnet.parameters():
        param.requires_grad = False
    
    return stegnet