import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.models import mobilenet_v2

    
class Detector(nn.Module):
    def __init__(self):
        super().__init__()
        self.model1 = mobilenet_v2(pretrained=True)
        self.model1 = nn.Sequential(*(list(self.model1.children())[:-1]))
        
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
                        nn.Linear(1280, 512, bias=False), # 
                        nn.ReLU(True),
                        nn.Dropout(),
                        nn.Linear(512, 1, bias=False),
                        nn.Sigmoid())
        
    
    def forward(self, ori, sec, train = False):
        det = ori - sec
        feat_det = self.model1(det)
        
        feat_diff_avg = self.gap(feat_det).squeeze(-1).squeeze(-1)
        pred = self.head(feat_diff_avg)
    
        return pred
        
        
class DetectorWM(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = mobilenet_v2(pretrained=True)
        self.model = nn.Sequential(*(list(self.model.children())[:-1]))
        
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
                        nn.Linear(1280, 1, bias=False),
                        nn.Sigmoid())
        
    
    def forward(self, ori):
        feat_ori = self.model(ori)
        feat_avg = self.gap(feat_ori).squeeze(-1).squeeze(-1)
        pred = self.head(feat_avg)
        
        return pred