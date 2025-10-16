import torch
import torch.nn.functional as F
from torch import nn
from .backbone import Backbone

class FaceNet(nn.Module):
    def __init__(self, gpu=None):
        super(FaceNet, self).__init__()
        print('Loading ResNet ArcFace')
        self.facenet = Backbone(input_size=112, num_layers=50, drop_ratio=0.6, mode='ir_se')
        
        if gpu is not None:
            self.facenet.load_state_dict(torch.load('modules/id/model_ir_se50.pth', map_location='cuda:'+str(gpu)))
        else:
            self.facenet.load_state_dict(torch.load('modules/id/model_ir_se50.pth'))
            
        self.face_pool = torch.nn.AdaptiveAvgPool2d((112, 112))
        self.facenet.eval()
            
            
    def extract_feats(self, x):
        if x.size(-1) == 256:
            x = x[:, :, 35:223, 32:220]  # Crop interesting region
        elif x.size(-1) == 128:
            x = x[:, :, 16:112, 16:112]
            
        x = self.face_pool(x)
        x_feats = self.facenet(x)
        
        return x_feats


    def forward(self, x):
        '''
        y_hat:根据y生成的人脸图像
        y:原始人脸图像
        x:原始人脸在另外一个角度的图像
        '''

        x_feats = self.extract_feats(x)
        
        return x_feats
    
        
class FaceNet_random(nn.Module):
    def __init__(self, gpu=None):
        super(FaceNet_random, self).__init__()
        print('Loading ResNet ArcFace')
        self.facenet = Backbone(input_size=112, num_layers=50, drop_ratio=0.6, mode='ir_se')
        
        if gpu is not None:
            self.facenet.load_state_dict(torch.load('modules/id/model_ir_se50.pth', map_location=gpu))
        else:
            self.facenet.load_state_dict(torch.load('modules/id/model_ir_se50.pth'))
            
        self.face_pool = torch.nn.AdaptiveAvgPool2d((112, 112))
        self.facenet.eval()
            
    def extract_feats(self, x):
        if x.size(-1) == 256:
            x = x[:, :, 35:223, 32:220]  # Crop interesting region
        elif x.size(-1) == 128:
            x = x[:, :, 16:112, 16:112]
            
        x = self.face_pool(x)
        x_feats = self.facenet(x)
        
        return x_feats


    def forward(self, x):
        x_feats = self.extract_feats(x)
        weight = F.softmax(torch.sum(x_feats.view(x_feats.size(0), -1, 4), dim=1), dim=-1).unsqueeze(-1).unsqueeze(-1)

        return weight
