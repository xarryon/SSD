import torch
from torch import nn
from .backbone import Backbone


class IDLoss(nn.Module):
    def __init__(self, gpu=None):
        super(IDLoss, self).__init__()
        print('Loading ResNet ArcFace')
        self.facenet = Backbone(input_size=112, num_layers=50, drop_ratio=0.6, mode='ir_se')
        
        if gpu is not None:
            self.facenet.load_state_dict(torch.load('modules/id/model_ir_se50.pth', map_location=gpu))
        else:
            self.facenet.load_state_dict(torch.load('modules/id/model_ir_se50.pth'))
            
        self.face_pool = torch.nn.AdaptiveAvgPool2d((112, 112))
        self.facenet.eval()
            
    def extract_feats(self, x):
        # x = x[:, :, 35:223, 32:220]  # Crop interesting region
        x = self.face_pool(x)
        x_feats = self.facenet(x)
        
        return x_feats

    def extract_hidden(self, x):
        # x = x[:, :, 35:223, 32:220]  # Crop interesting region
        x = self.face_pool(x)
        x_inp_feats = self.facenet.input_layer(x)
        x_hidden_feats = self.facenet.body(x_inp_feats)
        
        return x_inp_feats, x_hidden_feats
    
    def forward(self, x, y, repeat=1):
        sim = 0
        
        for i in range(repeat):
            x_feats = self.extract_feats(x[:,i].unsqueeze(1).repeat(1,3,1,1))
            y_feats = self.extract_feats(y[:,i].unsqueeze(1).repeat(1,3,1,1))
            sim += torch.diag(x_feats @ y_feats.T)
        
        sim = sim / repeat 
        # y_feats = y_feats.detach()
        # loss = 0
        # sim_improvement = 0
        # id_logs = []
        # count = 0

        # diff_target = y_hat_feats @ y_feats.T
        # loss = 1 - torch.diag(diff_target)
        
        return sim

