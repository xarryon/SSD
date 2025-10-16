from . import *
from .noise_layers import *


class Random_Noise(nn.Module):
    # batch中每个样本执行不用的noise变换 #
    def __init__(self, layers, len_layers_R, len_layers_F):
        super(Random_Noise, self).__init__()
        for i in range(len(layers)):
            layers[i] = eval(layers[i])
        self.layers = layers
        self.noise = nn.Sequential(*layers)
        
        self.len_layers_R = len_layers_R
        self.len_layers_F = len_layers_F

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])


    def forward(self, image_cover_mask):
        image, cover_image, mask = image_cover_mask[0], image_cover_mask[1], image_cover_mask[2]
        forward_image = image.clone().detach()
        forward_cover_image = cover_image.clone().detach()
        forward_mask = mask
        noised_image_C = torch.zeros_like(forward_image)
        
        for index in range(forward_image.shape[0]):
            random_noise_layer_C = np.random.choice(self.noise, 1)[0] # 随机抽取任意变换
            noised_image_C[index] = random_noise_layer_C([forward_image[index].clone().unsqueeze(0), forward_cover_image[index].clone().unsqueeze(0)])

        noised_image_gap_C = noised_image_C.clamp(-1, 1) - forward_image
        noised_image_gap_C.to(torch.float32)
        
        return image + noised_image_gap_C



class Noise_ADV(nn.Module):
    # batch中每个样本执行不用的noise变换 #
    def __init__(self):
        super(Noise_ADV, self).__init__()
        
        GB = GaussianBlur().eval()
        MB = MedianBlur().eval() 
        
        GN = GaussianNoise().eval() 
        SP = SaltPepper().eval() 
        
        B = Brightness().eval() 
        C = Contrast().eval() 
        S = Saturation().eval() 
        H = Hue().eval()
        
        Rz = Resize().eval()
        jpeg = JpegTest().eval()
        
        self.noise =  [GB, GN, MB, SP, jpeg]
        
    def forward(self, img):
        if torch.min(img) > 0:
            img = img * 2 - 1
            
        forward_image = img.clone().detach()
        noised_image = torch.zeros_like(forward_image)
        
        for index in range(forward_image.shape[0]):
            random_noise_layer = np.random.choice(self.noise, 1)[0] # 随机抽取任意变换
            noised_image[index] += random_noise_layer([forward_image[index].clone().unsqueeze(0), 0])[0]
        
        noised_image_gap = noised_image.clamp(-1, 1) - forward_image
    
        if torch.min(img) > 0:
            return 0.5 * (1 + (img + noised_image_gap))
        else:
            return img + noised_image_gap