import torch
import torch.nn as nn
import numpy as np
from kornia.augmentation import RandomResizedCrop
import random

class FaceBoxCrop(nn.Module):

    def __init__(self, prob=0.035):
        super(FaceBoxCrop, self).__init__()
        self.height_ratio = int(np.sqrt(prob) * 100) / 100
        self.width_ratio = int(np.sqrt(prob) * 100) / 100

    def forward(self, image_cover_mask):
        image, mask = image_cover_mask[0], image_cover_mask[-1]
        
        h_start, h_end, w_start, w_end = get_random_rectangle_inside(image.shape, self.height_ratio,
                                                                        self.width_ratio)
        h_center = h_start + (h_end - h_start) // 2
        w_center = w_start + (w_end - w_start) // 2
        
        # while(mask[:, 0, h_center, w_center] == 1):
        #     h_start, h_end, w_start, w_end = get_random_rectangle_inside(image.shape, self.height_ratio,
        #                                                                 self.width_ratio)
        #     h_center = h_start + (h_end - h_start) // 2
        #     w_center = w_start + (w_end - w_start) // 2

        maskk = torch.zeros_like(image)
        maskk[:, :, h_start: h_end, w_start: w_end] = 1
        output = image.clone()
        output[maskk==1] = -1
        
        return output


# class FaceEdgeCrop(nn.Module):

#     def __init__(self, prob = 0.035):
#         super(FaceEdgeCrop, self).__init__()
#         self.height_ratio = int(np.sqrt(prob) * 100) / 100
#         self.width_ratio = int(np.sqrt(prob) * 100) / 100
        
#     def forward(self, image_cover_mask):
#         image = image_cover_mask
        
#         h, w = image.shape[2], image.shape[3]
#         self.crop = RandomResizedCrop(size = (h, w), scale = (0.95, 1), ratio=(1, 1))
        
#         image = self.crop(image)
        
#         return image
    
# class FaceEdgeCrop(nn.Module):

#     def __init__(self, prob=0.035):
#         super(FaceEdgeCrop, self).__init__()
#         self.height_ratio = int(np.sqrt(prob) * 100) / 100
#         self.width_ratio = int(np.sqrt(prob) * 100) / 100
        
#     def forward(self, image_cover_mask):
#         image, mask = image_cover_mask[0], image_cover_mask[-1]
        
#         # image = image * (1 - mask)
#         image = image * mask
        
#         h, w = image.shape[2], image.shape[3]
#         self.crop = RandomResizedCrop(size = (h, w), scale = (0.9, 0.9), ratio=(1, 1))
        
#         image = self.crop(image)
        
#         return image

class FaceEdgeCrop_train(nn.Module):

    def __init__(self, ratio = 0.8):
        super(FaceEdgeCrop_train, self).__init__()
        self.height_ratio = ratio
        self.width_ratio = ratio
        self.ratio = ratio
        
    def forward(self, image_cover_mask):
        image = image_cover_mask[0]
        # image = image_cover_mask
        gain = random.random() * (1 - self.ratio)
        
        h, w = image.shape[2], image.shape[3]
        h_r, w_r = int(h * (1 - self.height_ratio - gain)), int(w * (1 - self.width_ratio - gain))
        img_new = torch.zeros_like(image) - 1
        img_new[:, :, h_r : h - h_r, w_r : w - w_r] = image[:, :, h_r : h - h_r, w_r : w - w_r]
        
        return img_new
    
class FaceEdgeCrop_new(nn.Module):

    def __init__(self, ratio = 0.7):
        super(FaceEdgeCrop_new, self).__init__()
        self.height_ratio = ratio
        self.width_ratio = ratio
        
    def forward(self, image_cover_mask):
        image, mask = image_cover_mask[0], image_cover_mask[2]
        
        h, w = image.shape[2], image.shape[3]

        indices = torch.nonzero(mask[0, 0])
        top_left = indices.min(dim=0)[0]
        bottom_right = indices.max(dim=0)[0]
        
        top, left = top_left[0], top_left[1]
        bottom, right = bottom_right[0], bottom_right[1]
        
        top_new, left_new = top * self.height_ratio, left * self.width_ratio
        bottom_new, right_new = bottom + (h - bottom) * (1 - self.height_ratio), right + (w - right) * (1 - self.width_ratio)
        
        # h_r, w_r = int(h * (1 - self.height_ratio)), int(w * (1 - self.width_ratio))
        # img_new = torch.zeros_like(image)
        # img_new[:, :, h_r : h - h_r, w_r : w - w_r] += image[:, :, h_r : h - h_r, w_r : w - w_r]
        
        img_new = torch.zeros_like(image) - 1
        img_new[:, :, int(top_new) : int(bottom_new), int(left_new) : int(right_new)] = image[:, :, int(top_new) : int(bottom_new), int(left_new) : int(right_new)]

        return img_new

class FaceCropout(nn.Module):

    def __init__(self, prob=0.5):
        super(FaceCropout, self).__init__()
        self.height_ratio = int(np.sqrt(prob) * 100) / 100
        self.width_ratio = int(np.sqrt(prob) * 100) / 100

    def forward(self, image_cover_mask):
        image, cover_image = image_cover_mask[0], image_cover_mask[1]

        # mask = mask[:, 0: 3, :, :]
        output = torch.ones_like(image) * -1

        h_start, h_end, w_start, w_end = get_random_rectangle_inside(image.shape, 
                                                                     self.height_ratio,
                                                                     self.width_ratio)
        # output = cover_image.clone()
        output[:, :, h_start: h_end, w_start: w_end] = image[:, :, h_start: h_end, w_start: w_end]

        return output  #image * mask + output * (1 - mask) 
        # return image * mask + cover_image * (1 - mask) #image * mask + output * (1 - mask)


class Dropout(nn.Module):

    def __init__(self, prob=0.95):
        super(Dropout, self).__init__()
        self.prob = prob

    def forward(self, image_cover_mask):
        image, cover_image = image_cover_mask[0], image_cover_mask[1]

        #mask = mask[:, 0: 3, :, :]

        maskk = torch.Tensor(np.random.choice([0.0, 1.0], image.shape[2:], p=[self.prob, 1 - self.prob])).to(image.device)
        maskk = maskk.type(image.dtype)
        maskk = maskk.expand_as(image)
        # output = image * (1 - maskk)
        output = image.clone()
        output[maskk == 1] = -1
        return output #output * mask + image * (1 - mask)


class FaceErase(nn.Module):

    def __init__(self):
        super(FaceErase, self).__init__()

    def forward(self, image_cover_mask):
        image, cover_image, mask = image_cover_mask[0], image_cover_mask[1], image_cover_mask[2]

        mask = mask[:, 0: 3, :, :]

        return image * (1 - mask)


class FaceEraseout(nn.Module):

    def __init__(self):
        super(FaceEraseout, self).__init__()

    def forward(self, image_cover_mask):
        image, cover_image, mask = image_cover_mask[0], image_cover_mask[1], image_cover_mask[2]

        mask = mask[:, 3: 6, :, :]

        output = image * (1 - mask) + cover_image * mask
        return output


def get_random_rectangle_inside(image_shape, height_ratio, width_ratio):
    image_height = image_shape[2]
    image_width = image_shape[3]

    remaining_height = int(height_ratio * image_height)
    remaining_width = int(width_ratio * image_width)

    if remaining_height == image_height:
        height_start = 0
    else:
        height_start = np.random.randint(0, image_height - remaining_height)

    if remaining_width == image_width:
        width_start = 0
    else:
        width_start = np.random.randint(0, image_width - remaining_width)

    return height_start, height_start + remaining_height, width_start, width_start + remaining_width

