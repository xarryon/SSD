import os
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset
import torch
# from utils.funcs import crop_face
from natsort import natsorted
from hsemotion.facial_emotions import HSEmotionRecognizer
import glob
import numpy as np
import cv2
import random


class attrsImgDataset(Dataset):

    def __init__(self, path, image_size, mask_label=False, attr_path="celeba", zero=False):
        super(attrsImgDataset, self).__init__()
        self.image_size = image_size
        
        if type(path) is not list:
            self.image_dir = path
        else:
            self.image_dir = path[0]
            
        if attr_path[0:len("celebahq")] != "celebahq":
            self.attr_path = 'network/noise_layers/stargan/list_attr_celeba.txt'
        else:
            self.attr_path = 'network/noise_layers/stargan/CelebAMask-HQ-attribute-anno.txt'

        self.selected_attrs = ['Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Male', 'Young']
        self.list = [] # os.listdir(path)]
        self.attr2idx = {}
        self.idx2attr = {}
        
        self.preprocess()
        self.mask_label = mask_label
        
        if self.mask_label:
            self.dir = path[1]
            self.box_format = 'npy'
            self.img_format = 'png'
            self.files = natsorted(sorted(glob.glob(self.dir + "/*." + self.box_format)))
            
        if not zero:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
        else:
            self.transform = transforms.Compose([
                transforms.ToTensor()])
            
    def build_mask(self, img, landmarks):
        mask = np.zeros(img.shape[0:2] + (1, ), dtype = np.float32)

        landmarks = landmarks.copy()
        face = (landmarks[0:17], landmarks[68:81])
        parts = [face]

        for item in parts:
            merged = np.concatenate(item)
            cv2.fillConvexPoly(mask, cv2.convexHull(merged), 1)  # pylint: disable=no-member

        # mask = cv2.GaussianBlur(mask, ksize=(31,31), sigmaX=2, sigmaY=2)[:,:,None] / 255
        mask = np.tile(mask, 3)

        return mask
    
    
    def preprocess(self):
        """Preprocess the CelebA attribute file."""
        lines = [line.rstrip() for line in open(self.attr_path, 'r')]
        all_attr_names = lines[1].split()

        for i, attr_name in enumerate(all_attr_names):
            self.attr2idx[attr_name] = i
            self.idx2attr[i] = attr_name
            
        lines = lines[2:]
        
        for i, line in enumerate(lines):
            split = line.split()
            basename = os.path.basename(split[0])
            filename = os.path.splitext(basename)[0]
            values = split[1:]

            label = []
            for attr_name in self.selected_attrs: # 从多个attributes中选择selected_attrs的属性
                idx = self.attr2idx[attr_name]
                label.append(values[idx] == '1')

            if os.path.exists(os.path.join(self.image_dir, str(filename).zfill(5) + ".png")):
                self.list.append([str(filename).zfill(5) + ".png", label])


    def __getitem__(self, index):
        """Return one image and its corresponding attribute label."""
        
        if not self.mask_label:
            filename, label = self.list[index]
            image = Image.open(os.path.join(self.image_dir, filename)).convert("RGB")

            if image is not None:
                return self.transform(image), torch.FloatTensor(label)
        
        else:
            landmark = np.load(self.files[index])
            base = os.path.basename(self.files[index])
            
            base_num = int(base[:-4])
            _, label = self.list[base_num]
            
            image = cv2.imread(self.image_dir + base.replace(self.box_format, self.img_format))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            mask = self.build_mask(image, landmark)
            mask = np.transpose(mask, (2, 0, 1))

            image = Image.fromarray(image)
            
            return self.transform(image), torch.FloatTensor(label), mask
        

    def __len__(self):
        return len(self.list)


class attrsImgDataset_v2(Dataset):

    def __init__(self, path, image_size, attr_path="celeba", zero=True):
        super(attrsImgDataset_v2, self).__init__()
        self.image_size = image_size
        self.image_dir = path
        
        if attr_path[0:len("celebahq")] != "celebahq":
            self.attr_path = 'network/noise_layers/stargan/list_attr_celeba.txt'
        else:
            self.attr_path = 'network/noise_layers/stargan/CelebAMask-HQ-attribute-anno.txt'

        self.selected_attrs = ['Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Male', 'Young']
        self.list = [] # os.listdir(path)]
        self.attr2idx = {}
        self.idx2attr = {}
        
        self.transform_rec = transforms.Compose([
            transforms.Resize((int(self.image_size)//2, int(self.image_size)//2)),
            transforms.ToTensor(),
        ])
        self.preprocess()
        
        if not zero:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
        else:
            self.transform = transforms.Compose([
                transforms.ToTensor()])
            
    def preprocess(self):
        """Preprocess the CelebA attribute file."""
        lines = [line.rstrip() for line in open(self.attr_path, 'r')]
        all_attr_names = lines[1].split()
        for i, attr_name in enumerate(all_attr_names):
            self.attr2idx[attr_name] = i
            self.idx2attr[i] = attr_name

        lines = lines[2:]
        for i, line in enumerate(lines):
            split = line.split()
            basename = os.path.basename(split[0])
            filename = os.path.splitext(basename)[0]
            values = split[1:]

            label = []
            for attr_name in self.selected_attrs:
                idx = self.attr2idx[attr_name]
                label.append(values[idx] == '1')

            if os.path.exists(os.path.join(self.image_dir, str(filename).zfill(5) + ".png")):
                self.list.append([str(filename).zfill(5) + ".png", label])

    def __getitem__(self, index):
        """Return one image and its corresponding attribute label."""
        filename, label = self.list[index]
        image = Image.open(os.path.join(self.image_dir, filename)).convert("RGB")
        
        if image is not None:
            return self.transform(image), self.transform_rec(image), torch.FloatTensor(label)

    def __len__(self):
        return len(self.list)
    
    
class attrsImgDataset_mask(Dataset):

    def __init__(self, path, image_size, format = ["npy", "png"], zero = False):
        super(attrsImgDataset_mask, self).__init__()
        self.image_size = image_size
        
        self.img_dir = path[0]
        self.dir = path[1]
        
        self.box_format = format[0]
        self.img_format = format[1]
        
        self.files = natsorted(sorted(glob.glob(self.dir + "/*." + self.box_format)))

        if not zero:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
        else:
            self.transform = transforms.Compose([
                transforms.ToTensor()])
        
        self.transform_mask = transforms.Compose([
            transforms.ToTensor()])
        

    def build_mask(self, img, landmarks):
        mask = np.zeros(img.shape[0:2] + (1, ), dtype = np.float32)

        landmarks = landmarks.copy()
        face = (landmarks[0:17], landmarks[68:81])
        parts = [face]

        for item in parts:
            merged = np.concatenate(item)
            cv2.fillConvexPoly(mask, cv2.convexHull(merged), 1)  # pylint: disable=no-member

        # mask = cv2.GaussianBlur(mask, ksize=(31,31), sigmaX=2, sigmaY=2)[:,:,None] / 255
        mask = np.tile(mask, 3) 

        return mask
    
      
    def __getitem__(self, index):
        """Return one image and its corresponding attribute label."""
        landmark = np.load(self.files[index])
        base = os.path.basename(self.files[index])

        image = cv2.imread(self.img_dir + base.replace(self.box_format, self.img_format))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        mask = self.build_mask(image, landmark)
        mask = np.transpose(mask, (2, 0, 1))

        image = Image.fromarray(image)

        return self.transform(image), mask


    def __len__(self):
        return len(self.files)



class attrsImgDataset_v3(Dataset):

    def __init__(self, path, image_size, mode = "train", format = ["npy", "png"]):
        super(attrsImgDataset_v3, self).__init__()
        self.image_size = image_size
        
        self.img_dir = path[0]
        self.dir = path[1]
        
        self.box_format = format[0]
        self.img_format = format[1]
        
        self.mode = mode
        
        # if mode == 'train':
        #     # train
        #     self.files = natsorted(sorted(glob.glob(self.dir + "/*." + self.box_format)))
        # else:
        #     # test
        #     self.files = sorted(glob.glob(self.dir + "/*." + self.box_format))
        
        self.files = natsorted(sorted(glob.glob(self.dir + "/*." + self.box_format)))

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
        
        self.transform_mask = transforms.Compose([
            transforms.ToTensor()])
        
        self.transform_face = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])])

        
    def bbox_acquirement(self, mask):
        h, w = np.nonzero(mask[:,:,0] == 1)
        h_top, h_bottom = np.min(h), np.max(h)
        w_top, w_bottom = np.min(w), np.max(w)
        
        coordinate = np.array([h_top, h_bottom,
                                   w_top, w_bottom])
        
        return coordinate
    

    def build_mask(self, img, landmarks):
        mask = np.zeros(img.shape[0:2] + (1, ), dtype = np.float32)

        landmarks = landmarks.copy()
        face = (landmarks[0:17], landmarks[68:81])
        parts = [face]

        for item in parts:
            merged = np.concatenate(item)
            cv2.fillConvexPoly(mask, cv2.convexHull(merged), 1)  # pylint: disable=no-member

        mask = np.tile(mask, 3) 
        
        return mask
    
    
    # def build_circle(self, mask):
    #     kernel = np.ones((11, 11), np.uint8)
    #     circle = cv2.dilate(mask.copy(), kernel, iterations = 2)
    #     circle = circle - mask
        
    #     kernel_1=7
    #     kernel_1=(kernel_1,kernel_1)
    #     kernel_2=7
    #     kernel_2=(kernel_2,kernel_2)
        
    #     circle_blured = cv2.GaussianBlur(circle, kernel_1, 7)
    #     circle_blured = circle_blured/(circle_blured.max())
    #     circle_blured[circle_blured<1]=0
        
    #     circle_blured = cv2.GaussianBlur(circle_blured, kernel_2, 7)
    #     circle_blured = circle_blured/(circle_blured.max())
        
    #     return circle_blured
    
    def build_circle(self, circle, size = 3):
        kernel_1=size
        kernel_1=(kernel_1,kernel_1)
        kernel_2=size
        kernel_2=(kernel_2,kernel_2)
        
        circle_blured = cv2.GaussianBlur(circle, kernel_1, size)
        circle_blured = circle_blured/(circle_blured.max())
        
        circle_blured = cv2.GaussianBlur(circle_blured, kernel_2, size)
        circle_blured = circle_blured/(circle_blured.max())
        
        return circle_blured
    
      
    def __getitem__(self, index):
        """Return one image and its corresponding attribute label."""
        landmark = np.load(self.files[index])
        base = os.path.basename(self.files[index])

        image = cv2.imread(self.img_dir + base.replace(self.box_format, self.img_format))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        mask = self.build_mask(image, landmark)
        # coor = self.bbox_acquirement(mask)

        # face = image.copy()
        # face = face[coor[0] : coor[1], coor[2] : coor[3]]
        
        # background = image.copy()
        # background[mask == 1] = 0
        
        # background = Image.fromarray(background)
        # face = Image.fromarray(face)
        image = Image.fromarray(image)
        # circle = self.build_circle(mask)
        kernel = np.ones((11, 11), np.uint8)

        if self.mode == 'train':
            circle = cv2.dilate(mask.copy(), kernel, iterations = random.choice([2, 3]))
        
        elif self.mode == 'test':
            circle = cv2.dilate(mask.copy(), kernel, iterations = 2)
            
        circle = circle - mask
        circle[circle > 0] = 1
        circle[circle < 1] = 0
        
        mask = np.transpose(mask, (2, 0, 1))
        # circle = self.build_circle(circle, size = 3)
        circle = np.transpose(circle, (2, 0, 1))
        
        # return self.transform(image), self.transform_face(face), self.transform(background), mask, coor
        return self.transform(image), mask, circle


    def __len__(self):
        return len(self.files)
    

class attrsImgDataset_nl(Dataset):

    def __init__(self, path, image_size, format = ["jpg", "png"], zero=False):
        super(attrsImgDataset_nl, self).__init__()
        self.image_size = image_size
        
        self.img_dir = path
        
        self.img_format = format[1] # 0 for lfw / 1 for ffhq
        
        self.files = natsorted(sorted(glob.glob(self.img_dir + "/*." + self.img_format))) # for ffhq
        # self.files = natsorted(sorted(glob.glob(self.img_dir + "/**/*." + self.img_format))) # for lfw
        
        if not zero:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
        else:
            self.transform = transforms.Compose([
                transforms.ToTensor()])
        
        self.transform_face = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])])

      
    def __getitem__(self, index):
        """Return one image and its corresponding attribute label."""
        img_pth = self.files[index]

        image = cv2.imread(img_pth)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (self.image_size, self.image_size))
        
        image = Image.fromarray(image)

        return self.transform(image)


    def __len__(self):
        return len(self.files)
