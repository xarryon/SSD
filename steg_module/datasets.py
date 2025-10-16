import glob
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from natsort import natsorted


def to_rgb(image):
    rgb_image = Image.new("RGB", image.size)
    rgb_image.paste(image)
    return rgb_image


class SSD_Dataset(Dataset):
    def __init__(self, c, transforms_=None, mode="train"):

        self.transform = transforms_
        self.mode = mode
        
        if mode == 'train':
            # train
            self.files = natsorted(sorted(glob.glob(c.TRAIN_PATH + "/*." + c.format_train)))
        elif mode == 'val':
            # test
            self.files = sorted(glob.glob(c.VAL_PATH + "/*." + c.format_val))
        elif mode == 'test':
            # test
            self.files = sorted(glob.glob(c.TEST_PATH + "/*." + c.format_test))
            
    def __getitem__(self, index):
        try:
            image = Image.open(self.files[index])
            image = to_rgb(image)
            item = self.transform(image)
            return item

        except:
            return self.__getitem__(index + 1)

    def __len__(self):
        if self.mode == 'shuffle':
            return max(len(self.files_cover), len(self.files_secret))

        else:
            return len(self.files)
