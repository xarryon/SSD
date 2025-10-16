import random


def get_random_float(float_range: [float]):
	return random.random() * (float_range[1] - float_range[0]) + float_range[0]


def get_random_int(int_range: [int]):
	return random.randint(int_range[0], int_range[1])

from .identity import Identity
from .crop import Dropout
from .salt_pepper import SaltPepper
from .jpeg import JpegTest
from .resize import Resize
from .kornia_noises import GaussianBlur, BoxBlur, MotionBlur, GaussianNoise, MedianBlur, Brightness, Contrast, Saturation, Hue, Rotation, Affine, Elastic, Flipping, Grayscale
from .crop import FaceBoxCrop, FaceEdgeCrop_new, FaceEdgeCrop_train, FaceCropout

############################################
###### GAN ######
# from network.noise_layers.stargan.stargan import StarGan
# from .simswap.test_one_image import SimSwap
# from network.noise_layers.e4s_main.scripts.face_swap import E4S
# from network.noise_layers.mega.inference.mega import MegaFS

###### Diffusion ######
# from network.noise_layers.diffFace.model import DiffFace
# from network.noise_layers.diffSwap.swap_pipeline import DiffSwap
# from network.noise_layers.fading.fading import Fading
# from network.noise_layers.diffAE.diffAE import DiffAE