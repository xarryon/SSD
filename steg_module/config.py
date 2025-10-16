# Super parameters
clamp= [2, 1.75, 1.5, 1.25, 1, 0.75, 0.5, 0.25]
channels_in= 3
log10_lr= -4.5
lr= 0.0001
epochs= 50
weight_decay= 0.00001
init_scale= 0.01
length= 8

# Mode:
dwt_label= True
prograss_flip= True

lamda_reconstruction= 10
lamda_guide= 1
lamda_adv= 0.1
lamda_c= 1
lamda_z= 1

# Train:
batch_size= 32
size= 256
beta_1= 0.5
beta_2= 0.999
weight_step= 40
gamma= 0.1
patch_size= 64


# Val:
cropsize_val= 1
batchsize_val= 4
batchsize_test= 1
shuffle_val: False
val_freq= 5


# Dataset
TRAIN_PATH= '/dataset/train_256/'
VAL_PATH= '/dataset/val_256/'
TEST_PATH= '/dataset/test_256/'
format_train= 'png'
format_val= 'png'
format_test= 'png'


# Display and logging:
loss_display_cutoff= 2.0
loss_names= ['L', 'lr']
silent= False
live_visualization= False
progress_bar=False


# Saving checkpoints:
MODEL_PATH= '/dataset/model/'
checkpoint_on_error= True
SAVE_freq: 5

IMAGE_PATH= '/image/'
IMAGE_PATH_cover=IMAGE_PATH + 'cover/'
IMAGE_PATH_secret= IMAGE_PATH + 'secret/'
IMAGE_PATH_steg= IMAGE_PATH + 'steg/'
IMAGE_PATH_secret_rev= IMAGE_PATH + 'secret-rev/'
IMAGE_PATH_cover_rev= IMAGE_PATH + 'cover-rev/'

EXP_DIR= 'weights/'

# Load:
suffix= 'model.pt'
tain_next= False
trained_epoch= 0

# Noise_layers
noise_layers=['Identity()', 'JpegTest()','GaussianBlur()', 'GaussianNoise()']
