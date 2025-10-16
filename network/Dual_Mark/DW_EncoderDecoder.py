from . import *
from .Encoder_U import DW_Encoder, DW_Encoder_v2, DW_EncoderV3, DW_Encoder_ref, DW_Encoder_de
from .Decoder_U import DW_Decoder, DW_Decoder_v2, DW_Decoder_Multi
# from .Noise import Noise
from .Random_Noise import Random_Noise
from torch.cuda.amp import autocast as autocast
from .Denoise import Reconstructor5
from .noise_layers.crop import FaceEdgeCrop_train
import steg_module.modules.Unet_common as common

dwt = common.DWT()
iwt = common.IWT()
    
class DW_EncoderDecoder(nn.Module):
    '''
    A Sequential of Encoder_MP-Noise-Decoder
    '''

    def __init__(self, message_length, noise_layers_R, noise_layers_F, attention_encoder, attention_decoder):
        super(DW_EncoderDecoder, self).__init__()
        self.encoder = DW_Encoder_ref(message_length, attention = attention_encoder)
        self.noise = Random_Noise(noise_layers_R + noise_layers_F, len(noise_layers_R), len(noise_layers_F))
        self.decoder_C = DW_Decoder(message_length, attention = attention_decoder)
        self.decoder_RF = DW_Decoder(message_length, attention = attention_decoder)

    def forward(self, image, message, mask): # 
        encoded_image = self.encoder(image, message)
        noised_image_C, noised_image_R, noised_image_F = self.noise([encoded_image, image, mask])
        decoded_message_C = self.decoder_C(noised_image_C)
        decoded_message_R = self.decoder_RF(noised_image_R)
        decoded_message_F = self.decoder_RF(noised_image_F)
  
        return encoded_image, noised_image_C, decoded_message_C, decoded_message_R, decoded_message_F


class RW_EncoderDecoder(nn.Module):
    '''
    A Sequential of Encoder_MP-Noise-Decoder
    '''

    def __init__(self, message_length, noise_layers_R, noise_layers_F, attention_encoder, attention_decoder, adain, block_num=2, scale = 1):
        super(RW_EncoderDecoder, self).__init__()
        self.encoder = DW_Encoder(message_length, blocks = block_num, attention = attention_encoder, Adain = adain, scale = scale)
        self.noise = Random_Noise(noise_layers_R + noise_layers_F, len(noise_layers_R), len(noise_layers_F))
        self.decoder_C = DW_Decoder(message_length, attention = attention_decoder, scale = scale)
        
        # self.face_crop = FaceEdgeCrop_train()
        
    def forward(self, image, message, mask, inn): # 
        encoded_image, feat = self.encoder(image, message, mask)
        # encoded_image, feat = self.encoder(image, message, mask = None)
        size = encoded_image.shape[-1]
        
        encoded_images_rez = encoded_image.clone()
        img_rez = image
        
        if inn is not None:
            stegnet, denoiser = inn[0], inn[1]
            image2 = F.interpolate(image * 2 - 1, scale_factor=2)
            images_wavelets = dwt(image2)
            
            output_steg, output_z_sets, _ = stegnet(images_wavelets,
                                                    0.5*(1+encoded_images_rez), 
                                                    condition = [None, [None, None], 0, 1]) # 隐写图像
            steg_img = iwt(output_steg)
            noised_steg_img = self.noise([steg_img, img_rez, None])
            de_steg_img = 0.5 * (denoiser(noised_steg_img) + 1)
            
            output_steg = dwt(de_steg_img)
            output_zeros = torch.zeros_like(output_z_sets[-1])
            cover_rev_sets, secret_rev_sets = stegnet(output_steg, output_zeros, rev = True,
                                                      condition = [None, [None, None], 0, 1])
            noised_image_C = secret_rev_sets[-1] * 2 - 1
            del output_steg
            del output_z_sets
            del cover_rev_sets
        else:                                            
            noised_image_C = self.noise([encoded_images_rez, img_rez, None])
        
        noised_image_C_mask = noised_image_C.clone()
        
        if mask is not None:
            # noised_image_C_mask = noised_image_C_mask * mask # 去掉人脸
            noised_image_C_mask = 0.5 * (noised_image_C_mask + 1) * mask
            noised_image_C_mask = noised_image_C_mask * 2 - 1
            
        # noised_image_C_mask = self.face_crop(noised_image_C_mask)
        decoded_message_C = self.decoder_C(noised_image_C_mask)

        # image_mask = image.clone()
        # image_mask = image_mask * mask
        # decoded_message_none = self.decoder_C(image_mask)
        # return encoded_image, noised_image_C, decoded_message_C, decoded_message_none

        return encoded_image, noised_image_C, decoded_message_C


class RW_EncoderDecoder_2(nn.Module):
    '''
    A Sequential of Encoder_MP-Noise-Decoder
    '''

    def __init__(self, message_length, noise_layers_R, noise_layers_F, attention_encoder, attention_decoder, denoise = False):
        super(RW_EncoderDecoder_2, self).__init__()
        self.encoder = DW_EncoderV3(message_length, attention = attention_encoder)
        self.noise = Random_Noise(noise_layers_R + noise_layers_F, len(noise_layers_R), len(noise_layers_F))
  
        if denoise == True:
            self.denoiser = Reconstructor5()
            denoise_pth = '/weights/face/best_0.pth'
            denoise_weight = torch.load(denoise_pth)
            denoise_state_dict = {k[7:] : v for k, v in denoise_weight.items()}
            self.denoiser.load_state_dict(denoise_state_dict)
            self.denoiser.eval()
        else:
            self.denoiser = None
            
        self.decoder_C = DW_Decoder(message_length, attention = attention_decoder)
        self.decoder_D = DW_Decoder(message_length, attention = attention_decoder)
        
        
    def forward(self, image, message_exp, message_ide, mask): # 
        encoded_image = self.encoder(image, message_exp, message_ide, 1 - mask)
        noised_image_C = self.noise([encoded_image, image, None])
        
        if self.denoiser is not None:
            with torch.no_grad():
                self.denoiser.eval()
                noised_image_C = self.denoiser(noised_image_C) + noised_image_C
            
        noised_image_C_mask = noised_image_C.clone()
        noised_image_C_mask_bg = noised_image_C_mask * (1 - mask)
        noised_image_C_mask_fg = noised_image_C_mask * (mask)
        
        decoded_message_C = self.decoder_C(noised_image_C_mask_bg)
        decoded_message_D = self.decoder_D(noised_image_C_mask_fg)
        
        return encoded_image, noised_image_C, decoded_message_C, decoded_message_D
    
    

class RW_EncoderDecoder_rec(nn.Module):
    '''
    A Sequential of Encoder_MP-Noise-Decoder
    '''

    def __init__(self, message_length, noise_layers_R, noise_layers_F, attention_encoder, attention_decoder):
        super(RW_EncoderDecoder_rec, self).__init__()
        self.encoder = DW_Encoder_v2(message_length, attention = attention_encoder)
        self.noise = Random_Noise(noise_layers_R + noise_layers_F, len(noise_layers_R), len(noise_layers_F))
        self.decoder_C = DW_Decoder_v2(message_length, attention = attention_decoder)

    # @autocast()
    def forward(self, image, message, mask): # 

        encoded_image = self.encoder(image, message)
        noised_image_C = self.noise([encoded_image, image, mask])
        decoded_message_C = self.decoder_C(noised_image_C)

        return encoded_image, noised_image_C, decoded_message_C


class DE_EncoderDecoder(nn.Module):
    '''
    A Sequential of Encoder_MP-Noise-Decoder
    '''

    def __init__(self, message_length, noise_layers_R, noise_layers_F, attention_encoder, attention_decoder, adain, block_num=2):
        super(DE_EncoderDecoder, self).__init__()
        self.encoder = DW_Encoder_de(message_length, blocks = block_num, attention = attention_encoder, Adain = adain)
        self.noise = Random_Noise(noise_layers_R + noise_layers_F, len(noise_layers_R), len(noise_layers_F))
        self.decoder_C = DW_Decoder(message_length, attention = attention_decoder)
        
        # self.face_crop = FaceEdgeCrop_train()
        
    def forward(self, image, message, mask): # 
        encoded_image_m, encoded_image_n = self.encoder(image, message, mask)
        noised_image_C = self.noise([encoded_image_m, image, None])
        
        noised_image_C_mask = noised_image_C.clone()
        # noised_image_C_mask = noised_image_C_mask * (1 - mask)
        
        if mask is not None:
            noised_image_C_mask = noised_image_C_mask * mask # 去掉人脸
            
        # noised_image_C_mask = self.face_crop(noised_image_C_mask)
        decoded_message_C = self.decoder_C(noised_image_C_mask)

        # image_mask = image.clone()
        # image_mask = image_mask * mask
        # decoded_message_none = self.decoder_C(image_mask)
        # return encoded_image, noised_image_C, decoded_message_C, decoded_message_none

        return encoded_image_m, encoded_image_n, decoded_message_C, noised_image_C