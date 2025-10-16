from .DW_EncoderDecoder import *
from .Patch_Discriminator import Patch_Discriminator, compute_gradient_penalty
import torch
import kornia.losses
import lpips
import random
from torch.cuda.amp import autocast, GradScaler

class CLoss(nn.Module):
    def __init__(self):
        super(CLoss, self).__init__()
        self.mse = nn.MSELoss(reduction='sum')
    
    
    def l2_norm(self, input, axis=1):
        norm = torch.norm(input, 2, axis, True)
        output = torch.div(input, norm+1e-10)
        return output
    
    
    def forward(self, input_ref, input):
        # 0 - real; 1 - fake.
        input_ref = self.l2_norm(input_ref)
        input = self.l2_norm(input)
        
        sim_ref = input_ref @ input_ref.t() # b/2 x b/2
        sim_ref = torch.triu(sim_ref, diagonal=1)

        sim = input @ input.t()
        sim = torch.triu(sim, diagonal=1)
        
        num = sim.shape[0] * (sim.shape[0] - 1) / 2
        
        loss = self.mse(sim, sim_ref) / num 
            
        return loss
    

class ConstrastLoss(nn.Module):
    def __init__(self):
        super(ConstrastLoss, self).__init__()
        self.mse = nn.MSELoss(reduction='sum')
    
    
    def l2_norm(self, input, axis=1):
        norm = torch.norm(input, 2, axis, True)
        output = torch.div(input, norm+1e-10)
        return output
    
    
    def forward(self, input, label, center):
        # 0 - real; 1 - fake.
        center = self.l2_norm(center)
        input = self.l2_norm(input)
        
        label = label.unsqueeze(-1)
        
        sim_ref = center @ center.t() # 8 x 8
        sim_ref = torch.triu(sim_ref, diagonal=1)

        num = sim_ref.shape[0] * (sim_ref.shape[0] - 1) / 2
        loss1 = (num + torch.sum(sim_ref)) / num
        
        sim = input @ center.t()
        mask = torch.zeros_like(sim)
        mask = mask.scatter_(1, label, 1)

        sim = sim * mask
        loss2 = 1 - (torch.sum(sim) / sim.shape[0])
        
        loss = loss1 + loss2
        
        return loss
    
    
class ConstrastLossV2(nn.Module):
    def __init__(self):
        super(ConstrastLoss, self).__init__()
        self.mse = nn.MSELoss(reduction='sum')
    
    
    def l2_norm(self, input, axis=1):
        norm = torch.norm(input, 2, axis, True)
        output = torch.div(input, norm+1e-10)
        
        return output
    
    
    def forward(self, input, label, center):
        # 0 - real; 1 - fake.
        center = self.l2_norm(center)
        input = self.l2_norm(input)
        
        label = label.unsqueeze(-1)
        
        sim_ref = center @ center.t() # 8 x 8
        sim_ref = torch.triu(sim_ref, diagonal=1)

        num = sim_ref.shape[0] * (sim_ref.shape[0] - 1) / 2
        loss1 = (num + torch.sum(sim_ref)) / num
        
        sim = input @ center.t()
        mask = torch.zeros_like(sim)
        mask = mask.scatter_(1, label, 1)

        sim = sim * mask
        loss2 = 1 - (torch.sum(sim) / sim.shape[0])
        
        loss = loss1 + loss2
        
        return loss


class TV_Loss(nn.Module):
    def __init__(self):
        super(TV_Loss, self).__init__()
    
    def _tensor_size(self, t):
        return t.size()[1] * t.size()[2] * t.size()[3]
    
    def forward(self, est_noise):
        h_x = est_noise.size()[2]
        w_x = est_noise.size()[3]
        count_h = self._tensor_size(est_noise[:, :, 1:, :])
        count_w = self._tensor_size(est_noise[:, :, : ,1:])
        h_tv = torch.pow((est_noise[:, :, 1:, :] - est_noise[:, :, :h_x - 1, :]), 2).sum()
        w_tv = torch.pow((est_noise[:, :, :, 1:] - est_noise[:, :, :, :w_x - 1]), 2).sum()
        tvloss = h_tv / count_h + w_tv / count_w
        
        return tvloss
    
        
class Network:

    def __init__(self, message_length, noise_layers_R, noise_layers_F, device, batch_size, lr, beta1, attention_encoder, attention_decoder, weight, scale):
        # device
        self.device = device

        # loss function
        self.criterion_MSE = nn.MSELoss().to(device)
        self.criterion_LPIPS = lpips.LPIPS().to(device)

        # weight of encoder-decoder loss 不同损失的权重
        self.encoder_weight = weight[0] # encoder
        self.decoder_weight_C = weight[1] # decoder_C
        self.decoder_weight_Lpips = weight[2] # decoder_D
        self.decoder_weight_F = weight[3] # decoder_F
        self.discriminator_weight = weight[4] # discriminator

        # network
        self.encoder_decoder = RW_EncoderDecoder(message_length, noise_layers_R, noise_layers_F, attention_encoder, attention_decoder, adain = False, scale = scale).to(device)
        self.discriminator = Patch_Discriminator().to(device)

        self.encoder_decoder = torch.nn.DataParallel(self.encoder_decoder)
        self.discriminator = torch.nn.DataParallel(self.discriminator)

        # mark "cover" as 1, "encoded" as -1
        self.label_cover = 1.0
        self.label_encoded = - 1.0

        for p in self.encoder_decoder.module.noise.parameters():
            p.requires_grad = False
        
        # optimizer
        self.opt_encoder_decoder = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.encoder_decoder.parameters()), lr=lr, betas=(beta1, 0.999))
        self.opt_discriminator = torch.optim.Adam(self.discriminator.parameters(), lr=lr, betas=(beta1, 0.999))

        self.epoch = None
        self.bce = nn.BCEWithLogitsLoss().to(device)
        
    def train(self, images: torch.Tensor, messages: torch.Tensor, masks: torch.Tensor, INN = None, epoch=None):
        self.encoder_decoder.train()
        self.discriminator.train()
        
        with torch.enable_grad():
            # use device to compute 
            images, messages = images.to(self.device), messages.to(self.device) # mask放到noise_layer里面执行变换
            
            if masks is not None:
                masks = masks.to(self.device)
                
            encoded_images, noised_images, decoded_messages_C = self.encoder_decoder(images, messages, masks, INN) 
            
            '''
            train discriminator
            '''
            for p in self.discriminator.parameters():
                p.requires_grad = True

            self.opt_discriminator.zero_grad()

            # with autocast():
            # RAW : target label for image should be "cover"(1)
            d_label_cover = self.discriminator(images)
            #d_cover_loss = self.criterion_MSE(d_label_cover, torch.ones_like(d_label_cover))
            #d_cover_loss.backward()

            # GAN : target label for encoded image should be "encoded"(0)
            d_label_encoded = self.discriminator(encoded_images.detach())
            # d_encoded_loss = self.criterion_MSE(d_label_encoded, torch.zeros_like(d_label_encoded))
            # d_encoded_loss.backward()

            d_loss = self.criterion_MSE(d_label_cover - torch.mean(d_label_encoded), self.label_cover * torch.ones_like(d_label_cover)) + \
                    self.criterion_MSE(d_label_encoded - torch.mean(d_label_cover), self.label_encoded * torch.ones_like(d_label_encoded))
            
            d_loss.backward()
            self.opt_discriminator.step()
   
            # scaler.scale(d_loss).backward()
            # scaler.step(self.opt_discriminator)
            # scaler.update()
   
            '''
            train encoder and decoder
            '''
            # Make it a tiny bit faster
            for p in self.discriminator.parameters():
                p.requires_grad = False

            self.opt_encoder_decoder.zero_grad()
   
            # with autocast():
            # GAN : target label for encoded image should be "cover"(0)
            g_label_cover = self.discriminator(images)
            g_label_encoded = self.discriminator(encoded_images)
            g_loss_on_discriminator = self.criterion_MSE(g_label_cover - torch.mean(g_label_encoded), self.label_encoded * torch.ones_like(g_label_cover)) +\
                                    self.criterion_MSE(g_label_encoded - torch.mean(g_label_cover), self.label_cover * torch.ones_like(g_label_encoded))

            # RAW : the encoded image should be similar to cover image
            g_loss_on_encoder_MSE = self.criterion_MSE(encoded_images, images)
            g_loss_on_encoder_LPIPS = torch.mean(self.criterion_LPIPS(encoded_images, images))

            # RESULT : the decoded message should be similar to the raw message / Dual
            mx_m, mi_m = torch.max(torch.abs(messages)), torch.min(torch.abs(messages))
            # g_loss_on_decoder_C = self.criterion_MSE((decoded_messages_C-mi_m)/(mx_m-mi_m+1e-12), (messages-mi_m)/(mx_m-mi_m+1e-12))
            
            if type(decoded_messages_C) == tuple:
                for sub in decoded_messages_C:
                    g_loss_on_decoder_C += self.criterion_MSE(sub, messages)
                decoded_messages_C = torch.mean(torch.stack(decoded_messages_C), dim=0)
            else:
                g_loss_on_decoder_C = self.criterion_MSE(decoded_messages_C, messages)
                
            # full loss
            # if epoch is not None:
            #     encoder_weight = 20 * self.encoder_weight
            #     discriminator_weight = 5 * self.discriminator_weight
            #     decoder_weight_C = 0.25 * self.decoder_weight_C
            # else:
            #     encoder_weight = self.encoder_weight
            #     discriminator_weight = self.discriminator_weight
            #     decoder_weight_C = self.decoder_weight_C
            
            # encoder_weight = self.encoder_weight
            # discriminator_weight = self.discriminator_weight
            # decoder_weight_C = self.decoder_weight_C
             
            # g_loss = discriminator_weight * g_loss_on_discriminator + encoder_weight * g_loss_on_encoder_MSE +\
            #         decoder_weight_C * g_loss_on_decoder_C
                    
            # decoder_weight_C = 0.75 * self.decoder_weight_C if g_loss_on_decoder_C.item() < 0.001 else self.discriminator_weight # adaptive
            
            g_loss = self.discriminator_weight * g_loss_on_discriminator + \
                self.encoder_weight * g_loss_on_encoder_MSE + \
                self.decoder_weight_C * g_loss_on_decoder_C + \
                self.decoder_weight_Lpips * g_loss_on_encoder_LPIPS
                    
            g_loss.backward()
            self.opt_encoder_decoder.step()

            # scaler_2.scale(g_loss).backward()
            # scaler_2.step(self.opt_encoder_decoder)
            # scaler_2.update()
   
            # psnr
            psnr = - kornia.losses.psnr_loss(encoded_images.detach(), images, 2)

            # ssim
            ssim = 1 - 2 * kornia.losses.ssim_loss(encoded_images.detach(), images, window_size = 11, reduction = "mean")

        '''
        decoded message error rate / Dual
        '''
        error_rate_C = self.decoded_message_error_rate_batch(messages, decoded_messages_C)

        result = {
            "g_loss": g_loss,
            "error_rate_C": error_rate_C,
            "error_rate_R": 0,
            "error_rate_F": 0,
            "psnr": psnr,
            "ssim": ssim,
            "g_loss_on_discriminator": g_loss_on_discriminator,
            "g_loss_on_encoder_MSE": g_loss_on_encoder_MSE,
            "g_loss_on_encoder_LPIPS": g_loss_on_encoder_LPIPS,
            "g_loss_on_decoder_C": g_loss_on_decoder_C,
            "g_loss_on_decoder_R": 0,
            "g_loss_on_decoder_F": 0,
            "d_loss": d_loss
        }
        return result

    def train_new(self, images: torch.Tensor, messages: torch.Tensor, masks: torch.Tensor, INN = None, epoch=None):
        self.encoder_decoder.train()
        self.discriminator.train()
        
        with torch.enable_grad():
            # use device to compute 
            images, messages = images.to(self.device), messages.to(self.device) # mask放到noise_layer里面执行变换
            if masks is not None:
                masks = masks.to(self.device)
            encoded_images, noised_images, decoded_messages_C = self.encoder_decoder(images, messages, masks, INN) 
            
            '''
            train discriminator
            '''
            self.encoder_decoder.eval()
            self.discriminator.train()
            
            for p in self.encoder_decoder.parameters():
                p.requires_grad = False
            for p in self.discriminator.parameters():
                p.requires_grad = True

            self.opt_discriminator.zero_grad()

            d_label_cover = self.discriminator(images)
            d_label_encoded = self.discriminator(encoded_images.detach())

            d_loss = self.bce(d_label_cover,  torch.zeros_like(d_label_cover)) + \
                    self.bce(d_label_encoded, torch.ones_like(d_label_encoded))
            d_loss.backward()
            
            self.opt_discriminator.step()
   
            '''
            train encoder and decoder
            '''
            self.encoder_decoder.train()
            self.discriminator.eval()
            
            # Make it a tiny bit faster
            for p in self.discriminator.parameters():
                p.requires_grad = False
            for p in self.encoder_decoder.parameters():
                p.requires_grad = True

            self.opt_encoder_decoder.zero_grad()

            g_label_encoded = self.discriminator(encoded_images)
            g_loss_on_discriminator = self.bce(g_label_encoded, torch.zeros_like(g_label_encoded))
            g_loss_on_encoder_MSE = self.criterion_MSE(encoded_images, images)
            g_loss_on_encoder_LPIPS = torch.mean(self.criterion_LPIPS(encoded_images, images))

            g_loss_on_decoder_C = self.criterion_MSE(decoded_messages_C, messages)
            
            # if epoch < 50: 
            #     encoder_weight = self.encoder_weight
            #     discriminator_weight = self.discriminator_weight
            #     decoder_weight_C = self.decoder_weight_C
            # else:
            #     encoder_weight = 10*self.encoder_weight
            #     discriminator_weight = 50*self.discriminator_weight
            #     decoder_weight_C = 0.25*self.decoder_weight_C
                
            # g_loss = discriminator_weight * g_loss_on_discriminator + encoder_weight * g_loss_on_encoder_MSE +\
            #         decoder_weight_C * g_loss_on_decoder_C
            
            g_loss = self.discriminator_weight * g_loss_on_discriminator + self.encoder_weight * g_loss_on_encoder_MSE + \
                    self.decoder_weight_C * g_loss_on_decoder_C
            g_loss.backward()
            
            self.opt_encoder_decoder.step()

            # psnr
            psnr = - kornia.losses.psnr_loss(encoded_images.detach(), images, 2)

            # ssim
            ssim = 1 - 2 * kornia.losses.ssim_loss(encoded_images.detach(), images, window_size = 11, reduction = "mean")

        '''
        decoded message error rate / Dual
        '''
        error_rate_C = self.decoded_message_error_rate_batch(messages, decoded_messages_C)

        result = {
            "g_loss": g_loss,
            "error_rate_C": error_rate_C,
            "error_rate_R": 0,
            "error_rate_F": 0,
            "psnr": psnr,
            "ssim": ssim,
            "g_loss_on_discriminator": g_loss_on_discriminator,
            "g_loss_on_encoder_MSE": g_loss_on_encoder_MSE,
            "g_loss_on_encoder_LPIPS": g_loss_on_encoder_LPIPS,
            "g_loss_on_decoder_C": g_loss_on_decoder_C,
            "g_loss_on_decoder_R": 0,
            "g_loss_on_decoder_F": 0,
            "d_loss": d_loss
        }
        return result
    

    def validation(self, images: torch.Tensor, messages: torch.Tensor, masks: torch.Tensor, INN = None):
        self.encoder_decoder.eval()
        self.encoder_decoder.module.noise.train()
        self.discriminator.eval()

        with torch.no_grad():
            # use device to compute
            images, messages = images.to(self.device), messages.to(self.device)
            if masks is not None:
                masks = masks.to(self.device)
            encoded_images, noised_images, decoded_messages_C = self.encoder_decoder(images, messages, masks, INN)

            '''
            validate discriminator
            '''
            # RAW : target label for image should be "cover"(1)
            d_label_cover = self.discriminator(images)
            #d_cover_loss = self.criterion_MSE(d_label_cover, torch.ones_like(d_label_cover))

            # GAN : target label for encoded image should be "encoded"(0)
            d_label_encoded = self.discriminator(encoded_images.detach())
            #d_encoded_loss = self.criterion_MSE(d_label_encoded, torch.zeros_like(d_label_encoded))

            d_loss = self.criterion_MSE(d_label_cover - torch.mean(d_label_encoded), self.label_cover * torch.ones_like(d_label_cover)) +\
                     self.criterion_MSE(d_label_encoded - torch.mean(d_label_cover), self.label_encoded * torch.ones_like(d_label_encoded))

            '''
            validate encoder and decoder
            '''

            # GAN : target label for encoded image should be "cover"(0)
            g_label_cover = self.discriminator(images)
            g_label_encoded = self.discriminator(encoded_images)
            g_loss_on_discriminator = self.criterion_MSE(g_label_cover - torch.mean(g_label_encoded), self.label_encoded * torch.ones_like(g_label_cover)) +\
                                      self.criterion_MSE(g_label_encoded - torch.mean(g_label_cover), self.label_cover * torch.ones_like(g_label_encoded))

            # RAW : the encoded image should be similar to cover image
            g_loss_on_encoder_MSE = self.criterion_MSE(encoded_images, images)
            g_loss_on_encoder_LPIPS = torch.mean(self.criterion_LPIPS(encoded_images, images))

            # RESULT : the decoded message should be similar to the raw message /Dual
            # full loss
            # unstable g_loss_on_discriminator is not used during validation
            
            g_loss_on_decoder_C = 0
            if type(decoded_messages_C) == tuple:
                decoded_messages_C = torch.mean(torch.stack(decoded_messages_C), dim=0)
                g_loss_on_decoder_C += self.criterion_MSE(decoded_messages_C, messages) 
            else:
                g_loss_on_decoder_C = self.criterion_MSE(decoded_messages_C, messages)
                
            g_loss = 0 * g_loss_on_discriminator + self.encoder_weight * g_loss_on_encoder_LPIPS +\
                     self.decoder_weight_C * g_loss_on_decoder_C


            # psnr
            psnr = - kornia.losses.psnr_loss(encoded_images.detach(), images, 2)

            # ssim
            ssim = 1 - 2 * kornia.losses.ssim_loss(encoded_images.detach(), images, window_size=11, reduction="mean")

        '''
        decoded message error rate /Dual
        '''
        error_rate_C = self.decoded_message_error_rate_batch(messages, decoded_messages_C)

        result = {
            "g_loss": g_loss,
            "error_rate_C": error_rate_C,
            "error_rate_R": 0,
            "error_rate_F": 0,
            "psnr": psnr,
            "ssim": ssim,
            "g_loss_on_discriminator": g_loss_on_discriminator,
            "g_loss_on_encoder_MSE": g_loss_on_encoder_MSE,
            "g_loss_on_encoder_LPIPS": g_loss_on_encoder_LPIPS,
            "g_loss_on_decoder_C": g_loss_on_decoder_C,
            "g_loss_on_decoder_R": 0,
            "g_loss_on_decoder_F": 0,
            "d_loss": d_loss
        }

        return result, (images, encoded_images, noised_images)

    def decoded_message_error_rate(self, message, decoded_message):
        length = message.shape[0]

        message = message.gt(0) # 大于0为1，小于为0
        decoded_message = decoded_message.gt(0)
        error_rate = float(sum(message != decoded_message)) / length
        return error_rate

    def decoded_message_error_rate_actual(self, message, decoded_message):
        length = message.shape[0]
        error_rate = float(sum(abs(message - decoded_message))) / length
        
        return error_rate
    
    def decoded_message_error_rate_actual_v2(self, message, decoded_message):
        length = message.shape[0] * message.shape[1] * message.shape[2]
        
        error_rate = float(torch.sum(abs(message - decoded_message))) / length
        return error_rate

    def decoded_message_error_rate_batch(self, messages, decoded_messages):
        error_rate = 0.0
        batch_size = len(messages)
        for i in range(batch_size):
            error_rate += self.decoded_message_error_rate_actual(messages[i], decoded_messages[i]) # decoded_message_error_rate
        error_rate /= batch_size
        
        return error_rate

    def save_model(self, path_encoder_decoder: str, path_discriminator: str):
        torch.save(self.encoder_decoder.module.state_dict(), path_encoder_decoder)
        torch.save(self.discriminator.module.state_dict(), path_discriminator)

    def load_model(self, path_encoder_decoder: str, path_discriminator: str):
        self.load_model_ed(path_encoder_decoder)
        self.load_model_dis(path_discriminator)

    def load_model_ed(self, path_encoder_decoder: str):
        self.encoder_decoder.module.load_state_dict(torch.load(path_encoder_decoder), strict=False)

    def load_model_dis(self, path_discriminator: str):
        self.discriminator.module.load_state_dict(torch.load(path_discriminator))



class Network2:

    def __init__(self, message_length, noise_layers_R, noise_layers_F, device, batch_size, lr, beta1, attention_encoder, attention_decoder, weight, factor = 0.1, adain = False):
        # device
        self.device = device

        # setting loss functions
        self.criterion_MSE = nn.MSELoss().to(device)
        self.criterion_LPIPS = lpips.LPIPS().to(device)
        self.criterion_SIM = ConstrastLoss().to(device)
        self.bce = nn.BCEWithLogitsLoss().to(device)
        
        # weight of encoder-decoder loss 不同损失的权重
        self.encoder_weight = weight[0]
        self.decoder_weight_C = weight[1]
        self.decoder_weight_none = weight[2]
        self.message_weight_rec = weight[3]
        self.discriminator_weight = weight[4]

        # network
        self.factor = factor

        self.encoder_decoder = RW_EncoderDecoder(message_length, noise_layers_R, noise_layers_F, attention_encoder, attention_decoder, adain).to(device)
        self.discriminator = Patch_Discriminator().to(device)
        
        self.encoder_decoder = torch.nn.DataParallel(self.encoder_decoder)
        self.discriminator = torch.nn.DataParallel(self.discriminator)

        # mark "cover" as 1, "encoded" as -1
        self.label_cover = 1.0
        self.label_encoded = - 1.0

        for p in self.encoder_decoder.module.noise.parameters():
            p.requires_grad = False
               
        # optimizer
        self.opt_encoder_decoder = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.encoder_decoder.parameters()), lr=lr, betas=(beta1, 0.999))
        self.opt_discriminator = torch.optim.Adam(self.discriminator.parameters(), lr=lr, betas=(beta1, 0.999)) # 修改了OPTIM
        
        self.message_legnth = message_length
    
    def binary_message(self, message, range):
        # message_mean = torch.mean(message, dim = -1)
        bin_message = message.clone()
        bin_message[message >= 0] = 1
        bin_message[message < 0] = -1
        bin_message = bin_message * range
        
        return bin_message
    
    def norm(self, message):
        message_norm = (message - torch.min(message, dim = -1, keepdim = True).values) / (torch.max(message, dim = -1, keepdim = True).values - torch.min(message, dim = -1, keepdim = True).values)
        message_norm = message_norm * 2 - 1
        
        return message_norm

        
    def train(self, images: torch.Tensor, messages: torch.Tensor, masks: torch.Tensor, messages_2: torch.Tensor, idx_list = None, penalty = False):
        self.encoder_decoder.train()
        self.discriminator.train()
        
        with torch.enable_grad():
            ### use device to compute 
            images = images.to(self.device)
            if masks is not None:
                masks = masks.to(self.device)
            messages_tex = messages.to(self.device)
            messages_ide = messages_2.to(self.device)
                
            ### squeeze message
            if idx_list is None:
                messages_tex_sqz = messages_tex[:, :self.message_legnth]
                messages_ide_sqz = messages_ide[:, :self.message_legnth]
                
            elif len(idx_list) == 2:
                messages_tex_sqz = messages_tex
                pca_ide = idx_list[1]
                messages_ide_sqz = pca_ide.transform(messages_ide)
                messages_ide_sqz = self.norm(messages_ide_sqz)
                
            elif len(idx_list) == 3:
                pca_tex = idx_list[0]
                pca_ide = idx_list[1]
                messages_tex_sqz = pca_tex.transform(messages_tex)
                messages_ide_sqz = pca_ide.transform(messages_ide)
                messages_tex_sqz = self.norm(messages_tex_sqz)
                messages_ide_sqz = self.norm(messages_ide_sqz)
            
            # random mode
            # if random.random() > 0.5:
            #     B, N = messages_ide_sqz.shape[0], messages_ide_sqz.shape[1]
            #     random_indices = torch.stack([torch.randperm(N) for _ in range(B)]).to(messages_ide_sqz.device)
            #     messages_tex_sqz = torch.gather(messages_tex_sqz, 1, random_indices)
            #     messages_ide_sqz = torch.gather(messages_ide_sqz, 1, random_indices)
            ###
            
            messages_tex_bin = self.binary_message(messages_tex_sqz.detach(), self.factor)
            messages_ide_bin = self.binary_message(messages_ide_sqz.detach(), self.factor)
            messages_bin = torch.cat((messages_tex_bin, messages_ide_bin), dim = -1)
            
            encoded_images, noised_images, decoded_messages_C = self.encoder_decoder(images, messages_bin, masks) # decoded_messages_none
            
            '''
            train discriminator
            '''
            self.encoder_decoder.eval()
            self.discriminator.train()
            
            for p in self.encoder_decoder.parameters():
                p.requires_grad = False
            for p in self.discriminator.parameters():
                p.requires_grad = True
            
            # for _ in range(3):
            #     self.opt_discriminator.zero_grad()
            #     encoded_images, noised_images, decoded_messages_C = self.encoder_decoder(images, messages_bin, masks)
            #     # with autocast():
            #     # RAW : target label for image should be "cover"(1)
            #     # GAN : target label for encoded image should be "encoded"(0)
            #     d_label_cover = self.discriminator(images)
            #     d_label_encoded = self.discriminator(encoded_images.detach())
            #     gradient_penalty = compute_gradient_penalty(self.discriminator, images, encoded_images.detach())
            #     d_loss = self.bce(d_label_cover,  torch.zeros_like(d_label_cover)) + \
            #              self.bce(d_label_encoded, torch.ones_like(d_label_encoded)) + \
            #              10 * gradient_penalty
            #     d_loss.backward()
            #     self.opt_discriminator.step()
            
            self.opt_discriminator.zero_grad()
            
            # with autocast():
            # RAW : target label for image should be "cover"(1)
            # GAN : target label for encoded image should be "encoded"(0)
            d_label_cover = self.discriminator(images)
            d_label_encoded = self.discriminator(encoded_images.detach())
            
            if penalty:
                d_gradient_penalty = compute_gradient_penalty(self.discriminator, images, encoded_images.detach())
                d_loss = self.bce(d_label_cover,  torch.zeros_like(d_label_cover)) + \
                            self.bce(d_label_encoded, torch.ones_like(d_label_encoded))+ \
                            self.message_weight_rec * d_gradient_penalty
            else:
                d_loss = self.bce(d_label_cover,  torch.zeros_like(d_label_cover)) + \
                            self.bce(d_label_encoded, torch.ones_like(d_label_encoded))
                            
            d_loss.backward()
            self.opt_discriminator.step()
            
            '''
            train encoder and decoder
            '''
            self.encoder_decoder.train()
            self.discriminator.eval()
            
            # Make it a tiny bit faster
            for p in self.discriminator.parameters():
                p.requires_grad = False
            for p in self.encoder_decoder.parameters():
                p.requires_grad = True
            
            self.opt_encoder_decoder.zero_grad()
            
            # GAN : target label for encoded image should be "cover"(0)
            g_label_encoded = self.discriminator(encoded_images)
            g_loss_on_discriminator = self.bce(g_label_encoded, torch.zeros_like(g_label_encoded))
                                    
            # RAW : the encoded image should be similar to cover image
            g_loss_on_encoder_MSE = self.criterion_MSE(encoded_images, images)
            g_loss_on_encoder_LPIPS = torch.mean(self.criterion_LPIPS(encoded_images, images))

            # RESULT : the decoded message should be similar to the raw message / Dual
            g_loss_on_decoder_C = self.criterion_MSE(decoded_messages_C, messages_bin.detach()) ## 解码后的结果与sqz相同
            
            # Total Variance for final features
            # g_loss_on_decoder_none = self.criterion_MSE(decoded_messages_none, torch.zeros_like(messages_bin).detach())
            g_loss_on_decoder_none = 0
                
            # full loss
            # g_loss = self.discriminator_weight * g_loss_on_discriminator + self.encoder_weight * g_loss_on_encoder_MSE + \
            #         self.decoder_weight_C * g_loss_on_decoder_C + self.decoder_weight_none * g_loss_on_decoder_none
            g_loss = self.discriminator_weight * g_loss_on_discriminator + self.encoder_weight * g_loss_on_encoder_MSE + \
                    self.decoder_weight_C * g_loss_on_decoder_C
            
            g_loss.backward()
            self.opt_encoder_decoder.step()
   
            # psnr
            psnr = - kornia.losses.psnr_loss(encoded_images.detach(), images, 2)

            # ssim
            ssim = 1 - 2 * kornia.losses.ssim_loss(encoded_images.detach(), images, window_size = 11, reduction = "mean")

        '''
        decoded message error rate / Dual
        '''
        
        error_rate_C = self.decoded_message_error_rate_batch(messages_bin, decoded_messages_C)

        result = {
            "g_loss": g_loss,
            "error_rate_C": error_rate_C,
            "error_rate_R": 0,
            "error_rate_F": 0,
            "psnr": psnr,
            "ssim": ssim,
            "g_loss_on_discriminator": g_loss_on_discriminator,
            "g_loss_on_encoder_MSE": g_loss_on_encoder_MSE,
            "g_loss_on_encoder_LPIPS": g_loss_on_encoder_LPIPS,
            "g_loss_on_decoder_C": g_loss_on_decoder_C,
            "g_loss_on_decoder_none": g_loss_on_decoder_none,
            "g_loss_on_decoder_F": 0,
            "d_loss": d_loss
        }
        
        return result


    def validation(self, images: torch.Tensor, messages: torch.Tensor, masks: torch.Tensor, messages_2: torch.Tensor, idx_list = None):
        self.encoder_decoder.eval()
        self.encoder_decoder.module.noise.train()
        self.discriminator.eval()
        
        with torch.no_grad():
            # use device to compute
            images = images.to(self.device)
            if masks is not None:
                masks = masks.to(self.device)
            messages_tex = messages.to(self.device)
            messages_ide = messages_2.to(self.device)
            
            if idx_list is None:
                messages_tex_sqz = messages_tex[:, :self.message_legnth]
                messages_ide_sqz = messages_ide[:, :self.message_legnth]
                
            elif len(idx_list) == 2:
                messages_tex_sqz = messages_tex
                pca_ide = idx_list[1]
                messages_ide_sqz = pca_ide.transform(messages_ide)
                messages_ide_sqz = self.norm(messages_ide_sqz)
                
            elif len(idx_list) == 3:
                pca_tex = idx_list[0]
                pca_ide = idx_list[1]
                messages_tex_sqz = pca_tex.transform(messages_tex)
                messages_ide_sqz = pca_ide.transform(messages_ide)
                messages_tex_sqz = self.norm(messages_tex_sqz)
                messages_ide_sqz = self.norm(messages_ide_sqz)
                
            messages_tex_bin = self.binary_message(messages_tex_sqz.detach(), self.factor)
            messages_ide_bin = self.binary_message(messages_ide_sqz.detach(), self.factor)
            messages_bin = torch.cat((messages_tex_bin, messages_ide_bin), dim = -1)
            
            encoded_images, noised_images, decoded_messages_C = self.encoder_decoder(images, messages_bin, masks)
            
            '''
            validate discriminator
            '''
            # RAW : target label for image should be "cover"(1)
            # GAN : target label for encoded image should be "encoded"(0)
            d_label_cover = self.discriminator(images)            
            d_label_encoded = self.discriminator(encoded_images.detach())

            d_loss = self.bce(d_label_cover,  torch.zeros_like(d_label_cover)) + \
                    self.bce(d_label_encoded, torch.ones_like(d_label_encoded))
            '''
            validate encoder and decoder
            '''
            # GAN : target label for encoded image should be "cover"(0)
            g_label_cover = self.discriminator(images)
            g_label_encoded = self.discriminator(encoded_images)
            g_loss_on_discriminator = self.criterion_MSE(g_label_cover - torch.mean(g_label_encoded), self.label_encoded * torch.ones_like(g_label_cover)) + \
                                      self.criterion_MSE(g_label_encoded - torch.mean(g_label_cover), self.label_cover * torch.ones_like(g_label_encoded))

            # RAW : the encoded image should be similar to cover image
            g_loss_on_encoder_MSE = self.criterion_MSE(encoded_images, images)
            g_loss_on_encoder_LPIPS = torch.mean(self.criterion_LPIPS(encoded_images, images))

            # RESULT : the decoded message should be similar to the raw message /Dual
            g_loss_on_decoder_C = self.criterion_MSE(decoded_messages_C, messages_bin.detach())
            
            # Total Variance for final features
            # g_loss_on_decoder_none = self.criterion_MSE(decoded_messages_none, torch.zeros_like(messages_bin).detach())
            g_loss_on_decoder_none = 0 
            
            # full loss
            # unstable g_loss_on_discriminator is not used during validation
            # g_loss = 0 * g_loss_on_discriminator + self.encoder_weight * g_loss_on_encoder_LPIPS + \
            #          self.decoder_weight_C * g_loss_on_decoder_C + self.decoder_weight_none * g_loss_on_decoder_none
            g_loss = 0 * g_loss_on_discriminator + self.encoder_weight * g_loss_on_encoder_LPIPS + \
                     self.decoder_weight_C * g_loss_on_decoder_C
                     
            # psnr
            psnr = - kornia.losses.psnr_loss(encoded_images.detach(), images, 2)

            # ssim
            ssim = 1 - 2 * kornia.losses.ssim_loss(encoded_images.detach(), images, window_size=11, reduction="mean")

        '''
        decoded message error rate /Dual
        '''
        error_rate_C = self.decoded_message_error_rate_batch(messages_bin, decoded_messages_C)

        result = {
            "g_loss": g_loss,
            "error_rate_C": error_rate_C, # 
            "error_rate_R": 0,
            "error_rate_F": 0,
            "psnr": psnr,
            "ssim": ssim,
            "g_loss_on_discriminator": g_loss_on_discriminator,
            "g_loss_on_encoder_MSE": g_loss_on_encoder_MSE,
            "g_loss_on_encoder_LPIPS": g_loss_on_encoder_LPIPS,
            "g_loss_on_decoder_C": g_loss_on_decoder_C,
            "g_loss_on_decoder_none": g_loss_on_decoder_none,
            "g_loss_on_decoder_F": 0,
            "d_loss": d_loss
        }

        return result, (images, encoded_images, noised_images)

    def decoded_message_error_rate(self, message, decoded_message):
        length = message.shape[1] * message.shape[0]
        message = message.gt(0) # 大于0为1，小于为0
        decoded_message = decoded_message.gt(0)
        error_rate = float(torch.sum(message != decoded_message)) / length
        return error_rate
    
    def decoded_message_error_rate_list(self, message, decoded_message):
        message, decoded_message = torch.Tensor(message)[None], torch.Tensor(decoded_message)[None]
        length = message.shape[1] * message.shape[0]
        message = message.gt(0) # 大于0为1，小于为0
        decoded_message = decoded_message.gt(0)
        error_rate = float(torch.sum(message != decoded_message)) / length
        return error_rate
    
    def decoded_message_error_rate_actual(self, message, decoded_message):
        length = message.shape[0]
        
        error_rate = float(sum(abs(message - decoded_message))) / length
        return error_rate
    
    def decoded_message_error_rate_actual_v2(self, message, decoded_message):
        length = message.shape[0] * message.shape[1] * message.shape[2]
        
        error_rate = float(torch.sum(abs(message - decoded_message))) / length
        return error_rate

    def decoded_message_error_rate_batch(self, messages, decoded_messages): # MAE
        error_rate = 0.0
        batch_size = len(messages)
        for i in range(batch_size):
            error_rate += self.decoded_message_error_rate_actual(messages[i], decoded_messages[i]) # decoded_message_error_rate
        error_rate /= batch_size
        return error_rate

    def save_model(self, path_encoder_decoder: str, path_discriminator: str):
        torch.save(self.encoder_decoder.module.state_dict(), path_encoder_decoder)
        torch.save(self.discriminator.module.state_dict(), path_discriminator)

    def load_model(self, path_encoder_decoder: str, path_discriminator: str):
        self.load_model_ed(path_encoder_decoder)
        self.load_model_dis(path_discriminator)

    def load_model_ed(self, path_encoder_decoder: str):
        self.encoder_decoder.module.load_state_dict(torch.load(path_encoder_decoder), strict=False)

    def load_model_dis(self, path_discriminator: str):
        self.discriminator.module.load_state_dict(torch.load(path_discriminator))


class Network3:

    def __init__(self, message_length, noise_layers_R, noise_layers_F, device, batch_size, lr, beta1, attention_encoder, attention_decoder, weight, factor = 0.1):
        # device
        self.device = device

        # setting loss functions
        self.criterion_MSE = nn.MSELoss().to(device)
        self.criterion_LPIPS = lpips.LPIPS().to(device)
        self.criterion_SIM = ConstrastLoss().to(device)
        self.bce = nn.BCEWithLogitsLoss().to(device)
        
        # weight of encoder-decoder loss 不同损失的权重
        self.encoder_weight = weight[0]
        self.decoder_weight_C = weight[1]
        self.decoder_weight_D= weight[2]
        # self.message_weight_rec = weight[3]
        self.discriminator_weight = weight[4]

        # network
        self.factor = factor

        # self.SE_module = Squeeze_Excitation(message_length).to(device)
        self.encoder_decoder = RW_EncoderDecoder_2(message_length, noise_layers_R, noise_layers_F, attention_encoder, attention_decoder).to(device)
        self.discriminator = Patch_Discriminator().to(device)
        # self.centers = nn.Parameter(torch.randn(8, message_length)).to(self.device)
        
        # self.SE_module = torch.nn.DataParallel(self.SE_module)
        self.encoder_decoder = torch.nn.DataParallel(self.encoder_decoder)
        self.discriminator = torch.nn.DataParallel(self.discriminator)

        # mark "cover" as 1, "encoded" as -1
        self.label_cover = 1.0
        self.label_encoded = - 1.0

        for p in self.encoder_decoder.module.noise.parameters():
            p.requires_grad = False
               
        # optimizer
        # self.opt_squeeze = torch.optim.Adam(self.SE_module.parameters(), lr=lr, betas=(beta1, 0.999))
        self.opt_encoder_decoder = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.encoder_decoder.parameters()), lr=lr, betas=(beta1, 0.999))
        self.opt_discriminator = torch.optim.Adam(self.discriminator.parameters(), lr=lr, betas=(beta1, 0.999)) # 修改了OPTIM
        
        self.message_legnth = message_length
    
    def binary_message(self, message, range, mean = False):
        bin_message = message.clone()
        
        if not mean:
            bin_message[message >= 0] = 1
            bin_message[message < 0] = -1
        else:
            message_mean = torch.mean(message, dim = -1)
            bin_message[message >= message_mean] = 1
            bin_message[message < message_mean] = -1
        
        bin_message = bin_message * range
        
        return bin_message
        
    def train(self, images: torch.Tensor, messages_exp: torch.Tensor, masks: torch.Tensor, messages_ide: torch.Tensor, idx_list = None):
        self.encoder_decoder.train()
        self.discriminator.train()
        # self.SE_module.train()
        
        with torch.enable_grad():
            ### use device to compute 
            images, masks = images.to(self.device), masks.to(self.device)
            messages_exp = messages_exp.to(self.device)
            messages_ide = messages_ide.to(self.device)
                
            ### squeeze message
            # messages_sqz = self.SE_module(messages)
            if idx_list is None:
                messages_exp_sqz = messages_exp[:, :self.message_legnth]
                messages_ide_sqz = messages_ide[:, :self.message_legnth]
            else:
                exp_idx = idx_list[0]
                ide_idx = idx_list[1]
                messages_exp_sqz = messages_exp[:, exp_idx]
                messages_ide_sqz = messages_ide[:, ide_idx]
                
            messages_exp_bin = self.binary_message(messages_exp_sqz.detach(), self.factor)
            messages_ide_bin = self.binary_message(messages_ide_sqz.detach(), self.factor)
            
            encoded_images, noised_images, decoded_messages_C, decoded_messages_D = self.encoder_decoder(images, messages_exp_bin, messages_ide_bin, masks)
            
            '''
            train discriminator
            '''
            self.encoder_decoder.eval()
            # self.SE_module.eval()
            self.discriminator.train()
            
            for p in self.discriminator.parameters():
                p.requires_grad = True
            for p in self.encoder_decoder.parameters():
                p.requires_grad = False
            
                
            self.opt_discriminator.zero_grad()
            
            # with autocast():
            # RAW : target label for image should be "cover"(1)
            # GAN : target label for encoded image should be "encoded"(0)
            d_label_cover = self.discriminator(images)
            d_label_encoded = self.discriminator(encoded_images.detach())
            d_loss = self.bce(d_label_cover,  torch.zeros_like(d_label_cover)) + \
                        self.bce(d_label_encoded, torch.ones_like(d_label_encoded))
            d_loss.backward()
            self.opt_discriminator.step()
            
            '''
            train encoder and decoder
            '''
            self.encoder_decoder.train()
            # self.SE_module.train()
            self.discriminator.eval()
            
            # Make it a tiny bit faster
            for p in self.discriminator.parameters():
                p.requires_grad = False
            for p in self.encoder_decoder.parameters():
                p.requires_grad = True
            
            self.opt_encoder_decoder.zero_grad()
            # self.opt_squeeze.zero_grad()
            
            # GAN : target label for encoded image should be "cover"(0)
            g_label_encoded = self.discriminator(encoded_images)
            g_loss_on_discriminator = self.bce(g_label_encoded, torch.zeros_like(g_label_encoded))
                                    
            # RAW : the encoded image should be similar to cover image
            g_loss_on_encoder_MSE = self.criterion_MSE(encoded_images, images)
            g_loss_on_encoder_LPIPS = torch.mean(self.criterion_LPIPS(encoded_images, images))

            # RESULT : the decoded message should be similar to the raw message / Dual
            g_loss_on_decoder_C = self.criterion_MSE(decoded_messages_C, messages_exp_bin.detach())
            g_loss_on_decoder_D = self.criterion_MSE(decoded_messages_D, messages_ide_bin.detach())
            
            # full loss
            g_loss = self.discriminator_weight * g_loss_on_discriminator + self.encoder_weight * g_loss_on_encoder_MSE + \
                    self.decoder_weight_C * g_loss_on_decoder_C + self.decoder_weight_D * g_loss_on_decoder_D

            g_loss.backward()
            self.opt_encoder_decoder.step()
            # self.opt_squeeze.step()
   
            # psnr
            psnr = - kornia.losses.psnr_loss(encoded_images.detach(), images, 2)

            # ssim
            ssim = 1 - 2 * kornia.losses.ssim_loss(encoded_images.detach(), images, window_size = 11, reduction = "mean")

        '''
        decoded message error rate / Dual
        '''
        
        error_rate_C = self.decoded_message_error_rate_batch(messages_exp_bin, decoded_messages_C)
        error_rate_D = self.decoded_message_error_rate_batch(messages_ide_bin, decoded_messages_D)
        
        result = {
            "g_loss": g_loss,
            "error_rate_exp": error_rate_C,
            "error_rate_ide": error_rate_D,
            "error_rate_F": 0,
            "psnr": psnr,
            "ssim": ssim,
            "g_loss_on_discriminator": g_loss_on_discriminator,
            "g_loss_on_encoder_MSE": g_loss_on_encoder_MSE,
            "g_loss_on_encoder_LPIPS": g_loss_on_encoder_LPIPS,
            "g_loss_on_decoder_exp": g_loss_on_decoder_C,
            "g_loss_on_message_ide": g_loss_on_decoder_D,
            "g_loss_on_decoder_F": 0,
            "d_loss": d_loss
        }
        
        return result


    def validation(self, images: torch.Tensor, messages_exp: torch.Tensor, masks: torch.Tensor, messages_ide: torch.Tensor, idx_list = None):
        self.encoder_decoder.eval()
        self.encoder_decoder.module.noise.train()
        self.discriminator.eval()
        # self.SE_module.eval()
        
        with torch.no_grad():
            # use device to compute
            images, masks = images.to(self.device), masks.to(self.device)
            messages_exp = messages_exp.to(self.device)
            messages_ide = messages_ide.to(self.device)
                
            # messages_sqz = self.SE_module(messages)
            if idx_list is None:
                messages_exp_sqz = messages_exp[:, :self.message_legnth]
                messages_ide_sqz = messages_ide[:, :self.message_legnth]
            else:
                exp_idx = idx_list[0]
                ide_idx = idx_list[1]
                messages_exp_sqz = messages_exp[:, exp_idx]
                messages_ide_sqz = messages_ide[:, ide_idx]
                
            messages_exp_bin = self.binary_message(messages_exp_sqz.detach(), self.factor)
            messages_ide_bin = self.binary_message(messages_ide_sqz.detach(), self.factor)

            # messages_gen = self.expand(messages_sqz)
            encoded_images, noised_images, decoded_messages_C, decoded_messages_D = self.encoder_decoder(images, messages_exp_bin, messages_ide_bin, masks)

            '''
            validate discriminator
            '''
            # RAW : target label for image should be "cover"(1)
            # GAN : target label for encoded image should be "encoded"(0)
            d_label_cover = self.discriminator(images)            
            d_label_encoded = self.discriminator(encoded_images.detach())

            d_loss = self.bce(d_label_cover,  torch.zeros_like(d_label_cover)) + \
                    self.bce(d_label_encoded, torch.ones_like(d_label_encoded))
            '''
            validate encoder and decoder
            '''
            # GAN : target label for encoded image should be "cover"(0)
            g_label_cover = self.discriminator(images)
            g_label_encoded = self.discriminator(encoded_images)
            g_loss_on_discriminator = self.criterion_MSE(g_label_cover - torch.mean(g_label_encoded), self.label_encoded * torch.ones_like(g_label_cover)) +\
                                      self.criterion_MSE(g_label_encoded - torch.mean(g_label_cover), self.label_cover * torch.ones_like(g_label_encoded))

            # RAW : the encoded image should be similar to cover image
            g_loss_on_encoder_MSE = self.criterion_MSE(encoded_images, images)
            g_loss_on_encoder_LPIPS = torch.mean(self.criterion_LPIPS(encoded_images, images))

            # RESULT : the decoded message should be similar to the raw message /Dual
            g_loss_on_decoder_C = self.criterion_MSE(decoded_messages_C, messages_exp_bin.detach())
            g_loss_on_decoder_D = self.criterion_MSE(decoded_messages_D, messages_ide_bin.detach())
            
            # g_loss_on_message = self.criterion_SIM(messages_sqz, labels, self.centers)
            
            # full loss
            # unstable g_loss_on_discriminator is not used during validation

            g_loss = 0 * g_loss_on_discriminator + self.encoder_weight * g_loss_on_encoder_LPIPS + \
                     self.decoder_weight_C * g_loss_on_decoder_C + self.decoder_weight_C * g_loss_on_decoder_D

            # psnr
            psnr = - kornia.losses.psnr_loss(encoded_images.detach(), images, 2)

            # ssim
            ssim = 1 - 2 * kornia.losses.ssim_loss(encoded_images.detach(), images, window_size=11, reduction="mean")

        '''
        decoded message error rate /Dual
        '''
        error_rate_C = self.decoded_message_error_rate_batch(messages_exp_bin, decoded_messages_C)
        error_rate_D = self.decoded_message_error_rate_batch(messages_ide_bin, decoded_messages_D)
        

        result = {
            "g_loss": g_loss,
            "error_rate_exp": error_rate_C, # 
            "error_rate_ide": error_rate_D,
            "error_rate_F": 0,
            "psnr": psnr,
            "ssim": ssim,
            "g_loss_on_discriminator": g_loss_on_discriminator,
            "g_loss_on_encoder_MSE": g_loss_on_encoder_MSE,
            "g_loss_on_encoder_LPIPS": g_loss_on_encoder_LPIPS,
            "g_loss_on_decoder_exp": g_loss_on_decoder_C,
            "g_loss_on_message_ide": g_loss_on_decoder_D,
            "g_loss_on_decoder_F": 0,
            "d_loss": d_loss
        }

        return result, (images, encoded_images, noised_images)

    def decoded_message_error_rate(self, message, decoded_message):
        length = message.shape[1] * message.shape[0]
        message = message.gt(0) # 大于0为1，小于为0
        decoded_message = decoded_message.gt(0)
        error_rate = float(torch.sum(message != decoded_message)) / length
        return error_rate

    def decoded_message_error_rate_list(self, message, decoded_message):
        message, decoded_message = torch.Tensor(message)[None], torch.Tensor(decoded_message)[None]
        length = message.shape[1] * message.shape[0]
        message = message.gt(0) # 大于0为1，小于为0
        decoded_message = decoded_message.gt(0)
        error_rate = float(torch.sum(message != decoded_message)) / length
        return error_rate
    
    def decoded_message_error_rate_actual(self, message, decoded_message):
        length = message.shape[0]
        
        error_rate = float(sum(abs(message - decoded_message))) / length
        return error_rate
    
    def decoded_message_error_rate_actual_v2(self, message, decoded_message):
        length = message.shape[0] * message.shape[1] * message.shape[2]
        
        error_rate = float(torch.sum(abs(message - decoded_message))) / length
        return error_rate

    def decoded_message_error_rate_batch(self, messages, decoded_messages): # MAE
        error_rate = 0.0
        batch_size = len(messages)
        
        for i in range(batch_size):
            error_rate += self.decoded_message_error_rate_actual(messages[i], decoded_messages[i]) # decoded_message_error_rate
        
        error_rate /= batch_size
        return error_rate

    def save_model(self, path_encoder_decoder: str, path_discriminator: str):
        torch.save(self.encoder_decoder.module.state_dict(), path_encoder_decoder)
        torch.save(self.discriminator.module.state_dict(), path_discriminator)

    def load_model(self, path_encoder_decoder: str, path_discriminator: str):
        self.load_model_ed(path_encoder_decoder)
        self.load_model_dis(path_discriminator)

    def load_model_ed(self, path_encoder_decoder: str):
        self.encoder_decoder.module.load_state_dict(torch.load(path_encoder_decoder), strict=False)

    def load_model_dis(self, path_discriminator: str):
        self.discriminator.module.load_state_dict(torch.load(path_discriminator))
        
        
class Network_ref:

	def __init__(self, message_length, noise_layers_R, noise_layers_F, device, batch_size, lr, beta1, attention_encoder, attention_decoder, weight):
		# device
		self.device = device

		# loss function
		self.criterion_MSE = nn.MSELoss().to(device)
		self.criterion_LPIPS = lpips.LPIPS().to(device)

		# weight of encoder-decoder loss 不同损失的权重
		self.encoder_weight = weight[0]
		self.decoder_weight_C = weight[1]
		self.decoder_weight_R = weight[2]
		self.decoder_weight_F = weight[3]
		self.discriminator_weight = weight[4]

		# network
		self.encoder_decoder = DW_EncoderDecoder(message_length, noise_layers_R, noise_layers_F, attention_encoder, attention_decoder).to(device)
		self.discriminator = Patch_Discriminator().to(device)

		self.encoder_decoder = torch.nn.DataParallel(self.encoder_decoder)
		self.discriminator = torch.nn.DataParallel(self.discriminator)

		# mark "cover" as 1, "encoded" as -1
		self.label_cover = 1.0
		self.label_encoded = - 1.0

		for p in self.encoder_decoder.module.noise.parameters():
			p.requires_grad = False

		# optimizer
		self.opt_encoder_decoder = torch.optim.Adam(
			filter(lambda p: p.requires_grad, self.encoder_decoder.parameters()), lr=lr, betas=(beta1, 0.999))
		self.opt_discriminator = torch.optim.Adam(self.discriminator.parameters(), lr=lr, betas=(beta1, 0.999))


	def train(self, images: torch.Tensor, messages: torch.Tensor, masks: torch.Tensor):
		self.encoder_decoder.train()
		self.discriminator.train()

		with torch.enable_grad():
			# use device to compute 
			images, messages, masks = images.to(self.device), messages.to(self.device), masks.to(self.device) # mask是啥及作用？作用放到noise_layer里面执行变换
			encoded_images, noised_images, decoded_messages_C, decoded_messages_R, decoded_messages_F = self.encoder_decoder(images, messages, masks) # 

			'''
			train discriminator
			'''
			for p in self.discriminator.parameters():
				p.requires_grad = True

			self.opt_discriminator.zero_grad()

			# RAW : target label for image should be "cover"(1)
			d_label_cover = self.discriminator(images)
			#d_cover_loss = self.criterion_MSE(d_label_cover, torch.ones_like(d_label_cover))
			#d_cover_loss.backward()

			# GAN : target label for encoded image should be "encoded"(0)
			d_label_encoded = self.discriminator(encoded_images.detach())
			#d_encoded_loss = self.criterion_MSE(d_label_encoded, torch.zeros_like(d_label_encoded))
			#d_encoded_loss.backward()

			d_loss = self.criterion_MSE(d_label_cover - torch.mean(d_label_encoded), self.label_cover * torch.ones_like(d_label_cover)) +\
			         self.criterion_MSE(d_label_encoded - torch.mean(d_label_cover), self.label_encoded * torch.ones_like(d_label_encoded))
			d_loss.backward()

			self.opt_discriminator.step()

			'''
			train encoder and decoder
			'''
			# Make it a tiny bit faster
			for p in self.discriminator.parameters():
				p.requires_grad = False

			self.opt_encoder_decoder.zero_grad()

			# GAN : target label for encoded image should be "cover"(0)
			g_label_cover = self.discriminator(images)
			g_label_encoded = self.discriminator(encoded_images)
			g_loss_on_discriminator = self.criterion_MSE(g_label_cover - torch.mean(g_label_encoded), self.label_encoded * torch.ones_like(g_label_cover)) +\
									  self.criterion_MSE(g_label_encoded - torch.mean(g_label_cover), self.label_cover * torch.ones_like(g_label_encoded))

			# RAW : the encoded image should be similar to cover image
			g_loss_on_encoder_MSE = self.criterion_MSE(encoded_images, images)
			g_loss_on_encoder_LPIPS = torch.mean(self.criterion_LPIPS(encoded_images, images))

			# RESULT : the decoded message should be similar to the raw message /Dual
			g_loss_on_decoder_C = self.criterion_MSE(decoded_messages_C, messages)
			g_loss_on_decoder_R = self.criterion_MSE(decoded_messages_R, messages)
			g_loss_on_decoder_F = self.criterion_MSE(decoded_messages_F, torch.zeros_like(messages))

			# full loss
			g_loss = self.discriminator_weight * g_loss_on_discriminator + self.encoder_weight * g_loss_on_encoder_MSE +\
					 self.decoder_weight_C * g_loss_on_decoder_C + self.decoder_weight_R * g_loss_on_decoder_R + self.decoder_weight_F * g_loss_on_decoder_F

			g_loss.backward()
			self.opt_encoder_decoder.step()

			# psnr
			psnr = - kornia.losses.psnr_loss(encoded_images.detach(), images, 2)

			# ssim
			ssim = 1 - 2 * kornia.losses.ssim_loss(encoded_images.detach(), images, window_size=11, reduction="mean")

		'''
		decoded message error rate /Dual
		'''
		error_rate_C = self.decoded_message_error_rate_batch(messages, decoded_messages_C)
		error_rate_R = self.decoded_message_error_rate_batch(messages, decoded_messages_R)
		error_rate_F = self.decoded_message_error_rate_batch(messages, decoded_messages_F)

		result = {
			"g_loss": g_loss,
			"error_rate_C": error_rate_C,
			"error_rate_R": error_rate_R,
			"error_rate_F": error_rate_F,
			"psnr": psnr,
			"ssim": ssim,
			"g_loss_on_discriminator": g_loss_on_discriminator,
			"g_loss_on_encoder_MSE": g_loss_on_encoder_MSE,
			"g_loss_on_encoder_LPIPS": g_loss_on_encoder_LPIPS,
			"g_loss_on_decoder_C": g_loss_on_decoder_C,
			"g_loss_on_decoder_R": g_loss_on_decoder_R,
			"g_loss_on_decoder_F": g_loss_on_decoder_F,
			"d_loss": d_loss
		}
		return result


	def validation(self, images: torch.Tensor, messages: torch.Tensor, masks: torch.Tensor):
		self.encoder_decoder.eval()
		self.encoder_decoder.module.noise.train()
		self.discriminator.eval()

		with torch.no_grad():
			# use device to compute
			images, messages, masks = images.to(self.device), messages.to(self.device), masks.to(self.device)
			encoded_images, noised_images, decoded_messages_C, decoded_messages_R, decoded_messages_F = self.encoder_decoder(images, messages, masks)

			'''
			validate discriminator
			'''
			# RAW : target label for image should be "cover"(1)
			d_label_cover = self.discriminator(images)
			#d_cover_loss = self.criterion_MSE(d_label_cover, torch.ones_like(d_label_cover))

			# GAN : target label for encoded image should be "encoded"(0)
			d_label_encoded = self.discriminator(encoded_images.detach())
			#d_encoded_loss = self.criterion_MSE(d_label_encoded, torch.zeros_like(d_label_encoded))

			d_loss = self.criterion_MSE(d_label_cover - torch.mean(d_label_encoded), self.label_cover * torch.ones_like(d_label_cover)) +\
			         self.criterion_MSE(d_label_encoded - torch.mean(d_label_cover), self.label_encoded * torch.ones_like(d_label_encoded))

			'''
			validate encoder and decoder
			'''

			# GAN : target label for encoded image should be "cover"(0)
			g_label_cover = self.discriminator(images)
			g_label_encoded = self.discriminator(encoded_images)
			g_loss_on_discriminator = self.criterion_MSE(g_label_cover - torch.mean(g_label_encoded), self.label_encoded * torch.ones_like(g_label_cover)) +\
									  self.criterion_MSE(g_label_encoded - torch.mean(g_label_cover), self.label_cover * torch.ones_like(g_label_encoded))

			# RAW : the encoded image should be similar to cover image
			g_loss_on_encoder_MSE = self.criterion_MSE(encoded_images, images)
			g_loss_on_encoder_LPIPS = torch.mean(self.criterion_LPIPS(encoded_images, images))

			# RESULT : the decoded message should be similar to the raw message /Dual
			g_loss_on_decoder_C = self.criterion_MSE(decoded_messages_C, messages)
			g_loss_on_decoder_R = self.criterion_MSE(decoded_messages_R, messages)
			g_loss_on_decoder_F = self.criterion_MSE(decoded_messages_F, torch.zeros_like(messages))

			# full loss
			# unstable g_loss_on_discriminator is not used during validation

			g_loss = 0 * g_loss_on_discriminator + self.encoder_weight * g_loss_on_encoder_LPIPS +\
					 self.decoder_weight_C * g_loss_on_decoder_C + self.decoder_weight_R * g_loss_on_decoder_R + self.decoder_weight_F * g_loss_on_decoder_F


			# psnr
			psnr = - kornia.losses.psnr_loss(encoded_images.detach(), images, 2)

			# ssim
			ssim = 1 - 2 * kornia.losses.ssim_loss(encoded_images.detach(), images, window_size=11, reduction="mean")

		'''
		decoded message error rate /Dual
		'''
		error_rate_C = self.decoded_message_error_rate_batch(messages, decoded_messages_C)
		error_rate_R = self.decoded_message_error_rate_batch(messages, decoded_messages_R)
		error_rate_F = self.decoded_message_error_rate_batch(messages, decoded_messages_F)

		result = {
			"g_loss": g_loss,
			"error_rate_C": error_rate_C,
			"error_rate_R": error_rate_R,
			"error_rate_F": error_rate_F,
			"psnr": psnr,
			"ssim": ssim,
			"g_loss_on_discriminator": g_loss_on_discriminator,
			"g_loss_on_encoder_MSE": g_loss_on_encoder_MSE,
			"g_loss_on_encoder_LPIPS": g_loss_on_encoder_LPIPS,
			"g_loss_on_decoder_C": g_loss_on_decoder_C,
			"g_loss_on_decoder_R": g_loss_on_decoder_R,
			"g_loss_on_decoder_F": g_loss_on_decoder_F,
			"d_loss": d_loss
		}

		return result, (images, encoded_images, noised_images)

	def decoded_message_error_rate(self, message, decoded_message):
		length = message.shape[0]

		message = message.gt(0) # 大于0为1，小于为0
		decoded_message = decoded_message.gt(0)
		error_rate = float(sum(message != decoded_message)) / length
		return error_rate

	def decoded_message_error_rate_actual(self, message, decoded_message):
		length = message.shape[0]

		error_rate = float(sum(abs(message - decoded_message))) / length
		return error_rate
 
	def decoded_message_error_rate_batch(self, messages, decoded_messages):
		error_rate = 0.0
		batch_size = len(messages)
		for i in range(batch_size):
			error_rate += self.decoded_message_error_rate_actual(messages[i], decoded_messages[i]) # decoded_message_error_rate
		error_rate /= batch_size
		return error_rate

	def save_model(self, path_encoder_decoder: str, path_discriminator: str):
		torch.save(self.encoder_decoder.module.state_dict(), path_encoder_decoder)
		torch.save(self.discriminator.module.state_dict(), path_discriminator)

	def load_model(self, path_encoder_decoder: str, path_discriminator: str):
		self.load_model_ed(path_encoder_decoder)
		self.load_model_dis(path_discriminator)

	def load_model_ed(self, path_encoder_decoder: str):
		self.encoder_decoder.module.load_state_dict(torch.load(path_encoder_decoder), strict=False)

	def load_model_dis(self, path_discriminator: str):
		self.discriminator.module.load_state_dict(torch.load(path_discriminator))


class Network4:

    def __init__(self, message_length, noise_layers_R, noise_layers_F, device, batch_size, lr, beta1, attention_encoder, attention_decoder, weight):
        # device
        self.device = device

        # loss function
        self.criterion_MSE = nn.MSELoss().to(device)
        self.criterion_LPIPS = lpips.LPIPS().to(device)
        self.bce = nn.BCEWithLogitsLoss().to(device)
        
        # weight of encoder-decoder loss 不同损失的权重
        self.encoder_weight = weight[0]
        self.decoder_weight_C = weight[1]
        self.decoder_weight_R = weight[2]
        self.decoder_weight_F = weight[3]
        self.discriminator_weight = weight[4]
        
        # network
        self.encoder_decoder = DE_EncoderDecoder(message_length, noise_layers_R, noise_layers_F, attention_encoder, attention_decoder, adain = False).to(device)
        self.discriminator_1 = Patch_Discriminator().to(device)
        self.discriminator_2 = Patch_Discriminator().to(device)
        

        self.encoder_decoder = torch.nn.DataParallel(self.encoder_decoder)
        self.discriminator_1 = torch.nn.DataParallel(self.discriminator_1)
        self.discriminator_2 = torch.nn.DataParallel(self.discriminator_2)
        

        # mark "cover" as 1, "encoded" as -1
        self.label_cover = 1.0
        self.label_encoded = - 1.0

        for p in self.encoder_decoder.module.noise.parameters():
            p.requires_grad = False
        
        # optimizer
        self.opt_encoder_decoder = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.encoder_decoder.parameters()), lr=lr, betas=(beta1, 0.999))
        self.opt_discriminator_1 = torch.optim.Adam(self.discriminator_1.parameters(), lr=lr, betas=(beta1, 0.999))
        self.opt_discriminator_2 = torch.optim.Adam(self.discriminator_2.parameters(), lr=lr, betas=(beta1, 0.999))


    def train(self, images: torch.Tensor, messages: torch.Tensor, masks: torch.Tensor):
        self.encoder_decoder.train()
        self.discriminator_1.train()
        self.discriminator_2.train()
        
        
        with torch.enable_grad():
            # use device to compute 
            images, messages = images.to(self.device), messages.to(self.device) # mask放到noise_layer里面执行变换
            if masks is not None:
                masks = masks.to(self.device)
            encoded_images_m, encoded_images_n, decoded_messages_C, _ = self.encoder_decoder(images, messages, masks) 
            
            '''
            train discriminator
            '''
            for p in self.discriminator_1.parameters():
                p.requires_grad = True
            for p in self.discriminator_2.parameters():
                p.requires_grad = True
                
            self.opt_discriminator_1.zero_grad()
            self.opt_discriminator_2.zero_grad()
            
            '''
            train discriminator
            '''
            d_label_cover_1 = self.discriminator_1(images)
            d_label_encoded = self.discriminator_1(encoded_images_m.detach())
            # d_loss = self.criterion_MSE(d_label_cover_1 - torch.mean(d_label_encoded), self.label_cover * torch.ones_like(d_label_cover_1)) + \
            #         self.criterion_MSE(d_label_encoded - torch.mean(d_label_cover_1), self.label_encoded * torch.ones_like(d_label_encoded))
            d_loss = self.bce(d_label_cover_1,  torch.zeros_like(d_label_cover_1)) + \
                self.bce(d_label_encoded, torch.ones_like(d_label_encoded))
            d_loss.backward()
            self.opt_discriminator_1.step()
            
            d_label_cover_2 = self.discriminator_2(images)
            d_label_encoded_n = self.discriminator_2(encoded_images_n.detach())
            # d_loss_n = self.criterion_MSE(d_label_cover_2 - torch.mean(d_label_encoded_n), self.label_cover * torch.ones_like(d_label_cover_2)) + \
            #         self.criterion_MSE(d_label_encoded_n - torch.mean(d_label_cover_2), self.label_encoded * torch.ones_like(d_label_encoded_n))
            d_loss_n = self.bce(d_label_cover_2,  torch.zeros_like(d_label_cover_2)) + \
                self.bce(d_label_encoded_n, torch.ones_like(d_label_encoded_n))
            d_loss_n.backward()
            self.opt_discriminator_2.step()
   
            '''
            train encoder and decoder
            '''
            for p in self.discriminator_1.parameters():
                p.requires_grad = False
            for p in self.discriminator_2.parameters():
                p.requires_grad = False
                
            self.opt_encoder_decoder.zero_grad()
   
            # Generator Training:
            # watermark branch
            g_label_cover = self.discriminator_1(images)
            g_label_encoded = self.discriminator_1(encoded_images_m)
            g_loss_on_discriminator = self.criterion_MSE(g_label_cover - torch.mean(g_label_encoded), self.label_encoded * torch.ones_like(g_label_cover)) +\
                                    self.criterion_MSE(g_label_encoded - torch.mean(g_label_cover), self.label_cover * torch.ones_like(g_label_encoded))
            g_loss_on_encoder_MSE = self.criterion_MSE(encoded_images_m, images)
            g_loss_on_encoder_LPIPS = torch.mean(self.criterion_LPIPS(encoded_images_m, images))
            g_loss_on_decoder_C = self.criterion_MSE(decoded_messages_C, messages)
            g_loss_m = self.discriminator_weight * g_loss_on_discriminator + self.encoder_weight * g_loss_on_encoder_MSE +\
                    self.decoder_weight_C * g_loss_on_decoder_C

            # naive branch
            g_label_cover_n = self.discriminator_2(images)
            g_label_encoded_n = self.discriminator_2(encoded_images_n)
            g_loss_on_discriminator_n = self.criterion_MSE(g_label_cover_n - torch.mean(g_label_encoded_n), self.label_encoded * torch.ones_like(g_label_cover_n)) +\
                                    self.criterion_MSE(g_label_encoded_n - torch.mean(g_label_cover_n), self.label_cover * torch.ones_like(g_label_encoded_n))
            g_loss_on_encoder_MSE_n = self.criterion_MSE(encoded_images_n, images)
            g_loss_on_encoder_LPIPS_n = torch.mean(self.criterion_LPIPS(encoded_images_n, images))
            g_loss_n = self.discriminator_weight * g_loss_on_discriminator_n + self.encoder_weight * g_loss_on_encoder_MSE_n

            g_loss = g_loss_m + g_loss_n
            g_loss.backward()
            self.opt_encoder_decoder.step()
   
            # psnr
            psnr_m = - kornia.losses.psnr_loss(encoded_images_m.detach(), images, 2)
            psnr_n = - kornia.losses.psnr_loss(encoded_images_n.detach(), images, 2)
            
            # ssim
            ssim_m = 1 - 2 * kornia.losses.ssim_loss(encoded_images_m.detach(), images, window_size = 11, reduction = "mean")
            ssim_n = 1 - 2 * kornia.losses.ssim_loss(encoded_images_n.detach(), images, window_size = 11, reduction = "mean")
            

        '''
        decoded message error rate / Dual
        '''
        error_rate_C = self.decoded_message_error_rate_batch(messages, decoded_messages_C)

        result = {
            "g_loss": g_loss,
            "error_rate_C": error_rate_C,
            "psnr_m": psnr_m,
            "psnr_n": psnr_n,
            "ssim_m": ssim_m,
            "ssim_n": ssim_n,
            "g_loss_on_discriminator_M": g_loss_on_discriminator,
            "g_loss_on_encoder_MSE_M": g_loss_on_encoder_MSE,
            "g_loss_on_encoder_LPIPS_M": g_loss_on_encoder_LPIPS,
            "g_loss_on_decoder_C": g_loss_on_decoder_C,
            "g_loss_on_discriminator_N": g_loss_on_discriminator_n,
            "g_loss_on_encoder_MSE_N": g_loss_on_encoder_MSE_n,
            "g_loss_on_encoder_LPIPS_N": g_loss_on_encoder_LPIPS_n,
            "d_loss": d_loss,
            "d_loss_n": d_loss_n
            
        }
        return result


    def validation(self, images: torch.Tensor, messages: torch.Tensor, masks: torch.Tensor):
        self.encoder_decoder.eval()
        self.encoder_decoder.module.noise.train()
        self.discriminator_1.eval()
        self.discriminator_2.eval()
        

        with torch.no_grad():
            # use device to compute
            images, messages = images.to(self.device), messages.to(self.device)
            if masks is not None:
                masks = masks.to(self.device)
            encoded_images_m, encoded_images_n, decoded_messages_C, noised_images_m = self.encoder_decoder(images, messages, masks)

            '''
            validate discriminator
            '''
            # Discriminator Training:
            d_label_cover_1 = self.discriminator_1(images)
            d_label_cover_2 = self.discriminator_2(images)

            d_label_encoded = self.discriminator_1(encoded_images_m.detach())
            # d_loss = self.criterion_MSE(d_label_cover_1 - torch.mean(d_label_encoded), self.label_cover * torch.ones_like(d_label_cover_1)) + \
            #         self.criterion_MSE(d_label_encoded - torch.mean(d_label_cover_1), self.label_encoded * torch.ones_like(d_label_encoded))
            d_loss = self.bce(d_label_cover_1,  torch.zeros_like(d_label_cover_1)) + \
                self.bce(d_label_encoded, torch.ones_like(d_label_encoded))
   
            d_label_encoded_n = self.discriminator_2(encoded_images_n.detach())
            # d_loss_n = self.criterion_MSE(d_label_cover_2 - torch.mean(d_label_encoded_n), self.label_cover * torch.ones_like(d_label_cover_2)) + \
            #         self.criterion_MSE(d_label_encoded_n - torch.mean(d_label_cover_2), self.label_encoded * torch.ones_like(d_label_encoded_n))
            d_loss_n = self.bce(d_label_cover_2,  torch.zeros_like(d_label_cover_2)) + \
                self.bce(d_label_encoded_n, torch.ones_like(d_label_encoded_n))
   
            '''
            validate encoder and decoder
            '''
            # watermark branch
            g_label_cover = self.discriminator_1(images)
            g_label_encoded = self.discriminator_1(encoded_images_m)
            g_loss_on_discriminator = self.criterion_MSE(g_label_cover - torch.mean(g_label_encoded), self.label_encoded * torch.ones_like(g_label_cover)) +\
                                    self.criterion_MSE(g_label_encoded - torch.mean(g_label_cover), self.label_cover * torch.ones_like(g_label_encoded))
            g_loss_on_encoder_MSE = self.criterion_MSE(encoded_images_m, images)
            g_loss_on_encoder_LPIPS = torch.mean(self.criterion_LPIPS(encoded_images_m, images))
            g_loss_on_decoder_C = self.criterion_MSE(decoded_messages_C, messages)
            g_loss_m = 0 * g_loss_on_discriminator + self.encoder_weight * g_loss_on_encoder_MSE +\
                    self.decoder_weight_C * g_loss_on_decoder_C

            # naive branch
            g_label_cover_n = self.discriminator_2(images)
            g_label_encoded_n = self.discriminator_2(encoded_images_n)
            g_loss_on_discriminator_n = self.criterion_MSE(g_label_cover_n - torch.mean(g_label_encoded_n), self.label_encoded * torch.ones_like(g_label_cover_n)) +\
                                    self.criterion_MSE(g_label_encoded_n - torch.mean(g_label_cover_n), self.label_cover * torch.ones_like(g_label_encoded_n))
            g_loss_on_encoder_MSE_n = self.criterion_MSE(encoded_images_n, images)
            g_loss_on_encoder_LPIPS_n = torch.mean(self.criterion_LPIPS(encoded_images_m, images))
            g_loss_n = 0 * g_loss_on_discriminator_n + self.encoder_weight * g_loss_on_encoder_MSE_n

            g_loss = g_loss_m + g_loss_n

            # psnr
            psnr_m = - kornia.losses.psnr_loss(encoded_images_m.detach(), images, 2)
            psnr_n = - kornia.losses.psnr_loss(encoded_images_n.detach(), images, 2)
            
            # ssim
            ssim_m = 1 - 2 * kornia.losses.ssim_loss(encoded_images_m.detach(), images, window_size = 11, reduction = "mean")
            ssim_n = 1 - 2 * kornia.losses.ssim_loss(encoded_images_n.detach(), images, window_size = 11, reduction = "mean")
            

        '''
        decoded message error rate /Dual
        '''
        error_rate_C = self.decoded_message_error_rate_batch(messages, decoded_messages_C)

        result = {
            "g_loss": g_loss,
            "error_rate_C": error_rate_C,
            "psnr_m": psnr_m,
            "psnr_n": psnr_n,
            "ssim_m": ssim_m,
            "ssim_n": ssim_n,
            "g_loss_on_discriminator_M": g_loss_on_discriminator,
            "g_loss_on_encoder_MSE_M": g_loss_on_encoder_MSE,
            "g_loss_on_encoder_LPIPS_M": g_loss_on_encoder_LPIPS,
            "g_loss_on_decoder_C": g_loss_on_decoder_C,
            "g_loss_on_discriminator_N": g_loss_on_discriminator_n,
            "g_loss_on_encoder_MSE_N": g_loss_on_encoder_MSE_n,
            "g_loss_on_encoder_LPIPS_N": g_loss_on_encoder_LPIPS_n,
            "d_loss": d_loss,
            "d_loss_n": d_loss_n
        }

        return result, (images, encoded_images_m, encoded_images_n, noised_images_m)

    def decoded_message_error_rate(self, message, decoded_message):
        length = message.shape[0]

        message = message.gt(0) # 大于0为1，小于为0
        decoded_message = decoded_message.gt(0)
        error_rate = float(sum(message != decoded_message)) / length
        return error_rate

    def decoded_message_error_rate_actual(self, message, decoded_message):
        length = message.shape[0]
        
        error_rate = float(sum(abs(message - decoded_message))) / length
        return error_rate
    
    def decoded_message_error_rate_actual_v2(self, message, decoded_message):
        length = message.shape[0] * message.shape[1] * message.shape[2]
        
        error_rate = float(torch.sum(abs(message - decoded_message))) / length
        return error_rate

    def decoded_message_error_rate_batch(self, messages, decoded_messages):
        error_rate = 0.0
        batch_size = len(messages)
        for i in range(batch_size):
            error_rate += self.decoded_message_error_rate_actual(messages[i], decoded_messages[i]) # decoded_message_error_rate
        error_rate /= batch_size
        
        return error_rate

    def save_model(self, path_encoder_decoder: str, path_discriminator_1: str, path_discriminator_2: str):
        torch.save(self.encoder_decoder.module.state_dict(), path_encoder_decoder)
        torch.save(self.discriminator_1.module.state_dict(), path_discriminator_1)
        torch.save(self.discriminator_2.module.state_dict(), path_discriminator_2)
        

    def load_model(self, path_encoder_decoder: str, path_discriminator_1: str, path_discriminator_2: str):
        self.load_model_ed(path_encoder_decoder)
        self.load_model_dis(path_discriminator_1)
        self.load_model_dis(path_discriminator_2)
        

    def load_model_ed(self, path_encoder_decoder: str):
        self.encoder_decoder.module.load_state_dict(torch.load(path_encoder_decoder), strict=False)

    def load_model_dis(self, path_discriminator_1: str, path_discriminator_2: str):
        self.discriminator_1.module.load_state_dict(torch.load(path_discriminator_1))
        self.discriminator_2.module.load_state_dict(torch.load(path_discriminator_2))
        