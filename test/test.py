import os
import time
import yaml
import numpy as np
import torch
import torch.nn.functional as F

from lpips import LPIPS
import kornia

from easydict import EasyDict
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from ssd.utils import *
from ssd.network.Dual_Mark import *

from shutil import copyfile
from ssd.network.noise_layers import *
import random, string

import ssd.steg_module.modules.Unet_common as common
from ssd.steg_module.model import Model
from ssd.steg_module.modules.denoiser.denoise_module import DnCNN, Enhancement

from detector.models import Detector, DetectorWM
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from tqdm import tqdm
'''
test
'''

def seed_torch(seed=2025):
    seed = int(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_path(path="temp/"):
    return path + ''.join(random.sample(string.ascii_letters + string.digits, 16)) + ".png"

def cos(a, b):
    a_flat = torch.mean(a.view(1, a.size(1), -1).permute(0, 2, 1), dim=1)
    b_flat = torch.mean(b.view(1, b.size(1), -1).permute(0, 2, 1), dim=1)
    cos_sim_flat = F.cosine_similarity(a_flat, b_flat, dim=1)
    
    return cos_sim_flat
    
def main(setting):
    
    seed_torch(2025)
    criterion_LPIPS = LPIPS().cuda()
    
    with open('cfg/test.yaml', 'r') as f:
        args = EasyDict(yaml.load(f, Loader=yaml.SafeLoader))
        
    result_folder = "results/" + args.result_folder
    model_epoch = args.model_epoch
    batch_size = args.batch_size
    steg_folder = args.steg_folder
    
    if setting is None:
        noise_layer = args.noise_layer
    else:
        noise_layer = setting
        
    with open(result_folder + '/train.yaml', 'r') as f:
        args = EasyDict(yaml.load(f, Loader=yaml.SafeLoader))
        
    image_size = args.image_size
    test_dataset_path = '/dataset/test_256/'
    
    dwt = common.DWT()
    iwt = common.IWT()
    
    test_log = result_folder + "test_log" + time.strftime("%_Y_%m_%d__%H_%M_%S", time.localtime()) + ".txt"
    copyfile("configs/test.yaml", result_folder + "test_DualMark" + time.strftime("_%Y_%m_%d__%H_%M_%S", time.localtime()) + ".yaml")
    writer = SummaryWriter('runs/' + result_folder + noise_layer + time.strftime("%_Y_%m_%d__%H_%M_%S", time.localtime()))

    with open('./configs/config.yaml', 'r') as f:
        c = EasyDict(yaml.load(f, Loader=yaml.SafeLoader))
    stegnet = Model(c.size, c.patch_size, c.clamp, 'cuda:0').to()
    pretrained_weight = torch.load(os.path.join(steg_folder, 'model_checkpoint.pt'))
    stegnet.load_state_dict(pretrained_weight['net'], strict=True)
    stegnet.eval()
    for param in stegnet.parameters():
        param.requires_grad = False
    stegnet.cuda()
    
    dncnn_weight = torch.load('first.pth')
    dncnn = DnCNN()
    dncnn.load_state_dict(dncnn_weight, strict=True)
    for param in dncnn.parameters():
        param.requires_grad = False
    dncnn = dncnn.cuda().eval()
    
    denoiser_weight = torch.load('second.pth')
    new_state_dict = {}
    for key, value in denoiser_weight.items():
        if key.startswith('module.'): 
            new_key = key[7:]
            new_state_dict[new_key] = value
        else:
            new_state_dict[key] = value
    denoiser = Enhancement()
    denoiser.load_state_dict(new_state_dict, strict=True)
    for param in denoiser.parameters():
        param.requires_grad = False
    denoiser = denoiser.cuda().eval()
    
    if 'StarGan' not in noise_layer:
        test_dataset = attrsImgDataset_nl(test_dataset_path, image_size, zero=True)
    else:
        test_dataset = attrsImgDataset(test_dataset_path, image_size, zero=True)
        
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

    print("\n Start Testing : \n\n")

    test_result = {
        "error_rate_C": 0.0,
        "error_rate_RF": 0.0,
        "psnr": 0.0,
        "ssim": 0.0,
        "lpips": 0.0}

    saved_iterations = [1,2,3,4,5,6,7,8]
    saved_all = None
    
    error_rate_C_all = 0
    
    NOISE = eval(noise_layer)
    NOISE = NOISE.cuda()
    
    noise_r_name = None
    if noise_r_name is not None:
        NOISE_R = eval(noise_r_name)
    else:
        NOISE_R = None
    
    # detector
    detection_label = True
    detection_wm_label = True 
    forgery_label = True
    
    pred_all = []
    pred_all_score = []
    label_all = []
    
    pred_all_wm = []
    pred_all_wm_score = []
    label_wm_all = []
    
    if detection_label:
        detector = Detector().cuda()
        det_weight = torch.load('/weights_det/model_checkpoint.pt')
        detector.load_state_dict(det_weight['net'], strict=True)
        detector.eval()
        for param in detector.parameters():
            param.requires_grad = False
        detector.cuda()
        
    if detection_wm_label:
        detectorWM = DetectorWM().cuda()
        det_weight = torch.load('/weights_detWM/model_checkpoint.pt')
        detectorWM.load_state_dict(det_weight['net'], strict=True)
        detectorWM.eval()
        for param in detectorWM.parameters():
            param.requires_grad = False
        detectorWM.cuda()
    
    for step, variables in tqdm(enumerate(test_dataloader, 1)):
        
        if 'StarGan' not in noise_layer:
            image, label = variables.cuda(), None
        else:
            image, label = variables[0].cuda(), variables[1].cuda()
        
        '''
		test
		'''
        with torch.no_grad():
            images = image.cuda()
            
            images_wavelets = dwt(images)
            encoded_images = images_wavelets.narrow(1,0,3).clone() * 0.5
            
            output_steg, output_z_sets, _ = stegnet(images_wavelets,
                                                    encoded_images, 
                                                    condition = [None, [None, None], 0, 1])
            steg_img = iwt(output_steg)
            
            psnr = - kornia.losses.psnr_loss(steg_img.detach(), images, torch.max(steg_img).item()).item()
            ssim = 1 - 2 * kornia.losses.ssim_loss(steg_img.detach(), images, window_size=11, reduction="mean").item()
            lpips = torch.mean(criterion_LPIPS(steg_img.detach(), images)).item()
            
            if 'StarGan' not in noise_layer:
                noised_images = NOISE([steg_img*2-1, images*2-1, step])
            else:
                noised_images = NOISE([steg_img*2-1, images*2-1, label])
            
            if NOISE_R is not None:
                noised_images_real = NOISE_R([steg_img * 2 - 1, steg_img * 2 - 1, label])
                noised_images = NOISE_R([noised_images, images * 2 - 1, label])
                
            if isinstance(noised_images, tuple): 
                noised_images, success_label = noised_images
                if not success_label:
                    continue
            
            de_steg_img = 0.5 * (denoiser(dncnn(noised_images)) + 1)
            output_steg = dwt(de_steg_img)
            output_zeros = torch.zeros_like(output_z_sets[-1])
            
            cover_rev_sets, secret_rev_sets = stegnet(output_steg, output_zeros, rev = True, condition = [None, [None, None], 0, 1]) # cover_input condition = True
            secret_rev = secret_rev_sets[-1]

            if detection_label:
                output_steg_ll = output_steg.narrow(1,0,3) * 0.5
                
                if NOISE_R is None:
                    de_steg_img_r = 0.5 * (denoiser(dncnn(steg_img * 2 - 1)) + 1)
                else:
                    de_steg_img_r = 0.5 * (denoiser(dncnn(noised_images_real)) + 1)
                
                output_steg_r = dwt(de_steg_img_r)
                output_steg_ll_r = output_steg_r.narrow(1, 0, 3) * 0.5
                _, secret_rev_sets_r = stegnet(output_steg_r, output_zeros, rev = True, condition = [None, [None, None], 0, 1])
                secret_rev_r = secret_rev_sets_r[-1]
                
                inp_steg = torch.cat((output_steg_ll_r, output_steg_ll), dim=0)
                inp_sec_rev = torch.cat((secret_rev_r, secret_rev), dim=0)
                pred = detector(inp_steg, inp_sec_rev)

                if forgery_label:
                    label = torch.cat((torch.zeros(output_steg_ll_r.shape[0]), torch.ones(output_steg_ll.shape[0])), dim = 0).to(pred.device)
                else:
                    label = torch.cat((torch.zeros(output_steg_ll_r.shape[0]), torch.zeros(output_steg_ll.shape[0])), dim = 0).to(pred.device)
                
                pred = pred.cpu().numpy()
                label = label.long().cpu().numpy()
                
                for i in range(pred.shape[0]):
                    pred_all.append(1) if pred[i] > 0.5 else pred_all.append(0)
                    pred_all_score.append(pred[i][0])
                    label_all.append(label[i])
                
                
            if detection_wm_label:
                de_cover_img_c = 0.5 * (1 + denoiser(dncnn(images * 2 - 1)))
                output_cover_c = dwt(de_cover_img_c)
                
                _, secret_rev_sets_cover = stegnet(output_cover_c, output_zeros, rev = True, condition = [None, [None, None], 0, 1])
                secret_rev_cover = secret_rev_sets_cover[-1]
                
                pred = detectorWM(torch.cat((secret_rev_cover, secret_rev)))
                label = torch.cat((torch.zeros(secret_rev_cover.shape[0]), torch.ones(secret_rev.shape[0])), dim = 0).to(pred.device)
                
                pred = pred.cpu().numpy()
                label = label.long().cpu().numpy()
                
                for i in range(pred.shape[0]):
                    pred_all_wm.append(1) if pred[i] > 0.5 else pred_all_wm.append(0)
                    pred_all_wm_score.append(pred[i])
                    label_wm_all.append(label[i])
                    
        error_rate_C_all += 0
        
        result = {
            "error_rate_C": 0,
            "error_rate_RF": 0,
            "psnr": psnr,
            "ssim": ssim,
            "lpips": lpips}

        for key in result:
            test_result[key] += float(result[key])

        if step in saved_iterations:
            if saved_all is None:
                saved_all = get_random_images(image, steg_img, noised_images)
            else:
                saved_all = concatenate_images(saved_all, image, steg_img, noised_images)

        content = "Image " + str(step) + " : \n"
        for key in test_result:
            content += key + "=" + str(result[key]) + ","
            writer.add_scalar("Test/" + key, float(result[key]), step)
        content += "\n"

        with open(test_log, "a") as file:
            file.write(content)
        
    '''
	test results
	'''
 
    content = "Average : \n"
    for key in test_result:
        content += key + "=" + str(test_result[key] / step) + ","
        writer.add_scalar("Test_epoch/" + key, float(test_result[key] / step), 1)
    content += "\n"
    writer.close()
    
    print(content)
    print('dataset', test_dataset_path)
    print('Noise:', noise_layer)
    
    if NOISE_R is not None:
        print('NoiseR:', noise_r_name)
    
    if detection_label:
        acc = accuracy_score(label_all, pred_all)
        f1 = f1_score(label_all, pred_all)
        
        print('ACC:{}'.format(acc))
        print('F1:{}'.format(f1))
        
        tmp_pos = []
        tmp_pos_score = []
        tmp_neg = []
        tmp_neg_score = []
        
        for i in range(len(label_all)):
            if label_all[i] == 0:
                tmp_neg.append(label_all[i])
                tmp_neg_score.append(pred_all_score[i])
            else:
                tmp_pos.append(label_all[i])
                tmp_pos_score.append(pred_all_score[i])
                
        label = [*tmp_neg, *tmp_pos]
        score = [*tmp_neg_score, *tmp_pos_score]
        auc = roc_auc_score(label, score)
        print('AUC:{}'.format(auc))
        
        if noise_r_name is not None:
            acc_content = '\n' + noise_layer + '\n' + noise_r_name + '\n' + 'acc:'+ str(acc) + ' \nf1:'+ str(f1) + ' \nauc:'+ str(auc)
        
            with open('test_acc_log.txt', "a") as file:
                file.write(acc_content)
            
    if detection_wm_label:
        print('ACC_WM:{}'.format(accuracy_score(label_wm_all, pred_all_wm)))
        print('F1_WM:{}'.format(f1_score(label_wm_all, pred_all_wm)))
        tmp_pos = []
        tmp_pos_score = []
        tmp_neg = []
        tmp_neg_score = []
        
        for i in range(len(label_wm_all)):
            if label_wm_all[i] == 0:
                tmp_neg.append(label_wm_all[i])
                tmp_neg_score.append(pred_all_wm_score[i])
            else:
                tmp_pos.append(label_wm_all[i])
                tmp_pos_score.append(pred_all_wm_score[i])
                
        label = [*tmp_neg, *tmp_pos]
        score = [*tmp_neg_score, *tmp_pos_score]
        print('AUC_WM:{}'.format(roc_auc_score(label, score)))

if __name__ == '__main__':
    import argparse
    
    def arg_parser():
        parser = argparse.ArgumentParser(description="config")
        
        parser.add_argument("--setting",
                            type=str,
                            default=None,
                            help="Specified the path of configuration file to be used.")
        
        return parser.parse_args()
    
    param = arg_parser()
    setting = param.setting
    
    main(setting)