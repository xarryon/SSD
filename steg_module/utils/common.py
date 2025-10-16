import torch
import numpy as np
import torch.nn.functional as F
import os
from PIL import Image
import torch.nn as nn
import importlib
from typing import Mapping, Any, Tuple, Callable, Dict, Literal


def sp_spatial(x, patch):
    b, c, h, w = x.shape
    
    interval = h // patch
    x = x.view(b, c, interval, patch, interval, patch)
    x = x.permute(0, 2, 4, 1, 3, 5).contiguous()
    x = x.view(b, -1, patch, patch)
    
    return x


def exp_spatial(x, size):
    b, c, h, w = x.shape
    interval = size // h
    c1 = c // (interval * interval)
    
    x = x.view(b, interval, interval, c1, h, w)
    x = x.permute(0, 3, 1, 4, 2, 5).contiguous()
    x = x.view(b, c1, size, size)
    
    return x


def to(obj, device):
    if torch.is_tensor(obj):
        return obj.to(device)
    if isinstance(obj, dict):
        return {k: to(v, device) for k, v in obj.items()}
    if isinstance(obj, tuple):
        return tuple(to(v, device) for v in obj)
    if isinstance(obj, list):
        return [to(v, device) for v in obj]
    return obj


def get_random_images(cover, secret, steg_img, secret_rev, noise_steg_img):
    selected_id = np.random.randint(1, cover.shape[0]) if cover.shape[0] > 1 else 1
    image = cover.cpu()[selected_id - 1:selected_id, :, :, :]
    secret = secret.cpu()[selected_id - 1:selected_id, :, :, :]
    steg_img = steg_img.cpu()[selected_id - 1:selected_id, :, :, :]
    secret_rev = secret_rev.cpu()[selected_id - 1:selected_id, :, :, :]
    noise_steg_img = noise_steg_img.cpu()[selected_id - 1:selected_id, :, :, :]
    
    return [image, secret, steg_img, secret_rev, noise_steg_img]


def concatenate_images(saved_all, cover, secret, steg_img, secret_rev, noise_steg_img):
    saved = get_random_images(cover, secret, steg_img, secret_rev, noise_steg_img)
    if saved_all[2].shape[2] != saved[2].shape[2]:
        return saved_all
    saved_all[0] = torch.cat((saved_all[0], saved[0]), 0)
    saved_all[1] = torch.cat((saved_all[1], saved[1]), 0)
    saved_all[2] = torch.cat((saved_all[2], saved[2]), 0)
    saved_all[3] = torch.cat((saved_all[3], saved[3]), 0)
    saved_all[4] = torch.cat((saved_all[4], saved[4]), 0)
    
    return saved_all


def _normalize(input_tensor):
    output = input_tensor.clone()
    for i in range(output.shape[0]):
        min_val, max_val = torch.min(output[i]), torch.max(output[i])
        output[i] = (output[i] - min_val) / (max_val - min_val)

    return output


def save_images(saved_all, epoch, folder, resize_to=None):
    cover, secret, steg_img, secret_rev, noise_steg_img = saved_all

    cover_show = cover[:cover.shape[0], :, :, :].cpu()
    secret_show = secret[:secret.shape[0], :, :, :].cpu()
    steg_img_show = steg_img[:steg_img.shape[0], :, :, :].cpu()
    secret_rev_show = secret_rev[:secret_rev.shape[0], :, :, :].cpu()
    noise_steg_img_show = noise_steg_img[:noise_steg_img.shape[0], :, :, :].cpu()
    
    # scale values to range [0, 1] from original range of [-1, 1]
    cover_show = cover_show
    secret_show = secret_show
    steg_img_show = steg_img_show
    secret_rev_show = secret_rev_show
    noise_steg_img_show = noise_steg_img_show 
    
    diff_w2co = _normalize(torch.abs(cover_show - steg_img_show))
    diff_w2no = _normalize(torch.abs(secret_show - secret_rev_show))
    
    stacked_images = torch.cat(
        [cover_show.unsqueeze(0), 
         secret_show.unsqueeze(0), 
         steg_img_show.unsqueeze(0), 
         secret_rev_show.unsqueeze(0), 
         noise_steg_img_show.unsqueeze(0),
         diff_w2co.unsqueeze(0), diff_w2no.unsqueeze(0)], dim=0)
    shape = stacked_images.shape
    stacked_images = stacked_images.permute(0, 3, 1, 4, 2).reshape(shape[3] * shape[0], shape[4] * shape[1], shape[2])
    
    # Add 0.5 after unnormalizing to [0, 255] to round to nearest integer
    stacked_images = stacked_images.mul(255).add_(0.5).clamp_(0, 255).to('cpu', torch.uint8).numpy()
    filename = os.path.join(folder, 'epoch-{}.png'.format(epoch))
    saved_image = Image.fromarray(np.array(stacked_images, dtype=np.uint8)).convert("RGB")
    saved_image.save(filename)


def o2t(x):
    return torch.clamp(x * 2 - 1, -1, 1)

def t2o(x):
    return torch.clamp(0.5 * (x + 1), 0, 1)


def get_obj_from_str(string: str, reload: bool = False) -> Any:
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


def instantiate_from_config(config: Mapping[str, Any]) -> Any:
    if not "target" in config:
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", dict()))
