from os import listdir
from os.path import join
# import cv2 as cv
import numpy as np
import scipy.io as sio
from PIL import Image
import torchvision.utils as utils
from torchvision.transforms import ToTensor
import torch
import os

def save_image_RS(modelname,pngname,sate, count, fusedinput):
    if 'Botswana' in sate :
        fused_rgb = fusedinput[:, :, [10,30,40]]
    elif 'Houston' in sate:
        fused_rgb = fusedinput[:, :, [80,20,100]]
    elif 'Pavia'in sate:
        fused_rgb = fusedinput[:, :, [80,20,100]]#bost:[10,30,40],Houston:[20,30,60],pavia:[80,20,100]
    fused_ms = fusedinput
    path = os.getcwd()
    fusedpath = join(path,r'fused/{}/{}/{}'.format(sate.replace('gt', ''),sate,modelname))
    if not os.path.exists(fusedpath):
        # 如果不存在，创建目录
        os.makedirs(fusedpath, exist_ok=True)
    rgbpath = join(path,r'fused/{}/{}/{}'.format(sate.replace('gt', ''),sate+'rgb',modelname))
    if not os.path.exists(rgbpath):
        # 如果不存在，创建目录
        os.makedirs(rgbpath, exist_ok=True)
    save_path_fused = join(fusedpath, pngname+'_%04d.mat' % count)  ## need to change here
    save_path_rgb = join(rgbpath, pngname+'_%04d.png'% count)  ## need to change here
    fused_rgb = ToTensor()(fused_rgb)
    utils.save_image(fused_rgb, save_path_rgb)
    fused = ToTensor()(fused_ms)
    ndarr = fused.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
    sio.savemat(save_path_fused, {'fused': ndarr})

