# GPL License
# Copyright (C) 2021 , UESTC
# All Rights Reserved
# @Author  : Xiao Wu, LiangJian Deng
# @reference:
import torch.utils.data as data
import torch
import h5py
import cv2
import numpy as np


class Dataset_Pro(data.Dataset):
    def __init__(self, file_path, img_scale, max_samples=None):
        super(Dataset_Pro, self).__init__()
        data_r = h5py.File(file_path)  # NxCxHxW = 0x1x2x3

        print(f"loading Dataset_Pro: {file_path} with {img_scale}")
        # tensor type:
        ms1 = data_r["ms"][...]  # convert to np tpye for CV2.filter
        ms1 = np.array(ms1, dtype=np.float32) / img_scale
        self.ms = torch.from_numpy(ms1)

        gt1 = data_r["gt"][...]  # convert to np tpye for CV2.filter
        gt1 = np.array(gt1, dtype=np.float32) / img_scale
        self.gt = torch.from_numpy(gt1)  # NxCxHxW:

        lms1 = data_r["lms"][...]  # convert to np tpye for CV2.filter
        lms1 = np.array(lms1, dtype=np.float32) / img_scale
        self.lms = torch.from_numpy(lms1)

        pan1 = data_r['pan'][...]  # Nx1xHxW
        pan1 = np.array(pan1, dtype=np.float32) / img_scale  # Nx1xHxW
        self.pan = torch.from_numpy(pan1)  # Nx1xHxW:

        if max_samples is not None:
            self.gt = self.gt[:max_samples]
            self.ms = self.ms[:max_samples]
            self.lms = self.lms[:max_samples]
            self.pan = self.pan[:max_samples]

        # if 'valid' in file_path:
        #     self.gt = self.gt.permute([0, 2, 3, 1])

        print(f'pan:{self.pan.shape}', f'lms:{self.lms.shape}', f'gt:{self.gt.shape}', f'ms:{self.ms.shape}')

    #####必要函数
    def __getitem__(self, index):

        return {'gt': self.gt[index, :, :, :].float(),
                'lms': self.lms[index, :, :, :].float(),
                'ms': self.ms[index, :, :, :].float(),
                'pan': self.pan[index, :, :, :].float()}

        #####必要函数

    def __len__(self):
        return self.gt.shape[0]
