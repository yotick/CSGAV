from os.path import join
import torch
from torch.utils.data.dataset import Dataset
import h5py
import os
import glob
def normal(input_tensor):
    input_tensor = input_tensor.float()
    min_value = input_tensor.min()
    max_value = input_tensor.max()
    if min_value<0:
        min_value = 0
    # 进行归一化
    normalized_tensor = (input_tensor - min_value) / (max_value - min_value)
    return normalized_tensor

class TrainDatasetFromFolder(Dataset):
    def __init__(self, dataset_dir):
        super(TrainDatasetFromFolder, self).__init__()
        self.gt_file = join(dataset_dir, 'hrhsi')
        self.hsi_file = join(dataset_dir, 'hsi')
        self.msi_file = join(dataset_dir, 'msi')
        gt_mat = glob.glob(os.path.join(self.gt_file, '*.mat'))
        self.gt_path = [os.path.basename(mat) for mat in gt_mat]

        hsi_mat = glob.glob(os.path.join(self.hsi_file, '*.mat'))
        self.hsi_path = [os.path.basename(mat) for mat in hsi_mat]

        msi_mat = glob.glob(os.path.join(self.msi_file, '*.mat'))
        self.msi_path = [os.path.basename(mat) for mat in msi_mat]



    def __getitem__(self, index):  # for train datasets
        msi_data =  h5py.File(join(self.msi_file,self.msi_path[index]))
        msi = msi_data['block'][:]
        hsi_data =  h5py.File(join(self.hsi_file,self.hsi_path[index]))
        hsi = hsi_data['block'][:]
        gt_data =  h5py.File(join(self.gt_file,self.gt_path[index]))
        gt = gt_data['block'][:]
        msi = torch.from_numpy(msi)
        hsi = torch.from_numpy(hsi)
        gt = torch.from_numpy(gt)
        hsi = normal(hsi)
        msi = normal(msi)
        gt = normal(gt)
        return hsi, msi, gt

    def __len__(self):  # training
        return len(self.msi_path)

class ValDatasetFromFolder(Dataset):  # for validation datasets
    def __init__(self, dataset_dir):
        super(ValDatasetFromFolder, self).__init__()
        self.gt_file = join(dataset_dir, 'hrhsi')
        self.hsi_file = join(dataset_dir, 'hsi')
        self.msi_file = join(dataset_dir, 'msi')
        gt_mat = glob.glob(os.path.join(self.gt_file, '*.mat'))
        self.gt_path = [os.path.basename(mat) for mat in gt_mat]

        hsi_mat = glob.glob(os.path.join(self.hsi_file, '*.mat'))
        self.hsi_path = [os.path.basename(mat) for mat in hsi_mat]

        msi_mat = glob.glob(os.path.join(self.msi_file, '*.mat'))
        self.msi_path = [os.path.basename(mat) for mat in msi_mat]

    def __getitem__(self, index):  # for validation datasets
        msi_data =  h5py.File(join(self.msi_file,self.msi_path[index]))
        msi = msi_data['block'][:]
        hsi_data =  h5py.File(join(self.hsi_file,self.hsi_path[index]))
        hsi = hsi_data['block'][:]
        gt_data =  h5py.File(join(self.gt_file,self.gt_path[index]))
        gt = gt_data['block'][:]
        msi = torch.from_numpy(msi)
        hsi = torch.from_numpy(hsi)
        gt = torch.from_numpy(gt)
        hsi = normal(hsi)
        msi = normal(msi)
        gt = normal(gt)
        return hsi, msi, gt

    def __len__(self):  # for validation datasets
        return 4

class TestDatasetFromFolder(Dataset):  # for validation datasets
    def __init__(self,dataset_dir):
        super(TestDatasetFromFolder, self).__init__()
        self.gt_file = join(dataset_dir, 'hrhsi')
        self.hsi_file = join(dataset_dir, 'hsi')
        self.msi_file = join(dataset_dir, 'msi')
        gt_mat = glob.glob(os.path.join(self.gt_file, '*.mat'))
        self.gt_path = [os.path.basename(mat) for mat in gt_mat]

        hsi_mat = glob.glob(os.path.join(self.hsi_file, '*.mat'))
        self.hsi_path = [os.path.basename(mat) for mat in hsi_mat]

        msi_mat = glob.glob(os.path.join(self.msi_file, '*.mat'))
        self.msi_path = [os.path.basename(mat) for mat in msi_mat]

    def __getitem__(self, index):  # for validation datasets
        msi_data = h5py.File(join(self.msi_file, self.msi_path[index]))
        msi = msi_data['block'][:]
        hsi_data = h5py.File(join(self.hsi_file, self.hsi_path[index]))
        hsi = hsi_data['block'][:]
        gt_data = h5py.File(join(self.gt_file, self.gt_path[index]))
        gt = gt_data['block'][:]
        msi = torch.from_numpy(msi)
        hsi = torch.from_numpy(hsi)
        gt = torch.from_numpy(gt)
        hsi = normal(hsi)
        msi = normal(msi)
        gt = normal(gt)
        return hsi, msi, gt

    def __len__(self):  # for validation datasets
        return len(self.msi_path)

