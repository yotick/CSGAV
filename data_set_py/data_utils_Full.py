from os import listdir
from os.path import join
import torch
from PIL import Image
from torch.utils.data.dataset import Dataset
from torchvision.transforms import Compose, RandomCrop, ToTensor, ToPILImage, CenterCrop, Resize, Grayscale

# from general import histMatch
from Pansharpening_Toolbox_Assessment_Python.MTF import lu_MTF
from data_set_py.imagecrop import FusionRandomCrop
from torchvision.transforms import functional as F
import numpy as np
from data_set_py.transforms import Stretch
from torch.nn import functional as FC


def data_transform():
    # return Compose([ToTensor(), Stretch()])
    return Compose([Stretch()])


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in
               ['.tif', '.bmp', '.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG'])


def calculate_valid_crop_size(crop_size, upscale_factor):
    return crop_size - (crop_size % upscale_factor)


def tif_to_gray(image):
    # img = np.array(image)
    w, h, c = image.shape
    intensity = np.zeros((w, h))
    for i in range(c):
        intensity = intensity + 1 / c * image[:, :, i]

    # img = Image.fromarray(intensity)
    return intensity


def get_detail_gt(ms, gt):
    detail_1 = gt - ms
    detail_2 = (detail_1 + 0.5) / 2
    return detail_2


def train_rand_crop(crop_size):
    return Compose([
        FusionRandomCrop(crop_size),
    ])


def train_gray_transform():
    return Compose([
        Grayscale(num_output_channels=3),
        ToTensor(),
    ])


def train_lr_transform(crop_size, upscale_factor):
    return Compose([
        ToPILImage(),
        Resize(crop_size // upscale_factor, interpolation=Image.BICUBIC),
        ToTensor()
    ])


def display_transform():
    return Compose([
        ToPILImage(),
        #   Resize(400),
        #  CenterCrop(400),
        # Grayscale(),
        ToTensor()
    ])


def pil_crop_transform(pil, crop_size):
    img_crop = F.crop(pil, crop_size[0], crop_size[1], crop_size[2], crop_size[3])
    return ToTensor()(img_crop)


class TrainDatasetFromFolder_Full(Dataset):  # for train datasets
    def __init__(self, dataset_dir, sate, crop_size, upscale_factor):
        super(TrainDatasetFromFolder_Full, self).__init__()
        self.upscale_factor = upscale_factor
        self.gray_transform = train_gray_transform()
        self.rand_crop_trans = train_rand_crop(crop_size)

        self.sate = sate
        self.sensor = sate

        if sate == 'wv3_8':  # for train datasets
            self.sensor = 'WV3'

            pan1024_path = join(dataset_dir, 'data2017/DIV2K_train_HR/train_img7/PAN/PAN1024')

            nir1_256_path = join(dataset_dir, 'data2017/DIV2K_train_HR/train_img7/NIR1/NIR1256')
            nir2_256_path = join(dataset_dir, 'data2017/DIV2K_train_HR/train_img7/NIR2/NIR2256')
            rgb256_path = join(dataset_dir, 'data2017/DIV2K_train_HR/train_img7/RGB/RGB256')
            coastbl256_path = join(dataset_dir, 'data2017/DIV2K_train_HR/train_img7/CoastalBlue/CoastalBlue256')
            rededge256_path = join(dataset_dir, 'data2017\DIV2K_train_HR/train_img7\RedEdge\RedEdge256')
            yellow256_path = join(dataset_dir, 'data2017\DIV2K_train_HR\\train_img7\Yellow\Yellow256')

            self.nir1_256_file_name = [join(nir1_256_path, x.split('.')[0]) for x in listdir(nir1_256_path) if
                                       is_image_file(x)]
            self.nir2_256_file_name = [join(nir2_256_path, x.split('.')[0]) for x in listdir(nir2_256_path) if
                                       is_image_file(x)]
            self.coastbl256_file_name = [join(coastbl256_path, x.split('.')[0]) for x in listdir(coastbl256_path) if
                                         is_image_file(x)]
            self.yellow256_file_name = [join(yellow256_path, x.split('.')[0]) for x in listdir(yellow256_path) if
                                        is_image_file(x)]
            self.rededge256_file_name = [join(rededge256_path, x.split('.')[0]) for x in listdir(rededge256_path) if
                                         is_image_file(x)]

        if sate == 'ik':  # for train datasets
            self.sensor = 'IKONOS'
            pan1024_path = join(dataset_dir, 'data2017/DIV2K_valid_HR/test_img4/PAN/PAN1024')
            nir256_path = join(dataset_dir, 'data2017/DIV2K_valid_HR/test_img4/NIR/NIR256')
            rgb256_path = join(dataset_dir, 'data2017/DIV2K_valid_HR/test_img4/RGB/RGB256')
            self.nir256_file_name = [join(nir256_path, x.split('.')[0]) for x in listdir(nir256_path) if
                                     is_image_file(x)]
        if sate == 'pl':  # for train datasets
            pan1024_path = join(dataset_dir, 'data2017/DIV2K_valid_HR/test_img3/PAN/PAN1024')
            nir256_path = join(dataset_dir, 'data2017/DIV2K_valid_HR/test_img3/NIR/NIR256')
            rgb256_path = join(dataset_dir, 'data2017/DIV2K_valid_HR/test_img3/RGB/RGB256')
            self.nir256_file_name = [join(nir256_path, x.split('.')[0]) for x in listdir(nir256_path) if
                                     is_image_file(x)]
            # target_path = join(dataset_dir, 'Pleiades\\pl_gt256\\')
            # ms_r_up_path = join(dataset_dir, 'Pleiades\\pl_up_ms\\')

        self.pan1024_file_name = [join(pan1024_path, x.split('.')[0]) for x in listdir(pan1024_path) if
                                  is_image_file(x)]
        self.rgb256_file_name = [join(rgb256_path, x.split('.')[0]) for x in listdir(rgb256_path) if is_image_file(x)]

        # self.target_file_name = [join(target_path, x.split('.')[0]) for x in listdir(target_path) if
        #                          is_image_file(x)]
        # self.ms_r_up_file_name = [join(ms_r_up_path, x.split('.')[0]) for x in listdir(ms_r_up_path) if
        #                           is_image_file(x)]

    def __getitem__(self, index):  # for train datasets
        pan1024 = (Image.open('%s.tif' % self.pan1024_file_name[index]))
        rgb256 = Image.open('%s.tif' % self.rgb256_file_name[index])

        rgb_up = rgb256.resize((1024, 1024), Image.BICUBIC)
        rgb_up_near = rgb256.resize((1024, 1024), Image.NEAREST)

        crop_size = self.rand_crop_trans(pan1024)  #
        pan_crop = ToTensor()(F.crop(pan1024, crop_size[0], crop_size[1], crop_size[2], crop_size[3]))
        # rgb256_crop = ToTensor()(F.crop(rgb256, crop_size[0], crop_size[1], crop_size[2], crop_size[3]))
        rgb_up_crop = ToTensor()(F.crop(rgb_up, crop_size[0], crop_size[1], crop_size[2], crop_size[3]))
        rgb_near_crop = ToTensor()(F.crop(rgb_up_near, crop_size[0], crop_size[1], crop_size[2], crop_size[3]))

        if self.sate == 'pl' or self.sate == 'ik':  # for validation datasets
            nir256 = Image.open('%s.tif' % self.nir256_file_name[index])

            nir_up = nir256.resize((1024, 1024), Image.BICUBIC)
            nir_up_near = nir256.resize((1024, 1024), Image.NEAREST)

            nir256_crop = ToTensor()(nir256)
            nir_up_crop = ToTensor()(nir_up)
            nir_near_crop = ToTensor()(nir_up_near)

            # gt_crop = torch.cat([rgb256_crop, nir256_crop])
            # ms_up_crop = torch.cat([rgb_up_crop, nir_up_crop])
            # ms_near_crop = torch.cat([rgb_near_crop, nir_near_crop])

        if self.sate == 'wv3_8':  # for train datasets

            nir1_256 = Image.open('%s.tif' % self.nir1_256_file_name[index])
            nir2_256 = Image.open('%s.tif' % self.nir2_256_file_name[index])
            coastbl256 = Image.open('%s.tif' % self.coastbl256_file_name[index])
            rededge256 = Image.open('%s.tif' % self.rededge256_file_name[index])
            yellow256 = Image.open('%s.tif' % self.yellow256_file_name[index])

            # nir1_up = nir1_256.resize((1024, 1024), Image.BICUBIC)
            # coastbl_up = coastbl256.resize((1024, 1024), Image.BICUBIC)
            # rededge_up = rededge256.resize((1024, 1024), Image.BICUBIC)
            # yellow_up = yellow256.resize((1024, 1024), Image.BICUBIC)
            # nir2_up = nir2_256.resize((1024, 1024), Image.BICUBIC)

            nir1_up_near = nir1_256.resize((1024, 1024), Image.NEAREST)  # for training datasets
            coastbl_up_near = coastbl256.resize((1024, 1024), Image.NEAREST)
            rededge_up_near = rededge256.resize((1024, 1024), Image.NEAREST)
            yellow_up_near = yellow256.resize((1024, 1024), Image.NEAREST)
            nir2_up_near = nir2_256.resize((1024, 1024), Image.NEAREST)

            # nir1_256_crop = ToTensor()(F.crop(nir1_256, crop_size[0], crop_size[1], crop_size[2], crop_size[3]))
            # nir1_up_crop = ToTensor()(F.crop(nir1_up, crop_size[0], crop_size[1], crop_size[2], crop_size[3]))
            nir1_near_crop = ToTensor()(F.crop(nir1_up_near, crop_size[0], crop_size[1], crop_size[2], crop_size[3]))

            # coastbl256_crop = ToTensor()(F.crop(coastbl256, crop_size[0], crop_size[1], crop_size[2], crop_size[3]))
            # coastbl_up_crop = ToTensor()(F.crop(coastbl_up, crop_size[0], crop_size[1], crop_size[2], crop_size[3]))
            coastbl_near_crop = ToTensor()(
                F.crop(coastbl_up_near, crop_size[0], crop_size[1], crop_size[2], crop_size[3]))
            # for train datasets
            # rededge256_crop = ToTensor()(F.crop(rededge256, crop_size[0], crop_size[1], crop_size[2], crop_size[3]))
            # rededge_up_crop = ToTensor()(F.crop(rededge_up, crop_size[0], crop_size[1], crop_size[2], crop_size[3]))
            rededge_near_crop = ToTensor()(
                F.crop(rededge_up_near, crop_size[0], crop_size[1], crop_size[2], crop_size[3]))

            # yellow256_crop = ToTensor()(F.crop(yellow256, crop_size[0], crop_size[1], crop_size[2], crop_size[3]))
            # yellow_up_crop = ToTensor()(F.crop(yellow_up, crop_size[0], crop_size[1], crop_size[2], crop_size[3]))
            yellow_near_crop = ToTensor()(
                F.crop(yellow_up_near, crop_size[0], crop_size[1], crop_size[2], crop_size[3]))

            # nir2_256_crop = ToTensor()(F.crop(nir2_256, crop_size[0], crop_size[1], crop_size[2], crop_size[3]))
            # nir2_up_crop = ToTensor()(F.crop(nir2_up, crop_size[0], crop_size[1], crop_size[2], crop_size[3]))
            nir2_near_crop = ToTensor()(F.crop(nir2_up_near, crop_size[0], crop_size[1], crop_size[2], crop_size[3]))

            # gt_crop = torch.cat(
            #     [rgb256_crop, nir1_256_crop, coastbl256_crop, rededge256_crop, yellow256_crop, nir2_256_crop])
            # ms_up_crop = torch.cat(
            #     [rgb_up_crop, nir1_up_crop, coastbl_up_crop, rededge_up_crop, yellow_up_crop, nir2_up_crop])
            ms_near_crop = torch.cat(
                [rgb_near_crop, nir1_near_crop, coastbl_near_crop, rededge_near_crop, yellow_near_crop,
                 nir2_near_crop])           # for train datasets
            # ms_64 = torch.cat([rgb64_t, nir1_64_t, coastbl64_t, rededge64_t, yellow64_t, nir2_64_t])

        ms_near_crop_t = ms_near_crop.unsqueeze(0)  # in order to use FC.inter
        size_n = int(ms_near_crop.size()[1] / 4)
        ms_org_crop_t = FC.interpolate(ms_near_crop_t, size=(size_n, size_n), mode='nearest')
        ms_org_crop = ms_org_crop_t.squeeze(0)
        # ms_org_crop = gt_crop

        return ms_org_crop, pan_crop

    def __len__(self):
        # return len(self.rgb256_file_name)
        return 1000


class TestDatasetFromFolder_Full(Dataset):  # for test datasets
    def __init__(self, dataset_dir, sate, upscale_factor):
        super(TestDatasetFromFolder_Full, self).__init__()
        self.upscale_factor = upscale_factor
        self.gray_transform = train_gray_transform()

        self.sate = sate
        self.sensor = sate

        if sate == 'wv3_8':  # for test datasets
            self.sensor = 'WV3'

            pan1024_path = join(dataset_dir, 'data2017/DIV2K_valid_HR\\test_img7\PAN\PAN1024')
            nir1_256_path = join(dataset_dir, 'data2017/DIV2K_valid_HR/test_img7/NIR1/NIR1256')
            nir2_256_path = join(dataset_dir, 'data2017/DIV2K_valid_HR/test_img7/NIR2/NIR2256')
            rgb256_path = join(dataset_dir, 'data2017/DIV2K_valid_HR/test_img7/RGB/RGB256')
            coastbl256_path = join(dataset_dir, 'data2017/DIV2K_valid_HR/test_img7/CoastalBlue/CoastalBlue256')
            rededge256_path = join(dataset_dir, 'data2017\DIV2K_valid_HR/test_img7\RedEdge\RedEdge256')
            yellow256_path = join(dataset_dir, 'data2017\DIV2K_valid_HR\\test_img7\Yellow\Yellow256')

            self.nir1_256_file_name = [join(nir1_256_path, x.split('.')[0]) for x in listdir(nir1_256_path) if
                                       is_image_file(x)]
            self.nir2_256_file_name = [join(nir2_256_path, x.split('.')[0]) for x in listdir(nir2_256_path) if
                                       is_image_file(x)]
            self.coastbl256_file_name = [join(coastbl256_path, x.split('.')[0]) for x in listdir(coastbl256_path) if
                                         is_image_file(x)]
            self.yellow256_file_name = [join(yellow256_path, x.split('.')[0]) for x in listdir(yellow256_path) if
                                        is_image_file(x)]
            self.rededge256_file_name = [join(rededge256_path, x.split('.')[0]) for x in listdir(rededge256_path) if
                                         is_image_file(x)]

        if sate == 'ik':  # for test datasets
            self.sensor = 'IKONOS'
            pan1024_path = join(dataset_dir, 'data2017/DIV2K_valid_HR/test_img4/PAN/PAN1024')
            nir256_path = join(dataset_dir, 'data2017/DIV2K_valid_HR/test_img4/NIR/NIR256')
            rgb256_path = join(dataset_dir, 'data2017/DIV2K_valid_HR/test_img4/RGB/RGB256')
            self.nir256_file_name = [join(nir256_path, x.split('.')[0]) for x in listdir(nir256_path) if
                                     is_image_file(x)]
        if sate == 'pl':  # for test datasets
            pan1024_path = join(dataset_dir, 'data2017/DIV2K_valid_HR/test_img3/PAN/PAN1024')
            nir256_path = join(dataset_dir, 'data2017/DIV2K_valid_HR/test_img3/NIR/NIR256')
            rgb256_path = join(dataset_dir, 'data2017/DIV2K_valid_HR/test_img3/RGB/RGB256')
            self.nir256_file_name = [join(nir256_path, x.split('.')[0]) for x in listdir(nir256_path) if
                                     is_image_file(x)]
            # target_path = join(dataset_dir, 'Pleiades\\pl_gt256\\')
            # ms_r_up_path = join(dataset_dir, 'Pleiades\\pl_up_ms\\')

        self.pan1024_file_name = [join(pan1024_path, x.split('.')[0]) for x in listdir(pan1024_path) if
                                  is_image_file(x)]
        self.rgb256_file_name = [join(rgb256_path, x.split('.')[0]) for x in listdir(rgb256_path) if is_image_file(x)]

        # self.target_file_name = [join(target_path, x.split('.')[0]) for x in listdir(target_path) if
        #                          is_image_file(x)]
        # self.ms_r_up_file_name = [join(ms_r_up_path, x.split('.')[0]) for x in listdir(ms_r_up_path) if
        #                           is_image_file(x)]

    def __getitem__(self, index):  # for test datasets
        pan1024 = Image.open('%s.tif' % self.pan1024_file_name[index])
        rgb256 = Image.open('%s.tif' % self.rgb256_file_name[index])

        rgb_up = rgb256.resize((1024, 1024), Image.BICUBIC)
        # rgb_up_near = rgb256.resize((1024, 1024), Image.NEAREST)

        pan_crop = ToTensor()(pan1024)
        rgb256_crop = ToTensor()(rgb256)
        rgb_up_crop = ToTensor()(rgb_up)
        # rgb_near_crop = ToTensor()(rgb_up_near)

        if self.sate == 'pl' or self.sate == 'ik':  # for validation datasets
            nir256 = Image.open('%s.tif' % self.nir256_file_name[index])

            nir_up = nir256.resize((1024, 1024), Image.BICUBIC)
            # nir_up_near = nir256.resize((1024, 1024), Image.NEAREST)

            nir256_crop = ToTensor()(nir256)
            nir_up_crop = ToTensor()(nir_up)
            # nir_near_crop = ToTensor()(nir_up_near)

            gt_crop = torch.cat([rgb256_crop, nir256_crop])
            ms_up_crop = torch.cat([rgb_up_crop, nir_up_crop])
            # ms_near_crop = torch.cat([rgb_near_crop, nir_near_crop])

        if self.sate == 'wv3_8':  # for testing datasets

            nir1_256 = Image.open('%s.tif' % self.nir1_256_file_name[index])
            nir2_256 = Image.open('%s.tif' % self.nir2_256_file_name[index])
            coastbl256 = Image.open('%s.tif' % self.coastbl256_file_name[index])
            rededge256 = Image.open('%s.tif' % self.rededge256_file_name[index])
            yellow256 = Image.open('%s.tif' % self.yellow256_file_name[index])

            nir1_up = nir1_256.resize((1024, 1024), Image.BICUBIC)
            coastbl_up = coastbl256.resize((1024, 1024), Image.BICUBIC)
            rededge_up = rededge256.resize((1024, 1024), Image.BICUBIC)
            yellow_up = yellow256.resize((1024, 1024), Image.BICUBIC)
            nir2_up = nir2_256.resize((1024, 1024), Image.BICUBIC)

            # nir1_up_near = nir1_64.resize((256, 256), Image.NEAREST)  # for validation datasets
            # coastbl_up_near = coastbl64.resize((256, 256), Image.NEAREST)
            # rededge_up_near = rededge64.resize((256, 256), Image.NEAREST)
            # yellow_up_near = yellow64.resize((256, 256), Image.NEAREST)
            # nir2_up_near = nir2_64.resize((256, 256), Image.NEAREST)

            # nir1_64_t = ToTensor()(nir1_64)
            nir1_256_crop = ToTensor()(nir1_256)
            nir1_up_crop = ToTensor()(nir1_up)
            # nir1_near_crop = ToTensor()(nir1_up_near)

            # coastbl64_t = ToTensor()(coastbl64)
            coastbl256_crop = ToTensor()(coastbl256)
            coastbl_up_crop = ToTensor()(coastbl_up)
            # coastbl_near_crop = ToTensor()(coastbl_up_near)
            # for testing datasets
            # rededge64_t = ToTensor()(rededge64)
            rededge256_crop = ToTensor()(rededge256)
            rededge_up_crop = ToTensor()(rededge_up)
            # rededge_near_crop = ToTensor()(rededge_up_near)

            # yellow64_t = ToTensor()((yellow64))
            yellow256_crop = ToTensor()(yellow256)
            yellow_up_crop = ToTensor()(yellow_up)
            # yellow_near_crop = ToTensor()(yellow_up_near)

            # nir2_64_t = ToTensor()(nir2_64)
            nir2_256_crop = ToTensor()(nir2_256)
            nir2_up_crop = ToTensor()(nir2_up)
            # nir2_near_crop = ToTensor()(nir2_up_near)

            gt_crop = torch.cat(
                [rgb256_crop, nir1_256_crop, coastbl256_crop, rededge256_crop, yellow256_crop, nir2_256_crop])
            ms_up_crop = torch.cat(
                [rgb_up_crop, nir1_up_crop, coastbl_up_crop, rededge_up_crop, yellow_up_crop, nir2_up_crop])
            # ms_near_crop = torch.cat(
            #     [rgb_near_crop, nir1_near_crop, coastbl_near_crop, rededge_near_crop, yellow_near_crop,
            #      nir2_near_crop])  # for testing datasets
            # ms_64 = torch.cat([rgb64_t, nir1_64_t, coastbl64_t, rededge64_t, yellow64_t, nir2_64_t])

        ms_org_crop = gt_crop
        size_n = ms_org_crop.size()[1]
        # MTF and downsample first, then upsample.
        ms_org_crop_n = ms_org_crop.numpy().transpose(1, 2, 0)
        ms_mtf = lu_MTF(ms_org_crop_n, self.sensor, 1 / 4)
        ms_mtf_t = ToTensor()(ms_mtf).unsqueeze(0).float()
        size_m = ms_mtf_t.size()[2] // 4
        ms_mtf_d = FC.interpolate(ms_mtf_t, size=(size_m, size_m), mode='nearest')
        ms_d_up_t = FC.interpolate(ms_mtf_d, size=(size_n, size_n), mode='bicubic')
        ms_d_up = ms_d_up_t.squeeze(0)

        # downsampling pan image too.
        pan_crop_t = pan_crop.unsqueeze(0)
        pan_d = FC.interpolate(pan_crop_t, size=(size_n, size_n), mode='bicubic')
        pan_d = pan_d.squeeze(0)

        # data = torch.cat([ms_gray_crop, pan_crop])
        # return ms_up_crop, detail_crop, detail_gt_crop
        return ms_up_crop, ms_org_crop, pan_crop, ms_d_up, pan_d

    def __len__(self):
        # return len(self.rgb256_file_name)
        return 100
