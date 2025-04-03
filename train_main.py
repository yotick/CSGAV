###############################################
# This code version belongs to Professor Lu Hangyuan
# of Jinhua University of Vocational Technology.
###############################################
import argparse
import torch
import torch.backends.cudnn as cudnn
from tensorboardX import SummaryWriter
from def_train import Train
from torch.utils.data import DataLoader
import time
from models.pnn_main import build_pnn    ### need to change ######
from data_set_py.data_utils_RS_2 import TrainDatasetFromFolder, ValDatasetFromFolder
from data_set_py.psdata_h5 import PansharpeningSession
import json
from data_set_py.HSI_datasets import *
import math

#### mse_loss, tv_loss, lap_loss, total_loss ####
# results = {'d_loss': [], 'g_loss': [], 'd_score': [], 'g_score': [], 'psnr': [], 'ssim': []}
results = {'time': [], 'epoch': [], 'lr': [], 'total_loss': [],
           'psnr': [], 'sam': [], 'ergas': [], 'scc': [], 'q_avg': [], 'q2n': []}

writer = SummaryWriter()
count = 0
# for epoch in range(1, NUM_EPOCHS + 1):

#########  start from saved models ########################
log_dir = ' '
# log_dir = './model_trained/pl/netG_pl_epoch_1_1230.pth'  # 模型保存路径


t = time.strftime("%Y%m%d%H%M")

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Train Pansharpening Models')
    # parser.add_argument('--model_name', type=str, default='FusionNet', help='Name of the model to use')
    # parser.add_argument("--log_path", type=str, default="training_results\\")
    parser.add_argument('-c', '--config', default='data_set_py/config_HSIT_PRE.json', type=str,
                        help='Path to the config file')
    args = parser.parse_args()

    # *******加载模型配置,需要修改模型名字!! *******************************
    ########## args.dataset in ['pl' , 'ik' , 'wv3-8' , 'wv3_8'] my own ############
    ########## args.dataset in ['botswana4' , 'chikusei' , 'pavia' , 'FR1'] ##########

    args.dataset = 'botswana4'  #### need to change here !!  DLJ h5 is wv3

    if "wv" in args.dataset:
        args.spectral_num = 8
    else:
        args.spectral_num = 4

    args.is_small_patch_train = False

    if args.dataset == 'botswana4':
        args.is_small_patch_train = True
        args.patch_size = 60  ### become smaller patch, 60 for botswana4, 64 for chikusei，80 for pavia
    elif args.dataset == 'chikusei':
        args.is_small_patch_train = True
        args.patch_size = 64  ### become smaller patch, 60 for botswana4, 64 for chikusei，80 for pavia
    elif args.dataset == 'pavia':
        args.is_small_patch_train = True
        args.patch_size = 80  ### become smaller patch, 60 for botswana4, 64 for chikusei，80 for pavia

    # 设置训练参数
    args.lr = 5e-4
    args.batch_size = 80  ## 设置train batch size
    args.samples_per_gpu = args.batch_size  ## 设置batch size
    # args.samples_per_gpu = 80  ## 设置batch size,视情况修改！！
    args.epochs = 300
    args.lr_step = 100   ### 调整lr的间隔
    args.val_step = 5  ### 验证集的频率
    args.log_dir = log_dir
    ###### The above need to chang ##############

    sate = args.dataset
    # args.num_channel = get_num_channel(args.dataset)  #### need to change here !!

    args.crop_size = 128   ### for my own dataset
    args.workers_per_gpu = 0
    args.val_dataset = 'valid_' + args.dataset  ### for h5 dataset
    args.img_range = 2047.0

    build_model = build_pnn()

    # ================== Pre-Define =================== #
    SEED = 15
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    # cudnn.benchmark = True  ###自动寻找最优算法
    cudnn.deterministic = True
    device = torch.device('cuda:0')

    # wv3 9714 16-64
    # wv2 15084 16-64
    # gf2 19809 16-64
    # qb  17139 16-64

    dataset_dir = 'E:\\00 remote sense image fusion\Source Data\\'  ### need to change here   #####

    if args.dataset in ['pl' , 'ik' , 'wv3-8' , 'wv3_8'] :      ####### load data of my own #########
        train_set = TrainDatasetFromFolder(dataset_dir, sate, crop_size=args.crop_size)  # 训练集导入
        val_set = ValDatasetFromFolder(dataset_dir, sate)  # 测试集导入
        train_loader = DataLoader(dataset=train_set, num_workers=0, batch_size=args.batch_size, shuffle=True)  # 训练集制作
        valid_loader = DataLoader(dataset=val_set, num_workers=0, batch_size=1, shuffle=False)

    elif args.dataset in ['botswana4' , 'chikusei' , 'pavia' , 'FR1']: ## PAN-HSI： botswana4, chikusei, pavia_centre,FR1
        config = json.load(open(args.config))
        __dataset__ = {'botswana4': botswana4_dataset,"pavia": pavia_dataset, "FR1": FR1_dataset, 'chikusei': chikusei_dataset}
        train_loader = DataLoader(__dataset__[args.dataset](dataset_dir, config, typeM='train'), batch_size = args.batch_size, num_workers=1, shuffle=True,
                                       pin_memory=False, )
        valid_loader = DataLoader(__dataset__[args.dataset](dataset_dir, config, typeM='val'), batch_size=config["val_batch_size"], num_workers=config["num_workers"],
                                      shuffle=True,  pin_memory=False, )
        args.spectral_num = config[args.dataset]['spectral_bands']

    else:       ####### load data of H5 #########
        sess = PansharpeningSession(args)
        dataset_dir1= dataset_dir+'MS-PAN-DLJ\\'
        train_loader, _ = sess.get_dataloader(dataset_dir1, args.dataset, False)
        valid_loader, _ = sess.get_eval_dataloader(dataset_dir1, args.val_dataset, False)
    ####### load data end   #########


    Train(train_loader, valid_loader, build_model, args)
    # print(len(train_loader))
