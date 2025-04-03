###############################################
# This code version belongs to Professor Lu Hangyuan
# of Jinhua University of Vocational Technology.
###############################################
import argparse
import os
from tqdm import tqdm
import numpy as np
import torch
from data_set_py.psdata_h5 import PansharpeningSession
from save_image_func2 import save_image_RS
import time
from data_set_py.data_utils_RS_2 import TestDatasetFromFolder
from torch.utils.data import DataLoader
from models.model_pnn import PNN    ### need to change ######
import thop
import json
from data_set_py.HSI_datasets import *

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# 强制切换到 CPU
device = torch.device("cuda")
# device = torch.device("cpu")

def test(test_data_loader, model, args):
    test_bar = tqdm(test_data_loader)  # 验证集的进度条
    count = 0
    for data in test_bar:
        gt, lms, ms, pan = data['gt'].float().cuda(), data['lms'].float().cuda(), \
                           data['ms'].float().cuda(), data['pan'].float().cuda()

        start = time.time()

        out = model(lms, pan)  #### need to change!!!!!!!

        end = time.time()
        output = out.cpu()
        time_ave = (end - start) / args.samples_per_gpu
        print('Average testing time is', time_ave)

        for i in range(args.samples_per_gpu):
            # image = (output.data[i] + 1) / 2.0
            count += 1
            out_image = output.data[i]
            # image = image.mul(255).byte()
            out_image = np.transpose(out_image.numpy(), (1, 2, 0))
            # gt = np.transpose(gt.numpy(), (1, 2, 0))

            if args.spectral_num > 4:
                save_f_name = args.dataset + '_%03d.mat' % count
            else:
                save_f_name = args.dataset + '_%03d.tif' % count
            save_image_RS(args, save_f_name, out_image)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test Pansharpening Models')
    # parser.add_argument('--model_name', type=str, default='FusionNet', help='Name of the model to use')
    # parser.add_argument("--log_path", type=str, default="training_results\\")
    parser.add_argument('-c', '--config', default='data_set_py/config_HSIT_PRE.json', type=str,
                        help='Path to the config file')
    # parser.add_argument('--testBatchSize', type=int, default=10, help='testing batch size')
    args = parser.parse_args()

    #######  需修改,重要!!  load model, need to change here !!  #####################################
    # *******加载模型配置,需要修改模型名字!! *******************************

    ####  args.dataset in ['ik','pl', 'qb','wv3-8', 'wv3_8']
    #### args.dataset in ['botswana4', 'chikusei', 'pavia', 'FR1']:
    args.dataset = 'FR1'  #### need to change here !!  including ik pl wv3_8

    if "wv" in args.dataset:
        args.spectral_num = 8
    else:
        args.spectral_num = 4

    args.test_dataset = 'test_' + args.dataset
    args.samples_per_gpu = 1   #### test batch size
    args.workers_per_gpu = 0
    args.img_range = 2047.0
    BATCH_SIZE = args.samples_per_gpu

    #################################################
    pretrained = '_epoch_165.pth'  ####  need to change ##########
    # pretrained = '_pnn.pth'
    model_path = r'.\training_results/' + args.dataset + '/pretrained_models/' + args.dataset + pretrained
    dataset_dir = 'E:\\00 remote sense image fusion\\Source Data\\'  ### need to change here   #####

     #####  load dataset and PTH weight ######
    if args.dataset in ['ik','pl', 'qb','wv3-8', 'wv3_8']:  ######## load model and dataset of my own
        test_set = TestDatasetFromFolder(dataset_dir, args.dataset)  # 测试集导入
        test_loader = DataLoader(dataset=test_set, num_workers=args.workers_per_gpu, batch_size=args.samples_per_gpu,
                                 shuffle=False)
    elif args.dataset in ['botswana4', 'chikusei', 'pavia', 'FR1']:   ######## load HSI
        config = json.load(open(args.config))
        __dataset__ = {'botswana4': botswana4_dataset, "pavia": pavia_dataset,
                       "FR1": FR1_dataset, 'chikusei': chikusei_dataset}
        test_loader = DataLoader(__dataset__[args.dataset](dataset_dir, config, typeM='test'),
                                  batch_size=config["val_batch_size"], num_workers=config["num_workers"],
                                  shuffle=True, pin_memory=False, )
        args.spectral_num = config[args.dataset]['spectral_bands']

    else: ########### load model and dataset h5
        sess = PansharpeningSession(args)
        test_loader, _ = sess.get_eval_dataloader(dataset_dir, args.test_dataset, False)

   ##### load model ############
    model = PNN(args.spectral_num).cuda()  ### need to change!!!!
    model.eval()

    ###### The following need to chang ##############
    CROP_SIZE = 256
    # 计算模型中所有需要学习的参数数量
    input1 = torch.randn(1, args.spectral_num, CROP_SIZE, CROP_SIZE).to(device)  ## UPMS
    input2 = torch.randn(1, args.spectral_num, CROP_SIZE // 4, CROP_SIZE // 4).to(device)  # MS
    input3 = torch.randn(1, 1, CROP_SIZE, CROP_SIZE).to(device)  # PAN
    # 计算FLOPs和参数数量
    # 修复thop.profile返回值不匹配的问题
    # 修复thop.profile返回两个值而不是三个值的问题
    flops, params = thop.profile(model, inputs=(input1, input3))
    print("FLOPs: {:.2f}G, Params: {:.3f}M".format(flops / 1e9, params / 1e6))

    checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage)
    model.load_state_dict(checkpoint, False)

    test(test_loader, model, args)
