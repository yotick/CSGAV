import argparse
import os
from tqdm import tqdm
import numpy as np
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
from new_dataloader import TestDatasetFromFolder
from models.CSGAV import CSGAV
from save_image_func import save_image_RS

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def test(test_data_loader, model, sate):
    val_bar = tqdm(test_data_loader)  # 验证集的进度条
    count = 0
    for hsi, msi, gt_crop in val_bar:
        batch_size = msi.size(0)  # current batch size


        with torch.no_grad():  # validation
            hsi = Variable(hsi)
            msi = Variable(msi)
            if torch.cuda.is_available():
                model.cuda()
                hsi = hsi.cuda()
                msi = msi.cuda()
            out = model(hsi, msi)
        output = out.cpu()

        for i in range(batch_size):
            count += 1
            image = output.data[i]
            gt = gt_crop.data[i]
            image = np.transpose(image.numpy(), (1, 2, 0))
            gt = np.transpose(gt.numpy(), (1, 2, 0))
            save_image_RS(opt.model,opt.model,sate, count, image)
            save_image_RS(opt.model,opt.model,sate + 'gt', count, gt)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch Super Res Example')
    parser.add_argument('--ratio', type=int, default=4)  # ratio here
    parser.add_argument('--testBatchSize', type=int, default=10, help='testing batch size')
    parser.add_argument('--threads', type=int, default=0, help='number of threads for data loader to use')
    parser.add_argument('--cuda', default=True, help='use cuda?')
    parser.add_argument('--model', default='CSGAV', help='use cuda?')
    #CSGAV
    parser.add_argument('--dataset', default='Houston', help='use cuda?')#Houston,Botswana,Pavia
    parser.add_argument("--msi_channel", type=int, default=4)
    parser.add_argument("--hsi_channel", type=int, default=144)#102,144,145
    parser.add_argument('--batch_size', default=4, type=int, help='train epoch number')  ### need to change
    parser.add_argument('--rate', type=float, default='1.0')  #
    opt = parser.parse_args()
    if opt.dataset == 'Botswana':
        opt.hsi_channel = 145
        opt.msi_channel = 4
    if opt.dataset == 'Houston':
        opt.hsi_channel = 144
        opt.msi_channel = 4
    if opt.dataset == 'Pavia':
        opt.hsi_channel = 102
        opt.msi_channel = 4



    if opt.model == 'CSGAV':  #
        model = CSGAV(opt.hsi_channel, opt.msi_channel, embed_dim=96).cuda()

    # 计算FLOPs和参数数量
    CROP_SIZE = 256  # 裁剪会带来拼尽问题嘛
    BATCH_SIZE = 1
    num_chanel = 4

    opt.checkpoint = '\\{}_epoch_150.pth'.format(opt.model)
    model_path = 'model_trained/'
    dataset_dir = os.path.join('dataset',opt.dataset,'test')
    # dataset_dir = R'D:\cbw\funion\高光谱数据集\{}\test'.format(opt.database)

    test_set = TestDatasetFromFolder(dataset_dir)  # 测试集导入
    test_data_loader = DataLoader(dataset=test_set, num_workers=opt.threads, batch_size=opt.testBatchSize,
                                  shuffle=False)
    data = opt.dataset
    checkpoint = torch.load(model_path + data + opt.checkpoint,map_location=lambda storage, loc: storage)
    model.load_state_dict(checkpoint, False)
    test(test_data_loader, model, opt.dataset)
