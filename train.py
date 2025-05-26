import argparse
import os

# import pytorch_ssim

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import numpy as np
import torch.optim as op
import torchvision.utils as utils
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch import nn
import torch
import torch.backends.cudnn as cudnn
from tensorboardX import SummaryWriter
from new_dataloader import TrainDatasetFromFolder, ValDatasetFromFolder
from os.path import join
import os
from img_index import ref_evaluate
import pandas as pd
from datetime import datetime
from Pansharpening_Toolbox_Assessment_Python.indexes_evaluation import indexes_evaluation
import time
import thop
from models.CSGAV import CSGAV
device = torch.device('cuda:0')



val_step = 10

parser = argparse.ArgumentParser(description='Train Super Resolution Models')
parser.add_argument('--crop_size', default=64, type=int, help='training images crop size')  ### need to change
parser.add_argument('--num_epochs', default=150, type=int, help='train epoch number')  # for 8 band 500
parser.add_argument('--lr', type=float, default=0.0001,  # for 8 band , need to change !!
                    help='Learning Rate. Default=0.01')  # for 8 band 0.0006, for 4 band half
parser.add_argument("--step", type=int, default=30, help="Sets the learning rate to the initial LR decayed by "
                                                          "momentum every n epochs, Default: n=500")
parser.add_argument("--log_path", type=str, default="training_results\\")
parser.add_argument("--hsi_channel", type=int, default=102)#102,145,144
parser.add_argument("--msi_channel", type=int, default=4)
parser.add_argument("--database", type=str, default='Houston')#Houston,Botswana,Pavia
parser.add_argument("--model", default='CSGAV', type=str, help='training images crop size')  ### need to change
# CSGAV
parser.add_argument('--batch_size', default=4, type=int, help='train epoch number')  ##

opt = parser.parse_args()
if opt.database == 'Botswana':
    opt.hsi_channel = 145
    opt.msi_channel = 4
if opt.database == 'Houston':
    opt.hsi_channel = 144
    opt.msi_channel = 4
if opt.database == 'Pavia':
    opt.hsi_channel = 102
    opt.msi_channel = 4


cwd = os.getcwd()
train_dir = 'dataset/{}/Train'.format(opt.database)
val_dir = 'dataset/{}/Test'.format(opt.database)  ### need to change here   #####


if opt.model == 'CSGAV': #
    model = CSGAV(opt.hsi_channel, opt.msi_channel, 5, embed_dim=96).cuda()

CROP_SIZE = opt.crop_size  # 裁剪会带来拼尽问题嘛
NUM_EPOCHS = opt.num_epochs  # 轮数
BATCH_SIZE = opt.batch_size

train_set = TrainDatasetFromFolder(train_dir)  # 训练集导入
val_set = ValDatasetFromFolder(val_dir)  # 测试集导入
train_loader = DataLoader(dataset=train_set, num_workers=0, batch_size=BATCH_SIZE, shuffle=True)  # 训练集制作
val_loader = DataLoader(dataset=val_set, num_workers=0, batch_size=4, shuffle=False)


# ================== Pre-Define =================== #
SEED = 15
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
# cudnn.benchmark = True  ###自动寻找最优算法
cudnn.deterministic = True


# print('# generator parameters:', sum(param.numel() for param in model.parameters()))
# 计算模型中所有需要学习的参数数量
num_params = sum(p.numel() for p in model.parameters())
print("Total number of parameters: {:.3f}M".format(num_params / 1e6))
# 创建随机输入张量，只需要指定形状即可
input1 = torch.randn(1, opt.hsi_channel, CROP_SIZE //4, CROP_SIZE // 4).cuda()
input2 = torch.randn(1, opt.msi_channel, CROP_SIZE, CROP_SIZE).cuda()

# 计算FLOPs和参数数量
flops, params = thop.profile(model, inputs=(input1, input2))
print("FLOPs: {:.2f}G, Params: {:.3f}M".format(flops / 1e9, params / 1e6))

optimizerG = op.Adam(model.parameters(), lr=opt.lr, betas=(0.9, 0.999))


#### mse_loss, tv_loss, lap_loss, total_loss ####
# results = {'d_loss': [], 'g_loss': [], 'd_score': [], 'g_score': [], 'psnr': [], 'ssim': []}
results = {'time': [], 'epoch': [], 'lr': [], 'mse_loss': [], 'l1_loss_d': [], 'l1_loss': [], 'total_loss': [],
           'psnr': [],
           'sam': [],
           'ergas': [], 'scc': [], 'q_avg': [], 'q2n': [],'FLOPS': [],'Params': []}
df = pd.DataFrame(results)
out_path = 'training_results/' # 输出路径
writer = SummaryWriter()
count = 0
lr = opt.lr
#########  start from  d models ########################
log_dir = ''


if os.path.exists(log_dir):
    checkpoint = torch.load(log_dir)
    model.load_state_dict(checkpoint)
    start_epoch = 134
    print('加载 epoch  成功！')
else:
    start_epoch = 0
    print('无保存模型，将从头开始训练！')

t = time.strftime("%Y%m%d%H%M")

for epoch in range(start_epoch + 1, opt.num_epochs + 1):
    train_bar = tqdm(train_loader)  # 进度条
    lr = 0.0001
    for param_group in optimizerG.param_groups:
        param_group["lr"] = lr

    running_results = {'mse_loss': 0, 'l1_loss_d': 0, 'l1_loss': 0, 'total_loss': 0, 'batch_sizes': 0}

    model.train()
    # model.eval()
    for hsi, msi, gt_crop in train_bar:
        g_update_first = True
        batch_size = msi.size(0)
        running_results['batch_sizes'] += batch_size  # pytorch batch_size 不一定一致
        hsi = Variable(hsi)
        msi = Variable(msi)
        gt = Variable(gt_crop)

        if torch.cuda.is_available():
            msi = msi.cuda()
            hsi = hsi.cuda()
            gt = gt.cuda()
        target = gt
        optimizerG.zero_grad()  # change
        l1 = nn.L1Loss()
        criterion = nn.MSELoss()
        out = model(hsi, msi)
        total_loss = l1(out, target)
        ###########################
        total_loss.requires_grad_(True)
        total_loss.backward()
        optimizerG.step()

        #### mse_loss, tv_loss, lap_loss, total_loss ####
        running_results['total_loss'] += total_loss.item() * batch_size

        train_bar.set_description(desc='lr:%f [%d/%d]   total_loss: %.5f' % (
            lr, epoch, NUM_EPOCHS,running_results['total_loss'] / running_results['batch_sizes']))
        writer.add_scalar('total_loss', running_results['total_loss'] / running_results['batch_sizes'], count)

        count += 1
    time_curr = "%s" % datetime.now()  # 获取当前时间
    model_path = 'model_trained/' + opt.database
    if not os.path.exists(model_path):
        os.makedirs(model_path, exist_ok=True)
    if epoch>0:
        torch.save(model.state_dict(),
                   'model_trained/' + opt.database + '/' + opt.model + '_epoch_%03d.pth' % epoch)  # 存储网络参数
    model.eval()
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    ####### 验证集 #################

    if epoch % val_step == 0 and epoch>0:
        val_bar = tqdm(val_loader)  # 验证集的进度条
        valing_results = {'mse': 0, 'ssims': 0, 'psnr': 0, 'ssim': 0, 'batch_sizes': 0}

        l_psnr = []
        l_sam = []
        l_ergas = []
        l_scc = []
        l_q = []
        l_q2n = []
        C_q = []
        for hsi,msi, gt in val_bar:
            batch_size = msi.size(0)
            size_n = hsi.shape[2]
            valing_results['batch_sizes'] += batch_size

            with torch.no_grad():  # validation
                msi = Variable(msi)
                hsi = Variable(hsi)
                if torch.cuda.is_available():
                    msi = msi.cuda()
                    hsi = hsi.cuda()
                _,out = model(hsi, msi)
            out = out.cpu()
            count = 0
            for i in range(batch_size):
                val_images = []
                val_out = out.data[i].cpu().squeeze(0)
                val_gt0 = gt.data[i].cpu().squeeze(0)

                val_fused = val_out
                val_gt = val_gt0

                val_rgb = val_fused[0:3]
                val_gt_rgb = val_gt[0:3]
                # val_images.extend([val_rgb.squeeze(0), val_gt_rgb.squeeze(0)])
                val_images.extend([val_rgb.squeeze(0), val_gt_rgb.squeeze(0)])

                ##############  index evaluation ######################
                val_gt_np = val_gt.numpy().transpose(1, 2, 0)
                val_fused_np = val_fused.numpy().transpose(1, 2, 0)

                val_images = utils.make_grid(val_images, nrow=2, padding=5)
                # utils.save_image(val_images, out_path + opt.database + '/images/' + opt.model + '_tensor_%d.tif' % i)

                [c_psnr, c_ssim, c_sam, c_ergas, c_scc, c_q] = ref_evaluate(val_fused_np, val_gt_np)
                # [Q2n_index, Q_index, ERGAS_index, SAM_index] = [0, 0, 0, 0]
                [Q2n_index, Q_index, ERGAS_index, SAM_index] = indexes_evaluation(val_fused_np, val_gt_np, 4, 8, 32, 1,
                                                                                  1,                                                                        1)
                l_psnr.append(c_psnr)
                l_sam.append(SAM_index)
                l_ergas.append(ERGAS_index)
                l_scc.append(c_scc)
                l_q.append(Q_index)
                l_q2n.append(Q2n_index)
                C_q.append(c_q)
        psnr_avg = np.mean(l_psnr)
        sam_avg = np.mean(l_sam)
        ergas_avg = np.mean(l_ergas)
        scc_avg = np.mean(l_scc)
        q_avg = np.mean(l_q)
        q2n_avg = np.mean(l_q2n)
        c_qavg = np.mean(C_q)
        #### save to excel  ####
        results['time'].append(time_curr)
        results['epoch'].append(epoch)
        results['lr'].append(lr)
        results['total_loss'].append(running_results['total_loss'] / running_results['batch_sizes'])
        results['psnr'].append(psnr_avg)
        results['sam'].append(sam_avg)
        results['ergas'].append(ergas_avg)
        results['scc'].append(scc_avg)
        results['q_avg'].append(q_avg)
        results['q2n'].append(q2n_avg)
        results['FLOPS'].append(flops / 1e9)
        results['Params'].append(params / 1e6)
        print(
            'psnr:{:.4f}, sam:{:.4f}, ergas:{:.4f}, scc:{:.4f}, q:{:.4f},q2n:{:.4f},c_q:{:.4f}'.format(psnr_avg, sam_avg,
                                                                                            ergas_avg,
                                                                                            scc_avg, q_avg, q2n_avg,c_qavg))
        df = pd.DataFrame(results)  ###############################  need to change!!!
        path = out_path + opt.database
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)
        df.to_excel(out_path + opt.database + '/' + opt.model+str(opt.batch_size) + f'_{t}.xlsx', index=False)  #### need to change here!!!
writer.close()
