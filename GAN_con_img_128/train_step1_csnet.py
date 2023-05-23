from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from torchvision.utils import save_image
import torchvision.datasets as dset
import torchvision.transforms as transforms
# import torchvision.utils as vutils
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from csnet import CSNet_Lite, _netG2, _netRS,_netG2_block
from tqdm import tqdm
from data_utils import TrainDatasetFromFolder, TestDatasetFromFolder
# from calculate_fid_pytorch.fid import fid_score
import vutils

parser = argparse.ArgumentParser()
parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
parser.add_argument('--batchSize', type=int, default=30, help='input batch size')
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=0.0002')
parser.add_argument('--lrRS', type=float, default=0.001, help='learning rate, default=0.00001')
parser.add_argument('--beta1', type=float, default=0.9, help='beta1 for adam. default=0.5')
parser.add_argument('--netG2', default='', help="path to netG2 (to continue training)")
parser.add_argument('--netRS', default='', help="path to netRS (to continue training)")
parser.add_argument('--outf', default='img_128_3_tuple_csnet_subrate_0.0625+gan_con_32', help='folder to output images and model checkpoints')
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('--nc', type=int, default=3, help='the number of image channel')
parser.add_argument('--lamb', type=int, default=100, help='weight on L1 term in objective')
parser.add_argument('--sub_dir1', default='imgStep1_csnet_normalize_subrate_0.0625_contrast_128', help='the sub directory 1 of saving images')

parser.add_argument('--pickle_dir', type=str, default='csnet_lite', help='input path')
parser.add_argument('--train_stage', type=int, default=1, help='training stage index')

parser.add_argument('--num_epochs', default=100, type=int, help='train epoch number')
parser.add_argument('--loadEpoch', default=0, type=int, help='load epoch number')
parser.add_argument('--crop_size', default=96, type=int, help='training images crop size')
parser.add_argument('--block_size', default=32, type=int, help='CS block size')
parser.add_argument('--sub_rate', default=0.0625, type=float, help='sampling sub rate')
parser.add_argument('--generatorWeights', type=str, default='', help="path to CSNet weights (to continue training)")
opt = parser.parse_args()

CROP_SIZE = opt.crop_size
BLOCK_SIZE = opt.block_size
SUBRATE = opt.sub_rate
NUM_EPOCHS = opt.num_epochs
LOAD_EPOCH = 0



# 保存图片对比重建前后效果
img_save_path = os.path.join(opt.outf, opt.sub_dir1)
if not os.path.exists(img_save_path):
    os.makedirs(img_save_path)

pickle_save_path = 'CSNet_normalize_patchsize_96_3tuple' + '_subrate_' + str(opt.sub_rate) + '_blocksize_' + str(BLOCK_SIZE)
if not os.path.exists(pickle_save_path):
    os.makedirs(pickle_save_path)

# Use GPU is available else use CPU.
device = torch.device("cuda:0" if(torch.cuda.is_available()) else "cpu")
#print(device, " will be used.\n")


#导入训练数据集
train_set = TrainDatasetFromFolder('E:/img_64/patch_data_96/train', crop_size=CROP_SIZE, blocksize=BLOCK_SIZE)
train_loader = DataLoader(dataset=train_set, num_workers=4, batch_size=opt.batchSize, shuffle=True)
#导入测试数据集
#1024*1024
#data_dir_1024 = 'celeba_1024_lite/celeba_1024'          # this path depends on your computer
#dset_1024 = TestDatasetFromFolder(data_dir_1024, blocksize=BLOCKSIZE)
#512*512
data_dir_128 = 'celeba-128/celeba-128'          # this path depends on your computer
dset_128 = TestDatasetFromFolder(data_dir_128, blocksize=BLOCK_SIZE)


# design model
#32*32*64生成512*512的图片
netG2 = _netG2_block(blocksize=BLOCK_SIZE, subrate=SUBRATE)

#512*512压缩为32*32*64
netRS = _netRS(blocksize=BLOCK_SIZE, subrate=SUBRATE)

if opt.generatorWeights != '':
    netG2.load_state_dict(torch.load(opt.generatorWeights))
    netRS.load_state_dict(torch.load(opt.generatorWeights))
    LOAD_EPOCH = opt.loadEpoch


# design the criterion
criterion_l1 = nn.L1Loss()
criterion_l2 = nn.MSELoss()

netG2.cuda()
netRS.cuda()
criterion_l1.cuda()
criterion_l2.cuda()

# setup optimizer

optimizerG2 = optim.Adam(netG2.parameters(), lr=opt.lrRS, betas=(opt.beta1, 0.999))
optimizerRS = optim.Adam(netRS.parameters(), lr=opt.lrRS, betas=(opt.beta1, 0.999))


if __name__=="__main__":
    schedulerRS = torch.optim.lr_scheduler.MultiStepLR(optimizerRS, milestones=[50,80], gamma=0.1)
    schedulerG2 = torch.optim.lr_scheduler.MultiStepLR(optimizerG2, milestones=[50, 80], gamma=0.1)
    for epoch in range(LOAD_EPOCH, NUM_EPOCHS + 1):
        train_bar = tqdm(train_loader)
        running_results = {'batch_sizes': 0, 'loss': 0, }

        netG2.train()
        netRS.train()

        for data, target in train_bar:
            batch_size = data.size(0)
            if batch_size <= 0:
                continue

            running_results['batch_sizes'] += batch_size

            if torch.cuda.is_available():
                real_img = target.cuda()

            if torch.cuda.is_available():
                z = data.cuda()

            netRS.zero_grad()
            netG2.zero_grad()

            fake_img = netG2(netRS(z))

            errRS = criterion_l2(fake_img, real_img)
            errRS.backward()
            optimizerRS.step()
            optimizerG2.step()

            running_results['loss'] += errRS.item() * batch_size

            train_bar.set_description(desc='[%d] Loss: %.4f' % (epoch, running_results['loss'] / running_results['batch_sizes']))
        schedulerG2.step()
        schedulerRS.step()

        #图片重构对比
        test_img_real = dset_128[1][0].cuda()
        if epoch == 0:
            save_image(test_img_real, '%s/%s/real_samples.png' % (opt.outf, opt.sub_dir1), nrow=1)

        if epoch % 5 ==0:
            fake = netG2(netRS(test_img_real.view(1, 3, 128, 128)))
            save_image(fake.data, '%s/%s/fake_samples_epoch_%s.png' % (opt.outf, opt.sub_dir1, epoch), nrow=1)

        # for saving model
        if epoch % 2 == 0:
            torch.save(netRS.state_dict(), pickle_save_path + '/netRS_epoch_%d_%6f.pth' % (epoch, running_results['loss']/running_results['batch_sizes']))
            torch.save(netG2.state_dict(), pickle_save_path + '/netG2_epoch_%d_%6f.pth' % (epoch, running_results['loss'] / running_results['batch_sizes']))
        elif epoch == (opt.num_epochs - 1):
            torch.save(netRS.state_dict(), pickle_save_path + '/netRS_final_%6f.pth' % (running_results['loss'] / running_results['batch_sizes']))
            torch.save(netG2.state_dict(), pickle_save_path + '/netG2_final_%6f.pth' % (running_results['loss'] / running_results['batch_sizes']))
