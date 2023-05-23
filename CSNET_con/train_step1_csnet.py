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
import numpy as np
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from torch.autograd import Variable
from torchvision.utils import save_image
import torchvision.datasets
from torchvision import datasets, transforms
from torch.autograd import Variable
from torchvision.utils import save_image
import torchvision.datasets as dsets
# from calculate_fid_pytorch.fid import fid_score
import vutils

parser = argparse.ArgumentParser()
parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
parser.add_argument('--batchSize', type=int, default=60, help='input batch size')
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=0.0002')
parser.add_argument('--lrRS', type=float, default=0.001, help='learning rate, default=0.00001')
parser.add_argument('--beta1', type=float, default=0.9, help='beta1 for adam. default=0.5')
parser.add_argument('--netG2', default='', help="path to netG2 (to continue training)")
parser.add_argument('--netRS', default='', help="path to netRS (to continue training)")
parser.add_argument('--outf', default='.', help='folder to output images and model checkpoints')
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('--nc', type=int, default=3, help='the number of image channel')
parser.add_argument('--lamb', type=int, default=100, help='weight on L1 term in objective')
parser.add_argument('--sub_dir1', default='imgStep1_csnet_lite_normalize_contrast_64', help='the sub directory 1 of saving images')
parser.add_argument('--num_epochs', default=100, type=int, help='train epoch number')
parser.add_argument('--loadEpoch', default=0, type=int, help='load epoch number')
parser.add_argument('--crop_size', default=96, type=int, help='training images crop size')
parser.add_argument('--block_size', default=32, type=int, help='CS block size')
parser.add_argument('--sub_rate', default=0.75, type=float, help='sampling sub rate')
parser.add_argument('--generatorWeights', type=str, default='', help="path to CSNet weights (to continue training)")
opt = parser.parse_args()

CROP_SIZE = opt.crop_size
BLOCK_SIZE = opt.block_size
SUBRATE = opt.sub_rate
NUM_EPOCHS = opt.num_epochs
LOAD_EPOCH = 0


# 保存图片对比重建前后效果
img_save_path = os.path.join(opt.outf, opt.sub_dir1,'CSNet_normalize_subrate_' + str(opt.sub_rate) + '_blocksize_' + str(BLOCK_SIZE)+'_cropsize_'+str(opt.crop_size))
if not os.path.exists(img_save_path):
    os.makedirs(img_save_path)

pickle_save_path = 'CSNet_normalize_patchsize_96_3tuple' + '_subrate_' + str(opt.sub_rate) + '_blocksize_' + str(BLOCK_SIZE) +'_cropsize_'+str(opt.crop_size)
if not os.path.exists(pickle_save_path):
    os.makedirs(pickle_save_path)

# Use GPU is available else use CPU.
device = torch.device("cuda:0" if(torch.cuda.is_available()) else "cpu")
#print(device, " will be used.\n")


#导入训练数据集
train_set = TrainDatasetFromFolder('E:/CSNET_con/patch_data_96/train', crop_size=CROP_SIZE, blocksize=BLOCK_SIZE)
train_loader = DataLoader(dataset=train_set, num_workers=4, batch_size=opt.batchSize, shuffle=True)
#导入测试数据集
#1024*1024
#data_dir_1024 = 'celeba_1024_lite/celeba_1024'          # this path depends on your computer
#dset_1024 = TestDatasetFromFolder(data_dir_1024, blocksize=BLOCKSIZE)
#512*512
data_dir_64 = 'F:/CSNET_LITE/celebA_hq/celeba_64'          # this path depends on your computer
dset_64 = datasets.ImageFolder(data_dir_64, transform=transforms.Compose([
        #transforms.Scale(IMAGE_SIZE),
        #transforms.Grayscale(num_output_channels=1),#转为单通道灰度图片
        transforms.ToTensor(),
        # ToTensor() returns [0, 1]-ranged tensor.
        # Normalize() performs x' = (x - m)/s for every channel
        # Thus, Normalize((0.5), (0.5)) scales tensor to [-1, 1]
        # as (0-0.5)/0.5=-1, (1-0.5)/0.5=1
        transforms.Normalize((0.5), (0.5))
    ]))
un_norm = transforms.Normalize(
    mean=[-1],
    std=[1 / 0.5]
)



# design model
#32*32*64生成512*512的图片
netG2 = _netG2_block(blocksize=BLOCK_SIZE, subrate=SUBRATE)

#512*512压缩为32*32*64
netRS = _netRS(blocksize=BLOCK_SIZE, subrate=SUBRATE)

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
    schedulerRS = torch.optim.lr_scheduler.MultiStepLR(optimizerRS, milestones=[50, 80], gamma=0.1)
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
        test_img_real = dset_64[0][0].cuda()
        test_img_real_save = un_norm(test_img_real)
        if epoch == 0:
            save_image(test_img_real_save, '%s/real_samples.png' % (img_save_path), nrow=1)

        if epoch % 10 ==0:
            fake = un_norm(netG2(netRS(test_img_real.view(1, 3, 64, 64))))
            save_image(fake.data, '%s/fake_samples_epoch_%s.png' % (img_save_path, epoch), nrow=1)

        # for saving model
        if epoch % 10 == 0:
            torch.save(netRS.state_dict(), pickle_save_path + '/netRS_epoch_%d_%6f.pth' % (epoch, running_results['loss']/running_results['batch_sizes']))
            torch.save(netG2.state_dict(), pickle_save_path + '/netG2_epoch_%d_%6f.pth' % (epoch, running_results['loss'] / running_results['batch_sizes']))
        elif epoch == (opt.num_epochs - 1):
            torch.save(netRS.state_dict(), pickle_save_path + '/netRS_final_%6f.pth' % (running_results['loss'] / running_results['batch_sizes']))
            torch.save(netG2.state_dict(), pickle_save_path + '/netG2_final_%6f.pth' % (running_results['loss'] / running_results['batch_sizes']))



        if epoch ==100:
            epoch_canculate = 1
            psnr_final = np.zeros(epoch_canculate)
            ssim_final = np.zeros(epoch_canculate)
            for p in range(epoch_canculate):
                N = 32
                seed = np.random.randint(1, 10000, size=N)
                psnr_sum = np.zeros(N)
                ssim_sum = np.zeros(N)
                j = 0
                for i in seed:
                    immg = dset_64[i][0].reshape(1, -1)
                    img = dset_64[i][0].reshape(64, 64, 3)
                    img = un_norm(img).numpy()
                    # 进入CSNET查看恢复情况
                    immg = immg.cuda()
                    immg_recon = netG2(netRS(immg.view(1, 3, 64, 64))).reshape(64, 64, 3)
                    img_recon = un_norm(immg_recon)
                    img_recon = img_recon.cpu().detach().numpy()

                    # psnr计算
                    psnr = compare_psnr(img, img_recon, data_range=1)
                    psnr_sum[j] = psnr

                    # ssim计算
                    ssim = compare_ssim(img, img_recon, gaussian_weights=True, multichannel=True, data_range=1)
                    ssim_sum[j] = ssim
                    j = j + 1

                psnr_ave = np.sum(psnr_sum) / N
                psnr_final[p] = psnr_ave
                # print(psnr_ave)
                ssim_ave = np.sum(ssim_sum) / N
                ssim_final[p] = ssim_ave
                # print(ssim_ave)

            psnr_last = np.sum(psnr_final) / epoch_canculate
            print(psnr_last)
            ssim_last = np.sum(ssim_final) / epoch_canculate
            print(ssim_last)



