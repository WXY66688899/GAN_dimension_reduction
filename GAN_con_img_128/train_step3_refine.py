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
from aegan_model import _netG1, _netD1,_netG1_64,_netD1_64, _netG1_64_,_netD2,reshape,recover, AEGAN_ResnetGenerator, AEGAN_ResnetDecoder, \
    NLayerDiscriminator
from csnet import CSNet_Lite, _netG2, _netRS, CSNet,_netG2_block
from refiner_model import _RefinerG_64, _RefinerD_64, UnetGenerator,_RefinerG_128,_RefinerD_128

from tqdm import tqdm
from data_utils import TrainDatasetFromFolder, TestDatasetFromFolder
# from calculate_fid_pytorch.fid import fid_score
import vutils

parser = argparse.ArgumentParser()
parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
parser.add_argument('--batchSize', type=int, default=18, help='input batch size')
parser.add_argument('--imageSize', type=int, default=128, help='the height / width of the input image to network')
parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector')
parser.add_argument('--ngf', type=int, default=64)
parser.add_argument('--ndf', type=int, default=64)
parser.add_argument('--nm', type=int, default=63)

parser.add_argument('--lr', type=float, default=0.0001, help='learning rate, default=0.0002')
parser.add_argument('--lrRS', type=float, default=0.001, help='learning rate, default=0.00001')
parser.add_argument('--beta1', type=float, default=0.9, help='beta1 for adam. default=0.5')
parser.add_argument('--netG2', default='', help="path to netG2 (to continue training)")
parser.add_argument('--netRS', default='', help="path to netRS (to continue training)")
parser.add_argument('--RefinerG', default='', help="path to Refiner (to continue training)")
parser.add_argument('--RefinerD', default='', help="path to RefinerD (to continue training)")
parser.add_argument('--outf', default='img_128_3_tuple_csnet_subrate_0.0625+gan_con_32', help='folder to output images and model checkpoints')
parser.add_argument('--nc', type=int, default=3, help='the number of image channel')
parser.add_argument('--lamb', type=int, default=100, help='weight on L1 term in objective')
parser.add_argument('--sub_dir1', default='imgStep1_csnet_contrast_512', help='the sub directory 1 of saving images')
parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
parser.add_argument('--pickle_dir', type=str, default='csnet_lite', help='input path')
parser.add_argument('--train_stage', type=int, default=1, help='training stage index')
parser.add_argument('--netG1', default='', help="path to netG1 (to continue training)")
parser.add_argument('--netD1', default='', help="path to netD1 (to continue training)")

parser.add_argument('--niter_stage2', type=int, default=200, help='number of epochs to train for G1 and D1')
parser.add_argument('--sub_dir2', default='imgStep2&3_2_nor_128', help='the sub directory 2 of saving images')
parser.add_argument('--sub_dir3', default='imgStep3_nor_128_lr_0.00005', help='the sub directory 3 of saving images')
parser.add_argument('--gan_type', type=str, default='GAN', help='options: GAN | Patch | PatchLSGAN')

parser.add_argument('--loadEpoch', default=0, type=int, help='load epoch number')
parser.add_argument('--crop_size', default=96, type=int, help='training images crop size')
parser.add_argument('--block_size', default=32, type=int, help='CS block size')
parser.add_argument('--sub_rate', default=0.0625, type=float, help='sampling sub rate')
parser.add_argument('--generatorWeights', type=str, default='', help="path to CSNet weights (to continue training)")

opt = parser.parse_args()

CROP_SIZE = opt.crop_size
BLOCK_SIZE = opt.block_size
SUBRATE = opt.sub_rate
LOAD_EPOCH = 0


# set gpu ids
str_ids = opt.gpu_ids.split(',')
opt.gpu_ids = []
for str_id in str_ids:
    id = int(str_id)
    if id >= 0:
        opt.gpu_ids.append(id)
if len(opt.gpu_ids) > 0:
    torch.cuda.set_device(opt.gpu_ids[0])

# 保存生成的图片
model_save_path3 = os.path.join(opt.outf, opt.sub_dir3)

if not os.path.exists(model_save_path3):
    os.makedirs(model_save_path3)

#存储refine模型
refine_save_path = os.path.join(opt.outf, 'only_Refine_lr_0.00005_nor_auto refine_STEP3_128_3tuple' + '_subrate_' + str(opt.sub_rate) + '_blocksize_' + str(BLOCK_SIZE))
if not os.path.exists(refine_save_path):
    os.makedirs(refine_save_path)




nz = int(opt.nz)
ngf = int(opt.ngf)
ndf = int(opt.ndf)
nc = int(opt.nc)
nm = int(opt.nm)


#导入数据512
data_dir_128 = 'celeba-128'
dataroot = data_dir_128
dataset = dset.ImageFolder(root= dataroot, transform=transforms.Compose([
                                   transforms.Scale(opt.imageSize),
                                   transforms.CenterCrop(opt.imageSize),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                               ]))

assert dataset
dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize, shuffle=True, num_workers=int(opt.workers))

#导入压缩感知网络RS G2
#导入压缩感知网络RS G2
# design model
#32*32*64生成512*512的图片
netG2 = _netG2_block(blocksize=BLOCK_SIZE, subrate=SUBRATE)
netG2.load_state_dict(torch.load('./CSNet_normalize_patchsize_96_3tuple_subrate_0.0625_blocksize_32/netG2_final_0.002918.pth'))
#512*512压缩为32*32*64
netRS = _netRS(blocksize=BLOCK_SIZE, subrate=SUBRATE)
netRS.load_state_dict(torch.load('./CSNet_normalize_patchsize_96_3tuple_subrate_0.0625_blocksize_32/netRS_final_0.002918.pth'))

#导入gan
#将noise生成为32*32*64
netG1 = _netG1(nz, ngf,opt.batchSize)
netG1.load_state_dict(torch.load('./img_128_3_tuple_csnet_subrate_0.0625+gan_con_32/GAN_32_nor_STEP2_128_3tuple_subrate_0.0625_blocksize_32/netG1_epoch_101_0.681336.pth'))
# netG1 = AEGAN_ResnetDecoder(nz, 64, n_blocks=1, n_downsampling=2)
# netG1.apply(weights_init)


#print(netD1)

#微调器编码后再恢复（512*512回到512*512）
#压缩感知网络去噪器
#RefinerG = CSNet(blocksize=32, subrate=0.1)
#if opt.generatorWeights != '':
    #RefinerG.load_state_dict(torch.load(opt.generatorWeights))

#自编码去噪器
RefinerG = _RefinerG_128(nc, ngf)
# RefinerG = UnetGenerator(nc, nc, 5)
# RefinerG.apply(weights_init)
RefinerD = _RefinerD_128(nc, ndf)

#print(RefinerD)


#图片变形
RESHAPE = reshape(reshape_size=32)
RECOVER = recover(recover_size=4)

#损失函数
criterion = nn.BCELoss()
criterion_l1 = nn.L1Loss()
criterion_l2 = nn.MSELoss()
criterion_NLL = nn.NLLLoss2d()


# variables
input = torch.FloatTensor(opt.batchSize, 3, opt.imageSize, opt.imageSize)
noise = torch.FloatTensor(opt.batchSize, nz, 1, 1)
#fixed_noise = torch.FloatTensor(opt.batchSize, nz, 1, 1).uniform_(-1, 1)
label = torch.FloatTensor(opt.batchSize)
# label = torch.LongTensor(opt.batchSize)
real_label = 1
fake_label = 0

device = torch.device('cuda:{}'.format(opt.gpu_ids[0]))


netG1.cuda()
netG2.cuda()
netRS.cuda()
RefinerG.cuda()
RefinerD.cuda()
criterion.cuda()
criterion_l1.cuda()
criterion_l2.cuda()
criterion_NLL.cuda()
input, label = input.cuda(), label.cuda()
noise = noise.cuda()

def GANLoss(output, label, bce, mse, gan_type):
    if gan_type == 'GAN' or gan_type == 'Patch':
        err = bce(output, label.expand_as(output))
    elif gan_type == 'PatchLSGAN':
        err = mse(output, label.expand_as(output))
    else:
        assert False, 'unsupported gan type: %s' % gan_type
    return err


input = Variable(input)
label = Variable(label)
noise = Variable(noise)


# setup optimizer
optimizerRefinerG = optim.Adam(RefinerG.parameters(), lr=opt.lr*0.5, betas=(opt.beta1, 0.999))
optimizerRefinerD = optim.Adam(RefinerD.parameters(), lr=opt.lr*0.5, betas=(opt.beta1, 0.999))


if __name__ == '__main__':
    print('start training')
    best_fid = 10000.0
    best_epoch = 0
    counter_img = 0
    counter_refine = 0
    for epoch in range(opt.niter_stage2):
        for i, data in enumerate(dataloader, 0):
            #####################
            # (0) Preprocessing #
            #####################
            # gain the real data
            real_cpu, _ = data
            batch_size = real_cpu.size(0)
            if batch_size < opt.batchSize:
                break
            input.data.resize_(real_cpu.size()).copy_(real_cpu)

            # design noise
            noise = torch.randn(batch_size, nz, 1, 1).cuda()

            RefinerD.zero_grad()
            fake_input = netG2(RECOVER(netG1(noise)))
            # label.data.resize_(batch_size).fill_(real_label)
            label.resize_(1).fill_(real_label)
            output = RefinerD(input)
            # errRefinerD_real = criterion(output, label)
            # errRefinerD_real = criterion(output, label.expand_as(output))
            errRefinerD_real = GANLoss(output, label, criterion, criterion_l2, opt.gan_type)
            # errRefinerD_real.backward(retain_variables=True)
            errRefinerD_real.backward(retain_graph=True)
            RefinerD_x = output.data.mean()
            # train with fake
            fake_refined = RefinerG(fake_input)
            label.data.fill_(fake_label)
            output = RefinerD(fake_refined.detach())
            # errRefinerD_fake = criterion(output, label)
            # errRefinerD_fake = criterion(output, label.expand_as(output))
            errRefinerD_fake = GANLoss(output, label, criterion, criterion_l2, opt.gan_type)
            # errRefinerD_fake.backward(retain_variables=True)
            errRefinerD_fake.backward(retain_graph=True)
            RefinerD_G_z = output.data.mean()
            errRefinerD = errRefinerD_real + errRefinerD_fake
            optimizerRefinerD.step()
            ######################################################
            # (4) Update RefinerG network: maximize log(D(G(z))) #
            ######################################################
            RefinerG.zero_grad()
            # netG1.zero_grad()
            # netG2.zero_grad()
            # netRS.zero_grad()
            # input_hat = netG2(netRS(input))
            # errRS = criterion_l1(input_hat, input)
            # errRS.backward()
            label.data.fill_(real_label)  # fake labels are real for generator cost
            output = RefinerD(fake_refined)
            # errRefinerG = criterion(output, label) + opt.lamb * criterion_l1(fake_refined, fake_input.detach())
            # errRefinerG = criterion(output, label.expand_as(output)) + opt.lamb * criterion_l1(fake_refined, fake_input.detach())
            errRefinerG = GANLoss(output, label, criterion, criterion_l2, opt.gan_type) + opt.lamb * criterion_l1(
                fake_refined, fake_input.detach())
            # errRefinerG = criterion(output, label)
            errRefinerG.backward()
            # optimizerG1.step()
            # optimizerG2.step()
            # optimizerRS.step()
            # RefinerG_z = output.data.mean()
            optimizerRefinerG.step()
            print('Train Refiner: [%d/%d][%d/%d] Loss_RefinerD: %.4f Loss_RefinerG: %.4f D(x): %.4f D(G(z)): %.4f'
                  % (epoch, opt.niter_stage2, i, len(dataloader),
                     errRefinerD.item(), errRefinerG.item(), RefinerD_x, RefinerD_G_z))
            fake = RefinerG(netG2(RECOVER(netG1(noise))))
            # if i % 100 == 0 and opt.dataset!='lsun':
            if i % 100 == 0:
                vutils.save_image(real_cpu,
                                  '%s/%s/real_samples.png' % (opt.outf, opt.sub_dir3), normalize=True)
                vutils.save_image(fake,
                                  '%s/%s/fake_samples_epoch_%03d.png' % (opt.outf, opt.sub_dir3, epoch),
                                  normalize=True)

        # for saving model
        if epoch % 1 == 0:
            torch.save(RefinerD.state_dict(),
                       refine_save_path + '/RefinerD_epoch_%d_%6f.pth' % (epoch, errRefinerD.item()))
            torch.save(RefinerG.state_dict(),
                       refine_save_path + '/RefinerG_epoch_%d_%6f.pth' % (epoch, errRefinerG.item()))

        elif epoch == (opt.niter_stage2 - 1):
            torch.save(RefinerD.state_dict(), refine_save_path + '/RefinerD_final.pth')
            torch.save(RefinerG.state_dict(), refine_save_path + '/RefinerG_final.pth')





